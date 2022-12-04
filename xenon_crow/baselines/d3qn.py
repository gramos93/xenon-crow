from random import choice

import numpy as np
import torch
from torch.nn import Conv2d, Linear, Module, ModuleDict, MSELoss, ReLU, Sequential

from ..common import BasicBuffer


class ConvDuelingDQN(Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = Sequential(
            Conv2d(input_dim[0], 32, kernel_size=5, stride=1),
            ReLU(),
            Conv2d(32, 64, kernel_size=5, stride=1),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU(),
        )

        self.fc_input_dim = self.feature_size()

        self.value_stream = Sequential(
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, 1)
        )

        self.advantage_stream = Sequential(
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    @torch.no_grad()
    def feature_size(self):
        return (
            self.conv(torch.zeros(1, *self.input_dim, requires_grad=False))
            .view(1, -1)
            .size(1)
        )


class DuelingDQNAgent:
    def __init__(
        self,
        env,
        buffer,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu",
    ):
        self.env = env
        self.gamma = gamma
        self._step = 0
        self.replay_buffer = buffer
        self.device = torch.device(device)
        self.model = ModuleDict(
            {
                "A": ConvDuelingDQN(env.observation_space.shape, env.action_space.n),
                "B": ConvDuelingDQN(env.observation_space.shape, env.action_space.n),
            }
        ).to(self.device)
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = MSELoss()

        self.set_train()

    @property
    def train(self):
        return self._train

    @train.setter
    def set_train(self, mode: bool = True):
        self._train = mode

    @torch.no_grad()
    def get_action(self, state):
        model = choice(["A", "B"])
        qvals = self.model[model](state)
        return np.argmax(qvals.cpu().detach().numpy()), {"model", model}

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones, info = batch

        for model in self.model.keys():
            if model == info["model"]:
                local = model
            else:
                target = model

        with torch.set_grad_enabled(self.train):
            curr_Q = self.model[local](states).gather(1, actions.unsqueeze(1))
            curr_Q = curr_Q.squeeze(1)

            next_Q = self.model[target](next_states)
            max_next_Q = torch.max(next_Q, 1)[0]

            # The previous implementation didn't used dones here
            expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * dones
            loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self):
        self.optimizer.zero_grad()

        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
