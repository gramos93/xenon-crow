from random import choice

import numpy as np
import torch
from torch.nn import Conv2d, Linear, Module, ModuleDict, MSELoss, ReLU, Sequential


class MplDuelingDQN(Module):

    def __init__(self, input_dim, output_dim):
        super(MplDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feauture_layer = Sequential(
            Linear(self.input_dim[0], 128),
            ReLU(),
            Linear(128, 128),
            ReLU()
        )

        self.value_stream = Sequential(
            Linear(128, 128),
            ReLU(),
            Linear(128, 1)
        )

        self.advantage_stream = Sequential(
            Linear(128, 128),
            ReLU(),
            Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

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


class DuelingDQNAgent(Module):
    def __init__(
        self,
        env,
        buffer,
        learning_rate=3e-4,
        gamma=0.99,
        format='conv',
    ):
        super(DuelingDQNAgent, self).__init__()
        self.env = env
        self.gamma = gamma
        self._step = 0
        self.replay_buffer = buffer
        model_class = ConvDuelingDQN if format == "conv" else MplDuelingDQN
        self.model = ModuleDict(
            {
                "A": model_class(env.observation_space.shape, env.action_space.n),
                "B": model_class(env.observation_space.shape, env.action_space.n),
            }
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = MSELoss()

        self._train = True

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
        local = choice(["A", "B"])
        for model in self.model.keys():
            if model == local:
                target = model

        with torch.set_grad_enabled(self.train):
            curr_Q = self.model[local](states).gather(1, actions)

            max_next_Q = self.model[target](next_states).argmax(1, keepdims=True)

            # The previous implementation didn't used dones here
            expected_Q = rewards + self.gamma * max_next_Q * dones
            loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self):
        self.optimizer.zero_grad()

        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
