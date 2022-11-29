import numpy as np
import torch
from torch.nn import MSELoss, Conv2d, ReLU, Sequential, Linear, Module
from xenon_crow.common import BasicBuffer


class ConvDuelingDQN(Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = Sequential(
            Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
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
        learning_rate=3e-4,
        gamma=0.99,
        eps=0.20,
        buffer_size=10000,
        device="cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device(device)

        self.model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = MSELoss()

    def __update_eps(self):
        pass

    @torch.no_grad()
    def get_action(self, state):

        self.__update_eps()
        if np.random.randn() < self.eps:
            qvals = self.model(state)
            return np.argmax(qvals.cpu().detach().numpy())

        return self.env.action_space.sample()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        curr_Q = self.model(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        # The previous implementation didn't used dones here
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * dones

        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update(self):
        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
