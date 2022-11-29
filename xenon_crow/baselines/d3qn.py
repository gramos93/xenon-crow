import numpy as np
import torch
from torch.nn import MSELoss, Conv2d, ReLU, Sequential, Linear, Module
from torch.autograd import autograd
from xenon_crow.common import BasicBuffer


def DQNRunner(environment, config):
    model = DuelingAgent()


class ConvDuelingDQN(Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = Sequential(
            Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU(),
        )

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

    def feature_size(self):
        return (self.conv(
            autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)
        )


class DuelingAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device(device)

        self.model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = MSELoss()

    @torch.no_grad()
    def get_action(self, state, eps=0.20):
        qvals = self.model(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if np.random.randn() > eps:
            return self.env.action_space.sample()
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update(self):
        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
