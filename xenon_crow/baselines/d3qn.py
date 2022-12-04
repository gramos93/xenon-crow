import numpy as np
import torch
from torch.nn import Conv2d, Linear, Module, ModuleDict, MSELoss, ReLU, Sequential, SiLU


class MplDuelingDQN(Module):
    def __init__(self, input_dim, output_dim):
        super(MplDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feauture_layer = Sequential(
            Linear(self.input_dim[0], 64),
            SiLU(),
            Linear(64, 128),
            SiLU(),
            Linear(128, 128),
            SiLU(),
        )
        self.value_stream = Sequential(Linear(128, 128), SiLU(), Linear(128, 1))
        self.advantage_stream = Sequential(
            Linear(128, 128), SiLU(), Linear(128, self.output_dim)
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
        tau=0.001,
        epsilon=0.99,
        format="conv",
    ):
        super(DuelingDQNAgent, self).__init__()
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.replay_buffer = buffer
        model_class = ConvDuelingDQN if format == "conv" else MplDuelingDQN
        self.model = ModuleDict(
            {
                "local": model_class(env.observation_space.shape, env.action_space.n),
                "target": model_class(env.observation_space.shape, env.action_space.n),
            }
        )
        self.optimizer = torch.optim.Adam(
            self.model["local"].parameters(), lr=learning_rate
        )
        self.MSE_loss = MSELoss()

        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def set_train(self, mode: bool = True):
        self._train = mode

    def update_epsilon(self):
        self.epsilon *= 0.99

    def __update_target_model(self):
        local_weights = self.model["local"].state_dict()
        target_weights = self.model["target"].state_dict()
        for w in local_weights:
            target_weights[w] = (1 - self.tau) * \
                target_weights[w] + self.tau * local_weights[w]

        self.model["target"].load_state_dict(target_weights, strict=True)

    @torch.no_grad()
    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()

        qvals = self.model["local"](state)
        return np.argmax(qvals.cpu().detach().numpy())

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        with torch.set_grad_enabled(self._train):
            curr_Q = self.model["local"](states).gather(1, actions)

            max_next_Q = self.model["target"](next_states).argmax(1, keepdims=True)

            expected_Q = rewards + self.gamma * max_next_Q * dones
            loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self):
        self.optimizer.zero_grad()

        batch = self.replay_buffer.sample()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.__update_target_model()
