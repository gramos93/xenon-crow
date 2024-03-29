import numpy as np
import torch
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential, SiLU, Identity
from torch.nn.functional import mse_loss
from torchvision.models import resnet18


class MplDuelingDQN(Module):
    def __init__(self, input_dim, output_dim):
        super(MplDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feauture_layer = Sequential(
            Linear(self.input_dim, 64),
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

    def get_weights(self):
        return self.state_dict()

    def get_weight_copies(self):
        weights = self.get_weights()
        for k in weights.keys():
            weights[k] = weights[k].cpu().clone()
        return weights

    def set_weights(self, weights, strict=True):
        return self.load_state_dict(weights, strict=strict)

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
        self.model = resnet18()
        self.model.conv1 = Conv2d(
            input_dim[1],
            self.model.conv1.out_channels,
            self.model.conv1.kernel_size,
            self.model.conv1.stride,
            self.model.conv1.padding,
        )
        self.model.fc = Identity()

        self.fc_input_dim = self.feature_size()

        self.value_stream = Sequential(
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, 1)
        )

        self.advantage_stream = Sequential(
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, self.output_dim)
        )

    def get_weights(self):
        return self.state_dict()

    def get_weight_copies(self):
        weights = self.get_weights()
        for k in weights.keys():
            weights[k] = weights[k].cpu().clone()
        return weights

    def set_weights(self, weights, strict=True):
        return self.load_state_dict(weights, strict=strict)

    def forward(self, state):
        features = self.model(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    @torch.no_grad()
    def feature_size(self):
        return (
            self.model(torch.zeros(self.input_dim, requires_grad=False))
            .view(1, -1)
            .size(1)
        )


class DuelingDQNAgent(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        buffer,
        learning_rate=3e-4,
        gamma=0.99,
        tau=1e-3,
        epsilon=1.0,
        format="conv",
    ):
        super(DuelingDQNAgent, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.replay_buffer = buffer
        self.actions = np.arange(0, output_dim, 1)
        self.device = "cuda"

        model_class = ConvDuelingDQN if format == "conv" else MplDuelingDQN
        self.model_local = model_class(input_dim, output_dim).to(self.device)
        self.model_target = model_class(input_dim, output_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model_local.parameters(), lr=learning_rate
        )

    def update_epsilon(self, eps):
        if eps > 0.0:
            self.epsilon = max(0.1, eps)
        else:
            self.epsilon = 0.0

    def __update_target_model(self):

        new_weights = {}
        local_weights = self.model_local.state_dict()
        target_weights = self.model_target.state_dict()
        for w in target_weights:
            new_weights[w] = (
                (1 - self.tau) * target_weights[w]
            ) + self.tau * local_weights[w]

        self.model_target.set_weights(new_weights)

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            qvals = self.model_local(state.to(self.device)).cpu()
            return torch.argmax(qvals, dim=1).item()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, not_dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        not_dones = not_dones.to(self.device)

        curr_Q = self.model_local(states).gather(1, actions)
        next_Q = self.model_target(next_states)
        target_Q = rewards + self.gamma * next_Q * not_dones

        return mse_loss(curr_Q, target_Q.max(1, keepdims=True).values)

    def update(self):

        batch = self.replay_buffer.sample()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        self.__update_target_model()

        return loss.item()
