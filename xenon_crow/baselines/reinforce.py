import torch
from torch.distributions import Categorical
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential, SiLU, LogSoftmax
from ..common import ReinforceBuffer, ReinforceHandler

class MplReinforce(Module):
    def __init__(self, input_dim, output_dim):
        super(MplReinforce, self).__init__()
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
        self.policy_layer = Sequential(
            Linear(128, 128), SiLU(), Linear(128, self.output_dim), LogSoftmax(dim=1)
        )
        self.value_layer = Sequential(Linear(128, 128), SiLU(), Linear(128, 1))

    def forward(self, state):
        features = self.feauture_layer(state)
        log_prob = self.policy_layer(features)
        value = self.value_layer(features)
        return log_prob, value


class ConvReinforce(Module):
    def __init__(self, input_dim, output_dim):
        super(ConvReinforce, self).__init__()
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
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, self.output_dim)
        )
        self.activation = LogSoftmax(dim=1)

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
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        return self.activation(values)

    @torch.no_grad()
    def feature_size(self):
        return (
            self.conv(torch.zeros(1, *self.input_dim, requires_grad=False))
            .view(1, -1)
            .size(1)
        )


class ReinforceAgent(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate=3e-4,
        gamma=0.99,
        format="conv",
    ):
        # REF: https://github.com/Nithin-Holla/reinforce_baselines
        super(ReinforceAgent, self).__init__()
        self.gamma = gamma
        self._replay_buffer = ReinforceBuffer(data_handler=ReinforceHandler())
        self._eps = torch.finfo(torch.float32).eps.item()
        model_class = ConvReinforce if format == "conv" else MplReinforce
        self.model = model_class(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(
            self.model_local.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        probs, value = self.model(state)
        action_pool = Categorical(probs)
        action = action_pool.sample()
        log_prob = action_pool.log_prob(action)
        return action.item(), log_prob, value

    def __compute_returns(self, rewards):
        gammas = self.gamma ** (torch.arange(0, len(rewards), 1))
        rewards = (rewards * gammas).cumsum()
        # Use normilized rewards
        return (rewards - rewards.mean()) / rewards.std()

    def __compute_loss(self):
        log_prob, baselines, rewards = self._replay_buffer.get_episode()
        Gs = self.__compute_returns(rewards) - baselines
        Gs /= Gs.abs().mean()

        return -(Gs * log_prob).sum()

    def update(self):

        self.optimizer.zero_grad()
        loss = self.__compute_loss()
        loss.backward()
        self.optimizer.step()
        self._replay_buffer.reset()
