import torch
from torch.distributions import Categorical
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential, SiLU, Softmax
from ..common import ReinforceBuffer, ReinforceHandler

class MplActorCritic(Module):
    def __init__(self, input_dim, output_dim):
        super(MplActorCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_layer = Sequential(
            Linear(self.input_dim, 64),
            SiLU(),
            Linear(64, 128),
            SiLU(),
            Linear(128, 128),
            SiLU(),
        )
        self.actor_head = Sequential(
            Linear(128, 128), SiLU(), Linear(128, self.output_dim), Softmax(dim=1)
        )
        self.critic_head = Sequential(Linear(128, 128), SiLU(), Linear(128, 1))

    def forward(self, state):
        features = self.feature_layer(state)
        log_prob = self.actor_head(features)
        value = self.critic_head(features)
        return log_prob, value


class ConvActorCritic(Module):
    def __init__(self, input_dim, output_dim):
        super(ConvActorCritic, self).__init__()
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
        self.activation = Softmax(dim=1)

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


class A2CAgent(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate=3e-4,
        gamma=0.99,
        format="conv",
    ):
        # REF: https://github.com/Nithin-Holla/reinforce_baselines
        # https://github.com/pytorch/examples/blob/main/reinforcement_learning
        super(A2CAgent, self).__init__()
        self.gamma = gamma
        self._replay_buffer = ReinforceBuffer(data_handler=ReinforceHandler())
        self._eps = torch.finfo(torch.float32).eps.item()
        model_class = ConvActorCritic if format == "conv" else MplActorCritic
        self.model = model_class(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(
            self.model_local.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        probs, value = self.model(state)
        action_pool = Categorical(probs)
        action = action_pool.sample()
        log_prob = action_pool.log_prob(action)
        entropy = - (torch.mean(probs) * torch.log(probs)).sum()

        return action.item(), log_prob, entropy, value

    def __compute_returns(self, rewards):
        gammas = self.gamma ** (torch.arange(0, len(rewards), 1))
        rewards = (rewards * gammas).cumsum()
        # Use normilized rewards
        return (rewards - rewards.mean()) / rewards.std()

    def __compute_loss(self, entropy):
        log_prob, values, rewards = self._replay_buffer.get_episode()
        
        advantage = self.__compute_returns(rewards) - values
        
        actor_loss = (-log_prob * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        
        return actor_loss + critic_loss + 0.001 * entropy

    def update(self, entropy):

        self.optimizer.zero_grad()
        loss = self.__compute_loss(entropy)
        loss.backward()
        self.optimizer.step()
        self._replay_buffer.reset()