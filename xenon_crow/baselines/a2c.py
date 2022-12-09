from collections import deque
import torch
from torch.distributions import Categorical
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential, SiLU, Softmax, LayerNorm


class MplActorCritic(Module):
    def __init__(self, input_dim, output_dim):
        super(MplActorCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_layer = Sequential(
            Linear(input_dim, 64),
            LayerNorm(64),
            SiLU(),
            Linear(64, 128),
            LayerNorm(128),
            SiLU(),
            Linear(128, 128),
            LayerNorm(128),
            SiLU(),
        )
        self.actor_layer = Sequential(
            Linear(128, 128),
            LayerNorm(128),
            SiLU(),
            Linear(128, output_dim),
            Softmax(dim=1),
        )
        self.critic_layer = Sequential(Linear(128, 128), SiLU(), Linear(128, 1))

    def forward(self, state):
        features = self.feature_layer(state)
        log_prob = self.actor_layer(features)
        value = self.critic_layer(features)
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
        critic_loss_mult=0.8,
        buffer=None,
        format="conv",
    ):
        super(A2CAgent, self).__init__()
        self.gamma = gamma
        self.crt_mult = critic_loss_mult
        self.replay_buffer = buffer
        self._eps = torch.finfo(torch.float32).eps
        model_class = ConvActorCritic if format == "conv" else MplActorCritic
        self.model = model_class(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        probs, value = self.model(state)
        action_pool = Categorical(probs)
        action = action_pool.sample()
        log_prob = action_pool.log_prob(action)
        entropy = action_pool.entropy().sum()
        # entropy = -(torch.mean(probs) * torch.log(probs)).sum()
        return action.item(), log_prob, entropy, value

    def __compute_returns(self, rewards):
        returns = deque(maxlen=len(rewards))
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.appendleft(R)

        # Use normilized rewards
        rewards = torch.tensor(returns)
        return (rewards - rewards.mean()) / (rewards.std() + self._eps)

    def __compute_loss(self, entropy):

        log_prob, values, rewards = self.replay_buffer.get_episode()
        Gs = self.__compute_returns(rewards)

        actor_loss, critic_loss = [], []
        for log_prob, R, val in zip(log_prob, Gs, values):
            adv = R - val
            critic_loss.append(adv**2)
            actor_loss.append(-log_prob * adv)

        actor_loss = torch.cat(actor_loss).sum()
        critic_loss = self.crt_mult * torch.cat(critic_loss).sum()

        return actor_loss + critic_loss + 0.001 * entropy

    def update(self, entropy):

        self.optimizer.zero_grad()
        loss = self.__compute_loss(entropy)
        loss.backward()
        self.optimizer.step()
        self.replay_buffer.reset()

        return loss.item()
