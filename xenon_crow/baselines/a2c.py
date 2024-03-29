from collections import deque
import torch
from torch.distributions import Categorical
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential, SiLU, Softmax, LayerNorm, Identity
from torchvision.models import resnet18


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
        self.features_layer = resnet18()
        self.features_layer.conv1 = Conv2d(
            input_dim[1],
            self.features_layer.conv1.out_channels,
            self.features_layer.conv1.kernel_size,
            self.features_layer.conv1.stride,
            self.features_layer.conv1.padding,
        )
        self.features_layer.fc = Identity()

        self.fc_input_dim = self.feature_size()
        self.critic_layer = Sequential(
            Linear(self.fc_input_dim, 128), ReLU(), Linear(128, 1)
        )
        self.actor_layer = Sequential(
            Linear(self.fc_input_dim, 128), 
            LayerNorm(128),
            SiLU(), 
            Linear(128, output_dim), 
            Softmax(dim=1)
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
        features = self.features_layer(state)
        log_prob = self.actor_layer(features)
        value = self.critic_layer(features)
        return log_prob, value

    @torch.no_grad()
    def feature_size(self):
        return (
            self.features_layer(torch.zeros(self.input_dim, requires_grad=False))
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
        self.device = "cuda"
        self.model = model_class(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

    def get_action(self, state):
        probs, value = self.model(state.to(self.device))
        action_pool = Categorical(probs.cpu())
        action = action_pool.sample()
        log_prob = action_pool.log_prob(action)
        entropy = action_pool.entropy().sum()
        # entropy = -(torch.mean(probs) * torch.log(probs)).sum()
        return action.item(), log_prob, entropy, value.cpu()

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

        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = self.crt_mult * torch.stack(critic_loss).sum()

        return actor_loss + critic_loss + 0.001 * entropy

    def update(self, entropy):

        self.optimizer.zero_grad()
        loss = self.__compute_loss(entropy)
        loss.backward()
        self.optimizer.step()
        self.replay_buffer.reset()

        return loss.item()
