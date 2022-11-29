import torch
from torch.nn import MSELoss
from xenon_crow.common import BasicBuffer

def DQNRunner(environment, config):
    model =DuelingAgent()


class ConvDuelingDQN():
    pass

class MlpDuelingDQN():
    pass

class DuelingAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = MlpDuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)


        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = MSELoss()

