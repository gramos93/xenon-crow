import sys
from pathlib import Path

# Add xenon_crow path to python module search path.
sys.path.append(str(Path(__file__).parent.resolve().parent))

import gym
import numpy as np
from torch import float32, manual_seed, save, tensor

from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import RandomBuffer, D3QNTrainer, XenonCrowEnv, XenonDataHandler

seed = 42
gym_name = "Thermal"
ENV = XenonCrowEnv("./data/train_4")
ENV.seed = seed
np.random.seed(seed)
manual_seed(seed)

MAX_EP = 100

GAMMA = 0.99
LR = 5e-4
TAU = 5e-3
EPS = 1.0

TRAIN_INTER = 4
BUFFER_SIZE = 50
BATCH_SIZE = 16

replay_buffer = RandomBuffer(
    max_size=BUFFER_SIZE, batch_size=BATCH_SIZE, data_handler=XenonDataHandler()
)
agent = DuelingDQNAgent(
    input_dim=ENV.observation_space.shape,
    output_dim=ENV.action_space.n,
    buffer=replay_buffer,
    learning_rate=LR,
    gamma=GAMMA,
    tau=TAU,
    epsilon=EPS,
    format="conv",
)

trainer = D3QNTrainer()
hist = trainer.run(ENV, agent, MAX_EP, TRAIN_INTER)
save(agent.model_target.state_dict(), f"./models/{gym_name}_D3QN_Target.pth")
save(agent.model_local.state_dict(), f"./models/{gym_name}_D3QN_Local.pth")
