import sys
# Add xenon_crow path to python module search path.
sys.path.append("../")

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import BasicBuffer, D3QNTrainer

sns.set_style("white")

ENV = gym.make("LunarLander-v2")
MAX_EP = 2000
MAX_STEPS = 1000

GAMMA = 0.99
LR = 1e-3

BUFFER_SIZE = 1e5
BATCH_SIZE = 64

DEVICE = "cpu"

replay_buffer = BasicBuffer(max_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
agent = DuelingDQNAgent(ENV, LR, GAMMA, replay_buffer, DEVICE)

trainer = D3QNTrainer()
hist = trainer.run(ENV, agent, MAX_EP, MAX_STEPS)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
x = np.arange(1, len(hist) + 1)
sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax[0])

ax[0].set_ylabel("Cumulative Reward")
ax[0].set_xlabel("Episodes")
ax[0].grid(visible=True, axis="y", linestyle="--")

fig.savefig("LunarLander-v2_D3QN.png")
