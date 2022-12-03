import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import D3QNTrainer

sns.set_style('white')

ENV = gym.make("LunalLander-v2")
MAX_EP = 2000
MAX_STEPS = 1000

GAMMA = 0.99
LR = 1e-3

BUFFER_SIZE = 1e5
BATCH_SIZE = 64

DEVICE = 'cpu'

agent = DuelingDQNAgent(ENV, LR, GAMMA, BUFFER_SIZE, DEVICE)

trainer = D3QNTrainer()
hist = trainer.run(ENV, agent, MAX_EP, MAX_STEPS)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
x = np.arange(1, len(hist) + 1)
sns.lineplot(y=hist, x=x, color='k', linewidth=1, ax=ax[0])

ax[0].set_ylabel("Cumulative Reward")
ax[0].set_xlabel("Episodes")
ax[0].grid(visible=True, axis='y', linestyle='--')

fig.savefig("LunarLander-v2_D3QN.png")