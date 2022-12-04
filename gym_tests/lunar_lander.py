import sys
from pathlib import Path
# Add xenon_crow path to python module search path.
sys.path.append(
    str(Path(__file__).parent.resolve().parent)
)

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm.auto import trange

from torch import FloatTensor
from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import BasicBuffer
from utils import GymDataHandler

sns.set_style("white")

ENV = gym.make("LunarLander-v2")
MAX_EP = 2000
MAX_STEPS = 1000

GAMMA = 1.0
LR = 1e-3
TAU = 0.001
EPS = 1.0

BUFFER_SIZE = 1e5
BATCH_SIZE = 64

DEVICE = "cpu"

replay_buffer = BasicBuffer(
    max_size=BUFFER_SIZE, batch_size=BATCH_SIZE, data_handler=GymDataHandler()
)
agent = DuelingDQNAgent(
    ENV,
    replay_buffer,
    LR,
    GAMMA,
    TAU,
    EPS,
    "mlp"
)


def run(env, agent, max_episodes, max_steps):
    episode_rewards = []
    progress_bar = trange(
        max_episodes,
        ncols=150,
        desc="Training",
        position=0,
        leave=True
    )
    for _ in progress_bar:
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = agent.get_action(FloatTensor(state))
            next_state, reward, done, trunc, _ = env.step(action)
            agent.replay_buffer.push(
                (state, action, reward, next_state, done or trunc)
            )
            episode_reward += reward

            if agent.replay_buffer.ready():
                agent.update()

            if done or trunc:
                break

            state = next_state
        
        agent.update_epsilon()
        episode_rewards.append(episode_reward)
        progress_bar.set_postfix(reward = episode_reward, refresh=True)

    return episode_rewards


hist = run(ENV, agent, MAX_EP, MAX_STEPS)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
x = np.arange(1, len(hist) + 1)
sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax)

ax.set_ylabel("Cumulative Reward")
ax.set_xlabel("Episodes")
ax.grid(visible=True, axis="y", linestyle="--")

fig.savefig("LunarLander-v2_D3QN.png")
