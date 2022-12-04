import sys

# Add xenon_crow path to python module search path.
sys.path.append("/home/gramos/Projects/IFT-7201/project/xenon-crow/")

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch import FloatTensor
from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import BasicBuffer, D3QNTrainer
from utils import GymDataHandler

sns.set_style("white")

ENV = gym.make("LunarLander-v2")
MAX_EP = 2000
MAX_STEPS = 1000

GAMMA = 0.99
LR = 1e-3

BUFFER_SIZE = 1e5
BATCH_SIZE = 64

DEVICE = "cpu"

replay_buffer = BasicBuffer(
    max_size=BUFFER_SIZE, batch_size=BATCH_SIZE, data_handler=GymDataHandler()
)
agent = DuelingDQNAgent(ENV, replay_buffer, LR, GAMMA, "mlp")


def run(env, agent, max_episodes, max_steps):
    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action, info = agent.get_action(FloatTensor(state))
            next_state, reward, done, trunc, info = env.step(action)
            agent.replay_buffer.push(
                (state, action, reward, next_state, done or trunc, info)
            )
            episode_reward += reward

            if agent.replay_buffer.ready():
                agent.update()

            if done or trunc:
                episode_rewards.append(episode_reward)
                print(f"[INFO]: Episode {episode}: {episode_reward}")
                break

            state = next_state

    return episode_rewards


hist = run(ENV, agent, MAX_EP, MAX_STEPS)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
x = np.arange(1, len(hist) + 1)
sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax[0])

ax[0].set_ylabel("Cumulative Reward")
ax[0].set_xlabel("Episodes")
ax[0].grid(visible=True, axis="y", linestyle="--")

fig.savefig("LunarLander-v2_D3QN.png")
