import sys
from pathlib import Path

# Add xenon_crow path to python module search path.
sys.path.append(str(Path(__file__).parent.resolve().parent))

import gym
import numpy as np
from torch import float32, manual_seed, save, tensor
from tqdm.auto import trange
from utils import GymDataHandler, plot_and_save, save_frames_as_gif

from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import RandomBuffer

seed = 42
ENV = gym.make("LunarLander-v2", render_mode="rgb_array")
ENV.seed = seed
np.random.seed(seed)
manual_seed(seed)

MAX_EP = 1200

GAMMA = 0.99
LR = 5e-4
TAU = 5e-3
EPS = 1.0

TRAIN_INTER = 4
BUFFER_SIZE = 1e5
BATCH_SIZE = 64

DEVICE = "cpu"

replay_buffer = RandomBuffer(
    max_size=BUFFER_SIZE, batch_size=BATCH_SIZE, data_handler=GymDataHandler()
)
agent = DuelingDQNAgent(
    input_dim=ENV.observation_space.shape[0],
    output_dim=ENV.action_space.n,
    buffer=replay_buffer,
    learning_rate=LR,
    gamma=GAMMA,
    tau=TAU,
    epsilon=EPS,
    format="mlp",
)


def run(env, agent, max_episodes, update_inter):
    episode_rewards = []
    progress_bar = trange(
        max_episodes, ncols=150, desc="Training", position=0, leave=True
    )
    for _ in progress_bar:
        step, reward = 1, 0.0
        terminated = False
        state, _ = env.reset()

        while not terminated:
            action = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
            next_state, r, done, trunc, *_ = env.step(action)
            terminated = done or trunc

            agent.replay_buffer.push((state, action, r, next_state, terminated))
            reward += r

            if agent.replay_buffer.ready() and step % update_inter == 0:
                agent.update()

            step += 1
            state = next_state

        agent.update_epsilon(agent.epsilon * 0.99)
        episode_rewards.append(reward)
        progress_bar.set_postfix(reward=reward, epsilon=agent.epsilon, refresh=True)

    return episode_rewards


hist = run(ENV, agent, MAX_EP, TRAIN_INTER)
save(agent.model_target.state_dict(), "./models/LunarLander_D3QN_Target.pth")
save(agent.model_local.state_dict(), "./models/LunarLander_D3QN_Local.pth")
plot_and_save(hist, "D3QN-LunarLander.png")


# Run the env
state, _ = ENV.reset()
frames = []
rewards = 0
agent.update_epsilon(0.0)
for t in range(1000):
    # Render to frames buffer
    frames.append(ENV.render())
    action = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
    next_s, r, done, trunc, _ = ENV.step(action)
    rewards += r

    if done or trunc:
        break
    state = next_s

ENV.close()
save_frames_as_gif(frames, "'D3QN-LunarLander.gif")
print(f"Test reward {rewards}")
