import sys
from pathlib import Path
# Add xenon_crow path to python module search path.
sys.path.append(
    str(Path(__file__).parent.resolve().parent)
)

import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm.auto import trange

from torch import tensor, manual_seed, float32, save, load
from xenon_crow.baselines import DuelingDQNAgent
from xenon_crow.common import RandomBuffer
from utils import GymDataHandler

sns.set_style("white")

ENV = gym.make("LunarLander-v2", render_mode="rgb_array")
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
    format="mlp"
)


def run(env, agent, max_episodes, update_inter):
    episode_rewards = []
    progress_bar = trange(
        max_episodes,
        ncols=150,
        desc="Training",
        position=0,
        leave=True
    )
    for _ in progress_bar:
        step, reward = 1, 0.0
        terminated = False
        state, _ = env.reset()

        while not terminated:
            action = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
            next_state, r, done, trunc, *_ = env.step(action)
            terminated = done or trunc

            agent.replay_buffer.push(
                (state, action, r, next_state, terminated)
            )
            reward += r

            if agent.replay_buffer.ready() and step % update_inter == 0 :
                agent.update()

            step += 1
            state = next_state
        
        agent.update_epsilon(agent.epsilon*0.99)
        episode_rewards.append(reward)
        progress_bar.set_postfix(reward=reward, epsilon=agent.epsilon, refresh=True)

    return episode_rewards

seed = 42
ENV.seed = seed
np.random.seed(seed)
manual_seed(seed)

hist = run(ENV, agent, MAX_EP, TRAIN_INTER)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
x = np.arange(1, len(hist) + 1)
sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax)

ax.set_ylabel("Cumulative Reward")
ax.set_xlabel("Episodes")
ax.grid(visible=True, axis="y", linestyle="--")

fig.savefig("LunarLander-v2_D3QN.png")
save(agent.model_target.state_dict(), "./models/LunarLander_D3QN_Target.pth")
save(agent.model_local.state_dict(), "./models/LunarLander_D3QN_Local.pth")

# agent.model_local.load_state_dict(
#     load("./models/LunarLander_D3QN_Local.pth"),
#     strict=True
# )
agent.update_epsilon(0.)

def save_frames_as_gif(frames, path='./assets/', filename='D3QN-LunarLander.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='ffmpeg', fps=60)


#Run the env
state, _ = ENV.reset()
frames = []
rewards = 0
for t in range(1000):
    #Render to frames buffer
    frames.append(ENV.render())
    action = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
    next_s, r, done, trunc, _ = ENV.step(action)
    rewards += r

    if done or trunc:
        break    
    state = next_s

ENV.close()
print(f"Test reward {rewards}")
save_frames_as_gif(frames)
