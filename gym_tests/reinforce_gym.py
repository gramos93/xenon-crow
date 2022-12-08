import sys
from pathlib import Path

# Add xenon_crow path to python module search path.
sys.path.append(str(Path(__file__).parent.resolve().parent))

import gym
import numpy as np
from torch import float32, manual_seed, save, tensor
from utils import GymDataHandlerReinforce, plot_and_save, save_frames_as_gif

from xenon_crow.baselines import ReinforceAgent
from xenon_crow.common import ReinforceBuffer
from xenon_crow.common import ReinforceTrainer

seed = 42
gym_name = "LunarLander-v2"
ENV = gym.make(gym_name, render_mode="rgb_array")
ENV.seed = seed
np.random.seed(seed)
manual_seed(seed)

MAX_EP = 1200

GAMMA = 0.95
LR = 1e-4

replay_buffer = ReinforceBuffer(
    data_handler=GymDataHandlerReinforce()
)
agent = ReinforceAgent(
    input_dim=ENV.observation_space.shape[0],
    output_dim=ENV.action_space.n,
    learning_rate=LR,
    gamma=GAMMA,
    buffer=replay_buffer,
    format="mlp",
)

trainer = ReinforceTrainer()

hist = trainer.run(ENV, agent, MAX_EP)
save(agent.model.state_dict(), f"./models/{gym_name}_REINFORCE_agent.pth")
plot_and_save(hist, f"Reinforce-{gym_name}.png")

# Run the env
state, _ = ENV.reset()
frames = []
rewards = 0
for t in range(1000):
    # Render to frames buffer
    frames.append(ENV.render())
    action, *_ = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
    next_s, r, done, trunc, _ = ENV.step(action)
    rewards += r

    if done or trunc:
        break
    state = next_s

ENV.close()
save_frames_as_gif(frames, f"Reinforce-{gym_name}.gif")
print(f"Test reward {rewards}")
