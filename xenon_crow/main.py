import sys
from pathlib import Path
import argparse

# Add xenon_crow path to python module search path.
sys.path.append(str(Path(__file__).parent.resolve().parent))

import numpy as np
from torch import manual_seed, save

from xenon_crow.baselines import DuelingDQNAgent, ReinforceAgent, A2CAgent
from xenon_crow.common import (
    RandomBuffer,
    ReinforceBuffer,
    D3QNTrainer,
    ReinforceTrainer,
    A2CTrainer,
    XenonCrowEnv,
    XenonDataHandler,
    XenonDataHandlerReinforce,
    plot_and_save,
)


parser = argparse.ArgumentParser(description="possible choices: d3qn, reinforce, a2c.")
parser.add_argument("algorithm")

seed = 42
gym_name = "Thermal"
ENV = XenonCrowEnv("./data/train")
ENV.seed = seed
np.random.seed(seed)
manual_seed(seed)

MAX_EP = 100

GAMMA = 0.99
LR = 1e-4
TAU = 1e-3
EPS = 1.0

TRAIN_INTER = 4
BUFFER_SIZE = 400
BATCH_SIZE = 64


def call_d3qn():
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
    save(agent.model_target.state_dict(), f"models/{gym_name}_D3QN_Target.pth")
    save(agent.model_local.state_dict(), f"models/{gym_name}_D3QN_Local.pth")
    plot_and_save(hist, figname="Xenon-D3QN.png")


def call_reinforce():
    replay_buffer = ReinforceBuffer(
        data_handler=XenonDataHandlerReinforce(),
    )
    agent = ReinforceAgent(
        input_dim=ENV.observation_space.shape,
        output_dim=ENV.action_space.n,
        buffer=replay_buffer,
        learning_rate=LR,
        gamma=GAMMA,
        format="conv",
    )

    trainer = ReinforceTrainer()
    hist = trainer.run(ENV, agent, MAX_EP)
    save(agent.model.state_dict(), f"./models/{gym_name}_REINFORCE.pth")
    plot_and_save(hist, figname="Xenon-REINFORCE.png")


def call_a2c():
    replay_buffer = ReinforceBuffer(
        data_handler=XenonDataHandlerReinforce(),
    )
    agent = A2CAgent(
        input_dim=ENV.observation_space.shape,
        output_dim=ENV.action_space.n,
        buffer=replay_buffer,
        learning_rate=LR,
        gamma=GAMMA,
        format="conv",
    )

    trainer = A2CTrainer()
    hist = trainer.run(ENV, agent, MAX_EP)
    save(agent.model.state_dict(), f"./models/{gym_name}_A2C.pth")
    plot_and_save(hist, figname="Xenon-A2C.png")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.algorithm == "d3qn":
        call_d3qn()
    elif args.algorithm == "reinforce":
        call_reinforce()
    elif args.algorithm == "a2c":
        call_a2c()

