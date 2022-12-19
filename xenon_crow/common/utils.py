from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import animation

sns.set_style("white")


class GymDataHandler(object):
    def __call__(self, batch: Tuple) -> Tuple:
        """
        Input :
            - batch, a list of n=batch_size elements from the replay buffer
            - target_network, the target network to compute the one-step lookahead target
            - gamma, the discount factor

        Returns :
            - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
            - (actions, targets) : where actions and targets both
                        have the shape (batch_size, ). Actions are the
                        selected actions according to the target network
                        and targets are the one-step lookahead targets.
        """
        batch = np.array(batch, dtype=object)

        states = torch.tensor(np.stack(batch[:, 0])).to(torch.float32)
        next_states = torch.tensor(np.stack(batch[:, 3])).to(torch.float32)

        action_hist = torch.tensor(batch[:, 1, None].astype(np.int64))
        rewards = torch.tensor(batch[:, 2, None].astype(np.float32))
        not_finals = np.logical_not(batch[:, -1, None].astype(np.ubyte))
        not_finals = torch.tensor(not_finals).float()
        return states, action_hist, rewards, next_states, not_finals


class GymDataHandlerReinforce(object):
    def __call__(self, batch: List) -> Tuple:
        """
        Input :
            - batch, a list of n=batch_size elements from the replay buffer
            - target_network, the target network to compute the one-step lookahead target
            - gamma, the discount factor

        Returns :
            - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
            - (actions, targets) : where actions and targets both
                        have the shape (batch_size, ). Actions are the
                        selected actions according to the target network
                        and targets are the one-step lookahead targets.
        """
        log_probs = [b[0] for b in batch]
        values = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        
        return log_probs, values, rewards


def plot_and_save(hist, figname, ma=0.2):
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    x = np.arange(1, len(hist) + 1)

    if ma < 1:
        kernel_size = int(len(hist) * ma)
        kernel = np.ones(kernel_size) / kernel_size
        sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax, alpha=0.3)
        sns.lineplot(y=np.convolve(hist, kernel, mode="same"), x=x, color="k", linewidth=1, ax=ax)
    else:
        sns.lineplot(y=hist, x=x, color="k", linewidth=1, ax=ax)
        
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Episodes")
    ax.grid(visible=True, axis="y", linestyle="--")

    fig.savefig(f"assets/{figname}")


def save_frames_as_gif(frames, filename, path="./assets/"):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="ffmpeg", fps=60)
