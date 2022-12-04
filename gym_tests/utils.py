from typing import List, Tuple

import numpy as np
import torch

class GymDataHandler(object):
    def __call__(self, batch: Tuple) -> List:
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
        not_finals = torch.tensor(not_finals)
        return states, action_hist, rewards, next_states, not_finals
    