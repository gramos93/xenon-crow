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

        states = torch.stack(batch[:, 0]).to(torch.float32)
        next_states = torch.stack(batch[:, 3]).to(torch.float32)

        action_hist = batch[:, 1, None].astype(torch.int64)
        rewards = torch.FloatTensor(batch[:, 2, None], requires_grad=False)
        not_finals = torch.logical_not(batch[:, -2, None]).to(torch.float32)

        return states, action_hist, rewards, next_states, not_finals, batch[:, -1]
    