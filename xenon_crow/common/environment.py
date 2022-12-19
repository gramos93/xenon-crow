from typing import Tuple
from itertools import cycle
from pathlib import Path
from natsort import natsort
from random import choice
import numpy as np
import torch
from gym import spaces, Env
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor


class XenonDataHandler(object):
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


class XenonCrowDataset(Dataset):
    def __init__(self, path: str, image_name: str):

        self.root = Path(path)
        self.image_path = self.root / "imgs" / image_name
        self.gt_path = self.root / "masks/gt" / (self.image_path.stem + ".png")
        self.states = natsort(
            Path.glob(self.root / "masks/states" / self.image_path.stem)
        )
        self.transform = Compose([ToTensor])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if self.mode == "single":

            if not hasattr(self, "image"):
                self.image = Image.open(self.image_path).convert("RGB")
                self.image = self.transform(self.image)

            if not hasattr(self, "gt"):
                self.gt = Image.open(self.gt_path).convert("L")
                self.gt = self.transform(self.gt)

            state = Image.open(self.states[idx]).convert("L")
            state = self.transform(state)

        return self.image, state, self.gt


class XenonCrowEnv(Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, path):
        super(XenonCrowEnv, self).__init__()
        self.seed = None
        # Path to whole dataset
        self.root = path
        # List of available images to serve as episodes.
        self.episodes = [file.name for file in Path.glob(path + "imgs/*")]
        self.progress_mask = torch.zeros((1, 240, 320))
        self.action_space = spaces.Discrete(2)
        # image as input:
        height = 240
        width = 320
        n_channels = 5
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_channels, height, width)
        )

        self.state = torch.zeros((1, 5, 240, 320))

    def set_random_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.dataset = DataLoader(
                XenonCrowDataset(self.root, choice(self.episodes)),
                shuffle=True,
                batch_size=1,
            )
        self.iterator = cycle(self.dataset)
        self.progress_mask = torch.zeros(self.observation_space.shape).type(torch.int32)
        # This step initializes the self.image and self.gt of the Dataset.
        img, next_state, _ = next(self.iterator)
        # The ground truth is not used in the state to the model.
        self.state = torch.vstack([img, self.progress_mask, next_state])
        return self.state

    def step(self, action):  # 0 is off, 1 is on
        step_mask = self.state[:, -1:, :, :].type(torch.int64)

        if action == 0:
            self.progress_mask = torch.bitwise_or(
                self.progress_mask, torch.bitwise_not(step_mask)
            )
        else:
            self.progress_mask = torch.bitwise_or(self.progress_mask, step_mask)

        reward = self.iou_pytorch(self.dataset.gt, self.progress_mask)  # IoU between gt and mask action

        if reward > torch.tensor(0.98):
            done = True
        else:
            done = False

        info = {}
        img, next_state, _ = next(self.iterator)
        observation = torch.vstack(
            [img, self.progress_mask, next_state]
        )
        return observation, reward, done, info

    def render(self):
        return torch.vstack([self.state.squeeze(0), self.progress_mask]).unsqueeze(0)

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        SMOOTH = 1e-6
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape

        # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (
            (outputs & labels).float().sum((1, 2))
        )  # Will be zero if Truth=0 or Prediction=0

        union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (
            union + SMOOTH
        )  # We smooth our devision to avoid 0/0

        if iou < 0.01:
            iou = 0.0

        return iou
