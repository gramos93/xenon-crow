from itertools import cycle, repeat
from pathlib import Path
from random import choice
from typing import Tuple
import numpy as np
import torch
from gym import Env, spaces
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode


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
        states, next_states, action_hist, rewards, not_finals = [], [], [], [], []
        for s in batch:
            states.append(s[0])
            next_states.append(s[3])
            action_hist.append(s[1])
            rewards.append(s[2])
            not_finals.append(s[-1])

        states = torch.stack(states).to(torch.float32).squeeze(1)
        next_states = torch.stack(next_states).to(torch.float32).squeeze(1)
        action_hist = torch.tensor(action_hist).to(torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards).to(torch.float32).unsqueeze(1)
        not_finals = torch.tensor(not_finals).to(torch.float32).unsqueeze(1)
        not_finals = torch.logical_not(not_finals).float()

        return states, action_hist, rewards, next_states, not_finals


class XenonCrowDataset(Dataset):
    def __init__(self, path: str, image_name: str):

        self.root = Path(path)
        self.image_path = self.root / "imgs" / image_name
        self.gt_path = self.root / "masks/gt" / (self.image_path.stem + ".png")
        self.states = natsorted(
            Path.glob(self.root / "masks/states" / self.image_path.stem, "*")
        )
        self.input_transform = Compose([Resize((120, 160)), ToTensor()])
        self.target_transform = Compose(
            [Resize((120, 160), InterpolationMode.NEAREST), ToTensor()]
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        if not hasattr(self, "image"):
            self.image = Image.open(self.image_path).convert("RGB")
            self.image = self.input_transform(self.image)

        if not hasattr(self, "gt"):
            self.gt = Image.open(self.gt_path).convert("L")
            self.gt = self.target_transform(self.gt)

        state = Image.open(self.states[idx]).convert("L")
        state = self.target_transform(state)

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
        self.episodes = [file.name for file in Path.glob(Path(path) / "imgs", "*")]
        # self.progress_mask = torch.zeros((1, 240, 320))
        self.action_space = spaces.Discrete(2)
        # image as input:
        height = 240//2
        width = 320//2
        n_channels = 5
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, n_channels, height, width)
        )
        self.save_masks = False

    def set_random_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.runs = 0
        self.dataset = DataLoader(
            XenonCrowDataset(self.root, choice(self.episodes)),
            shuffle=True,
            batch_size=1,
        )
        self.iterator = cycle(self.dataset)
        self.steps = 0
        self.progress_mask = torch.zeros(
            (1, 1, *self.observation_space.shape[2:]), dtype=torch.uint8
        )
        # This step initializes the self.image and self.gt of the Dataset.
        img, next_state, _ = next(self.iterator)
        # The ground truth is not used in the state to the model.
        self.state = torch.cat([img, self.progress_mask, next_state], dim=1)
        return self.state, None

    def step(self, action):  # 0 is off, 1 is on
        step_mask = self.state[:, -1:, :, :].byte()
        trunc = False
        if action == 0:
            self.progress_mask = torch.clamp(self.progress_mask - step_mask, 0, 1)
            reward = -1.0 * self.iou_pytorch(
                step_mask.squeeze(0), self.dataset.dataset.gt.byte()
            )
        else:
            self.progress_mask = torch.logical_or(self.progress_mask, step_mask).byte()
            reward = 2.0 * self.iou_pytorch(
                step_mask.squeeze(0), self.dataset.dataset.gt.byte()
            )

        total_iou = self.iou_pytorch(
            self.progress_mask.byte(), self.dataset.dataset.gt.byte()
        )
        if total_iou > 0.8:
            reward += (10 * total_iou)
            done = True
        else:
            done = False

        if self.steps ==  1 * len(self.dataset.dataset):
            reward += total_iou
            trunc = True
        else:
            self.steps += 1

        if self.save_masks and (done or trunc):
            self.log_mask()

        info = {}
        img, next_state, _ = next(self.iterator)
        self.state = torch.cat([img, self.progress_mask, next_state], dim=1)
        # Needs to return done, truncated. Here we use `done` for both.
        return self.state, reward, done, trunc, info
    
    def log_mask(self):
        img = self.progress_mask[0, 0].detach().cpu().to(torch.uint8).numpy() * 255
        gt = self.dataset.dataset.gt[0].detach().cpu().to(torch.uint8).numpy() * 255
        img = np.stack([gt, np.zeros_like(gt), img], axis=-1)
        img = Image.fromarray(img)
        img.save(f"./logs/{self.dataset.dataset.image_path.stem}.png")

    def render(self):
        return torch.vstack([self.state.squeeze(0), self.progress_mask]).unsqueeze(0)

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        SMOOTH = 1e-6
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape

        # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (
            (outputs & labels).float().sum()
        )  # Will be zero if Truth=0 or Prediction=0

        union = (outputs | labels).float().sum()  # Will be zero if both are 0

        iou = (intersection + SMOOTH) / (
            union + SMOOTH
        ).item()  # We smooth our devision to avoid 0/0
        if iou < 0.01:
            iou = -0.10

        return iou
