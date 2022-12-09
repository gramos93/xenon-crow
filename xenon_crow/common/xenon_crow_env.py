from poutyne import Model
from copy import deepcopy  # NEW
import random
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym import spaces
from torch.utils.data import DataLoader, Dataset
import glob
import torchvision.transforms as tr
from PIL import Image
import random

class XenonCrowDataset(Dataset):

    def __init__(self, mode, path, transformation=None):

        self.path = path
        self.mode = mode
        self.transformation = transformation

        if (self.mode == 'single'):
            images = glob.glob(str(self.path)+ "/masks/states/*")
            self.len = len(images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if (self.mode == 'single'):
            im_paths = glob.glob(str(self.path)+ "/image/*")
            gt_paths = glob.glob(str(self.path)+ "/masks/gt/*")
            mask_paths = glob.glob(str(self.path)+ "/masks/states/*")

            image = Image.open(im_paths[0])
            gt = Image.open(gt_paths[0]).convert('L')
            step_mask = Image.open(mask_paths[idx]).convert('L')

            preprocess = tr.Compose([
                #T.Resize(), will need to fix train data image size eventually
                tr.ToTensor(),
            ])

            image = preprocess(image)
            gt = preprocess(gt)
            step_mask = preprocess(step_mask)

            state = torch.vstack([image, gt, step_mask])

        return state

class XenonCrowEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mode, path):
    super(XenonCrowEnv, self).__init__()
    self.seed = None
    self.mode = mode
    self.path = path
    self.dataset = iter(DataLoader(XenonCrowDataset(mode, path), shuffle=True, batch_size=1))
    self.progress_mask = torch.zeros((1, 240, 320))
    self.action_space = spaces.Discrete(2)
    # image as input:
    height = 240
    width = 320
    n_channels = 5
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (height, width, n_channels))

    self.state = torch.zeros((1, 5, 240, 320))

  def set_random_seed(self, seed):
    self.seed = seed

  def step(self, action): # 0 is off, 1 is on
    self.state = self.peek(self.dataset)
    gt = self.state[:, 3, :, :].type(torch.int64)
    step_mask = self.state[:, 4, :, :].type(torch.int64)

    if (action == 0): self.progress_mask = torch.bitwise_or(self.progress_mask, torch.bitwise_not(step_mask))
    else: self.progress_mask = torch.bitwise_or(self.progress_mask, step_mask)

    reward = self.iou_pytorch(gt, step_mask) # IoU between gt and mask action

    if (self.iou_pytorch(self.progress_mask, gt) > torch.tensor(0.98)): done = True
    else: done = False

    info = {}
    observation = torch.vstack([self.state.squeeze(0), self.progress_mask]).unsqueeze(0)
    return observation, reward, done, info

  def reset(self):
    self.progress_mask = torch.zeros(self.progress_mask.size()).type(torch.int64)
    self.state = self.peek(self.dataset)
    return self.state

  def render(self):
    return torch.vstack([self.state.squeeze(0), self.progress_mask]).unsqueeze(0)

  def peek(self, iterable):
    try:
        return next(iterable)
    except StopIteration:
        self.dataset = iter(DataLoader(XenonCrowDataset(self.mode, self.path), shuffle=True, batch_size=1))
        return next(self.dataset)

  def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    if iou < 0.01: iou = 0.0

    return iou