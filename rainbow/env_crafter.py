# -*- coding: utf-8 -*-
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__)))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import pathlib
from collections import deque

import embodied
import numpy as np
import torch


class Env():

  def __init__(self, args):
    self.device = args.device
    # env = embodied.envs.load_single_env('xarm_real')
    env = embodied.envs.load_single_env('ur5_real')
    self.env = env
    self.score = 0
    self.length = 0
    self.once = True
    self.logger = embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(args.logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(args.logdir),
    ])

  def action_space(self):
    return self.env.act_space['action'].shape[0]

  def reset(self):
    return self.step(action=0, reset=True)[0]

  def step(self, action, reset=False):
    self.logger.step.increment()
    if reset:
      if not self.once:
        self.logger.add({
            'score': self.score,
            'length': self.length,
            'mean_reward': self.score / self.length,
        })
        self.logger.write(fps=True)
      self.once = False
      self.score = 0
      self.length = 0
    action_one_hot = np.zeros(self.action_space())
    action_one_hot[action] = 1.0
    obs = self.env.step({'action': action_one_hot, 'reset': reset})
    reward = obs.pop('reward')
    done = obs.pop('is_last')
    obs = self._augment_obs(obs)
    self.score += reward
    self.length += 1
    if self.length >=100:
      done = True
    return obs, reward, done

  def _resize(self, image):
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((84, 84), Image.NEAREST)
    image = np.array(image)
    return image

  def _augment_obs(self, obs):
    image = self._resize(obs['image'])
    # depth = self._resize(np.repeat(obs['depth'], 3, -1))
    # depth = depth[:, :, [0]]
    proprio_obs = np.concatenate([
      obs['cartesian_position'],
      obs['joint_positions'],
      obs['gripper_pos'],
      obs['gripper_side'],
    ], -1)
    proprio_obs = torch.tensor(np.broadcast_to(
        proprio_obs,
        (image.shape[0],
         image.shape[1],
         proprio_obs.shape[-1])),dtype=torch.float32, device=self.device)
    obs = torch.tensor(
        # np.concatenate([image, depth], -1),
        image,
        dtype=torch.float32, device=self.device).div_(255)
    return torch.cat([obs, proprio_obs], -1).permute(2, 0, 1)

  def train(self):
    pass

  def test(self):
    pass

