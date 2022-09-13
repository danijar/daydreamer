import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__)))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import time
import pickle as pkl
import keyboard


# from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import hydra

import motion_imitation.envs.env_builder as env_builder

def make_env(cfg):
    """Helper function to create dm_control environment"""


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        env = env_builder.build_env(
          enable_rendering=False,
          num_action_repeat=50, use_real_robot=True)
        env.seed(cfg.seed)
        env._max_episode_steps = 250
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
        self.env = env

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        self.step = 0
        # self.load('/home/rll/realworldrl/exp/2022.06.14/1133_sac_test_exp/everything.pt')
        self.load('/home/rll/realworldrl/exp/2022.06.14/1334_sac_test_exp/everything.pt')

    def run(self):

        obs = self.env.reset()

        episode, episode_reward, episode_step, done, just_paused = 0, 0, 0, True, False
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if episode_step > self.env._max_episode_steps or just_paused:

                if episode_step > 0:
                  self.logger.log('train/mean_reward', episode_reward / episode_step, self.step)


                self.logger.log('train/episode_reward', episode_reward,
                                self.step)


                # obs = self.env.reset()  # Never.
                self.agent.reset()
                done = False
                just_paused = False
                episode_reward = 0
                episode_step = 0
                episode += 1


                # Intervention at 49000, 50970
                self.logger.log('train/episode', episode, self.step)

                # self.logger.log('train/duration',
                #                time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(
                    self.step, save=(self.step > self.cfg.num_seed_steps))

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            if keyboard.is_pressed('p'):
              done = True
              just_paused = True
              while not keyboard.is_pressed('c'):
                time.sleep(0.01)

            if self.step % self.cfg.checkpoint_frequency == 0:
              print('saving checkpoin5000t')
              self.save('everything.pt')

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


    def save(self, path, infos=None):
        torch.save(
        {
            'critic': self.agent.critic.state_dict(),
            'critic_target': self.agent.critic_target.state_dict(),
            'actor': self.agent.actor.state_dict(),
            'log_alpha': self.agent.log_alpha,
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'log_alpha_optimizer': self.agent.log_alpha_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'step': self.step,
        }, path)

    def load(self, path, load_optimizer=True):
      loaded_dict = torch.load(path)
      self.replay_buffer = loaded_dict['replay_buffer']
      self.agent.critic.load_state_dict(loaded_dict['critic'])
      self.agent.critic_target.load_state_dict(loaded_dict['critic_target'])
      self.agent.actor.load_state_dict(loaded_dict['actor'])
      self.agent.log_alpha = loaded_dict['log_alpha']
      self.agent.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer'])
      self.agent.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer'])
      self.agent.log_alpha_optimizer.load_state_dict(loaded_dict['log_alpha_optimizer'])
      # self.agent.load_state_dict(loaded_dict['agent'])
      self.step = loaded_dict['step']

@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
