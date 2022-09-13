import argparse

import gym.spaces
import numpy as np
import stable_baselines3
import torch
from r3m import load_r3m

import embodied

parser = argparse.ArgumentParser()
boolean = lambda x: bool(['False', 'True'].index(x))
parser.add_argument('--logdir', type=str, default='logdir')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()


class Env:

    metadata = {}

    def __init__(self, outdir):
        self.env = embodied.envs.load_single_env('ur5_real', length=100)
        self.num_actions = self.env.act_space['action'].shape[0]
        self.logger = embodied.Logger(embodied.Counter(), [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(outdir, 'metrics.jsonl'),
            embodied.logger.TensorBoardOutput(outdir),
        ], multiplier=1)
        self.score = 0
        self.length = 0
        self.r3m = load_r3m("resnet50")
        self.r3m.eval()

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.obs_space.items():
            if key.startswith('log_'):
                continue
            if key.startswith('is_'):
                continue
            if key in ('reward', 'depth'):
                continue
            if key == 'image':
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, (2048,), np.float)
                continue

            print(value)
            spaces[key] = gym.spaces.Box(
                value.low, value.high, value.shape, value.dtype)
        # print('-'*100)
        # print({k: v.shape for k, v in spaces.items()})
        # print('-'*100)
        return gym.spaces.Dict(spaces)
        # return gym.spaces.Box((64, 64, 19), 0, 255

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    def reset(self):
        action = self.env.act_space['action'].sample()
        obs = self.env.step({'action': action, 'reset': True})
        del obs['is_first']
        del obs['is_last']
        del obs['is_terminal']
        del obs['reward']
        del obs['depth']
        with torch.no_grad():
            obs['image'] = self.r3m(torch.tensor(obs['image'])[None])[0]
        # obs = self._stack_obs(obs)
        self.score = 0
        self.length = 0
        return obs

    def step(self, action):
        self.logger.step.increment()
        action_onehot = np.zeros((self.num_actions,))
        action_onehot[action] = 1
        obs = self.env.step({'action': action_onehot, 'reset': False})
        if obs['is_last']:
            self.logger.add({
                'score': self.score,
                'length': self.length,
                'avg_reward': self.score / self.length,
            })
            self.logger.write(fps=True)
        done = obs.pop('is_last')
        reward = obs.pop('reward')
        self.score += reward
        self.length += 1
        del obs['is_first']
        del obs['is_terminal']
        del obs['depth']
        with torch.no_grad():
            obs['image'] = self.r3m(torch.tensor(obs['image'])[None])[0]
        # obs = self._stack_obs(obs)
        return obs, reward, done, {}

    def _stack_obs(self, obs):
        obs['image']


env = Env(args.logdir)
model = stable_baselines3.PPO(
    stable_baselines3.common.policies.MultiInputActorCriticPolicy,
    env, verbose=1)
model.learn(total_timesteps=args.steps)
