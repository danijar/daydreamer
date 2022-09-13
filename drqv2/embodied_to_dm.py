import dm_env
from dm_env import specs
import numpy as np


class EmbodiedToDM(dm_env.Environment):

  def __init__(self, env, timelimit=None):
    self.env = env
    self.last = True
    self._step = None
    self.timelimit = timelimit

  def reset(self):
    dummy = self.env.act_space['action'].sample()
    obs = self.env.step({'action': dummy, 'reset': True})
    self.last = obs['is_last']
    self._step = 0
    return self._convert_obs(obs)

  def step(self, action):
    if self.last:
      return self.reset()
    obs = self.env.step({'action': action, 'reset': False})
    self._step += 1
    if self.timelimit and self._step >= self.timelimit:
      obs['is_last'] = True
    self.last = obs['is_last']
    return self._convert_obs(obs)

  def observation_spec(self):
    spec = {}
    for key, space in self.env.obs_space.items():
      if key.startswith('is_') or key == 'reward':
        continue
      spec[key] = self._convert_space(key, space)
    return spec

  def action_spec(self):
    return self._convert_space('action', self.env.act_space['action'])

  def _convert_obs(self, obs):
    obs = obs.copy()
    if obs.pop('is_first'):
      stype = dm_env.StepType.FIRST
    elif obs.pop('is_last'):
      stype = dm_env.StepType.LAST
    else:
      stype = dm_env.StepType.MID
    reward = obs.pop('reward')
    discount = 0.0 if obs.pop('is_terminal') else 1.0
    return dm_env.TimeStep(stype, reward, discount, obs)

  def _convert_space(self, name, space):
    if space.dtype in (np.int32, np.int64, int):
      return specs.DiscreteArray(
          dtype=space.dtype,
          num_values=space.high,
          name=name,
      )
    else:
      return specs.BoundedArray(
          shape=space.shape,
          dtype=space.dtype,
          name=name,
          minimum=space.low,
          maximum=space.high,
      )
