import collections

import numpy as np

from .convert import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._obs = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.obs_space.items()}
    self._obs['is_last'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts, self._state = policy(self._obs, self._state, **self._kwargs)
    acts['reset'] = np.zeros(len(self._env), bool)
    if self._obs['is_last'].any():
      acts = {
          k: v * self._expand(1 - self._obs['is_last'], len(v.shape))
          for k, v in acts.items()}
      acts['reset'] = self._obs['is_last'].copy()
    acts = {k: convert(v) for k, v in acts.items()}
    assert all(len(x) == len(self._env) for x in acts.values()), acts
    self._obs = self._env.step(acts)
    assert all(len(x) == len(self._env) for x in self._obs.values()), self._obs
    self._obs = {k: convert(v) for k, v in self._obs.items()}
    trns = {**self._obs, **acts}
    if self._obs['is_first'].any():
      for i, first in enumerate(self._obs['is_first']):
        if not first:
          continue
        self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if self._obs['is_last'].any():
      for i, done in enumerate(self._obs['is_last']):
        if not done:
          continue
        ep = {k: convert(v) for k, v in self._eps[i].items()}
        [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
        episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
