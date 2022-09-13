import embodied
import numpy as np

from . import gym


class A1(embodied.Env):

  def __init__(self, task, repeat=1, length=1000, resets=True):
    assert task in ('sim', 'real'), task
    import motion_imitation.envs.env_builder as env_builder
    self._gymenv = env_builder.build_env(
        enable_rendering=False,
        num_action_repeat=repeat, use_real_robot=bool(task == 'real'))
    self._env = gym.Gym(
        self._gymenv, obs_key='vector', act_key='action', checks=True)

  @property
  def obs_space(self):
    return {
        **self._env.obs_space,
        'image': embodied.Space(np.uint8, (64, 64, 3)),
    }

  @property
  def act_space(self):
    # return self._env.act_space
    return {
        'action': embodied.Space(np.float32, (12,), -1.0, 1.0),
        'reset': embodied.Space(bool, ()),
    }

  def step(self, action):
    obs = self._env.step(action)
    obs['image'] = self._gymenv.render('rgb_array')
    assert obs['image'].shape == (64, 64, 3), obs['image'].shape
    assert obs['image'].dtype == np.uint8, obs['image'].dtype
    return obs
