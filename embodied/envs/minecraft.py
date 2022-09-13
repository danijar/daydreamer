import logging
import threading

import embodied
import numpy as np

from . import gym


class Minecraft(embodied.Env):

  _LOCK = threading.Lock()

  def __init__(
      self, task,
      repeat=1,
      size=(64, 64),
      length=24000,
      sticky_attack=30,
      sticky_jump=10,
      pitch_limit=(-60, 60),
      logs=False):
    self._task = task
    self._repeat = repeat
    self._size = size

    # Make env.
    if logs:
      logging.basicConfig(level=logging.DEBUG)
    with self._LOCK:
      import gym as openai_gym
      from .import minerl_internal
      ids = [x.id for x in openai_gym.envs.registry.all()]
      minerl_internal.SIZE = size
      if 'MinecraftDiamond-v1' not in ids:
        minerl_internal.Diamond().register()
      if 'MinecraftDiscover-v1' not in ids:
        minerl_internal.Discover().register()
      if 'MinecraftWood-v1' not in ids:
        minerl_internal.Wood().register()
      if 'MinecraftTable-v1' not in ids:
        minerl_internal.Table().register()
      if 'MinecraftAxe-v1' not in ids:
        minerl_internal.Axe().register()
      self._inner = openai_gym.make(f'Minecraft{task.title()}-v1')
    self._env = gym.Gym(self._inner, checks=False)
    self._env = embodied.wrappers.TimeLimit(self._env, length)

    # Observations.
    self._inv_keys = [
        k for k in self._env.obs_space if k.startswith('inventory/')]
    self._step = 0
    self._collected_items = set()
    self._equip_enum = self._inner.observation_space[
        'equipped_items']['mainhand']['type'].values.tolist()
    self._obs_space = self.obs_space

    # Actions.
    self._noop_action = minerl_internal.NOOP_ACTION
    actions = self._insert_defaults({
        'discover': minerl_internal.DISCOVER_ACTIONS,
        'diamond': minerl_internal.DIAMOND_ACTIONS,
        'wood': minerl_internal.WOOD_ACTIONS,
        'table': minerl_internal.TABLE_ACTIONS,
        'axe': minerl_internal.AXE_ACTIONS,
    }[task])
    self._action_names = tuple(actions.keys())
    self._action_values = tuple(actions.values())
    message = f'Minecraft action space ({len(self._action_values)}):'
    print(message, ', '.join(self._action_names))
    self._sticky_attack_length = sticky_attack
    self._sticky_attack_counter = 0
    self._sticky_jump_length = sticky_jump
    self._sticky_jump_counter = 0
    self._pitch_limit = pitch_limit
    self._pitch = 0

  @property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'inventory': embodied.Space(np.float32, len(self._inv_keys), 0),
        'equipped': embodied.Space(np.float32, len(self._equip_enum), 0, 1),
        **{f'log_{k}': embodied.Space(np.int64) for k in self._inv_keys},
        'reward': embodied.Space(np.float32),
        'new_items': embodied.Space(np.int64),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int64, (), 0, len(self._action_values)),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    action = action.copy()
    action.update(self._action_values[action.pop('action')])
    action = self._action(action)
    if action['reset']:
      obs = self._reset()
    else:
      following = self._noop_action.copy()
      for key in ('attack', 'forward', 'back', 'left', 'right'):
        following[key] = action[key]
      for act in [action] + ([following] * (self._repeat - 1)):
        obs = self._env.step(act)
        if 'error' in self._env.info:
          obs = self._reset()
          break
    new_items = self._track_new_items(obs)
    obs = self._obs(obs, new_items)
    self._step += 1
    return obs

  def _reset(self):
    with self._LOCK:
      obs = self._env.step({'reset': True})
    self._step = 0
    self._collected_items.clear()
    self._sticky_attack_counter = 0
    self._sticky_jump_counter = 0
    self._pitch = 0
    return obs

  def _obs(self, obs, new_items):
    inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
    inventory = np.log(1 + np.array(inventory))
    index = self._equip_enum.index(obs['equipped_items/mainhand/type'])
    equipped = np.zeros(len(self._equip_enum), np.float32)
    equipped[index] = 1.0
    if self._task == 'discover':
      reward = new_items
    else:
      reward = obs['reward']
    obs = {
        'image': obs['pov'],
        'inventory': inventory,
        'equipped': equipped,
        **{f'log_{k}': np.int64(obs[k]) for k in self._inv_keys},
        'reward': np.float32(reward),
        'new_items': new_items,
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
    }
    for key, value in obs.items():
      space = self._obs_space[key]
      if not isinstance(value, np.ndarray):
        value = np.array(value)
      assert value in space, (key, value, value.dtype, value.shape, space)
    return obs

  def _action(self, action):
    if self._sticky_attack_length:
      if action['attack']:
        self._sticky_attack_counter = self._sticky_attack_length
      if self._sticky_attack_counter > 0:
        action['attack'] = 1
        action['jump'] = 0
        self._sticky_attack_counter -= 1
    if self._sticky_jump_length:
      if action['jump']:
        self._sticky_jump_counter = self._sticky_jump_length
      if self._sticky_jump_counter > 0:
        action['jump'] = 1
        action['forward'] = 1
        self._sticky_jump_counter -= 1
    if self._pitch_limit and action['camera'][0]:
      lo, hi = self._pitch_limit
      if not (lo <= self._pitch + action['camera'][0] <= hi):
        action['camera'] = (0, action['camera'][1])
      self._pitch += action['camera'][0]
    return action

  def _track_new_items(self, obs):
    new_items = 0
    for key in self._inv_keys:
      if key in self._collected_items:
        continue
      if key == 'inventory/air':
        continue
      if obs[key].item() > 0:
        new_items += 1
        self._collected_items.add(key)
    return new_items

  def _insert_defaults(self, actions):
    actions = {name: action.copy() for name, action in actions.items()}
    for key, default in self._noop_action.items():
      for action in actions.values():
        if key not in action:
          action[key] = default
    return actions
