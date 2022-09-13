import functools
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest

ALL_STORES = [
    lambda directory, capacity=None: embodied.replay.RAMStore(capacity),
    embodied.replay.DiskStore,
    embodied.replay.CkptRAMStore,
]

PERSISTENT_STORES = [
    embodied.replay.DiskStore,
    embodied.replay.CkptRAMStore,
]

ALL_REPLAYS = [
    embodied.replay.FixedLength,
    embodied.replay.Consecutive,
    embodied.replay.Prioritized,
]

UNIFORM_FIXED_LENGTH = [
    functools.partial(
        embodied.replay.FixedLength, prio_starts=0.0, prio_ends=0.0),
    functools.partial(
        embodied.replay.Prioritized, exponent=0.0,
        prio_starts=0.0, prio_ends=0.0),
]


class TestReplay:

  # Example trajectory:
  # idx: -1    0        1      ...  9      10      11
  # obs: zeros is_first mid         mid    is_last is_first
  # act: reset policy   policy      policy reset   policy

  @pytest.mark.parametrize('Replay', ALL_REPLAYS)
  @pytest.mark.parametrize('Store', ALL_STORES)
  def test_internal_content(self, tmpdir, Replay, Store):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    store = Store(tmpdir)
    replay = Replay(store, length=5)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=2)
    assert len(replay) == 22
    assert len(store) == 2
    for key in store.keys():
      assert len(store[key]['action']) == 11
      assert (store[key]['step'] == np.arange(11)).all()
    for ongoing in replay.ongoing.values():
      assert len(ongoing) == 0

  @pytest.mark.parametrize('Replay', UNIFORM_FIXED_LENGTH)
  @pytest.mark.parametrize('Store', ALL_STORES)
  def test_sample_uniform(self, tmpdir, Replay, Store):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    store = Store(tmpdir)
    replay = Replay(store, length=10)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=1)
    count1, count2 = 0, 0
    iterator = replay.dataset()
    for _ in range(100):
      sample = next(iterator)['step']
      count1 += (sample == np.arange(0, 10)).all()
      count2 += (sample == np.arange(1, 11)).all()
    assert count1 + count2 == 100
    assert count1 > 30
    assert count2 > 30

  @pytest.mark.parametrize('Replay', ALL_REPLAYS)
  @pytest.mark.parametrize('Store', PERSISTENT_STORES)
  def test_reload(self, tmpdir, Replay, Store):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    old = Replay(Store(tmpdir), length=5)
    driver = embodied.Driver(env)
    driver.on_step(old.add)
    driver(agent.policy, episodes=2)
    store = Store(tmpdir)
    replay = Replay(store, length=5)
    assert len(replay) == 22
    assert len(store) == 2
    for key in store.keys():
      assert len(store[key]['action']) == 11
      assert (store[key]['step'] == np.arange(11)).all()

  @pytest.mark.parametrize(('capacity', 'steps', 'episodes'), [
      (11, 11, 1),
      (21, 11, 1),
      (22, 22, 2),
  ])
  @pytest.mark.parametrize('Replay', ALL_REPLAYS)
  @pytest.mark.parametrize('Store', ALL_STORES)
  def test_capacity(
      self, tmpdir, Replay, Store, capacity, steps, episodes):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    store = Store(tmpdir, capacity)
    replay = Replay(store, length=5)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=3)
    assert replay.stats['replay_steps'] == steps
    assert replay.stats['replay_trajs'] == episodes
    loaded = [store[key] for key in store.keys()]
    assert len(loaded) == episodes

  @pytest.mark.parametrize(('capacity', 'steps', 'episodes'), [
      (11, 11, 1),
      (21, 11, 1),
      (22, 22, 2),
  ])
  @pytest.mark.parametrize('Replay', ALL_REPLAYS)
  @pytest.mark.parametrize('Store', [
      embodied.replay.DiskStore, embodied.replay.CkptRAMStore])
  def test_reload_capacity(
      self, tmpdir, Replay, Store, capacity, steps, episodes):
    env = embodied.envs.load_env('dummy_discrete', length=10)
    agent = embodied.RandomAgent(env.act_space)
    old = Replay(Store(tmpdir, capacity), length=5)
    driver = embodied.Driver(env)
    driver.on_step(old.add)
    driver(agent.policy, episodes=3)
    store = Store(tmpdir, capacity)
    replay = Replay(store, length=5)
    assert replay.stats['replay_steps'] == steps
    assert replay.stats['replay_trajs'] == episodes
    assert len(store) == episodes

  @pytest.mark.parametrize('Replay', ALL_REPLAYS)
  def test_unexpected_reset(self, Replay):

    class UnexpectedReset(embodied.Wrapper):
      """Send is_first without preceeding is_last."""
      def __init__(self, env, when):
        super().__init__(env)
        self._when = when
        self._step = 0
      def step(self, action):
        if self._step == self._when:
          action = action.copy()
          action['reset'] = np.ones_like(action['reset'])
        self._step += 1
        return self.env.step(action)

    env = embodied.envs.load_env('dummy_discrete', length=10)
    env = UnexpectedReset(env, when=8)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    store = embodied.replay.RAMStore()
    replay = Replay(store, length=5)
    driver.on_step(replay.add)
    driver(agent.policy, episodes=2)
    assert len(replay) == 22
    assert len(store) == 2
    for key in store.keys():
      assert len(store[key]['action']) == 11
      assert (store[key]['step'] == np.arange(11)).all()
