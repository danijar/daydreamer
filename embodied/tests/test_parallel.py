import collections
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import pytest


class TestParallel:

  @pytest.mark.parametrize('parallel', ['thread', 'process'])
  def test_parallel_strategy(self, parallel):
    env = embodied.envs.load_env(
        'dummy_discrete', parallel=parallel, amount=4, length=10)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    episodes = collections.defaultdict(list)
    driver.on_episode(lambda ep, worker: episodes[worker].append(ep))
    driver(agent.policy, episodes=8)
    env.close()
    assert len(episodes) == 4
    assert set(episodes.keys()) == {0, 1, 2, 3}
    for worker, eps in episodes.items():
      assert len(eps) == 2
      assert len(eps[0]['reward']) == 11
      assert len(eps[1]['reward']) == 11

  @pytest.mark.parametrize('parallel', ['thread', 'process'])
  def test_parallel_fast(self, parallel):
    def ctor():
      env = embodied.envs.Dummy('discrete')
      env = Delay(env, 0.1)
      return env
    envs = [embodied.Parallel(ctor, parallel) for _ in range(4)]
    env = embodied.BatchEnv(envs, parallel=True)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    start = time.time()
    driver(agent.policy, steps=4)
    duration = time.time() - start
    env.close()
    assert duration <= 0.2

  def test_sequential_slow(self):
    def ctor():
      env = embodied.envs.Dummy('discrete')
      env = Delay(env, 0.1)
      return env
    envs = [ctor() for _ in range(4)]
    env = embodied.BatchEnv(envs, parallel=False)
    agent = embodied.RandomAgent(env.act_space)
    driver = embodied.Driver(env)
    start = time.time()
    driver(agent.policy, steps=4)
    duration = time.time() - start
    env.close()
    assert 0.4 <= duration

  def test_kill_process(self):
    start = time.time()
    class Dummy:
      def foo(self):
        time.sleep(10)
    parallel = embodied.Parallel(Dummy, 'process')
    parallel.foo()  # Call method but don't block.
    parallel.close()
    duration = time.time() - start
    assert duration < 2, duration

  def test_kill_nested_process(self):
    start = time.time()
    class Child:
      def foo(self):
        time.sleep(10)
    class Parent:
      def __init__(self):
        self.child = embodied.Parallel(Child, 'process', daemon=False)
      def foo(self):
        return self.child.foo()
    parallel = embodied.Parallel(Parent, 'process', daemon=False)
    parallel.foo()  # Call method but don't block.
    parallel.close()
    duration = time.time() - start
    assert duration < 2, duration


class Delay(embodied.base.Wrapper):

  def __init__(self, env, duration):
    super().__init__(env)
    self._duration = duration

  def step(self, action):
    time.sleep(self._duration)
    return self.env.step(action)
