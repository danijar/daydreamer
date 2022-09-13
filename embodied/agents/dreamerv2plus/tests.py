import os
import pathlib
import sys
import tempfile
import time

sys.argv[0] = __file__
path = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  path = path.resolve()
path = path.parent
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent.parent))
__package__ = path.name

import embodied
import numpy as np
import tensorflow as tf

os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'  # Enable XLA.

SLACK = 1.3

TEST_CONFIG = {
    'env.parallel': 'none',
    'replay_chunk': 8,
    'batch_size': 8,
    r'.*\.layers': 2,
    r'.*\.units': 128,
    r'.*\.cnn_depth': 16,
    r'.*\.wd$': 0.0,
    'train.steps': 500,
    'train.log_every': 100,
    'train.train_every': 50,
    'train.eval_every': 200,
    'train.train_fill': 200,
}


class AgentTest(tf.test.TestCase):

  def test_run(self):
    from .train import main
    with tempfile.TemporaryDirectory() as logdir:
      flags = {'logdir': logdir, 'run': 'train', **TEST_CONFIG}
      argv = sum([[f'--{k}', str(v)] for k, v in flags.items()], [])
      start = time.time()
      main(argv)
      duration = time.time() - start
    print('RUN DURATION:', duration)
    assert duration < SLACK * 60.0, duration

  def test_train(self):
    from . import agent as agnt
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(TEST_CONFIG)
    step = embodied.Counter()
    env = embodied.envs.load_env(config.task, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(env, [config.batch_size, config.replay_chunk])
    times = []
    state = None
    for _ in range(5):
      start = time.time()
      _, state, _ = agent.train(data, state)
      times.append(time.time() - start)
    print('TRAIN DURATIONS:', times)
    assert times[0] <= SLACK * 154.0, times
    assert min(times) <= SLACK * 0.02, times

  def test_policy(self):
    from . import agent as agnt
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(TEST_CONFIG)
    step = embodied.Counter()
    env = embodied.envs.load_env(config.task, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(env, [config.batch_size])
    times = []
    state = None
    for _ in range(5):
      start = time.time()
      _, state = agent.policy(data, state)
      times.append(time.time() - start)
    print('POLICY DURATIONS:', times)
    assert times[0] <= SLACK * 9.0, times
    assert min(times) <= SLACK * 0.007, times

  def test_report(self):
    from . import agent as agnt
    config = embodied.Config(agnt.Agent.configs['defaults'])
    config = config.update(TEST_CONFIG)
    step = embodied.Counter()
    env = embodied.envs.load_env(config.task, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    data = self._make_data(env, [config.batch_size, config.replay_chunk])
    times = []
    for _ in range(5):
      start = time.time()
      agent.report(data)
      times.append(time.time() - start)
    print('REPORT DURATIONS:', times)
    assert times[0] <= SLACK * 46.0, times
    assert min(times) <= SLACK * 0.01, times

  def _make_data(self, env, batch_dims):
    spaces = list(env.obs_space.items()) + list(env.act_space.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data


if __name__ == '__main__':
  tf.test.main()
