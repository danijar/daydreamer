import pathlib
import sys
import tempfile
import time

directory = pathlib.Path(__file__).parent.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name
sys.argv[0] = str(directory / 'train.py')

import embodied
import numpy as np

from . import agent as agnt


def time_train(repeats, kwargs=None):
  config = embodied.Config(agnt.Agent.configs['defaults'])
  config = config.update({r'.*\.wd': 0.0}).update(kwargs or {})
  env = embodied.envs.load_env(config.task, **config.env)
  step = embodied.Counter()
  agent = agnt.Agent(env.obs_space, env.act_space, step, config)
  data = make_data(
      env, batch_dims=[config.batch_size, config.replay_fixed.length])
  state = None
  times = []
  for _ in range(repeats):
    start = time.time()
    _, state, _ = agent.train(data, state)
    times.append(time.time() - start)
  print('Train durations:', times)
  return times


def time_policy(repeats, kwargs=None):
  config = embodied.Config(agnt.Agent.configs['defaults'])
  config = config.update({r'.*\.wd': 0.0}).update(kwargs or {})
  env = embodied.envs.load_env(config.task, **config.env)
  step = embodied.Counter()
  agent = agnt.Agent(env.obs_space, env.act_space, step, config)
  data = make_data(env, batch_dims=[1])
  state = None
  times = []
  for _ in range(repeats):
    start = time.time()
    _, state = agent.policy(data, state)
    times.append(time.time() - start)
  print('Policy durations:', times)
  return times


def time_report(repeats, kwargs=None):
  config = embodied.Config(agnt.Agent.configs['defaults'])
  config = config.update({r'.*\.wd': 0.0}).update(kwargs or {})
  env = embodied.envs.load_env(config.task, **config.env)
  step = embodied.Counter()
  agent = agnt.Agent(env.obs_space, env.act_space, step, config)
  data = make_data(
      env, batch_dims=[config.batch_size, config.replay_fixed.length])
  times = []
  for _ in range(repeats):
    start = time.time()
    agent.report(data)
    times.append(time.time() - start)
  print('Report durations:', times)
  return times


def time_run_small(kwargs=None):
  from .train import main
  with tempfile.TemporaryDirectory() as logdir:
    flags = {
        'logdir': logdir,
        'configs': 'small',
        'run': 'train',
        'train.steps': 500,
        'train.log_every': 100,
        'train.train_every': 50,
        'train.eval_every': 200,
        'train.train_fill': 200,
    }
    if kwargs:
      flags.update(kwargs)
    argv = []
    for key, value in flags.items():
      argv += [f'--{key}', str(value)]
    start = time.time()
    main(argv)
    duration = time.time() - start
  print('Run duration:', duration)
  return duration


def make_data(env, batch_dims):
  spaces = list(env.obs_space.items()) + list(env.act_space.items())
  data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
  for dim in reversed(batch_dims):
    data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
  return data
