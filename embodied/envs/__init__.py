import functools

import embodied

from .atari import Atari
from .crafter import Crafter
from .dmc import DMC
from .dmlab import DMLab
from .dummy import Dummy
from .gym import Gym
from .hrlgrid import HRLGrid
from .loconav import LocoNav
from .minecraft import Minecraft
from .a1 import A1


def load_env(
    task, amount=1, parallel='none', daemon=False, restart=False, seed=None,
    kbreset=False, **kwargs):
  ctors = []
  for index in range(amount):
    ctor = functools.partial(load_single_env, task, **kwargs)
    if seed is not None:
      ctor = functools.partial(ctor, seed=hash((seed, index)) % (2 ** 31 - 1))
    if parallel != 'none':
      ctor = functools.partial(embodied.Parallel, ctor, parallel, daemon)
    if restart:
      ctor = functools.partial(embodied.wrappers.RestartOnException, ctor)
    if kbreset:
      from .kbreset import KBReset
      ctor = functools.partial(KBReset, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(parallel != 'none'))


def load_single_env(
    task, size=(64, 64), repeat=1, mode='train', camera=-1, gray=False,
    length=0, logdir='/dev/null', discretize=0, sticky=True, lives=False,
    episodic=True, resets=True, seed=None):
  suite, task = task.split('_', 1)
  if suite == 'dummy':
    env = Dummy(task, size, length or 100)
  elif suite == 'gym':
    env = Gym(task)
  elif suite == 'a1':
    assert size == (64, 64), size
    env = A1(task, repeat, length or 1000, True)
  elif suite == 'xarm':
    assert size == (64, 64), size
    # from .xarm import XArm
    from .robot_interface import PickPlace, EnvConfig, RobotType
    assert task in ('real', 'dummy')
    env = PickPlace(
      EnvConfig(
        use_real=task == 'real',
        robot_type=RobotType.XARM,
        enable_z=True
      )
    )
  elif suite == 'ur5':
    assert size == (64, 64), size
    # from .xarm import XArm
    from .robot_interface import PickPlace, EnvConfig, RobotType
    assert task in ('real', 'dummy')
    env = PickPlace(EnvConfig(use_real=task == 'real', robot_type=RobotType.UR5))
  elif suite == 'sphero':
    from .sphero import SpheroEnv, EnvConfig
    assert task in ('real', 'dummy')
    env = SpheroEnv(EnvConfig(use_real=task == 'real'))
  elif suite == 'dmc':
    env = DMC(task, repeat, size, camera)
  elif suite == 'atari':
    env = Atari(task, repeat, size, gray, lives=lives, sticky=sticky)
  elif suite == 'crafter':
    assert repeat == 1
    outdir = embodied.Path(logdir) / 'crafter' if mode == 'train' else None
    env = Crafter(task, size, outdir)
  elif suite == 'dmlab':
    env = DMLab(task, repeat, size, mode, seed=seed, episodic=episodic)
  elif suite == 'minecraft':
    env = Minecraft(task, repeat, size)
  elif suite == 'loconav':
    env = LocoNav(task, repeat, size, camera)
  elif suite == 'hrlgrid':
    assert repeat == 1
    assert size == (64, 64)
    env = HRLGrid(int(task), length or 1000)
  else:
    raise NotImplementedError(suite)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    if space.discrete:
      env = embodied.wrappers.OneHotAction(env, name)
    elif discretize:
      env = embodied.wrappers.DiscretizeAction(env, name, discretize)
    else:
      env = embodied.wrappers.NormalizeAction(env, name)
  if length:
    env = embodied.wrappers.TimeLimit(env, length, resets)
  return env


__all__ = [
    k for k, v in list(locals().items())
    if type(v).__name__ in ('type', 'function') and not k.startswith('_')]
