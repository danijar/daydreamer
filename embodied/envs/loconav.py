import functools
import os
import warnings

import embodied

from . import dmc


class LocoNav(embodied.Env):

  DEFAULT_CAMERAS = dict(
      ant_trivial=4,
      ant_umaze=4,
  )

  def __init__(self, name, repeat=1, size=(64, 64), camera=-1):
    # TODO: This env variable is necessary when running on a headless GPU but
    # breaks when running on a CPU machine.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    from dm_control import composer
    from dm_control.locomotion.props import target_sphere
    from dm_control.locomotion.tasks import random_goal_maze
    if camera == -1:
      camera = self.DEFAULT_CAMERAS.get(name, 0)
    walker, arena = name.split('_')
    walker = self._make_walker(walker)
    arena = self._make_arena(arena)
    target = target_sphere.TargetSphere(radius=1.0, height_above_ground=0.0)
    task = random_goal_maze.RepeatSingleGoalMaze(
        walker=walker, maze_arena=arena, target=target, max_repeats=1,
        randomize_spawn_rotation=False, target_reward_scale=1.,
        physics_timestep=0.005, control_timestep=0.02)
    def after_step(self, physics, random_state):
      super(random_goal_maze.RepeatSingleGoalMaze, self).after_step(
          physics, random_state)
      self._rewarded_this_step = self._target.activated
      self._targets_obtained = int(self._target.activated)
    task.after_step = functools.partial(after_step, task)
    env = composer.Environment(
        time_limit=30, task=task, random_state=None,
        strip_singleton_obs_buffer_dim=True)
    self._env = dmc.DMC(env, repeat, size, camera)

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
      return self._env.step(action)

  def _make_walker(self, name):
    from dm_control.locomotion.walkers import ant
    if name == 'ant':
      return ant.Ant()
      # observable_options={'egocentric_camera': {'enabled': False}})
    else:
      raise NotImplementedError(name)

  def _make_arena(self, name):
    import labmaze
    from dm_control.locomotion.arenas import mazes
    if name == 'umaze':
      maze = labmaze.FixedMazeWithRandomGoals(
          entity_layer=UMAZE, num_spawns=1, num_objects=1,
          random_state=None)
      arena = mazes.MazeWithTargets(
          maze, xy_scale=1.2, z_height=2.0, aesthetic='default', name='maze')
      return arena
    elif name == 'trivial':
      maze = labmaze.FixedMazeWithRandomGoals(
          entity_layer=TRIVIAL, num_spawns=1, num_objects=1,
          random_state=None)
      arena = mazes.MazeWithTargets(
          maze, xy_scale=1.2, z_height=2.0, aesthetic='default', name='maze')
      return arena
    else:
      raise NotImplementedError(name)


TRIVIAL = """
***********
*         *
*         *
*         *
*******   *
*******   *
*******   *
*         *
* P G     *
*         *
***********
"""[1:]


UMAZE = """
***********
*         *
* G       *
*         *
*******   *
*******   *
*******   *
*         *
* P       *
*         *
***********
"""[1:]
