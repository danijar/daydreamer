"""Generates a random bumpy terrain at environment reset."""

import numpy as np
from pybullet_envs.minitaur.envs import env_randomizer_base


class HeightfieldRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Generates an uneven terrain in the gym env."""

  def __init__(self, max_height_perturbation=.05):
    """Initializes the randomizer.

    Args:
      max_height_perturbation: Max height of bumps in meters.
    """
    self._max_height_perturbation = max_height_perturbation
    self._terrain_shape = -1
    self._initial = True
    self._n_rows = 128
    self._n_cols = 128
    self._heightfield_data = [0] * self._n_rows * self._n_cols
    self.terrain = None

  def randomize_env(self, env):
    for j in range(int(self._n_rows / 2)):
      for i in range(int(self._n_cols / 2)):
        height = np.random.uniform(0, self._max_height_perturbation)
        self._heightfield_data[2 * i + 2 * j * self._n_rows] = height
        self._heightfield_data[2 * i + 1 + 2 * j * self._n_rows] = height
        self._heightfield_data[2 * i + (2 * j + 1) * self._n_rows] = height
        self._heightfield_data[2 * i + 1 + (2 * j + 1) * self._n_rows] = height

    # Rendering while loading is slow.
    if env.rendering_enabled:
      env.pybullet_client.configureDebugVisualizer(
          env.pybullet_client.COV_ENABLE_RENDERING, 0)

    self._terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        flags=env.pybullet_client.GEOM_CONCAVE_INTERNAL_EDGE,
        meshScale=[.15, .15, 1],
        heightfieldData=self._heightfield_data,
        numHeightfieldRows=self._n_rows,
        numHeightfieldColumns=self._n_cols,
        replaceHeightfieldIndex=self._terrain_shape)
    if self._initial:
      env.pybullet_client.removeBody(env.get_ground())
      self.terrain = env.pybullet_client.createMultiBody(0, self._terrain_shape)
      env.set_ground(self.terrain)
      self._initial = False
      texture_id = env.pybullet_client.loadTexture("checker_blue.png")
      env.pybullet_client.changeVisualShape(
          self.terrain, -1, textureUniqueId=texture_id, rgbaColor=(1, 1, 1, 1))
    # Center terrain under robot in case robot is resetting in place.
    x, y, _ = env.robot.GetBasePosition()
    env.pybullet_client.resetBasePositionAndOrientation(self.terrain, [x, y, 0],
                                                        [0, 0, 0, 1])

    if env.rendering_enabled:
      env.pybullet_client.configureDebugVisualizer(
          env.pybullet_client.COV_ENABLE_RENDERING, 1)
