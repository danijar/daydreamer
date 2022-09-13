"""Drops the robot with random orientation at episode start."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
from motion_imitation.envs.utilities import env_randomizer_base


class FallenRobotRandomizer(env_randomizer_base.EnvRandomizerBase):

  def __init__(self, max_roll_pitch=np.pi):
    self._max_roll_pitch = max_roll_pitch

  def randomize_env(self, env):
    env.robot.ResetPose(add_constraint=False)
    sampled_orientation = np.random.uniform(
        low=[-self._max_roll_pitch, -self._max_roll_pitch, -np.pi],
        high=[self._max_roll_pitch, self._max_roll_pitch, np.pi])
    env.pybullet_client.resetBasePositionAndOrientation(
        bodyUniqueId=env.robot.quadruped,
        posObj=[0, 0, np.random.uniform(low=.3, high=.8)],
        ornObj=env.pybullet_client.getQuaternionFromEuler(sampled_orientation))
    for _ in range(1000):
      env.pybullet_client.stepSimulation()
