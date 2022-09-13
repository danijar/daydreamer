"""Env wrapper that saves logs."""

import atexit
import os

import numpy as np
from phasespace import phasespace_robot_tracker


class LoggingWrapper(object):
  """Env wrapper that saves logs."""

  def __init__(self,
               env,
               output_dir,
               mocap_grpc_server=None,
               verbose=True,
               separate_episodes=False):
    """Constructor.

    Args:
      env: An instance (possibly wrapped) of LocomotionGymEnv.
      output_dir: Where to save logs.
      mocap_grpc_server: Hostname and port of the gRPC server outputting marker
        data protos
        (e.g. "localhost:12345"). If None, don't look for mocap data.
      verbose: If True, print a message every time a log is saved.
      separate_episodes: If True, save one log file per episode. If False, save
        all episodes as one log file.
    """
    if mocap_grpc_server:
      self._mocap_tracker = phasespace_robot_tracker.PhaseSpaceRobotTracker(
          server=mocap_grpc_server)
    else:
      self._mocap_tracker = None
    self._env = env
    self._robot = self._env.robot
    self._output_dir = output_dir
    os.makedirs(self._output_dir, exist_ok=True)
    self._verbose = verbose
    self._separate_episodes = separate_episodes
    self._clear_logs()
    self._episode_counter = 0
    atexit.register(self.log, verbose=True)

  def __getattr__(self, attr):
    return getattr(self._env, attr)

  def _clear_logs(self):
    self._linear_vels = []
    self._rpys = []
    self._angular_vels = []
    self._timestamps = []
    self._input_actions = []
    self._processed_actions = []
    self._joint_angles = []
    self._motor_temperatures = []
    self._mocap_positions = []
    self._mocap_rpys = []

  def step(self, action):
    self._input_actions.append(action)
    if self._mocap_tracker:
      self._mocap_tracker.update()
    obs, reward, done, info = self._env.step(action)

    self._processed_actions.append(self._robot.last_action)
    self._linear_vels.append(self._robot.GetBaseVelocity())
    self._rpys.append(self._robot.GetBaseRollPitchYaw())
    self._angular_vels.append(self._robot.GetBaseRollPitchYawRate())
    self._joint_angles.append(self._robot.GetMotorAngles())
    self._timestamps.append(self._robot.GetTimeSinceReset())
    if hasattr(self._robot, "motor_temperatures"):
      self._motor_temperatures.append(self._robot.motor_temperatures)
    if self._mocap_tracker:
      self._mocap_positions.append(self._mocap_tracker.get_base_position())
      self._mocap_rpys.append(self._mocap_tracker.get_base_roll_pitch_yaw())

    return obs, reward, done, info

  def log(self, verbose):
    if self._separate_episodes:
      out_file = os.path.join(
          self._output_dir,
          "log_episode_{:07d}.npz".format(self._episode_counter))
    else:
      out_file = os.path.join(self._output_dir, "log_all_episodes.npz")
    np.savez(
        out_file,
        input_actions=self._input_actions,
        processed_actions=self._processed_actions,
        timestamps=self._timestamps,
        linear_vels=self._linear_vels,
        rpys=self._rpys,
        angular_vels=self._angular_vels,
        joint_angles=self._joint_angles,
        motor_temperatures=self._motor_temperatures,
        mocap_positions=self._mocap_positions,
        mocap_rpys=self._mocap_rpys,
        )
    if verbose:
      print("logged to: {}".format(out_file))
    self._clear_logs()

  def reset(self, *args, **kwargs):
    if self._separate_episodes:
      self.log(self._verbose)
    self._episode_counter += 1
    return self._env.reset(*args, **kwargs)
