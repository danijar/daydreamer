# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pytype: disable=attribute-error
"""Real robot interface of A1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from absl import logging
import math
import re
import multiprocessing
import numpy as np
import time

from motion_imitation.robots import laikago_pose_utils
from motion_imitation.robots import a1
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config
from robot_interface import RobotInterface  # pytype: disable=import-error

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

_DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.13, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.13, 0),
)

ABDUCTION_P_GAIN = 80.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 80.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 80.0
KNEE_D_GAIN = 2.0

MOTOR_KPS = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * 4
MOTOR_KDS = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * 4
# If any motor is above this temperature (Celsius), a warning will be printed.
# At 60C, Unitree will shut down a motor until it cools off.
MOTOR_WARN_TEMP_C = 50.0

COMMAND_CHANNEL_NAME = 'LCM_Low_Cmd'
STATE_CHANNEL_NAME = 'LCM_Low_State'

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3


class A1Robot(a1.A1):
  """Interface for real A1 robot."""
  MPC_BODY_MASS = 108 / 9.8
  MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))

  MPC_BODY_HEIGHT = 0.24
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="FR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="FL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
  ]
  # Strictly enforce joint limits on the real robot, for safety.
  JOINT_EPSILON = 0.0

  def __init__(self,
               pybullet_client,
               time_step=0.001,
               enable_clip_motor_commands=True,
               reset_func_name='_StandupReset',
               **kwargs):
    """Initializes the robot class."""
    # self._timesteps = None
    # Initialize pd gain vector
    self._pybullet_client = pybullet_client
    self.time_step = time_step

    # Robot state variables
    self._init_complete = False
    self._base_position = np.zeros((3,))
    self._base_orientation = None
    self._last_position_update_time = time.time()
    self._raw_state = None
    self._last_raw_state = None
    self._motor_angles = np.zeros(12)
    self._motor_velocities = np.zeros(12)
    self._motor_temperatures = np.zeros(12)
    self._joint_states = None
    self._last_reset_time = time.time()

    # Initiate UDP for robot state and actions
    self._robot_interface = RobotInterface(0xff)
    self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
    # Re-entrant lock to ensure one process commands the robot at a time.
    self._robot_command_lock = multiprocessing.RLock()
    self._pipe = None
    self._child_pipe = None
    self._hold_process = None
    if 'velocity_source' in kwargs:
      del kwargs['velocity_source']

    reset_func_name='_StandupReset'
    super(A1Robot, self).__init__(
        pybullet_client,
        time_step=time_step,
        enable_clip_motor_commands=enable_clip_motor_commands,
        velocity_source=a1.VelocitySource.IMU_FOOT_CONTACT,
        reset_func_name=reset_func_name,
        **kwargs)
    self._init_complete = True

  def ReceiveObservation(self):
    """Receives observation from robot.

    Synchronous ReceiveObservation is not supported in A1,
    so changging it to noop instead.
    """
    state = self._robot_interface.receive_observation()
    self._raw_state = state
    # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
    q = state.imu.quaternion
    self._base_orientation = np.array([q[1], q[2], q[3], q[0]])
    self._accelerometer_reading = np.array(state.imu.accelerometer)
    self._motor_angles = np.array([motor.q for motor in state.motorState[:12]])
    self._motor_velocities = np.array(
        [motor.dq for motor in state.motorState[:12]])
    self._joint_states = np.array(
        list(zip(self._motor_angles, self._motor_velocities)))
    self._observed_motor_torques = np.array(
        [motor.tauEst for motor in state.motorState[:12]])
    self._motor_temperatures = np.array(
        [motor.temperature for motor in state.motorState[:12]])
    if self._init_complete:
      # self._SetRobotStateInSim(self._motor_angles, self._motor_velocities)
      self._velocity_estimator.update(state.tick / 1000.)
      self._UpdatePosition()

  def _CheckMotorTemperatures(self):
    if any(self._motor_temperatures > MOTOR_WARN_TEMP_C):
      print("WARNING: Motors are getting hot. Temperatures:")
      for name, temp in zip(MOTOR_NAMES, self._motor_temperatures.astype(int)):
        print(f"{name}: {temp} C")

  def _UpdatePosition(self):
    now = time.time()
    self._base_position += self.GetBaseVelocity() * (now - self._last_position_update_time)
    self._last_position_update_time = now

  def _SetRobotStateInSim(self, motor_angles, motor_velocities):
    self._pybullet_client.resetBasePositionAndOrientation(
        self.quadruped, self.GetBasePosition(), self.GetBaseOrientation())
    for i, motor_id in enumerate(self._motor_id_list):
      self._pybullet_client.resetJointState(self.quadruped, motor_id,
                                            motor_angles[i],
                                            motor_velocities[i])

  def GetTrueMotorAngles(self):
    # TODO
    # return self._motor_angles.copy()
    state = self._robot_interface.receive_observation()
    return np.array([motor.q for motor in state.motorState[:12]])

  def GetMotorAngles(self):
    return minitaur.MapToMinusPiToPi(self._motor_angles).copy()

  def GetMotorVelocities(self):
    return self._motor_velocities.copy()

  def GetBasePosition(self):
    return self._base_position.copy()

  def GetBaseRollPitchYaw(self):
    return self._pybullet_client.getEulerFromQuaternion(self._base_orientation)

  def GetTrueBaseRollPitchYaw(self):
    return self._pybullet_client.getEulerFromQuaternion(self._base_orientation)

  def GetBaseRollPitchYawRate(self):
    return self.GetTrueBaseRollPitchYawRate()

  def GetTrueBaseRollPitchYawRate(self):
    return np.array(self._raw_state.imu.gyroscope).copy()

  def GetBaseVelocity(self):
    return self._velocity_estimator.estimated_velocity.copy()

  def GetFootContacts(self):
    return np.array(self._raw_state.footForce) > 20

  def GetTimeSinceReset(self):
    return time.time() - self._last_reset_time

  def GetBaseOrientation(self):
    return self._base_orientation.copy()

  @property
  def motor_velocities(self):
    return self._motor_velocities.copy()

  @property
  def motor_temperatures(self):
    return self._motor_temperatures.copy()

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    if motor_control_mode is None:
      motor_control_mode = self._motor_control_mode

    motor_commands = self._ClipMotorCommands(motor_commands, motor_control_mode)

    command = np.zeros(60, dtype=np.float32)
    if motor_control_mode == robot_config.MotorControlMode.POSITION:
      for motor_id in range(NUM_MOTORS):
        command[motor_id * 5] = motor_commands[motor_id]
        command[motor_id * 5 + 1] = MOTOR_KPS[motor_id]
        command[motor_id * 5 + 3] = MOTOR_KDS[motor_id]
    elif motor_control_mode == robot_config.MotorControlMode.TORQUE:
      for motor_id in range(NUM_MOTORS):
        command[motor_id * 5 + 4] = motor_commands[motor_id]
    elif motor_control_mode == robot_config.MotorControlMode.HYBRID:
      command = np.array(motor_commands, dtype=np.float32)
    else:
      raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
          motor_control_mode))

    with self._robot_command_lock:
      self._robot_interface.send_command(command)

  def _HoldPose(self, pose, pipe):
    """Continually sends position command `pose` until `pipe` has a message.

    This method is intended to be run in its own process by HoldCurrentPose().
    """
    # Clear self._hold_process to make ReleasePose() a no-op in this process
    # (it must be called in the parent process). This does not affect the parent
    # process's self._hold_process.
    self._hold_process = None
    error = None
    with self._robot_command_lock:
      while not pipe.poll():
        self._Nap()
        # If a safety error has been encountered, spin without applying actions
        # until signalled to stop. This way self._robot_command_lock is retained
        # to avoid another process accidentally commanding the robot.
        if error is not None:
          continue
        try:
          self._ValidateMotorStates()
        except (robot_config.SafetyError) as e:
          error = e
          continue
        self.ApplyAction(
            pose, motor_control_mode=robot_config.MotorControlMode.POSITION)
    pipe.send(error)

  def HoldCurrentPose(self):
    """Starts a process to continually command the A1's current joint angles.

    Calling Step(), Brake(), or ReleasePose() will kill the subprocess and stop
    holding the pose. Ending the main python process (for example with a normal
    return or ctrl-c) will also kill the subprocess.
    """
    if self._hold_process is not None:
      return
    # Set self._child_pipe to prevent its being garbage collected.
    self._pipe, self._child_pipe = multiprocessing.Pipe()
    self._hold_process = multiprocessing.Process(
        target=self._HoldPose, args=(self.GetMotorAngles(), self._child_pipe))
    self._hold_process.start()

  def ReleasePose(self):
    """If a subprocess is holding a pose, stops the subprocess."""
    if self._hold_process is None:
      return
    self._pipe.send(None)
    self._hold_process.join()
    maybe_error = self._pipe.recv()
    if maybe_error is not None:
      print(maybe_error)
      self._is_safe = False
    self._pipe.close()
    self._child_pipe.close()
    self._hold_process.close()
    self._pipe = None
    self._child_pipe = None
    self._hold_process = None

  def Step(self, action, control_mode=None):
    """Steps simulation."""
    self.ReleasePose()
    super().Step(action, control_mode)
    self._CheckMotorTemperatures()

  def _StandupReset(self, default_motor_angles, reset_time):
    print("Stand up reset called!", reset_time, default_motor_angles)
    if reset_time <= 0:
      return
    # Stand up in 1.5 seconds, and keep the behavior in this way.
    standup_time = 1.5

    if not default_motor_angles:
      default_motor_angles = a1.INIT_MOTOR_ANGLES
    current_motor_angles = self.GetMotorAngles()

    for t in np.arange(0, standup_time, self.time_step * self._action_repeat):
      blend_ratio = min(t / standup_time, 1)
      action = blend_ratio * default_motor_angles + (
          1 - blend_ratio) * current_motor_angles
      self.Step(action, robot_config.MotorControlMode.POSITION)

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    """Reset the robot to default motor angles."""
    self._base_position[2] = 0
    self._last_position_update_time = time.time()
    super(A1Robot, self).Reset(reload_urdf=reload_urdf,
                               default_motor_angles=default_motor_angles,
                               reset_time=-1)
    self._currently_resetting = True
    self._reset_func(default_motor_angles, reset_time)

    if self._enable_action_filter:
      self._ResetActionFilter()

    self._velocity_estimator.reset()
    self._state_action_counter = 0
    self._step_counter = 0
    self._last_reset_time = time.time()
    self._currently_resetting = False
    self._last_action = None

  def Terminate(self):
    self.Brake()
    self._is_alive = False

  def _StepInternal(self, action, motor_control_mode=None):
    if self._is_safe:
      self.ApplyAction(action, motor_control_mode)
    self.ReceiveObservation()
    self._state_action_counter += 1
    if not self._is_safe:
      return
    try:
      self._ValidateMotorStates()
    except(robot_config.SafetyError) as e:
      print(e)
      if self.running_reset_policy:
        # Let the resetter handle retries.
        raise e
      self._is_safe = False
      return
    self._Nap()

  def _Nap(self):
    """Sleep for the remainder of self.time_step."""
    now = time.time()
    sleep_time = self.time_step - (now - self._last_step_time_wall)
    if self._timesteps is not None:
      self._timesteps.append(now - self._last_step_time_wall)
    self._last_step_time_wall = now
    if sleep_time >= 0:
      time.sleep(sleep_time)

  def Brake(self):
    self.ReleasePose()
    self._robot_interface.brake()
    self.LogTimesteps()
    self._Nap()
