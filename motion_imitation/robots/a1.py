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

"""Pybullet simulation of a Laikago robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import enum
import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error
import time

from motion_imitation.robots import a1_robot_velocity_estimator
from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config

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
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2  # TODO
# TODO: Find appropriate limits.
MAX_JOINT_VELOCITY = np.inf  # rad/s (was 11)
MAX_TORQUE = 42  # N-m  # TODO: 45

# _DEFAULT_HIP_POSITIONS = (
#     (0.17, -0.135, 0),
#     (0.17, 0.13, 0),
#     (-0.195, -0.135, 0),
#     (-0.195, 0.13, 0),
# )

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)

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

MOTOR_MINS = np.array([
    -0.802851455917,
    -1.0471975512,
    -2.69653369433,
] * 4)

MOTOR_MAXS = np.array([
    0.802851455917,
    4.18879020479,
    -0.916297857297,
] * 4)

MOTOR_OFFSETS = np.array([
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])

MOTOR_USED = np.array([
    [0.01, 0.99],
    [0.01, 0.90],
    [0.01, 0.60],
] * 4)

STANDING_POSE = np.array([0, -0.2, 1.0] * 4)


def unnormalize_action(action, clip=True):
  if clip:
    action = np.clip(action, -1, 1)
  action = action / 2 + 0.5
  lo = MOTOR_MINS * (1 - MOTOR_USED[:, 0]) + MOTOR_MAXS * MOTOR_USED[:, 0]
  hi = MOTOR_MINS * (1 - MOTOR_USED[:, 1]) + MOTOR_MAXS * MOTOR_USED[:, 1]
  action = action * (hi - lo) + lo
  action += MOTOR_OFFSETS
  return action

def normalize_action(action, clip=True):
  action -= MOTOR_OFFSETS
  lo = MOTOR_MINS * (1 - MOTOR_USED[:, 0]) + MOTOR_MAXS * MOTOR_USED[:, 0]
  hi = MOTOR_MINS * (1 - MOTOR_USED[:, 1]) + MOTOR_MAXS * MOTOR_USED[:, 1]
  action = (action - lo) / (hi - lo)
  action = (action - 0.5) * 2
  if clip:
    action = np.clip(action, -1, 1)
  return action



# INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)
# print(normalize_action(INIT_MOTOR_ANGLES))
# print(STANDING_POSE)
# import sys; sys.exit()


HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = os.path.join(parentdir, "motion_imitation/utilities/a1/a1.urdf")

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

# Empirical values from real A1.
ACCELEROMETER_VARIANCE = 0.03059
JOINT_VELOCITY_VARIANCE = 0.006206


class VelocitySource(enum.Enum):
  PYBULLET = 0
  IMU_FOOT_CONTACT = 1


# Found that these numba.jit decorators slow down the timestep from 1ms without
# to 5ms with decorators.
# @numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * l_hip_sign
  x, y, z = foot_position[0], foot_position[1], foot_position[2]
  theta_knee = -np.arccos(
      (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
      (2 * l_low * l_up))
  l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
  theta_hip = np.arcsin(-x / l) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)
  return np.array([theta_ab, theta_hip, theta_knee])


# @numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame(angles, l_hip_sign=1):
  theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * l_hip_sign
  leg_distance = np.sqrt(l_up**2 + l_low**2 +
                         2 * l_up * l_low * np.cos(theta_knee))
  eff_swing = theta_hip + theta_knee / 2

  off_x_hip = -leg_distance * np.sin(eff_swing)
  off_z_hip = -leg_distance * np.cos(eff_swing)
  off_y_hip = l_hip

  off_x = off_x_hip
  off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  return np.array([off_x, off_y, off_z])


# @numba.jit(nopython=True, cache=True)
def analytical_leg_jacobian(leg_angles, leg_id):
  """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a left (1) or right(-1) leg.
  """
  l_up = 0.2
  l_low = 0.2
  l_hip = 0.08505 * (-1)**(leg_id + 1)

  t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
  l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
  t_eff = t2 + t3 / 2
  J = np.zeros((3, 3))
  J[0, 0] = 0
  J[0, 1] = -l_eff * np.cos(t_eff)
  J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
      t_eff) / 2
  J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
  J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
  J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
      t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
  J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
  J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
  J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
      t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
  return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


# @numba.jit(nopython=True, cache=True, parallel=True)
def foot_positions_in_base_frame(foot_angles):
  foot_angles = foot_angles.reshape((4, 3))
  foot_positions = np.zeros((4, 3))
  for i in range(4):
    foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                   l_hip_sign=(-1)**(i + 1))
  return foot_positions + HIP_OFFSETS

class A1(minitaur.Minitaur):
  """A simulation for the Laikago robot."""

  # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
  # doesn't seem to matter much. However, these values should be better tuned
  # when the replan frequency is low (e.g. using a less beefy CPU).
  MPC_BODY_MASS = 108 / 9.8
  MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.
  MPC_BODY_HEIGHT = 0.24
  MPC_VELOCITY_MULTIPLIER = 0.5
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name=key, upper_bound=hi, lower_bound=lo)
      for key, hi, lo in zip(MOTOR_NAMES, MOTOR_MAXS, MOTOR_MINS)]
  INIT_RACK_POSITION = [0, 0, 1]
  INIT_POSITION = [0, 0, 0.25870023]
  INIT_ORIENTATION = (0, 0, 0, 1)
  # Joint angles are allowed to be JOINT_EPSILON outside their nominal range.
  # This accounts for imprecision seen in either pybullet's enforcement of joint
  # limits or its reporting of joint angles.
  JOINT_EPSILON = 0.1


  def __init__(
      self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      self_collision_enabled=False,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      reset_at_current_position=False,
      reset_func_name="_PybulletReset",
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      motor_torque_limits=MAX_TORQUE,
      reset_time=1,
      allow_knee_contact=False,
      log_time_per_step=False,
      observation_noise_stdev=(0.0,) * 6,
      velocity_source=VelocitySource.PYBULLET,
  ):
    """Constructor.

    Args:
      observation_noise_stdev: The standard deviation of a Gaussian noise model
        for the sensor. It should be an array for separate sensors in the
        following order [motor_angle, motor_velocity, motor_torque,
        base_roll_pitch_yaw, base_angular_velocity, base_linear_acceleration]
      velocity_source: How to determine the velocity returned by
        self.GetBaseVelocity().
    """
    self.running_reset_policy = False
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]
    self._joint_angle_lower_limits = np.array(
        [field.lower_bound for field in self.ACTION_CONFIG])
    self._joint_angle_upper_limits = np.array(
        [field.upper_bound for field in self.ACTION_CONFIG])
    if log_time_per_step:
      self._timesteps = []
    else:
      self._timesteps = None
    self._last_step_time_wall = 0
    self._currently_resetting = False
    self._max_vel = 0
    self._max_tau = 0
    self._velocity_estimator = None

    if velocity_source is VelocitySource.IMU_FOOT_CONTACT:
     self._velocity_estimator = a1_robot_velocity_estimator.VelocityEstimator(
         robot=self,
         accelerometer_variance=ACCELEROMETER_VARIANCE,
         sensor_variance=JOINT_VELOCITY_VARIANCE)

    super(A1, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        self_collision_enabled=self_collision_enabled,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        motor_torque_limits=motor_torque_limits,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        observation_noise_stdev=observation_noise_stdev,
        on_rack=on_rack,
        reset_at_current_position=reset_at_current_position,
        reset_func_name=reset_func_name,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time)

  def __del__(self):
    self.LogTimesteps()

  def _LoadRobotURDF(self):
    a1_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)

    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  # def GetHipPositionsInBaseFrame(self):
  #   return _DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def _SafeJointsReset(self, default_motor_angles=None, reset_time=None):
    super()._SafeJointsReset(default_motor_angles, reset_time)
    self.HoldCurrentPose()

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "hip_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "upper_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "lower_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._hip_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._lower_link_ids = []
    self._foot_link_ids = []
    self._imu_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        self._hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        self._lower_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        #assert self._urdf_filename == URDF_WITH_TOES
        self._foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        self._imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._lower_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)

    #assert len(self._foot_link_ids) == NUM_LEGS
    self._hip_link_ids.sort()
    self._motor_link_ids.sort()
    self._lower_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

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
    super(A1, self).ApplyAction(motor_commands, motor_control_mode)

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    self._currently_resetting = True
    super().Reset(
        reload_urdf=reload_urdf,
        default_motor_angles=default_motor_angles,
        reset_time=reset_time)
    self._currently_resetting = False

  def _CollapseReset(self, default_motor_angles, reset_time):
    """Sets joint torques to 0, then moves joints within bounds."""
    del default_motor_angles
    del reset_time
    # Important to fill the observation buffer.
    self.ReceiveObservation()
    # Spend 1 second collapsing.
    half_steps_to_reset = int(0.5 / self.time_step)
    for _ in range(half_steps_to_reset):
      self.Brake()
    for _ in range(half_steps_to_reset):
      self._StepInternal(
          np.zeros((self.num_motors,)),
          motor_control_mode=robot_config.MotorControlMode.TORQUE)
    self._SafeJointsReset()

  def _ClipMotorAngles(self, desired_angles, current_angles):
    if self._enable_clip_motor_commands:
      angle_ub = np.minimum(self._joint_angle_upper_limits + MOTOR_OFFSETS,
                            current_angles + MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
      angle_lb = np.maximum(self._joint_angle_lower_limits + MOTOR_OFFSETS,
                            current_angles - MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
    else:
      angle_ub = self._joint_angle_upper_limits
      angle_lb = self._joint_angle_lower_limits
    return np.clip(desired_angles, angle_lb, angle_ub)

  def _ClipMotorCommands(self, motor_commands, motor_control_mode):
    """Clips commands to respect any set joint angle and torque limits.

    Always clips position to be within ACTION_CONFIG. If
    self._enable_clip_motor_commands, also clips positions to be within
    MAX_MOTOR_ANGLE_CHANGE_PER_STEP of current positions.
    Always clips torques to be within self._motor_torque_limits (but the torque
    limits can be infinity).

    Args:
      motor_commands: np.array. Can be motor angles, torques, or hybrid.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      Clipped motor commands.
    """
    if motor_control_mode == robot_config.MotorControlMode.TORQUE:
      return np.clip(motor_commands, -1 * self._motor_torque_limits, self._motor_torque_limits)
    if motor_control_mode == robot_config.MotorControlMode.POSITION:
      return self._ClipMotorAngles(
          desired_angles=motor_commands,
          current_angles=self.GetTrueMotorAngles())
    if motor_control_mode == robot_config.MotorControlMode.HYBRID:
      # Clip angles
      angles = motor_commands[np.array(range(NUM_MOTORS)) * 5]
      clipped_positions = self._ClipMotorAngles(
          desired_angles=angles,
          current_angles=self.GetTrueMotorAngles())
      motor_commands[np.array(range(NUM_MOTORS)) * 5] = clipped_positions
      # Clip torques
      torques = motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4]
      clipped_torques = np.clip(torques, -1 * self._motor_torque_limits, self._motor_torque_limits)
      motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4] = clipped_torques
      return motor_commands

  def Brake(self):
    # Braking on the real robot has more resistance than this.
    # Call super to avoid doing safety checks while braking.
    super()._StepInternal(
        np.zeros((self.num_motors,)),
        motor_control_mode=robot_config.MotorControlMode.TORQUE)
    self.LogTimesteps()

  def HoldCurrentPose(self):
    """For compatibility with A1Robot."""
    pass

  def _ValidateMotorStates(self):
    # Check torque.
    if any(np.abs(self.GetTrueMotorTorques()) > self._motor_torque_limits):
      raise robot_config.SafetyError(
          "Torque limits exceeded\ntorques: {}".format(
              self.GetTrueMotorTorques()))

    # Check joint velocities.
    if any(np.abs(self.GetTrueMotorVelocities()) > MAX_JOINT_VELOCITY):
      raise robot_config.SafetyError(
          "Velocity limits exceeded\nvelocities: {}".format(
              self.GetTrueMotorVelocities()))

    # Joints often start out of bounds (in sim they're 0 and on real they're
    # slightly out of bounds), so we don't check angles during reset.
    if self._currently_resetting or self.running_reset_policy:
      return
    # Check joint positions.
    # if (any(self.GetTrueMotorAngles() > (self._joint_angle_upper_limits +
    #                                     self.JOINT_EPSILON)) or
    #    any(self.GetTrueMotorAngles() < (self._joint_angle_lower_limits -
    #                                     self.JOINT_EPSILON))):
    #  raise robot_config.SafetyError(
    #      "Joint angle limits exceeded\nangles: {}".format(
    #          self.GetTrueMotorAngles()))

  def _StepInternal(self, action, motor_control_mode=None):
    if self._timesteps is not None:
      now = time.time()
      self._timesteps.append(now - self._last_step_time_wall)
      self._last_step_time_wall = now
    if not self._is_safe:
      return
    super()._StepInternal(action, motor_control_mode)
    # real world
    try:
      self._ValidateMotorStates()
    except robot_config.SafetyError as e:
      print(e)
      self.Brake()
      self._is_safe = False

  def ReceiveObservation(self):
    super().ReceiveObservation()
    if self._velocity_estimator:
      self._velocity_estimator.update(self.GetTimeSinceReset())

  def GetBaseVelocity(self):
    if self._velocity_estimator:
      return self._velocity_estimator.estimated_velocity
    return super().GetBaseVelocity()

  def LogTimesteps(self):
    if self._timesteps is None or not len(self._timesteps):
      return
    timesteps = np.asarray(self._timesteps[1:])
    print('=====\nTimestep stats (secs)\nlen: ', len(timesteps), '\nmean: ',
          np.mean(timesteps), "\nmin: ", np.min(timesteps), "\nmax: ",
          np.max(timesteps), "\nstd: ", np.std(timesteps), "\n=====")

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants

  def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                              foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    assert len(self._foot_link_ids) == self.num_legs

    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = foot_position_in_hip_frame_to_joint_angle(
        foot_local_position - HIP_OFFSETS[leg_id],
        l_hip_sign=(-1)**(leg_id + 1))

    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) -
        np.asarray(self._motor_offset)[joint_position_idxs],
        self._motor_direction[joint_position_idxs])

    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()

  def GetFootPositionsInBaseFrame(self):
    """Get the robot's foot position in the base frame."""
    motor_angles = self.GetMotorAngles()
    return foot_positions_in_base_frame(motor_angles)

  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
    return analytical_leg_jacobian(motor_angles, leg_id)
