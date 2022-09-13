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

import numpy as np
from pybullet_utils import transformations

from motion_imitation.robots import a1
from motion_imitation.utilities import motion_data
from motion_imitation.utilities import pose3d

SITTING_POSE = np.array([0.0, 0.0, 0.11, 0.0, 0.0, 0.0, 1.0] + [0,  1.17752553, -2.69719727]*4)
STANDING_POSE = np.array([0.0, 0.0, 0.25870023, 0.0, 0.0, 0.0, 1.0] + [0, 0.9, -1.8] * 4)

JOINT_WEIGHTS = np.array([1.0, 0.75, 0.5] * 4)

class ResetTask(object):
  """Imitation reference motion task."""

  def __init__(self, terminal_conditions=(), real_robot=False):
    self._terminal_conditions = terminal_conditions
    self._env = None
    self._default_pose = None

    self._joint_pose_idx = None
    self._joint_pose_size = None
    
    self._stand_prob = 0.2
    self._sit_prob = 0.2

    self._fall_init_rot_x_min = -3.0 / 4.0 * np.pi
    self._fall_init_rot_x_max = 3.0 / 4.0 * np.pi

    self._real_robot = real_robot

    return

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    assert(self._stand_prob + self._sit_prob <= 1.0)

    self._env = env

    if (self._joint_pose_idx is None or self._env.hard_reset):
      self._build_joint_data()

    if self._real_robot:
      return 

    rand_val = self._rand_uniform(0, 1)
    if (rand_val < self._stand_prob):
      self._init_stand_pose()
    elif (rand_val < self._stand_prob + self._sit_prob):
      self._init_sit_pose()
    else:
      self._init_fall_pose(self._fall_init_rot_x_min, self._fall_init_rot_x_max)

    return

  def update(self, env):
    """Updates the internal state of the task."""
    del env
    return

  def done(self, env):
    """Checks if the episode is over."""
    del env
    done = any([done_fn(self._env) for done_fn in self._terminal_conditions])
    return done

  def build_target_obs(self):
    tar_obs = np.array([])
    return tar_obs

  def get_target_obs_bounds(self):
    low = np.array([])
    high = np.array([])
    return low, high

  def reward(self, env):
    """Get the reward without side effects."""
    del env

    roll_w = 0.5
    stand_w = 0.5
    roll_threshold = np.cos(0.2 * np.pi)
    
    r_roll, root_cos_dist = self._calc_reward_roll()

    if (root_cos_dist > roll_threshold):
      r_stand = self._calc_reward_stand()
    else:
      r_stand = 0.0

    reward = roll_w * r_roll + stand_w * r_stand

    return reward

  def _calc_reward_roll(self):
    up = np.array([0, 0, 1])
    root_rot = self._env.robot.GetTrueBaseOrientation()
    root_up = pose3d.QuaternionRotatePoint(up, root_rot)
    cos_dist = up.dot(root_up)

    r_roll = (0.5 * cos_dist + 0.5) ** 2

    return r_roll, cos_dist

  def _calc_reward_stand(self):
    tar_h = STANDING_POSE[2]
    pos_size = motion_data.MotionData.POS_SIZE
    rot_size = motion_data.MotionData.ROT_SIZE

    root_pos = self._env.robot.GetBasePosition()
    root_h = root_pos[2]
    h_err = tar_h - root_h
    h_err /= tar_h
    h_err = np.clip(h_err, 0.0, 1.0)
    r_height = 1.0 - h_err

    tar_pose = STANDING_POSE[(pos_size + rot_size):]
    joint_pose = self._env.robot.GetTrueMotorAngles()
    pose_diff = tar_pose - joint_pose
    pose_diff = JOINT_WEIGHTS * JOINT_WEIGHTS * pose_diff * pose_diff
    pose_err = np.sum(pose_diff)
    r_pose = np.exp(-0.6 * pose_err)

    tar_vel = 0.0
    joint_vel = self._env.robot.GetMotorVelocities()
    vel_diff = tar_vel - joint_vel
    vel_diff = vel_diff * vel_diff
    vel_err = np.sum(vel_diff)
    r_vel = np.exp(-0.02 * vel_err)

    r_stand = 0.2 * r_height + 0.6 * r_pose + 0.2 * r_vel

    return r_stand

  def _calc_reward_end_effector(self, ref_joint_angles):
    """Get the end effector reward for sim or real A1 robot."""
    pos_size = motion_data.MotionData.POS_SIZE
    rot_size = motion_data.MotionData.ROT_SIZE

    ref_base_pos = np.array(STANDING_POSE[:pos_size])
    ref_base_rot = np.array(STANDING_POSE[pos_size:(pos_size + rot_size)])
    rel_feet_pos_ref = a1.foot_positions_in_base_frame(ref_joint_angles)
    rel_feet_pos_robot = self._env.robot.GetFootPositionsInBaseFrame()
    end_eff_err = 0

    for rel_foot_pos_ref, rel_foot_pos_robot in zip(rel_feet_pos_ref,
                                                    rel_feet_pos_robot):
      rel_foot_pos_diff = rel_foot_pos_ref - rel_foot_pos_robot
      end_eff_err += rel_foot_pos_diff[0]**2 + rel_foot_pos_diff[1]**2

      foot_height_ref = pose3d.PoseTransformPoint(
          point=rel_foot_pos_ref,
          position=ref_base_pos,
          quat=ref_base_rot)[2]

      foot_height_robot = pose3d.PoseTransformPoint(
          point=rel_foot_pos_robot,
          position=self._env.robot.GetBasePosition(),
          quat=self._env.robot.GetBaseOrientation())[2]

      end_eff_err += 3.0 * (foot_height_ref - foot_height_robot)**2

    r_end_eff = np.exp(-40 * end_eff_err)
    return r_end_eff

  def _get_pybullet_client(self):
    """Get bullet client from the environment"""
    return self._env._pybullet_client

  def _get_num_joints(self):
    """Get the number of joints in the character's body."""
    pyb = self._get_pybullet_client()
    return pyb.getNumJoints(self._env.robot.quadruped)

  def _init_stand_pose(self):
    self._set_pose(STANDING_POSE)
    self._env.robot.ReceiveObservation()
    return
  
  def _init_sit_pose(self):
    self._set_pose(SITTING_POSE)
    self._env.robot.ReceiveObservation()
    return

  def _init_fall_pose(self, rot_x_min, rot_x_max):
    pyb = self._get_pybullet_client()
    pos_size = motion_data.MotionData.POS_SIZE
    rot_size = motion_data.MotionData.ROT_SIZE

    pose = self._get_pose()
    root_pos = np.array([0, 0, self._rand_uniform(low=0.4, high=0.5)])
    root_rot = self._rand_uniform(low=[rot_x_min, -np.pi/4, -np.pi],
                                 high=[rot_x_max, np.pi/4, np.pi])
    root_rot = pyb.getQuaternionFromEuler(root_rot)

    joint_lim_low = self._env.robot._joint_angle_lower_limits
    joint_lim_high = self._env.robot._joint_angle_upper_limits
    joint_pose_size = len(joint_lim_low)
    
    stand_pose = STANDING_POSE[-joint_pose_size:]
    joint_dir = self._randint(0, 2, joint_pose_size).astype(np.float32)
    lim_pose = (1.0 - joint_dir) * joint_lim_low + joint_dir * joint_lim_high
    
    pose_lerp = self._rand_uniform(low=0, high=1, size=joint_pose_size)
    pose_lerp = pose_lerp * pose_lerp * pose_lerp
    joint_pose = (1.0 - pose_lerp) * stand_pose + pose_lerp * lim_pose

    pose = np.concatenate([root_pos, root_rot, joint_pose])
    self._set_pose(pose)

    for _ in range(1000):
      pyb.stepSimulation()

    self._env.robot.ReceiveObservation()
    
    return
  
  def _build_joint_data(self):
    """Precomputes joint data to facilitating accessing data from motion frames."""
    num_joints = self._get_num_joints()
    self._joint_pose_idx = np.zeros(num_joints, dtype=np.int32)
    self._joint_pose_size = np.zeros(num_joints, dtype=np.int32)

    for j in range(num_joints):
      pyb = self._get_pybullet_client()
      j_info = pyb.getJointInfo(self._env.robot.quadruped, j)
      j_state = pyb.getJointStateMultiDof(self._env.robot.quadruped, j)

      j_pose_idx = j_info[3]
      j_pose_size = len(j_state[0])

      if (j_pose_idx < 0):
        assert (j_pose_size == 0)
        if (j == 0):
          j_pose_idx = 0
        else:
          j_pose_idx = self._joint_pose_idx[j - 1] + self._joint_pose_size[j - 1]

      self._joint_pose_idx[j] = j_pose_idx
      self._joint_pose_size[j] = j_pose_size

    return

  def _get_pose(self):
    root_pos = self._env.robot.GetBasePosition()
    root_rot = self._env.robot.GetTrueBaseOrientation()
    joint_pose = self._env.robot.GetTrueMotorAngles()
    pose = np.concatenate([root_pos, root_rot, joint_pose])
    return pose
  
  def _set_pose(self, pose):
    """Set the state of a character to the given pose and velocity.

    Args:
      phys_model: handle of the character
      pose: pose to be applied to the character
      vel: velocity to be applied to the character
    """
    pyb = self._get_pybullet_client()
    phys_model = self._env.robot.quadruped

    root_pos = pose[0:motion_data.MotionData.POS_SIZE]
    root_rot = pose[motion_data.MotionData.POS_SIZE:(motion_data.MotionData.POS_SIZE + motion_data.MotionData.ROT_SIZE)]
    pyb.resetBasePositionAndOrientation(phys_model, root_pos, root_rot)

    num_joints = self._get_num_joints()
    for j in range(num_joints):
      q_idx = self._get_joint_pose_idx(j)
      q_size = self._get_joint_pose_size(j)

      if (q_size > 0):
        j_pose = pose[q_idx:(q_idx + q_size)]
        j_vel = np.zeros_like(j_pose)
        pyb.resetJointStateMultiDof(phys_model, j, j_pose, j_vel)

    return

  def _get_joint_pose_idx(self, j):
    """Get the starting index of the pose data for a give joint in a pose array."""
    idx = self._joint_pose_idx[j]
    return idx

  def _get_joint_pose_size(self, j):
    """Get the size of the pose data for a give joint in a pose array."""
    pose_size = self._joint_pose_size[j]
    assert (pose_size == 1 or
            pose_size == 0), "Only support 1D and 0D joints at the moment."
    return pose_size

  def _rand_uniform(self, low, high, size=None):
    """Samples random float between [val_min, val_max]."""
    if hasattr(self._env, "np_random"):
      rand_val = self._env.np_random.uniform(low=low, high=high, size=size)
    else:
      rand_val = np.random.uniform(low=low, high=high, size=size)
    return rand_val

  def _randint(self, low, high, size=None):
    """Samples random integer between [val_min, val_max]."""
    if hasattr(self._env, "np_random"):
      rand_val = self._env.np_random.randint(low, high, size=size)
    else:
      rand_val = np.random.randint(low, high, size=size)
    return rand_val


class RollTask(ResetTask):
  """Imitation reference motion task."""

  def __init__(self, terminal_conditions=()):
    super().__init__(terminal_conditions)

    self._stand_prob = 0.0
    self._sit_prob = 0.2

    return

  def reward(self, env):
    """Get the reward without side effects."""
    del env
    
    r_roll, _ = self._calc_reward_roll()
    torques = self._env.robot.GetMotorTorques()
    torque_penalty = np.sum(np.abs(torques))

    reward = r_roll - 1e-3 * torque_penalty

    return reward

class StandTask(ResetTask):
  """Imitation reference motion task."""

  def __init__(self, terminal_conditions=()):
    super().__init__(terminal_conditions)

    self._fall_init_rot_x_min = -1.0 / 4.0 * np.pi
    self._fall_init_rot_x_max = 1.0 / 4.0 * np.pi

    return
  
  def reward(self, env):
    """Get the reward without side effects."""
    del env
    
    r_stand = self._calc_reward_stand()
    reward = r_stand

    return reward
