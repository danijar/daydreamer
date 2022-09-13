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
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self, des_forward_speed=0.6, des_min_angle=np.pi/4.0):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.des_forward_speed = des_forward_speed
    self.up_dir_cutoff = np.array([np.cos(des_min_angle), 0, np.sin(des_min_angle)])
    self.up_dir_cutoff /= np.linalg.norm(self.up_dir_cutoff)
    self.up_dir_cutoff = np.dot(self.up_dir_cutoff, [0, 0, 1])

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    # rot_quat = env.robot.GetBaseOrientation()
    # rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    return False

  def compute_heading_dir(self):
    current_base_orientation = self._env.robot.GetTrueBaseOrientation()
    rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(current_base_orientation)
    heading_dir = np.array([rot_matrix[0], rot_matrix[3], 0])
    return heading_dir
  
  def compute_lateral_dir(self):
    current_base_orientation = self._env.robot.GetTrueBaseOrientation()
    rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(current_base_orientation)
    lateral_dir = np.array([rot_matrix[1], rot_matrix[4], 0])
    return lateral_dir
  
  def compute_up_dir(self):
    current_base_orientation = self._env.robot.GetTrueBaseOrientation()
    rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(current_base_orientation)
    up_dir = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
    return up_dir  
  
  def compute_forward_vel_reward(self):
    velocity = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / self._env.env_time_step
    projected_velocity = np.dot(velocity, self.compute_heading_dir())

    if projected_velocity >= self.des_forward_speed:
          reward = 1.0
    elif projected_velocity <= 0.0:
          reward = 0.0
    else:
          reward = projected_velocity / self.des_forward_speed    
          # reward = np.exp(-2.0 * (projected_velocity - self.des_forward_speed)**2)  # exp formulation   
    return reward
  
  def compute_lateral_vel_reward(self):
    velocity = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / self._env.env_time_step
    projected_velocity = np.dot(velocity, self.compute_lateral_dir())
    reward = np.exp(-10.0 * np.power(projected_velocity, 2))
    return reward
  
  def compute_upright_reward(self):
    # reward = np.dot([0, 0, 1], self.compute_up_dir()) / 2 + 0.5
    
    up_dir_dot = np.dot([0, 0, 1], self.compute_up_dir())
    
    if up_dir_dot <= self.up_dir_cutoff:
          reward = 0.0
    else:
          reward = (up_dir_dot - self.up_dir_cutoff) / (1 - self.up_dir_cutoff)
    
    return reward

  def reward(self, env):
    """Get the reward without side effects."""
    del env
    reward = self.compute_forward_vel_reward() * self.compute_upright_reward() * self.compute_lateral_vel_reward()
    return reward
