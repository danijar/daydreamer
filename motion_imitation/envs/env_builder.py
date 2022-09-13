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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
from motion_imitation.envs import locomotion_gym_config, locomotion_gym_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper, simple_openloop, simple_forward_task, trajectory_generator_wrapper_env, rma_task
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.robots import a1, a1_robot

def build_env(enable_rendering=False,
              num_action_repeat=20,
              reset_at_current_position=False,
              use_real_robot=False,
              realistic_sim=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.reset_at_current_position = reset_at_current_position
  sim_params.num_action_repeat = num_action_repeat
  sim_params.render_height = 64
  sim_params.render_width = 64

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_kwargs = {"self_collision_enabled": False}

  if use_real_robot:
    robot_class = a1_robot.A1Robot
  else:
    robot_class = a1.A1

  if use_real_robot or realistic_sim:
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT
  else:
    robot_kwargs["reset_func_name"] = "_PybulletReset"
  num_motors = a1.NUM_MOTORS
  # traj_gen = simple_openloop.A1PoseOffsetGenerator(
  #     action_limit=np.array([0.802851455917, 4.18879020479, -0.916297857297] *
  #                           4) - np.array([0, 0.9, -1.8] * 4))
  traj_gen = simple_openloop.A1PoseOffsetGenerator(
      action_limit=np.array([1.8, 1.8, 1.8] * 4))

  sensors = [
    robot_sensors.MotorAngleSensor(
      num_motors=num_motors,
      lower_bound=np.array([-np.pi] * num_motors),
      upper_bound=np.array([np.pi] * num_motors),
      dtype=np.float32),
    robot_sensors.IMUSensor(
      lower_bound=np.array([-2 * np.pi, -2 * np.pi, -float('inf'), -float('inf')]),
      upper_bound=np.array([2 * np.pi, 2 * np.pi, float('inf'), float('inf')]),
      dtype=np.float32)
  ]
  # task = simple_forward_task.SimpleForwardTask()
  task = rma_task.RMATask()

  randomizers = []

  env = locomotion_gym_env.LocomotionGymEnv(
      gym_config=gym_config,
      robot_class=robot_class,
      robot_kwargs=robot_kwargs,
      env_randomizers=randomizers,
      robot_sensors=sensors,
      task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env, trajectory_generator=traj_gen)

  return env
