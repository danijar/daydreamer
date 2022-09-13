"""Executes scripted motions and logs IMU readings.

Pitches the robot forward and backward and rolls it slightly left and right with
fixed joint positions. By running this in both sim and real, the real robot's
IMU readings can be compared to simulation.
"""

import inspect
import os

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, grandparentdir)

from absl import app
from absl import flags
from absl import logging
import numpy as np
import time
from tqdm import tqdm

from motion_imitation.envs import env_builder
from motion_imitation.robots import a1_robot
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
FREQ = 0.5

flags.DEFINE_bool('real_robot', True,
                  'Whether to control a real robot or simulated.')
FLAGS = flags.FLAGS


def main(_):
  if FLAGS.real_robot:
    robot_class = a1_robot.A1Robot
    logging.info('WARNING: this code executes low-level control on the robot.')
    input('Press enter to continue...')
  else:
    robot_class = a1.A1

  env = env_builder.build_regular_env(
      robot_class,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      enable_rendering=not FLAGS.real_robot,
      on_rack=False,
      wrap_trajectory_generator=False)
  robot = env.robot

  # Move the motors slowly to initial position
  robot.ReceiveObservation()
  current_motor_angle = np.array(robot.GetMotorAngles())
  desired_motor_angle = np.array([0., 0.9, -1.8] * 4)

  for t in tqdm(range(300)):
    blend_ratio = np.minimum(t / 200., 1)
    action = (1 - blend_ratio
             ) * current_motor_angle + blend_ratio * desired_motor_angle
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    logging.info(robot.GetTrueBaseRollPitchYaw())
    time.sleep(0.005)

  # Pitch up
  for t in tqdm(range(200)):
    angle_hip = 0.25
    angle_hip_2 = 1.5
    angle_calf = -1
    angle_calf_2 = -2
    action = np.array([
        0., angle_hip, angle_calf, 0., angle_hip, angle_calf, 0., angle_hip_2,
        angle_calf_2, 0., angle_hip_2, angle_calf_2
    ])
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)
    logging.info(robot.GetTrueBaseRollPitchYaw())
    logging.info('pitch up: %f', robot.GetTrueBaseRollPitchYaw()[1])
  pitch_up = robot.GetTrueBaseRollPitchYaw()[1]

  # Pitch down
  for t in tqdm(range(200)):
    angle_hip = 0.25
    angle_hip_2 = 0.8
    angle_calf = -1
    angle_calf_2 = -2.4
    action = np.array([
        0., angle_hip_2, angle_calf_2, 0., angle_hip_2, angle_calf_2, 0.,
        angle_hip, angle_calf, 0., angle_hip, angle_calf
    ])
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)
    logging.info(robot.GetTrueBaseRollPitchYaw())
    logging.info('pitch down: %f', robot.GetTrueBaseRollPitchYaw()[1])
  pitch_down = robot.GetTrueBaseRollPitchYaw()[1]

  # Roll right
  angle_hip = 0.5
  angle_hip_2 = 0.9
  angle_calf = -1.5
  angle_calf_2 = -1.8
  action = np.array([
      0., angle_hip_2, angle_calf_2, 0., angle_hip, angle_calf, 0., angle_hip_2,
      angle_calf_2, 0., angle_hip, angle_calf
  ])
  for t in tqdm(range(200)):
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)
    logging.info(robot.GetTrueBaseRollPitchYaw())
    logging.info('roll right: %f', robot.GetTrueBaseRollPitchYaw()[0])
  roll_right = robot.GetTrueBaseRollPitchYaw()[0]

  # Roll left
  action = np.array([
      0.,
      angle_hip,
      angle_calf,
      0.,
      angle_hip_2,
      angle_calf_2,
      0.,
      angle_hip,
      angle_calf,
      0.,
      angle_hip_2,
      angle_calf_2,
  ])
  for t in tqdm(range(200)):
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)
    logging.info(robot.GetTrueBaseRollPitchYaw())
    logging.info('roll left: %f', robot.GetTrueBaseRollPitchYaw()[0])
  roll_left = robot.GetTrueBaseRollPitchYaw()[0]
  robot.Terminate()

  logging.info(
      '\npitch up: %f \npitch down: %f \nroll right: %f \nroll left: %f\n',
      pitch_up, pitch_down, roll_right, roll_left)


if __name__ == '__main__':
  app.run(main)
