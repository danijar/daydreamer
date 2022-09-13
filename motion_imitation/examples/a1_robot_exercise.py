"""Commands A1 robot to raise and lower its legs so it crouches and stands up.

Can be run in sim by setting --real_robot=False.
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

flags.DEFINE_bool(
    "real_robot", True, "Whether to control a real robot or simulated.")
FLAGS = flags.FLAGS


def main(_):
  if FLAGS.real_robot:
    robot_class = a1_robot.A1Robot
    logging.info("WARNING: this code executes low-level control on the robot.")
    input("Press enter to continue...")
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
    print(robot.GetBaseOrientation())
    blend_ratio = np.minimum(t / 200., 1)
    action = (1 - blend_ratio
              ) * current_motor_angle + blend_ratio * desired_motor_angle
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.005)

  # Move the legs in a sinusoidal curve
  for t in tqdm(range(1000)):
    print(robot.GetBaseOrientation())
    angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * FREQ * 0.01 * t)
    angle_calf = -2 * angle_hip
    action = np.array([0., angle_hip, angle_calf] * 4)
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)

  robot.Terminate()


if __name__ == "__main__":
  app.run(main)
