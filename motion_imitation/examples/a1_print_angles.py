"""Reads and prints joint angles from A1 robot without powering them.

By default prints all joint angles. To select specific joints:
`python a1_print_angles.py --joint FR_hip_motor --joint RL_upper_joint`
"""

import inspect
import os

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, grandparentdir)

from absl import app
from absl import flags
from motion_imitation.robots import a1_robot
import pybullet
import pybullet_data
from pybullet_utils import bullet_client


JOINT_DICT = {
    "FR_hip_motor": 0, "FR_upper_joint": 1, "FR_lower_joint": 2,
    "FL_hip_motor": 3, "FL_upper_joint": 4, "FL_lower_joint": 5,
    "RR_hip_motor": 6, "RR_upper_joint": 7, "RR_lower_joint": 8,
    "RL_hip_motor": 9, "RL_upper_joint": 10, "RL_lower_joint": 11,
    }

flags.DEFINE_multi_string("joint", JOINT_DICT.keys(),
                          "Names of joints to measure.")
FLAGS = flags.FLAGS


def main(_):
  p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)

  while True:
    robot.ReceiveObservation()
    angles = robot.GetTrueMotorAngles()
    if len(angles) != 12:
      continue
    print_list = [
        "{}:{:9.6f}".format(joint, angles[JOINT_DICT[joint]])
        for joint in FLAGS.joint
    ]
    print(", ".join(print_list))


if __name__ == "__main__":
  app.run(main)
