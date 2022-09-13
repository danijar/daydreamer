"""A simple locomotion task."""
import numpy as np

from motion_imitation.robots import a1

class RMATask(object):
  def __init__(self, des_forward_speed=0.3):
    self.des_forward_speed = des_forward_speed
    self.curr_base_velocity = np.zeros(3)
    self.curr_rot_mat = np.zeros((3, 3))

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    self._env = env

    self.curr_base_velocity = env.robot.GetBaseVelocity()
    self.curr_rot_mat = self._env.pybullet_client.getMatrixFromQuaternion(
      self._env.robot.GetTrueBaseOrientation())

  def update(self, env):
    self.curr_base_velocity = env.robot.GetBaseVelocity()
    self.curr_rot_mat = self._env.pybullet_client.getMatrixFromQuaternion(
      self._env.robot.GetTrueBaseOrientation())

  def done(self, env):
    return False

  def compute_heading_dir(self):
    return np.array([self.curr_rot_mat[0], self.curr_rot_mat[3], 0])

  def compute_up_dir(self):
    return np.array([self.curr_rot_mat[2], self.curr_rot_mat[5], self.curr_rot_mat[8]])

  def reward(self, env):
    normed_actions = a1.normalize_action(self._env.robot.GetTrueMotorAngles())
    deviations = np.abs(normed_actions - a1.STANDING_POSE)
    worst = np.maximum(1 - a1.STANDING_POSE, 1 + a1.STANDING_POSE)
    deviations = np.clip(deviations / worst, 0, 1)  # Map to [0, 1]
    r_upr =  np.dot([0, 0, 1], self.compute_up_dir()) / 2 + 0.5
    r_hip = (r_upr > 0.7) * (1 - deviations[0::3].mean())
    r_sho = (r_hip > 0.7) * (1 - deviations[1::3].mean())
    r_kne = (r_sho > 0.7) * (1 - deviations[2::3].mean())

    forward_vel = np.dot(self.curr_base_velocity, self.compute_heading_dir())
    total_vel = np.linalg.norm(self.curr_base_velocity)
    forward_frac = np.maximum(0, forward_vel) / total_vel
    forward_going = np.clip(forward_vel / self.des_forward_speed, -1, 1)
    r_vel = (r_kne > 0.7) * forward_frac * forward_going

    # print(
    #     '[x]' if r_kne > 0.7 else '[ ]',
    #     '#' * int(50 * forward_vel))

    return r_upr + r_hip + r_sho + r_kne + 10 * (r_vel + 1) / 2
