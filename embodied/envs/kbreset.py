import time

import embodied
import numpy as np
import pygame

NORMAL = (128, 128, 128)
PAUSED = (255, 0, 0)
REWARD = (0, 255, 0)

KEY_PAUSE = pygame.K_p
KEY_CONTINUE = pygame.K_c
KEY_RESET = pygame.K_r


class KBReset(embodied.Wrapper):

  def __init__(self, ctor, mouse=False):
    super().__init__(ctor())
    self._ctor = ctor
    pygame.init()
    self._screen = pygame.display.set_mode((800, 600))
    self._set_color(NORMAL)
    self._paused = False
    if mouse:
      from embodied.envs.spacemouse import SpaceMouse
      self._mouse = SpaceMouse()
    else:
      self._mouse = None

  def step(self, action):
    pressed_last = self._get_pressed()
    pressed_now = pygame.key.get_pressed()

    if self._paused:
      waiting = not (KEY_CONTINUE in pressed_last or KEY_RESET in pressed_last)
      hard = KEY_RESET in pressed_last
      while waiting:
        pressed = self._get_pressed()
        if KEY_CONTINUE in pressed or KEY_RESET in pressed:
          waiting = False
          hard = KEY_RESET in pressed
        self._set_color(PAUSED)
        time.sleep(0.1)
      assert action['reset'], action
      self._paused = False

      # obs = self.env.step({
      #     **action,
      #     'manual_resume': True,
      # })

      if hard:
        self.env.close()
        self.env = self._ctor()
      obs = self.env.step({**action, 'manual_resume': True})

      obs['is_first'] = True
      return obs

    if KEY_PAUSE in pressed_last:
      self._set_color(PAUSED)
      self._paused = True
      obs = self.env.step({**action, 'manual_pause': True})
      obs['is_last'] = True
      return obs

    # if (
    #     pygame.K_SPACE in pressed_last or
    #     pressed_now[pygame.K_SPACE] or
    #     self._mouse and np.linalg.norm(self._mouse.input_pos) > 100):
    #   print('GIVING MANUAL REWARD: 1.0')
    #   self._set_color(REWARD)
    #   self.env.manual_resume = True
    #   obs = self.env.step(action)
    #   obs['reward'] = 1.0
    #   return obs

    self._set_color(NORMAL)
    obs = self.env.step(action)

    sparse_rewards = [-1, 1, 10]
    if any([r - 0.01 < obs['reward'] < r + 0.01 for r in sparse_rewards]):
      if "gripper_pos" in obs:
        extra_info = f' | gripper_pos={obs["gripper_pos"]} '
      else:
        extra_info = ""
      print(f'NON-ZERO REWARD: {obs["reward"]}{extra_info}')

    return obs

  def _get_pressed(self):
    pressed = []
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        pressed.append(event.key)
    return pressed

  def _set_color(self, color):
    self._screen.fill(color)
    pygame.display.flip()
