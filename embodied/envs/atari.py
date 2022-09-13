import embodied
import numpy as np


class Atari(embodied.Env):

  LOCK = None

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=30, lives=False,
      sticky=True, actions='all', length=108000, seed=None):
    assert size[0] == size[1]

    if self.LOCK is None:
      import multiprocessing as mp
      mp = mp.get_context('spawn')
      self.LOCK = mp.Lock()

    import cv2
    self._cv2 = cv2

    # if 'ATARI_ROMS' in os.environ:  # TODO
    #   import absl.flags
    #   import atari_py  # noqa: For flag definition.
    #   absl.flags.FLAGS.atari_roms_path = os.environ['ATARI_ROMS']

    # from PIL import Image
    # self._image = Image

    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    self._repeat = repeat
    self._size = size
    self._gray = gray
    self._noops = noops
    self._lives = lives
    self._sticky = sticky
    self._length = length
    self._random = np.random.RandomState(seed)
    with self.LOCK:
      self._env = gym.envs.atari.AtariEnv(
          game=name,
          obs_type='image',  # TODO: Internal old version.
          # obs_type='grayscale' if gray else 'rgb',
          frameskip=1, repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=(actions == 'all'))
    assert self._env.unwrapped.get_action_meanings()[0] == 'NOOP'
    if gray:
      shape = self._env.observation_space.shape[:2]
      self._buffer = [np.empty(shape, np.uint8) for _ in range(2)]
    else:
      shape = self._env.observation_space.shape
      self._buffer = [np.empty(shape, np.uint8) for _ in range(2)]
    self._ale = self._env.unwrapped.ale
    self._last_lives = None
    self._done = True
    self._step = 0

  @property
  def obs_space(self):
    shape = self._size + (1 if self._gray else 3,)
    return {
        'image': embodied.Space(np.uint8, shape),
        # 'ram': embodied.Space(np.uint8, 128),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      with self.LOCK:
        self._reset()
      self._done = False
      self._step = 0
      return self._obs(0.0, is_first=True)
    total = 0.0
    dead = False
    for repeat in range(self._repeat):
      _, reward, over, info = self._env.step(action['action'])
      self._step += 1
      total += reward
      if repeat == self._repeat - 2:
        self._screen(self._buffer[1])
      if over:
        break
      if self._lives:
        current = self._ale.lives()
        if current < self._last_lives:
          dead = True
          self._last_lives = current
          break
    if not self._repeat:
      self._buffer[1][:] = self._buffer[0][:]
    self._screen(self._buffer[0])
    self._done = over or (self._length and self._step >= self._length)
    return self._obs(total, is_last=self._done, is_terminal=dead or over)

  def _reset(self):
    self._env.reset()
    if self._noops:
      for _ in range(self._random.randint(1, self._noops + 1)):
         _, _, dead, _ = self._env.step(0)
         if dead:
           self._env.reset()
    self._last_lives = self._ale.lives()
    self._screen(self._buffer[0])
    self._buffer[1].fill(0)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])

    # image = self._image.fromarray(self._buffer[0])
    # image = image.resize(self._size, self._image.NEAREST)
    # image = np.array(image)

    image = self._cv2.resize(
        self._buffer[0], self._size,
        interpolation=self._cv2.INTER_AREA)

    if self._gray:
      image = image[:, :, None]
    return dict(
        image=image,
        # ram=self._env.env._get_ram(),
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )

  def _screen(self, array):
    if self._gray:
      self._ale.getScreenGrayscale(array)
    else:
      self._ale.getScreenRGB2(array)

  def close(self):
    return self._env.close()
