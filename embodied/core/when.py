import time


class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    step = int(step)
    if not self._until:
      return True
    return step < self._until


class Clock:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    now = time.time()
    if self._last is None:
      self._last = now
      return True
    if now >= self._last + self._every:
      self._last += self._every
      return True
    return False
