"""Ends episode if robot is outside workspace bounds."""


class BoundaryTerminalCondition(object):
  """Ends episode if robot is outside workspace bounds."""

  def __init__(self, x_space_m=5, y_space_m=5):
    """Constructor.

    :param x_space_m: Length of workspace in meters.
    :param y_space_m: Width of workspace in meters.
    """
    self._x_bound = x_space_m / 2.0
    self._y_bound = y_space_m / 2.0

  def __call__(self, env):
    x, y, _ = env.robot.GetBasePosition()
    return abs(x) > self._x_bound or abs(y) > self._y_bound


class CircularBoundaryTerminalCondition(object):

  def __init__(self, radius_m=2.5):
    self._radius_squared = radius_m ** 2

  def __call__(self, env):
    x, y, _ = env.robot.GetBasePosition()
    return x ** 2 + y ** 2 > self._radius_squared
