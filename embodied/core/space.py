import numpy as np


class Space:

  def __init__(self, dtype, shape=(), low=None, high=None):
    # For integer types, high is the excluside upper bound.
    self.dtype = np.dtype(dtype)
    self.low = self._infer_low(dtype, shape, low, high)
    self.high = self._infer_high(dtype, shape, low, high)
    self.shape = self._infer_shape(dtype, shape, low, high)
    self.discrete = (
        np.issubdtype(self.dtype, np.integer) or self.dtype == bool)
    self._random = np.random.RandomState()

  def __repr__(self):
    return (
        f'Space(dtype={self.dtype.name}, '
        f'shape={self.shape}, '
        f'low={self.low.min()}, '
        f'high={self.high.max()})')

  def __contains__(self, value):
    if not isinstance(value, np.ndarray):
      value = np.array(value)
    if value.dtype != self.dtype:
      return False
    if value.shape != self.shape:
      return False
    if (value > self.high).any():
      return False
    if (value < self.low).any():
      return False
    return True

  def sample(self):
    low, high = self.low, self.high
    if np.issubdtype(self.dtype, np.floating):
      low = np.maximum(np.ones(self.shape) * np.finfo(self.dtype).min, low)
      high = np.minimum(np.ones(self.shape) * np.finfo(self.dtype).max, high)
    return self._random.uniform(low, high, self.shape).astype(self.dtype)

  def _infer_low(self, dtype, shape, low, high):
    if low is None:
      if np.issubdtype(dtype, np.floating):
        low = -np.inf * np.ones(shape)
      elif np.issubdtype(dtype, np.integer):
        low = np.iinfo(dtype).min * np.ones(shape, dtype)
      elif np.issubdtype(dtype, bool):
        low = np.zeros(shape, bool)
      else:
        raise ValueError('Cannot infer low bound from shape and dtype.')
    if shape:
      low = np.broadcast_to(low, shape)
    return np.array(low)

  def _infer_high(self, dtype, shape, low, high):
    if high is None:
      if np.issubdtype(dtype, np.floating):
        high = np.inf * np.ones(shape)
      elif np.issubdtype(dtype, np.integer):
        high = np.iinfo(dtype).max * np.ones(shape, dtype)
      elif np.issubdtype(dtype, bool):
        high = np.ones(shape, bool)
      else:
        raise ValueError('Cannot infer high bound from shape and dtype.')
    if shape:
      high = np.broadcast_to(high, shape)
    return np.array(high)

  def _infer_shape(self, dtype, shape, low, high):
    if shape is None and low is not None:
      shape = low.shape
    if shape is None and high is not None:
      shape = high.shape
    if not hasattr(shape, '__len__'):
      shape = (shape,)
    assert all(dim and dim > 0 for dim in shape), shape
    return tuple(shape)
