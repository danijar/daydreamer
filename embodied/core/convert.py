import numpy as np


CONVERSION = {
    np.floating: np.float32,
    np.signedinteger: np.int64,
    np.uint8: np.uint8,
    bool: bool,
}


def convert(value):
  if not isinstance(value, np.ndarray):
    value = np.array(value)
  if value.dtype not in CONVERSION.values():
    for src, dst in CONVERSION.items():
      if np.issubdtype(value.dtype, src):
        if value.dtype != dst:
          value = value.astype(dst)
        break
    else:
      raise TypeError(f'Unsupported dtype: {value.dtype}')
  return value
