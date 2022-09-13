import collections
import threading

import numpy as np


class Priorities:

  def __init__(self, aggregate, fraction=0.25, prio_starts=1.0, prio_ends=1.0):
    self.aggregate = aggregate
    self.fraction = fraction
    self.prio_starts = prio_starts
    self.prio_ends = prio_ends
    self.random = np.random.RandomState(seed=0)
    self.entries = {}
    self.probs = None
    self.keys = None
    self.lock = threading.Lock()
    self.metrics = {
        'samples': collections.defaultdict(int),
        'update_min': np.inf,
        'update_max': -np.inf,
    }

  def __contains__(self, key):
    return key in self.entries

  def __len__(self):
    return len(self.entries)

  @property
  def stats(self):
    if len(self) <= 1:
      return {}
    with self.lock:
      self._ensure()
      entropy = -(self.probs @ np.log(self.probs)).item()
      maximum = np.log(len(self.probs))
    samples = list(self.metrics['samples'].values()) or [0]
    return {
        'randomness': entropy / maximum,
        'seen_frac': len(self.metrics['samples']) / len(self.entries),
        'seen_max': max(samples),
        'sample_frac': sum(samples) / len(self.entries),
        'update_min': self.metrics['update_min'],
        'update_max': self.metrics['update_max'],
    }

  def sample(self):
    assert len(self)
    with self.lock:
      self._ensure()
      if len(self.probs) == 1:
        key = self.keys[0]
        prob = 1.0
      else:
        pos = self.random.choice(len(self.probs), p=self.probs)
        prob = self.probs[pos]
        key = self.keys[pos]
      entry = self.entries[key]
      index = self.random.choice(len(entry.probs), p=entry.probs)
      prob *= entry.probs[index]
    self.metrics['samples'][key] += 1
    return key, index, prob

  def add(self, key, prios):
    assert prios.dtype == np.float64, prios.dtype
    entry = Entry(prios)
    self._precompute(entry)
    with self.lock:
      self.entries[key] = entry
      self.probs = None

  def update(self, key, index, prios):
    assert prios.dtype == np.float64, prios.dtype
    self.metrics['update_min'] = min(self.metrics['update_min'], prios.min())
    self.metrics['update_max'] = max(self.metrics['update_max'], prios.max())
    try:
      entry = self.entries[key]
      entry.steps[index: index + len(prios)] = prios
      self._precompute(entry)
    except (KeyError, IndexError):
      raise KeyError
    with self.lock:
      self.probs = None

  def remove(self, key):
    self.metrics['samples'].pop(key, None)
    with self.lock:
      del self.entries[key]
      self.probs = None

  def save(self):
    return {
        'entries': self.entries.copy(),
        'metrics': self.metrics,
    }

  def load(self, data):
    with self.lock:
      self.metrics = data['metrics']
      self.entries.update(data['entries'])
      self.probs = None

  def _precompute(self, entry):
    agg = self.aggregate(entry.steps)
    assert (agg >= 0).all(), agg
    total = agg.sum()  # Doing this before converting infs.
    infs = np.isposinf(agg)
    if infs.any():
      agg = infs.astype(np.float64)
    uniform = np.ones_like(agg) / len(agg)
    if self.prio_starts or self.prio_ends:
      uniform[0] *= (len(entry.steps) - len(uniform)) * self.prio_starts
      uniform[-1] *= (len(entry.steps) - len(uniform)) * self.prio_ends
      uniform /= uniform.sum()
    normalized = agg.sum()
    if normalized == 0:
      probs = uniform
    else:
      probs = agg / normalized
    probs = self.fraction * probs + (1 - self.fraction) * uniform
    entry.probs = probs
    entry.total = total

  def _ensure(self):
    if self.probs is not None:
      return
    lengths = np.array([len(x.probs) for x in self.entries.values()])
    prios = np.array([x.total for x in self.entries.values()])
    infs = np.isposinf(prios)
    if infs.any():
      prios = infs.astype(np.float64)
    total = prios.sum()
    if total == 0:
      probs = np.ones_like(prios) / len(prios)
    else:
      probs = prios / total
    uniform = lengths / lengths.sum()
    probs = self.fraction * probs + (1 - self.fraction) * uniform
    self.probs = probs
    self.keys = tuple(self.entries.keys())


class Entry:

  __slots__ = ('steps', 'probs', 'total')

  def __init__(self, steps, probs=None, total=None):
    self.steps = steps
    self.probs = probs
    self.total = total
