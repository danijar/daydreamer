import collections
import threading
import time
import uuid

import numpy as np
import embodied


class Consecutive(embodied.Replay):

  def __init__(self, store, chunk=64, randomize=False, sync=0):
    self.store = store
    self.chunk = chunk
    # TODO: Initial time step is too unlikely right now I think.
    self.randomize = randomize
    self.random = np.random.RandomState(seed=0)
    self.ongoing = collections.defaultdict(
        lambda: collections.defaultdict(list))
    if sync:
      self.last_scan = time.time()
      # TODO: How can we propagate exceptions from this worker thread?
      self.thread = threading.Thread(
          target=self._sync, args=(sync,), daemon=True)
      self.thread.start()

  def __len__(self):
    return self.store.steps

  @property
  def stats(self):
    return {f'replay_{k}': v for k, v in self.store.stats().items()}

  def add(self, tran, worker=0):
    if tran['is_first']:
      self.ongoing[worker].clear()
    episode = self.ongoing[worker]
    [episode[k].append(v) for k, v in tran.items()]
    if tran['is_last']:
      self.add_traj(self.ongoing.pop(worker))

  def add_traj(self, traj):
    traj = {k: v for k, v in traj.items() if not k.startswith('log_')}
    traj = {k: embodied.convert(v) for k, v in traj.items()}
    self.store[uuid.uuid4().hex] = traj

  def dataset(self):
    source, index = None, None
    while True:
      chunk, missing = None, self.chunk
      while missing > 0:
        if not source or index >= len(source['action']):
          source, index = self._sample(), 0
        if not chunk:
          chunk = {k: v[index: index + missing] for k, v in source.items()}
        else:
          chunk = {
              k: np.concatenate([chunk[k], v[index: index + missing]], 0)
              for k, v in source.items()}
        index += missing
        missing = self.chunk - len(chunk['action'])
      assert missing == 0, missing
      yield chunk

  def _sample(self):
    keys = self.store.keys()
    while not len(keys):
      print('Waiting for episodes.')
      time.sleep(1)
      keys = self.store.keys()
    traj = self.store[keys[self.random.randint(0, len(keys))]]
    if self.randomize:
      length = len(next(iter(traj.values())))
      start = self.random.randint(0, max(1, length - self.chunk))
      traj = {k: v[start:] for k, v in traj.items()}
      traj['is_first'][:1] = True
    return traj

  def _sync(self, interval):
    while True:
      time.sleep(max(0, self.last_scan + interval - time.time()))
      self.last_scan = time.time()
      self.store.sync()
