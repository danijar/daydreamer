import queue as queuelib

import numpy as np


class Prefetch:

  """
  Implements zip() for iterables that yield dictionaries of Numpy arrays with
  optional prefetching using multiple threads. The source generator functions
  are split among the workers. The resulting arrays for each key are stacked,
  adding a leading batch dimension.
  """

  def __init__(self, sources, workers=0, prefetch=4):
    if workers:
      self._running = True
      self._queues = []
      self._creators = []
      # Round-robin assign sources to workers for balanced workload.
      assignments = [([], []) for _ in range(workers)]
      for index, source in enumerate(sources):
        queue = queuelib.Queue(prefetch)
        self._queues.append(queue)
        assignments[index % workers][0].append(source)
        assignments[index % workers][1].append(queue)
      import threading
      for args in assignments:
        creator = threading.Thread(
            target=self._creator, args=args, daemon=True)
        creator.start()
        self._creators.append(creator)
    else:
      self._creators = None
      self._iterators = [source() for source in sources]
    self._once = False

  def close(self):
    if self._creators:
      for creator in self._creators:
        creator.close()

  def __iter__(self):
    if self._once:
      raise RuntimeError(
          'You can only create one iterator per Batcher object to ensure that '
          'data is consumed in order. You can create another Batcher object '
          'instead.')
    self._once = True
    return self

  def __next__(self):
    if self._creators:
      elems = [x.get() for x in self._queues]
    else:
      elems = [next(x) for x in self._iterators]
    return {k: np.stack([x[k] for x in elems], 0) for k in elems[0]}

  def _creator(self, sources, queues):
    try:
      iterators = [source() for source in sources]
      while self._running:
        for iterator, queue in zip(iterators, queues):
          queue.put(next(iterator))
    except Exception as e:
      queues[0].put(e)
      raise
