import io
import time as timelib
import threading
import pickle

import embodied
import numpy as np


class RAMStore:

  def __init__(self, capacity=None):
    self.capacity = capacity
    self.steps = 0
    self.trajs = {}

  def stats(self):
    return {
        'steps': self.steps,
        'trajs': len(self.trajs),
    }

  def close(self):
    pass

  def keys(self):
    return tuple(self.trajs.keys())

  def __contains__(self, key):
    return key in self.trajs.keys()

  def __len__(self):
    return len(self.trajs)

  def __getitem__(self, key):
    return self.trajs[key]

  def __setitem__(self, key, traj):
    self.trajs[key] = traj
    self.steps += len(next(iter(traj.values())))
    self._enforce_limit()

  def __delitem__(self, key):
    traj = self.trajs.pop(key)
    self.steps -= len(next(iter(traj.values())))

  def sync(self):
    # Intentionally empty. Use CkptRAMStore for a RAM buffer that can sync with
    # trajectories on disk.
    pass

  def _enforce_limit(self):
    if not self.capacity:
      return
    while len(self.trajs) > 1 and self.steps > self.capacity:
      # Relying on Python preserving dict insertion order.
      del self[next(iter(self.trajs.keys()))]


class DiskStore:

  def __init__(self, directory, capacity=None, parallel=False):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.capacity = capacity
    self.filenames = {}
    self.steps = 0
    self.worker = embodied.Worker('thread' if parallel else 'none')
    self.sync()

  def stats(self):
    return {
        'steps': self.steps,
        'trajs': len(self.filenames),
    }

  def close(self):
    self.worker.close()

  def keys(self):
    return tuple(self.filenames.keys())

  def __len__(self):
    return len(self.filenames)

  def __contains__(self, key):
    return key in self.filenames.keys()

  def __getitem__(self, key):
    filename = embodied.Path(self.filenames[key])
    with filename.open('rb') as f:
      data = np.load(f)
      data = {k: data[k] for k in data.keys()}
    return data

  def __setitem__(self, key, traj):
    length = len(next(iter(traj.values())))
    filename = self._format(key, traj)
    self.filenames[key] = filename
    self.steps += length
    self._enforce_limit()
    # TODO: It can take a while for the trajectory to be written and it causes
    # a not found error if the user tries to access the episode before that.
    self.worker.run(self._save, filename, traj)

  def __delitem__(self, key):
    filename = self.filenames.pop(key)
    _, _, length, _ = self._parse(filename)
    self.steps -= length

  def sync(self):
    filenames = sorted(self.directory.glob('*.npz'))
    selected = {}
    steps = 0
    for filename in reversed(filenames):
      _, key, length, _ = self._parse(filename)
      if self.capacity and steps + length > self.capacity:
        break
      selected[key] = filename
      steps += length
    self.filenames = dict(reversed(list(selected.items())))
    self.steps = steps
    print(f'Synced last {len(selected)}/{len(filenames)} trajectories.')

  @staticmethod
  def _save(filename, traj):
    filename = embodied.Path(filename)
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **traj)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    print(f'Saved episode: {filename.name}')

  def _enforce_limit(self):
    if not self.capacity:
      return
    while len(self.filenames) > 1 and self.steps > self.capacity:
      # Relying on Python preserving dict insertion order.
      del self[next(iter(self.filenames.keys()))]

  def _format(self, key, traj):
    time = timelib.strftime('%Y%m%dT%H%M%S', timelib.gmtime(timelib.time()))
    length = len(next(iter(traj.values())))
    reward = str(int(traj['reward'].sum())).replace('-', 'm')
    return self.directory / f'{time}-{key}-len{length}-rew{reward}.npz'

  def _parse(self, filename):
    time, key, length, reward = filename.stem.split('-')
    time = timelib.mktime(timelib.strptime(
        time, '%Y%m%dT%H%M%S')) - timelib.timezone
    length = int(length.strip('len'))
    reward = int(reward.strip('rew').replace('m', '-'))
    return time, key, length, reward


class CkptRAMStore:

  def __init__(self, directory, capacity=None, parallel=False):
    self.disk_store = DiskStore(directory, capacity, parallel)
    self.ram_store = RAMStore(capacity)
    self.sync()

  @property
  def steps(self):
    return self.ram_store.steps

  def stats(self):
    return self.ram_store.stats()

  def close(self):
    self.ram_store.close()
    self.disk_store.close()

  def keys(self):
    return tuple(self.ram_store.keys())

  def __len__(self):
    return len(self.ram_store)

  def __contains__(self, key):
    return key in self.ram_store

  def __getitem__(self, key):
    return self.ram_store[key]

  def __setitem__(self, key, traj):
    self.ram_store[key] = traj
    self.disk_store[key] = traj

  def sync(self):
    self.disk_store.sync()
    for key in self.disk_store.keys():
      if key not in self.ram_store:
        self.ram_store[key] = self.disk_store[key]


class Stats:

  def __init__(self, store):
    self.store = store
    self.steps = self.store.steps
    self.episodes = 0
    self.reward = 0.0

  def stats(self):
    return {
        **self.store.stats(),
        'episodes': self.episodes,
        'ep_length': self.episodes and self.steps / self.episodes,
        'ep_return': self.episodes and self.reward / self.episodes,
    }

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.store, name)
    except AttributeError:
      raise ValueError(name)

  def __len__(self):
    return len(self.store)

  def __contains__(self, key):
    return key in self.store

  def __getitem__(self, key):
    return self.store[key]

  def __setitem__(self, key, traj):
    self.store[key] = traj
    self.reward += traj['reward'].sum()
    self.episodes += traj['is_first'].sum()
    self.steps += len(traj['is_first'])
    # print('add traj', len(traj['is_first']), self.steps)

  def __delitem__(self, key):
    traj = self.store[key]
    del self.store[key]
    self.reward -= traj['reward'].sum()
    self.episodes -= traj['is_first'].sum()
    self.steps -= len(traj['is_first'])
    # print('del traj', len(traj['is_first']), self.steps)


class StoreServer:

  def __init__(self, store, port):
    self.store = store
    self.thread = threading.Thread(
        target=self._server, args=(port,), daemon=True)
    self.thread.start()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.store, name)
    except AttributeError:
      raise ValueError(name)

  def __len__(self):
    return len(self.store)

  def __contains__(self, key):
    return key in self.store

  def __getitem__(self, key):
    return self.store[key]

  def __setitem__(self, key, traj):
    self.store[key] = traj

  def _server(self, port):
    import zmq
    import pickle
    print(f'Replay server listening on *:{port}')
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f'tcp://*:{port}')
    while True:
      method, args = pickle.loads(socket.recv())
      ret = None
      if method == 'keys':
        ret = self.keys()
      elif method == '__getitem__':
        key, = args
        ret = self[key]
      elif method == '__setitem__':
        key, traj = args
        self[key] = traj
      elif method == 'steps':
        ret = self.steps
      else:
        raise NotImplementedError(method)
      socket.send(pickle.dumps(ret))


class StoreClient:

  def __init__(self, address):
    import zmq
    self.address = address
    print(f'Using remote store via ZMQ on {address}')
    self.socket = zmq.Context().socket(zmq.REQ)
    self.socket.connect(f'tcp://{address}')
    self.pending = False
    self.once = True

  @property
  def steps(self):
    self._call('steps')
    return self._result()

  def stats(self):
    return {}

  def close(self):
    pass

  def keys(self):
    self._call('keys')
    return self._result()

  def __len__(self):
    raise NotImplementedError('Use store.keys() to cause fewer remote calls.')

  def __contains__(self, key):
    raise NotImplementedError('Use store.keys() to cause fewer remote calls.')

  def __getitem__(self, key):
    self._call('__getitem__', key)
    return self._result()

  def __setitem__(self, key, traj):
    self._call('__setitem__', key, traj)

  def sync(self):
    pass

  def _call(self, method, *args):
    if self.pending:
      # Need to wait for previous response before calling again
      self._result()
    msg = (method, args)
    self.socket.send(pickle.dumps(msg))
    self.pending = True

  def _result(self):
    assert self.pending
    # TODO: If the server is unavailable or the address is incorrect, it will
    # just hang here, not raising any error earlier during send.
    self.once and print(f'Waiting for response from {self.address}...')
    ret = pickle.loads(self.socket.recv())
    self.once and print(f'Connection to {self.address} successful!')
    self.once = False
    self.pending = False
    return ret
