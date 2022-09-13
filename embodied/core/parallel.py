import functools

from . import worker


class Parallel:

  def __init__(self, ctor, strategy='process', daemon=False):
    self._worker = worker.Worker(strategy, daemon)
    self._worker.run_with_state(self._make, ctor)()
    self._callables = {}

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      if name not in self._callables:
        self._callables[name] = self._worker.run_with_state(
            self._callable, name)()
      if self._callables[name]:
        return functools.partial(self._worker.run_with_state, self._call, name)
      else:
        return self._worker.run_with_state(self._access, name)()
    except AttributeError:
      raise ValueError(name)

  def __len__(self):
    return self._worker.run_with_state(self._call, '__len__')()

  def close(self):
    self._worker.close()

  @classmethod
  def _make(cls, ctor, state):
    state['env'] = ctor()

  @classmethod
  def _callable(cls, name, state):
    return callable(getattr(state['env'], name))

  @classmethod
  def _call(cls, name, *args, **kwargs):
    state = kwargs.pop('state')
    return getattr(state['env'], name)(*args, **kwargs)

  @classmethod
  def _access(cls, name, state):
    return getattr(state['env'], name)
