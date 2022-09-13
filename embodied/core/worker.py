import atexit
import enum
import functools
import os
import sys
import time
import traceback


class Message(enum.Enum):

  RUN = 2
  RUN_WITH_STATE = 1
  RESULT = 3
  STOP = 4
  ERROR = 5


class Worker:

  INITIALIZERS = []

  def __init__(self, strategy='process', daemon=True):
    self._strategy = strategy
    if strategy == 'process':
      import multiprocessing as mp
      mp = mp.get_context('spawn')
      kw = dict(daemon=daemon)
    elif strategy == 'thread':
      # The downside of using threads is that they cannot truly run in parallel
      # due to the Python GIL and that threads cannot be killed forcefully.
      import multiprocessing.dummy as mp
      kw = dict()
    elif strategy == 'none':
      self._result = None
      self._state = {}
    else:
      raise NotImplementedError(strategy)
    if self._strategy != 'none':
      initializers = self.INITIALIZERS if (strategy == 'process') else []
      self._pipe, pipe = mp.Pipe()
      self._process = mp.Process(
          target=self._loop, args=(pipe, initializers), **kw)
      atexit.register(self.close)
      self._process.start()
      assert self._receive() == 'ready'

  def run(self, function, *args, **kwargs):
    if self._strategy == 'none':
      self._result = function(*args, **kwargs)
      return lambda: self._result
    import cloudpickle
    if args or kwargs:
      function = functools.partial(function, *args, **kwargs)
    function = cloudpickle.dumps(function)
    self._pipe.send((Message.RUN, function))
    return self._receive  # Callable promise.

  def run_with_state(self, function, *args, **kwargs):
    if self._strategy == 'none':
      self._result = function(*args, **kwargs, state=self._state)
      return lambda: self._result
    import cloudpickle
    if args or kwargs:
      function = functools.partial(function, *args, **kwargs)
    function = cloudpickle.dumps(function)
    self._pipe.send((Message.RUN_WITH_STATE, function))
    return self._receive  # Callable promise.

  def close(self):
    try:
      self._pipe.send((Message.STOP, None))
      self._pipe.close()
    except (AttributeError, IOError):
      pass  # The connection was already closed.
    try:
      self._process.join(0.1)
      if self._strategy == 'process' and self._process.exitcode is None:
        try:
          os.kill(self._process.pid, 9)
          time.sleep(0.1)
        except Exception as e:
          print(f'Exception: {e}')
    except (AttributeError, AssertionError):
      pass

  @classmethod
  def add_initializer(cls, function):
    import cloudpickle
    function = cloudpickle.dumps(function)
    cls.INITIALIZERS.append(function)

  def _receive(self):
    try:
      message, payload = self._pipe.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to worker.')
    if message == Message.ERROR:
      raise Exception(payload)
    elif message == Message.RESULT:
      return payload
    else:
      raise KeyError(f'Unknown message type {message}.')

  @classmethod
  def _loop(cls, pipe, initializers):
    try:
      import cloudpickle
      state = {}
      for function in initializers:
        function = cloudpickle.loads(function)
        function()
      pipe.send((Message.RESULT, 'ready'))
      while True:
        if not pipe.poll(0.1):
          continue  # Wake up for keyboard interrupts.
        message, payload = pipe.recv()
        if message == Message.STOP:
          return
        elif message == Message.RUN:
          function = cloudpickle.loads(payload)
          result = function()
          pipe.send((Message.RESULT, result))
        elif message == Message.RUN_WITH_STATE:
          function = cloudpickle.loads(payload)
          result = function(state=state)
          pipe.send((Message.RESULT, result))
        else:
          raise RuntimeError(f'Invalid message: {message}')
    except (EOFError, KeyboardInterrupt):
      return
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error inside worker: {stacktrace}.', flush=True)
      pipe.send((Message.ERROR, stacktrace))
      return
    finally:
      try:
        pipe.close()
      except Exception:
        pass
