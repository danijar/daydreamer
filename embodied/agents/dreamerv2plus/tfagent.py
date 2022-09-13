import contextlib

import embodied
import tensorflow as tf

try:
  from tensorflow.python.distribute import values
except Exception:
  from google3.third_party.tensorflow.python.distribute import values

from . import tfutils


def Wrapper(agent_cls):
  class Agent(TFAgent):
    configs = agent_cls.configs
    def __init__(self, obs_space, act_space, step, config):
      super().__init__(agent_cls, obs_space, act_space, step, config)
  return Agent


class TFAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.tf
    self.strategy = self._setup()
    with self._strategy_scope():
      self.agent = agent_cls(obs_space, act_space, step, config)
      self.agent.strategy = self.strategy  # TODO
    self._cache_fns = config.tf.jit and not self.strategy
    self._cached_fns = {}

  def dataset(self, generator):
    with self._strategy_scope():
      dataset = self.agent.dataset(generator)
    # if self.strategy:
    #   dataset = self.strategy.experimental_distribute_dataset(dataset)
    return dataset

  def policy(self, obs, state=None, mode='train'):
    obs = {k: v for k, v in obs.items() if not k.startswith('log_')}
    obs = self._convert_inps(obs)
    if state is None:
      state = self._strategy_run(self.agent.initial_policy_state, obs)
    fn = self.agent.policy
    if self._cache_fns:
      if hasattr(fn, 'get_concrete_function'):
        key = f'policy_{mode}'
        if key not in self._cached_fns:
          self._cached_fns[key] = fn.get_concrete_function(obs, state, mode)
        fn = self._cached_fns[key]
    act, state = self._strategy_run(fn, obs, state, mode)
    act = self._convert_outs(act)
    return act, state

  def train(self, data, state=None):
    data = self._convert_inps(data)
    if state is None:
      state = self._strategy_run(self.agent.initial_train_state, data)
    fn = self.agent.train
    if self._cache_fns:
      if hasattr(fn, 'get_concrete_function'):
        key = 'train'
        if key not in self._cached_fns:
          self._cached_fns[key] = fn.get_concrete_function(data, state)
        fn = self._cached_fns[key]
    outs, state, metrics = self._strategy_run(fn, data, state)
    outs = self._convert_outs(outs)
    metrics = self._convert_mets(metrics)
    return outs, state, metrics

  def report(self, data):
    data = self._convert_inps(data)
    fn = self.agent.report
    if self._cache_fns:
      if hasattr(fn, 'get_concrete_function'):
        key = 'report'
        if key not in self._cached_fns:
          self._cached_fns[key] = fn.get_concrete_function(data)
        fn = self._cached_fns[key]
    metrics = self._strategy_run(fn, data)
    metrics = self._convert_mets(metrics)
    return metrics

  def save(self):
    return self.agent.save()

  def load(self, values):
    self.agent.load(values)

  @contextlib.contextmanager
  def _strategy_scope(self):
    if self.strategy:
      with self.strategy.scope():
        yield None
    else:
      yield None

  def _strategy_run(self, fn, *args, **kwargs):
    if self.strategy:
      return self.strategy.run(fn, args, kwargs)
    else:
      return fn(*args, **kwargs)

  def _convert_inps(self, value):
    if not self.strategy:
      return value
    if isinstance(value, (tuple, dict)):
      return tf.nest.map_structure(self._convert_inps, value)
    if isinstance(value, values.PerReplica):
      return value
    replicas = self.strategy.num_replicas_in_sync
    assert len(value) % replicas == 0, (len(value), replicas)
    value = tf.split(value, replicas, 0)
    return self.strategy.experimental_distribute_values_from_function(
        lambda ctx: value[ctx.replica_id_in_sync_group])

  def _convert_outs(self, value):
    if isinstance(value, (tuple, list, dict)):
      return tf.nest.map_structure(self._convert_outs, value)
    if isinstance(value, values.PerReplica):
      value = self.strategy.gather(value, axis=0)
    if hasattr(value, 'numpy'):  # Tensor, Variable, MirroredVariable
      value = value.numpy()
    return value

  def _convert_mets(self, value):
    if isinstance(value, (tuple, list, dict)):
      return tf.nest.map_structure(self._convert_mets, value)
    if isinstance(value, values.PerReplica):
      value = value.values[0]  # Only use metrics from first replica.
    if hasattr(value, 'numpy'):  # Tensor, Variable, MirroredVariable
      value = value.numpy()
    return value

  def _setup(self):
    assert self.config.precision in ('float16', 'float32'), (
        self.config.precision)
    tf.config.run_functions_eagerly(not self.config.jit)
    tf.config.set_soft_device_placement(self.config.soft_placement)
    if self.config.debug_nans:
      tf.debugging.enable_check_numerics()

    tf.config.experimental.enable_tensor_float_32_execution(
        self.config.tensorfloat)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if self.config.logical_gpus:
      conf = tf.config.LogicalDeviceConfiguration(memory_limit=1024)
      tf.config.set_logical_device_configuration(
          gpus[0], [conf] * self.config.logical_gpus)

    if self.config.platform == 'cpu':
      return None

    elif self.config.platform == 'gpu':
      assert len(gpus) >= 1, gpus
      if not self.config.logical_gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, self.config.growth)
      if self.config.precision == 'float16':
        tfutils.COMPUTE_DTYPE = tf.float16
      return None

    elif self.config.platform == 'multi_gpu':
      assert len(gpus) >= 1, gpus
      if self.config.precision == 'float16':
        tfutils.COMPUTE_DTYPE = tf.float16
      return tf.distribute.MirroredStrategy()

    elif self.config.platform == 'tpu':
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      return tf.distribute.TPUStrategy(resolver)

    else:
      raise NotImplementedError(self.config.platform)
