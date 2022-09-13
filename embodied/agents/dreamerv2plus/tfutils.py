import inspect
import logging
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')

import numpy as np
try:
  import sonnet.v2 as snt
except ImportError:
  import sonnet as snt
import tensorflow as tf
from tensorflow_probability import distributions as tfd

try:
  from tensorflow.python.distribute import values
except Exception:
  from google3.third_party.tensorflow.python.distribute import values

COMPUTE_DTYPE = tf.float32


for base in (tf.Tensor, tf.Variable, values.PerReplica):
  base.mean = tf.math.reduce_mean
  base.std = tf.math.reduce_std
  base.var = tf.math.reduce_variance
  base.sum = tf.math.reduce_sum
  base.prod = tf.math.reduce_prod
  base.any = tf.math.reduce_any
  base.all = tf.math.reduce_all
  base.min = tf.math.reduce_min
  base.max = tf.math.reduce_max
  base.abs = tf.math.abs
  base.logsumexp = tf.math.reduce_logsumexp
  base.transpose = tf.transpose
  base.reshape = tf.reshape
  base.astype = tf.cast
  base.flatten = lambda x: tf.reshape(x, [-1])


def tensor(value):
  if isinstance(value, values.PerReplica):
    return value
  return tf.convert_to_tensor(value)
tf.tensor = tensor


def scan(fn, inputs, start, static=True, reverse=False, axis=0):
  assert axis in (0, 1), axis
  if axis == 1:
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    inputs = tf.nest.map_structure(swap, inputs)
  if not static:
    return tf.scan(fn, inputs, start, reverse=reverse)
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(tf.nest.flatten(inputs)[0].shape[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    tf.nest.assert_same_structure(last, start)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, axis) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def cast_to_compute(values):
  return tf.nest.map_structure(lambda x: x.astype(COMPUTE_DTYPE), values)


def symlog(x):
  return tf.sign(x) * tf.math.log(1 + tf.abs(x))


def symexp(x):
  return tf.sign(x) * (tf.math.exp(tf.abs(x)) - 1)


def action_noise(action, amount, act_space):
  if amount == 0:
    return action
  amount = tf.cast(amount, action.dtype)
  if act_space.discrete:
    probs = amount / action.shape[-1] + (1 - amount) * action
    return OneHotDist(probs=probs).sample()
  else:
    return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)


class Module(snt.Module):

  _COUNTERS = {}

  def __new__(subcls, *args, **kwargs):
    path = f'{tf.get_current_name_scope()}/{subcls.__name__}'
    count = Module._COUNTERS.get(path, 0)
    name = f'{subcls.__name__}{count or ""}'
    Module._COUNTERS[path] = count + 1
    obj = super().__new__(subcls)
    snt.Module.__init__(obj, name=name)
    obj._modules = {}
    return obj

  def __init__(self, *args, **kwargs):
    raise RuntimeError('Calling super().__init__() is not needed.')

  def __repr__(self):
    return f'Module({self.name})'

  def save(self):
    values = {x.name: x.numpy() for x in self.variables}
    count = int(sum(np.prod(x.shape) for x in values.values()))
    print(f'Saving module with {len(values)} tensors and {count} parameters.')
    return values

  def load(self, values):
    existing = {x.name: x for x in self.variables}
    provided = values.copy()
    count = int(sum(np.prod(x.shape) for x in provided.values()))
    print(f'Loading module with {len(values)} tensors and {count} parameters.')
    existing = [x[1] for x in sorted(existing.items(), key=lambda x: x[0])]
    provided = [x[1] for x in sorted(provided.items(), key=lambda x: x[0])]
    assert len(provided) == len(existing), (len(provided) == len(existing))
    for src, dst in zip(provided, existing):
      dst.assign(src)

  def get(self, name, ctor, *args, **kwargs):
    if name not in self._modules:
      params = inspect.signature(ctor).parameters
      if 'name' in params or ctor is tf.Variable:
        self._modules[name] = ctor(*args, name=name, **kwargs)
      else:
        self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]


class Optimizer(Module):

  def __init__(
      self, name, lr, opt='adam', eps=1e-5, clip=0.0, warmup=0, wd=0.0,
      wd_pattern='kernel'):
    assert opt == 'adam', opt
    assert (0 <= wd < 1), wd
    assert (not clip or 1 <= clip), clip
    self._name = name
    self._lr = lr
    self._eps = eps
    self._clip = clip
    self._warmup = warmup
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self._m = []
    self._v = []
    if warmup:
      self._lr = lambda: lr * tf.clip_by_value(
          self._step.astype(tf.float32) / warmup, 0.0, 1.0)
    self._mixed = (COMPUTE_DTYPE == tf.float16)
    if self._mixed:
      self._grad_scale = tf.Variable(1e4, trainable=False, dtype=tf.float32)
      self._good_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    self._once = True
    # self._opt = tf.optimizers.Adam(lr, epsilon=eps)  # TODO

  @property
  def variables(self):
    variables = [self.step]
    variables += list(self._m.values())
    variables += list(self._v.values())
    if self._mixed:
      variables += [self._grad_scale, self._good_steps]
    return variables

  def __call__(self, tape, loss, modules):
    assert loss.dtype is tf.float32, (self._name, loss.dtype)
    assert len(loss.shape) == 0, (self._name, loss.shape)
    metrics = {}

    # Find variables.
    modules = modules if hasattr(modules, '__len__') else (modules,)
    params = tf.nest.flatten([
        module.trainable_variables for module in modules])
    params = sorted(params, key=lambda x: x.name)
    if self._once:
      # print('-' * 79)
      # for param in params:
      #   print(param.name)
      count = sum(int(np.prod(x.shape)) for x in params)
      print(f'Found {count} {self._name} parameters.')
      zero_var = lambda x, n: tf.Variable(
          tf.zeros_like(x),
          name=n.replace('/', '_').strip(':0'),
          trainable=False)
      self._m = {x.name: zero_var(x, f'm/{x.name}') for x in params}
      self._v = {x.name: zero_var(x, f'v/{x.name}') for x in params}
    else:
      assert set(self._m.keys()) == {x.name for x in params}
      assert set(self._v.keys()) == {x.name for x in params}

    # Check loss.
    tf.debugging.check_numerics(loss, self._name + '_loss')
    metrics[f'{self._name}_loss'] = loss

    # Compute scaled gradient.
    if self._mixed:
      with tape:
        loss = self._grad_scale * loss
    grads = tape.gradient(loss, params)
    for param, grad in zip(params, grads):
      if grad is None:
        raise RuntimeError(
            f'{self._name} optimizer found no gradient for {param.name}.')

    # Distributed sync.
    if tf.distribute.has_strategy():
      context = tf.distribute.get_replica_context()
      grads = context.all_reduce('mean', grads)

    overflow = False
    if self._mixed:
      grads = [x / self._grad_scale for x in grads]
      overflow = ~tf.reduce_all([
          tf.math.is_finite(x).all() for x in tf.nest.flatten(grads)])
      metrics[f'{self._name}_grad_scale'] = self._grad_scale
      metrics[f'{self._name}_grad_overflow'] = overflow.astype(tf.float32)
      keep = (~overflow & (self._good_steps < 1000))
      incr = (~overflow & (self._good_steps >= 1000))
      decr = overflow
      self._good_steps.assign(keep.astype(tf.int64) * (self._good_steps + 1))
      self._grad_scale.assign(tf.clip_by_value(
          keep.astype(tf.float32) * self._grad_scale +
          incr.astype(tf.float32) * self._grad_scale * 2 +
          decr.astype(tf.float32) * self._grad_scale / 2,
          1e-4, 1e4))

    # Gradient clipping.
    norm = tf.linalg.global_norm(grads)
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
    if self._mixed:
      norm = tf.where(tf.math.is_finite(norm), norm, np.nan)
    else:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    metrics[f'{self._name}_grad_norm'] = norm

    # TODO: Apply gradients or weight decay first?
    # Weight decay.
    if self._wd:
      if ~overflow:
        self._apply_wd(params)

    # Apply gradients.
    if ~overflow:
      self._step.assign_add(1)
      self._apply_adam(params, grads)
      # TODO
      # self._opt.apply_gradients(
      #     zip(grads, params),
      #     experimental_aggregate_gradients=False)
    metrics[f'{self._name}_grad_steps'] = self._step

    self._once = False
    return metrics

  def _apply_adam(self, params, grads, beta1=0.9, beta2=0.999):
    lr = self._lr() if callable(self._lr) else self._lr
    t = self._step.astype(tf.float32)
    for param, grad in zip(params, grads):
      assert not isinstance(grad, tf.IndexedSlices), type(grad)
      name = param.name
      assert self._m[name].shape == grad.shape, (
          param.name, self._m[name].shape, grad.shape, param.shape)
      self._m[name].assign(beta1 * self._m[name] + (1. - beta1) * grad)
      self._v[name].assign(beta2 * self._v[name] + (1. - beta2) * grad * grad)
      m_hat = self._m[name] / (1. - beta1 ** t)
      v_hat = self._v[name] / (1. - beta2 ** t)
      param.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + self._eps))

  def _apply_wd(self, params):
    lr = self._lr() if callable(self._lr) else self._lr
    log = (self._wd_pattern != r'.*') and self._once
    if log:
      print(f"Optimizer applied weight decay to {self._name} variables:")
    included, excluded = [], []
    for param in sorted(params, key=lambda x: x.name):
      if re.search(self._wd_pattern, self._name + '/' + param.name):
        param.assign((1 - self._wd * lr) * param)
        included.append(param.name)
      else:
        excluded.append(param.name)
    if log:
      for name in included:
        print(f'[x] {name}')
      for name in excluded:
        print(f'[ ] {name}')
      print('')


class MSEDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return self._mode

  def mean(self):
    return self._mode

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = ((self._mode - value) ** 2)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class SymlogDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = (self._mode - symlog(value)) ** 2
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=tf.float32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    if not isinstance(sample_shape, (list, tuple)):
      sample_shape = (sample_shape,)
    logits = self.logits_parameter().astype(self.dtype)
    shape = tuple(logits.shape)
    logits = logits.reshape([np.prod(shape[:-1]), shape[-1]])
    indices = tf.random.categorical(logits, np.prod(sample_shape), seed=None)
    sample = tf.one_hot(indices, shape[-1], dtype=self.dtype)
    if np.prod(sample_shape) != 1:
      sample = sample.transpose((1, 0, 2))
    sample = tf.stop_gradient(sample.reshape(sample_shape + shape))
    # Straight through biased gradient estimator.
    probs = self._pad(super().probs_parameter(), sample.shape)
    sample += tf.cast(probs - tf.stop_gradient(probs), sample.dtype)
    return sample

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(tf.float32) > thres).astype(tf.float32)
  neg = (target.astype(tf.float32) <= thres).astype(tf.float32)
  pred = (dist.mean().astype(tf.float32) > thres).astype(tf.float32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(tf.float32).mean(),
      pred=dist.mean().astype(tf.float32).mean(),
  )


class AutoAdapt(Module):

  def __init__(
      self, shape, impl, scale, target, min, max,
      vel=0.1, thres=0.1, inverse=False):
    self._shape = tuple(shape)
    self._impl = impl
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres
    if self._impl == 'fixed':
      self._scale = tf.tensor(scale)
    elif self._impl == 'mult':
      self._scale = tf.Variable(tf.ones(shape, tf.float32), trainable=False)
    elif self._impl == 'prop':
      self._scale = tf.Variable(tf.ones(shape, tf.float32), trainable=False)
    else:
      raise NotImplementedError(self._impl)

  @property
  def shape(self):
    return self._shape

  def __call__(self, reg, update=True):
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(), 'std': reg.std(),
        'scale_mean': scale.mean(), 'scale_std': scale.std()}
    return loss, metrics

  def scale(self):
    if self._impl == 'fixed':
      scale = self._scale
    elif self._impl == 'mult':
      scale = self._scale
    elif self._impl == 'prop':
      scale = self._scale
    else:
      raise NotImplementedError(self._impl)
    return tf.stop_gradient(tf.tensor(scale))

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    if self._impl == 'fixed':
      pass
    elif self._impl == 'mult':
      below = avg < (1 / (1 + self._thres)) * self._target
      above = avg > (1 + self._thres) * self._target
      if self._inverse:
        below, above = above, below
      inside = ~below & ~above
      adjusted = (
          above.astype(tf.float32) * self._scale * (1 + self._vel) +
          below.astype(tf.float32) * self._scale / (1 + self._vel) +
          inside.astype(tf.float32) * self._scale)
      self._scale.assign(tf.clip_by_value(adjusted, self._min, self._max))
    elif self._impl == 'prop':
      direction = avg - self._target
      if self._inverse:
        direction = -direction
      self._scale.assign(tf.clip_by_value(
          self._scale + self._vel * direction, self._min, self._max))
    else:
      raise NotImplementedError(self._impl)


class Normalize:

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, vareps=0.0, stdeps=0.0):
    self._impl = impl
    self._decay = decay
    self._max = max
    self._stdeps = stdeps
    self._vareps = vareps
    self._mean = tf.Variable(0.0, trainable=False, dtype=tf.float64)
    self._sqrs = tf.Variable(0.0, trainable=False, dtype=tf.float64)
    self._step = tf.Variable(0, trainable=False, dtype=tf.int64)

  def __call__(self, values, update=True):
    update and self.update(values)
    return self.transform(values)

  def update(self, values):
    x = values.astype(tf.float64)
    m = self._decay
    self._step.assign_add(1)
    self._mean.assign(m * self._mean + (1 - m) * x.mean())
    self._sqrs.assign(m * self._sqrs + (1 - m) * (x ** 2).mean())

  def transform(self, values):
    correction = 1 - self._decay ** self._step.astype(tf.float64)
    mean = self._mean / correction
    var = (self._sqrs / correction) - mean ** 2
    if self._max > 0.0:
      scale = tf.math.rsqrt(
          tf.maximum(var, 1 / self._max ** 2 + self._vareps) + self._stdeps)
    else:
      scale = tf.math.rsqrt(var + self._vareps) + self._stdeps
    if self._impl == 'off':
      pass
    elif self._impl == 'mean_std':
      values -= mean.astype(values.dtype)
      values *= scale.astype(values.dtype)
    elif self._impl == 'std':
      values *= scale.astype(values.dtype)
    else:
      raise NotImplementedError(self._impl)
    return values
