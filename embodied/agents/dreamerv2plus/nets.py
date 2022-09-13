import functools
import re

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import tfutils


class RSSM(tfutils.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=True, initial='zeros',
      unimix=0.0, prior_layers=1, post_layers=1, gru_layers=1, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._prior_layers = prior_layers
    self._post_layers = post_layers
    self._gru_layers = gru_layers
    self._kw = kw
    self._cast = tfutils.cast_to_compute

  def initial(self, batch_size):
    if self._classes:
      state = tfutils.cast_to_compute(dict(
          deter=tf.zeros([batch_size, self._deter], tf.float32),
          logit=tf.zeros([batch_size, self._stoch, self._classes], tf.float32),
          stoch=tf.zeros(
              [batch_size, self._stoch, self._classes], tf.float32)))
    else:
      state = tfutils.cast_to_compute(dict(
          deter=tf.zeros([batch_size, self._deter], tf.float32),
          mean=tf.zeros([batch_size, self._stoch], tf.float32),
          std=tf.ones([batch_size, self._stoch], tf.float32),
          stoch=tf.zeros([batch_size, self._stoch], tf.float32)))
    if self._initial == 'zeros':
      return state
    elif self._initial == 'learned':
      # This will cut gradients when the state is created outside of the
      # training graph, but this only happens once at the beginning of the
      # training loop. Afterwards, the state is reset inside the obs_step().
      state['deter'] = tf.repeat(self._cast(self.get(
          'initial_deter', tf.Variable, state['deter'][0].astype(tf.float32),
          trainable=True))[None], batch_size, 0)
      state['stoch'] = tf.repeat(self._cast(self.get(
          'initial_stoch', tf.Variable, state['stoch'][0].astype(tf.float32),
          trainable=True))[None], batch_size, 0)
      return state
    elif self._initial == 'learned2':
      # This will cut gradients when the state is created outside of the
      # training graph, but this only happens once at the beginning of the
      # training loop. Afterwards, the state is reset inside the obs_step().
      state['deter'] = tf.repeat(self._cast(tf.math.tanh(self.get(
          'initial_deter', tf.Variable, state['deter'][0].astype(tf.float32),
          trainable=True)))[None], batch_size, 0)
      state['stoch'] = self.get_stoch(state['deter'])
      return state
    else:
      raise NotImplementedError(self._initial)

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    post, prior = tfutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tfutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = tf.cast(state['logit'], tf.float32)
      dist = tfd.Independent(tfutils.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first):
    prev_state, prev_action, is_first = tf.nest.map_structure(
        self._cast, (prev_state, prev_action, is_first))
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum('b...,b->b...', x, 1.0 - is_first),
        (prev_state, prev_action))
    prev_state = tf.nest.map_structure(
        lambda x, y: x + tf.einsum('b...,b->b...', self._cast(y), is_first),
        prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    for i in range(self._post_layers - 1):
      x = self.get(f'obs_out_{i}', Linear, **self._kw)(x)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats_layer('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = self._cast(dist.sample())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._classes:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._classes]
      prev_stoch = tf.reshape(prev_stoch, shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + [np.prod(prev_action.shape[-2:])]
      prev_action = prev_action.reshape(shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    # x = self.get('img_out', Linear, **self._kw)(x)
    for i in range(self._prior_layers):
      x = self.get(f'img_out_{i}', Linear, **self._kw)(x)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    stoch = self._cast(dist.sample())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def get_stoch(self, deter):
    # x = self.get('img_out', Linear, **self._kw)(deter)
    x = deter
    for i in range(self._prior_layers):
      x = self.get(f'img_out_{i}', Linear, **self._kw)(x)
    stats = self._stats_layer('img_stats', x)
    dist = self.get_dist(stats)
    return self._cast(dist.mode())

  def _gru(self, x, deter):
    x = tf.concat([deter, x], -1)
    for i in range(self._gru_layers - 1):
      x = self.get(f'gru_{i}', Linear, **self._kw)(x)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru_out', Linear, **kw)(x)
    reset, cand, update = tf.split(x, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = tf.math.tanh(reset * cand)
    update = tf.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats_layer(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._classes])
      if self._unimix:
        probs = tf.nn.softmax(logit, -1)
        uniform = tf.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = tf.math.log(probs)
      return {'logit': logit}
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = tf.split(x, 2, -1)
      std = 2 * tf.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, balance=0.8):
    post_const = tf.nest.map_structure(tf.stop_gradient, post)
    prior_const = tf.nest.map_structure(tf.stop_gradient, prior)
    lhs = tfd.kl_divergence(self.get_dist(post_const), self.get_dist(prior))
    rhs = tfd.kl_divergence(self.get_dist(post), self.get_dist(prior_const))
    return balance * lhs + (1 - balance) * rhs


class MultiEncoder(tfutils.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='simple', cnn_depth=48, cnn_kernels=(4, 4, 4, 4),
      cnn_blocks=2, **kw):
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) in (0, 1)}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    if cnn == 'simple':
      self._cnn = ImageEncoderSimple(cnn_depth, cnn_kernels, **kw)
    elif cnn == 'resnet':
      self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, **kw)
    else:
      raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **kw)

  def __call__(self, data):
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = tuple(data[some_key].shape[:-len(some_shape)])
    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = tf.concat([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      outputs.append(output)
    if self.mlp_shapes:
      inputs = [
          data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
          for k in self.mlp_shapes]
      inputs = tf.concat(tfutils.cast_to_compute(inputs), -1)
      outputs.append(self._mlp(inputs))
    outputs = tf.concat(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:])
    return outputs


class MultiDecoder(tfutils.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='simple', cnn_depth=48,
      cnn_kernels=(5, 5, 6, 6), cnn_blocks=2, image_dist='mse', **kw):
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      merged = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'simple':
        self._cnn = ImageDecoderSimple(merged, cnn_depth, cnn_kernels, **kw)
      elif cnn == 'resnet':
        self._cnn = ImageDecoderResnet(merged, cnn_depth, cnn_blocks, **kw)
      else:
        raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **kw)
    self._inputs = Input(inputs)
    self._image_dist = image_dist

  def __call__(self, inputs):
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      flat = features.reshape([-1, features.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(features.shape[:-1] + output.shape[1:])
      means = tf.split(output, [v[-1] for v in self.cnn_shapes.values()], -1)
      dists.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists

  def _make_image_dist(self, name, mean):
    mean = mean.astype(tf.float32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return tfutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class ImageEncoderSimple(tfutils.Module):

  def __init__(self, depth, kernels, **kw):
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, features):
    Conv = functools.partial(Conv2D, stride=2, pad='valid')
    x = tfutils.cast_to_compute(features)
    depth = self._depth
    for i, kernel in enumerate(self._kernels):
      x = self.get(f'conv{i}', Conv, depth, kernel, **self._kw)(x)
      depth *= 2
    return x


class ImageDecoderSimple(tfutils.Module):

  def __init__(self, shape, depth, kernels, **kw):
    self._shape = shape
    self._depth = depth
    self._kernels = kernels
    self._kw = kw

  def __call__(self, features):
    ConvT = functools.partial(Conv2D, transp=True, stride=2, pad='valid')
    x = tfutils.cast_to_compute(features)
    x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
    depth = self._depth * 2 ** (len(self._kernels) - 2)
    for i, kernel in enumerate(self._kernels[:-1]):
      x = self.get(f'conv{i}', ConvT, depth, kernel, **self._kw)(x)
      depth //= 2
    x = self.get('out', ConvT, self._shape[-1], self._kernels[-1])(x)
    x = tf.math.sigmoid(x)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    return x


class ImageEncoderResnet(tfutils.Module):

  def __init__(self, depth, blocks, **kw):
    self._depth = depth
    self._blocks = blocks
    self._kw = {**kw, 'preact': True}

  def __call__(self, image):
    x = tfutils.cast_to_compute(image)
    stages = int(np.log2(image.shape[-2])) - 2
    depth = self._depth
    x = self.get('in', Conv2D, depth, 3)(x)
    for i in range(stages):
      x = tf.nn.avg_pool(x, [2, 2], [2, 2], 'SAME')
      for j in range(self._blocks):
        x = self._block(f's{i}b{j}', depth, x)
        # print(i, j, x.shape)
      depth *= 2
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    x = self.get('out', Linear, 1024)(x)
    return x

  def _block(self, name, depth, x):
    skip = x
    if skip.shape[-1] != depth:
      skip = self.get(f'{name}s', Conv2D, depth, 1, bias=False)(skip)
    x = self.get(f'{name}a', Conv2D, depth, 3, **self._kw)(x)
    x = self.get(f'{name}b', Conv2D, depth, 3, **self._kw)(x)
    return skip + 0.1 * x


class ImageDecoderResnet(tfutils.Module):

  def __init__(self, shape, depth, blocks, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._kw = {**kw, 'preact': True}

  def __call__(self, features):
    x = tfutils.cast_to_compute(features)
    stages = int(np.log2(self._shape[0])) - 2
    depth = 2 ** stages * self._depth
    x = self.get('in', Linear, 16 * depth)(x)
    x = x.reshape([-1, 4, 4, depth])
    for i in range(stages):
      for j in range(self._blocks):
        x = self._block(f's{i}b{j}', depth, x)
        # print(i, j, x.shape)
      x = tf.repeat(tf.repeat(x, 2, 1), 2, 2)  # Upsample
      depth //= 2
    x = self.get('out', Conv2D, self._shape[-1], 3)(x)
    x = tf.math.sigmoid(x)
    return x

  def _block(self, name, depth, x):
    skip = x
    if skip.shape[-1] != depth:
      skip = self.get(f'{name}s', Conv2D, depth, 1, bias=False)(skip)
    x = self.get(f'{name}a', Conv2D, depth, 3, **self._kw)(x)
    x = self.get(f'{name}b', Conv2D, depth, 3, **self._kw)(x)
    return skip + 0.1 * x


class MLP(tfutils.Module):

  def __init__(self, shape, layers, units, inputs=['tensor'], dims=None, **kw):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    distkeys = ('dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    feat = self._inputs(inputs)
    x = tfutils.cast_to_compute(feat)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'dense{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + [x.shape[-1]])
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', DistLayer, shape, **self._dist)(x)


class DistLayer(tfutils.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, minstd=0.1, maxstd=1.0,
      unimix=0.0):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    out = self.get(
        'out', Linear, int(np.prod(self._shape)),
        outscale=self._outscale)(inputs)
    out = tf.reshape(out, tuple(inputs.shape[:-1]) + self._shape)
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tuple(inputs.shape[:-1]) + self._shape)
      std = tf.cast(std, tf.float32)
    if self._dist == 'symlog':
      return tfutils.SymlogDist(out, len(self._shape), 'sum')
    if self._dist == 'mse':
      return tfutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * tf.nn.sigmoid(std) + lo
      dist = tfd.Normal(tf.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'trunc_normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * tf.nn.sigmoid(std) + lo
      dist = tfd.TruncatedNormal(tf.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, 1)
      dist.minent = np.prod(self._shape) * tfd.Normal(0.99, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'onehot':
      if self._unimix:
        probs = tf.nn.softmax(out, -1)
        uniform = tf.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = tf.math.log(probs)
      dist = tfutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * np.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(tfutils.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._pad = pad.upper()
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm)
    self._preact = preact
    self._bias = bias

  def __call__(self, hidden):
    if self._preact:
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      limit = np.sqrt(3.0 / (np.prod(shape[:-2]) * np.mean(shape[-2:])))
      winit = np.random.uniform(-limit, limit, shape).astype(np.float32)
      kernel = self.get(
          'kernel', tf.Variable, winit, dtype=tf.float32, trainable=True)
      kernel = tfutils.cast_to_compute(kernel)
      if self._pad == 'VALID':
        out = tuple(x.shape[:-3]) + (
            int(x.shape[-3] * self._stride + self._kernel - 2),
            int(x.shape[-2] * self._stride + self._kernel - 2),
            self._depth)
      if self._pad == 'SAME':
        out = tuple(x.shape[:-3]) + (
            int(x.shape[-3] * self._stride),
            int(x.shape[-2] * self._stride),
            self._depth)
      x = tf.nn.conv2d_transpose(x, kernel, out, self._stride, self._pad)
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      limit = np.sqrt(3.0 / np.mean(shape[-2:]))
      winit = np.random.uniform(-limit, limit, shape).astype(np.float32)
      kernel = self.get(
          'kernel', tf.Variable, winit, dtype=tf.float32, trainable=True)
      kernel = tfutils.cast_to_compute(kernel)
      x = tf.nn.conv2d(x, kernel, self._stride, self._pad)
    if self._bias:
      bias = self.get(
          'bias', tf.Variable, np.zeros(self._depth, np.float32),
          trainable=True)
      bias = tfutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(tfutils.Module):

  def __init__(self, units, act='none', norm='none', bias=True, outscale=1.0):
    self._units = units
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale

  def __call__(self, x):
    shape = (x.shape[-1], self._units)
    limit = np.sqrt(3.0 * self._outscale / np.mean(shape))
    winit = np.random.uniform(-limit, limit, shape).astype(np.float32)
    kernel = self.get(
        'kernel', tf.Variable, winit, dtype=tf.float32, trainable=True)
    kernel = tfutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get(
          'bias', tf.Variable, np.zeros(self._units, np.float32),
          dtype=tf.float32, trainable=True)
      bias = tfutils.cast_to_compute(bias)
      x += bias
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class Norm(tfutils.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      scale = self.get('scale', tf.Variable, tf.ones(x.shape[-1], tf.float32))
      bias = self.get('bias', tf.Variable, tf.zeros(x.shape[-1], tf.float32))
      x = x.astype(tf.float32)
      mean, var = tf.nn.moments(x, -1, keepdims=True)
      return tf.nn.batch_normalization(
          x, mean, var, bias, scale, 1e-3).astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + [np.prod(value.shape[dims - 1:])])
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return tf.concat(values, -1)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return tf.identity
  elif name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif name == 'gelu':
    return lambda x: tf.nn.gelu(x, approximate=True)
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)
