import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import nets
from . import tfutils


class Disag(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.config = config.update({'disag_head.inputs': ['tensor']})
    self.opt = tfutils.Optimizer('disag', **config.expl_opt)
    self.inputs = nets.Input(config.disag_head.inputs, dims='deter')
    self.target = nets.Input(self.config.disag_target, dims='deter')
    self.nets = None

  def __call__(self, traj):
    self._build(traj)
    inputs = self.inputs(traj)
    preds = [head(inputs).mode() for head in self.nets]
    disag = tf.math.reduce_std(preds, 0).mean(-1)
    if 'action' in self.config.disag_head.inputs:
      return disag[:-1]
    else:
      return disag[1:]

  def train(self, data):
    # TODO: Currently, actions lead to states in the same time step for replay
    # batches, but states lead to actions in the same time step for imagined
    # rollouts, as is more standard in RL. Here, we're using replay data.
    data = {**data, 'action': tf.concat(
        [data['action'][:, 1:], 0 * data['action'][:, :1]], 1)}
    self._build(data)
    inputs = self.inputs(data)[:, :-1]
    target = self.target(data)[:, 1:].astype(tf.float32)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self.nets]
      loss = -sum([pred.log_prob(target).mean() for pred in preds])
    return self.opt(tape, loss, self.nets)

  def _build(self, data):
    if not self.nets:
      self.nets = [
          nets.MLP(self.target(data).shape[-1], **self.config.disag_head)
          for _ in range(self.config.disag_models)]


class LatentVAE(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.enc = nets.MLP(**self.config.expl_enc)
    self.dec = nets.MLP(self.config.rssm.deter, **self.config.expl_dec)
    shape = self.config.expl_enc.shape
    if self.config.expl_enc.dist == 'onehot':
      self.prior = tfutils.OneHotDist(tf.zeros(shape))
      self.prior = tfd.Independent(self.prior, len(shape) - 1)
    else:
      self.prior = tfd.Normal(tf.zeros(shape), tf.ones(shape))
      self.prior = tfd.Independent(self.prior, len(shape))
    self.kl = tfutils.AutoAdapt(**self.config.expl_kl)
    self.opt = tfutils.Optimizer('disag', **self.config.expl_opt)
    self.flatten = lambda x: x.reshape(
        x.shape[:-len(shape)] + [np.prod(x.shape[len(shape):])])

  def __call__(self, traj):
    dist = self.enc(traj)
    target = tf.stop_gradient(traj['deter'].astype(tf.float32))
    ll = self.dec(self.flatten(dist.sample())).log_prob(target)
    if self.config.expl_vae_elbo:
      kl = tfd.kl_divergence(dist, self.prior)
      reward = kl - ll / self.kl.scale()
    else:
      reward = -ll
    return reward[1:]

  def train(self, data):
    metrics = {}
    target = tf.stop_gradient(data['deter'].astype(tf.float32))
    with tf.GradientTape() as tape:
      dist = self.enc(data)
      kl = tfd.kl_divergence(dist, self.prior)
      kl, mets = self.kl(kl)
      metrics.update({f'kl_{k}': v for k, v in mets.items()})
      ll = self.dec(self.flatten(dist.sample())).log_prob(target)
      assert kl.shape == ll.shape
      loss = (kl - ll).mean()
    metrics['vae_kl'] = kl.mean()
    metrics['vae_ll'] = ll.mean()
    metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
    return metrics


class CtrlDisag(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.disag = Disag(
        wm, act_space, config.update({'disag_target': ['ctrl']}))
    self.embed = nets.MLP((config.ctrl_size,), **config.ctrl_embed)
    self.head = nets.MLP(act_space.shape, **config.ctrl_head)
    self.opt = tfutils.Optimizer('ctrl', **config.ctrl_opt)

  def __call__(self, traj):
    return self.disag(traj)

  def train(self, data):
    metrics = {}
    with tf.GradientTape() as tape:
      ctrl = self.embed(data).mode()
      dist = self.head({'current': ctrl[:, :-1], 'next': ctrl[:, 1:]})
      loss = -dist.log_prob(data['action'][:, 1:]).mean()
    self.opt(tape, loss, [self.embed, self.head])
    metrics.update(self.disag.train({**data, 'ctrl': ctrl}))
    return metrics


class PBE(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.inputs = nets.Input(config.pbe_inputs, dims='deter')

  def __call__(self, traj):
    feat = self.inputs(traj)
    flat = feat.reshape([-1, feat.shape[-1]])
    dists = tf.norm(
        flat[:, None, :].reshape((len(flat), 1, -1)) -
        flat[None, :, :].reshape((1, len(flat), -1)), axis=-1)
    rew = -tf.math.top_k(-dists, self.config.pbe_knn, sorted=True)[0].mean(-1)
    return rew.reshape(feat.shape[:-1]).astype(tf.float32)

  def train(self, data):
    return {}
