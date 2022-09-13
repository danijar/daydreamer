import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import tfutils


class Greedy(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config)}
    elif config.critic_type == 'qfunction':
      critics = {'extr': agent.QFunction(rewfn, config)}
    elif config.critic_type == 'qtwin':
      critics = {'extr': agent.TwinQFunction(rewfn, config)}
    self.ac = agent.ImagActorCritic(critics, {'extr': 1.0}, act_space, config)

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    metrics = {}
    context, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    traj, _ = self.wm.imagine(
        self.policy, start, self.initial(len(next(iter(start)))),
        self.config.imag_horizon)
    dists = self.wm.heads['decoder'](traj)
    for key in self.wm.heads['decoder'].cnn_shapes.keys():
      video = dists[key].mode().transpose((1, 0, 2, 3, 4))
      metrics[f'imag_{key}'] = tfutils.video_grid(video)
    return metrics


class KnownReward(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self._auto_repr = None  # TODO: Sonnet weirdness.
    self.wm = wm
    self.config = config
    self.ac = agent.ImagActorCritic(
        {'manual': agent.VFunction(self.rewfn, config)},
        {'manual': 1.0}, act_space, config)

  def __repr__(self):
    return f'Module({self.name})'

  def rewfn(self, s):
    dists = self.wm.heads['decoder'](s)
    preds = {k: v.mode() for k, v in dists.items()}
    print({k: v.shape for k, v in preds.items()})
    if self.config.known_reward == 'none':
      return tf.zeros(s['deter'][1:, ..., 0].shape, tf.float32)
    else:
      raise NotImplementedError(self.config.known_reward)

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    metrics = {}
    context, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    traj, _ = self.wm.imagine(
        self.policy, start, self.initial(len(next(iter(start)))),
        self.config.imag_horizon)
    dists = self.wm.heads['decoder'](traj)
    for key in self.wm.heads['decoder'].cnn_shapes.keys():
      video = dists[key].mode().transpose((1, 0, 2, 3, 4))
      metrics[f'imag_{key}'] = tfutils.video_grid(video)
    return metrics


class Random(tfutils.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return tf.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = tfutils.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(tfutils.Module):

  REWARDS = {
      'disag': expl.Disag,
      'vae': expl.LatentVAE,
      'ctrl': expl.CtrlDisag,
      'pbe': expl.PBE,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        reward = lambda traj: wm.heads['reward'](traj).mean()[1:]
        critics[key] = agent.VFunction(reward, config)
      else:
        reward = self.REWARDS[key](wm, act_space, config)
        critics[key] = agent.VFunction(reward, config.update(
            discount=config.expl_discount,
            retnorm=config.expl_retnorm,
            scorenorm=config.expl_scorenorm))
        self.rewards[key] = reward
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(critics, scales, act_space, config)

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, reward in self.rewards.items():
      metrics.update(reward.train(data))
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}


class DisagWhen(tfutils.Module):

  def __init__(self, wm, act_space, config):
    config = config.update({'disag_head.inputs': ['deter']})
    self.act_space = act_space
    self.config = config
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    self.achiever = agent.ImagActorCritic({
        'extr': agent.VFunction(rewfn, config),
    }, {'extr': 1.0}, act_space, config)
    self.disag = expl.Disag(wm, act_space, config)
    self.explorer = agent.ImagActorCritic({
        'expl': agent.VFunction(self.disag, config),
    }, {'expl': 1.0}, act_space, config)
    capacity = int(self.config.expl_when_buffer)
    self.buffer = tf.Variable(
        tf.zeros((capacity, self.config.rssm.deter)),
        trainable=False, dtype=tf.float32)
    self.disags = tf.Variable(
        tf.zeros(capacity), trainable=False, dtype=tf.float32)
    self.once = tf.Variable(True, trainable=False, dtype=tf.bool)

  def initial(self, batch_size):
    return {
        'achiever': self.achiever.initial(batch_size),
        'explorer': self.explorer.initial(batch_size),
        'exploring': tf.zeros(batch_size, tf.bool),
        'counter': tf.zeros(batch_size, tf.int64),
    }

  def policy(self, latent, state):
    disag = self.disagreement(latent['deter'])
    higher = (disag[:, None] > self.disags[None, :])
    frac = higher.astype(tf.float32).sum(1) / self.disags.shape[0]
    exploring = tf.where(
        state['counter'] > 0, state['exploring'],
        (frac > self.config.expl_when_frac))
    counter = (state['counter'] + 1) % self.config.expl_when_every
    ac_dist, ac_state = self.achiever.policy(latent, state['achiever'])
    ex_dist, ex_state = self.explorer.policy(latent, state['explorer'])

    if self.config.expl_when_random:
      shape = (len(state['counter']),) + self.act_space.shape
      if self.act_space.discrete:
        dist = tfutils.OneHotDist(tf.zeros(shape))
      else:
        dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
        dist = tfd.Independent(dist, 1)
      ac_dist = dist

    act = tf.where(exploring[:, None], ex_dist.sample(), ac_dist.sample())
    act = tfd.Deterministic(act)
    state = {
        'achiever': ac_state, 'explorer': ex_state,
        'exploring': exploring, 'counter': counter}
    return act, state

  def train(self, imagine, start, data):
    metrics = {}
    metrics.update(self.disag.train(data))
    traj, mets = self.explorer.train(imagine, start, data)
    metrics.update({f'explorer_{k}': v for k, v in mets.items()})
    traj, mets = self.achiever.train(imagine, start, data)
    metrics.update({f'achiever_{k}': v for k, v in mets.items()})
    states = data['deter']
    states = states[:, states.shape[1] // 2]
    if self.once:
      states = tf.repeat(states[:1], self.buffer.shape[0], 0)
      self.buffer.assign(states.astype(tf.float32))
      self.disags.assign(self.disagreement(states))
      self.once.assign(False)
    else:
      states = tf.concat([self.buffer, states.astype(tf.float32)], 0)
      disags = self.disagreement(states)
      indices = tf.argsort(disags, 0)[-self.buffer.shape[0]:]
      self.buffer.assign(tf.gather(states, indices))
      self.disags.assign(tf.gather(disags, indices))
    return traj, metrics

  def disagreement(self, deter):
    return self.disag({'deter': tf.concat([deter[:1], deter], 0)})

  def report(self, data):
    return {}
