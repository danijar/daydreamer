import collections
import re
import warnings

import embodied
import numpy as np


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_train = embodied.when.Every(args.train_every)
  should_expl = embodied.when.Until(args.expl_until)
  should_log = embodied.when.Every(args.log_every)
  should_video_train = embodied.when.Every(args.eval_every)
  should_video_eval = embodied.when.Every(args.eval_every)
  step = logger.step

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    metrics = {}
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    metrics['length'] = length
    metrics['score'] = score
    metrics['reward_rate'] = (ep['reward'] - ep['reward'].min() >= 0.1).mean()
    logs = {}
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        logs[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        logs[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        logs[f'max_{key}'] = ep[key].max(0).mean()
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in args.log_keys_video:
        metrics[f'policy_{key}'] = ep[key]
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(metrics, prefix=f'{mode}_episode')
    logger.add(logs, prefix=f'{mode}_logs')
    logger.add(replay.stats, prefix='replay')
    logger.write()

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  fill = max(0, args.eval_fill - len(eval_replay))
  if fill:
    print(f'Fill eval dataset ({fill} steps).')
    driver_eval(random_agent.policy, steps=fill, episodes=1)
  fill = max(0, args.train_fill - len(train_replay))
  if fill:
    print(f'Fill train dataset ({fill} steps).')
    driver_train(random_agent.policy, steps=fill, episodes=1)

  dataset_train = iter(agent.dataset(train_replay.dataset))
  dataset_eval = iter(agent.dataset(eval_replay.dataset))
  state = [None]  # To be writable from train step function below.
  assert args.pretrain > 0  # At least one step to initialize variables.
  for _ in range(args.pretrain):
    _, state[0], _ = agent.train(next(dataset_train), state[0])

  metrics = collections.defaultdict(list)
  batch = [None]
  def train_step(tran, worker):
    if should_train(step):
      for _ in range(args.train_steps):
        batch[0] = next(dataset_train)
        outs, state[0], mets = agent.train(batch[0], state[0])
        [metrics[key].append(value) for key, value in mets.items()]
        if 'priority' in outs:
          train_replay.prioritize(outs['key'], outs['priority'])
    if should_log(step):
      with warnings.catch_warnings():  # Ignore empty slice warnings.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for name, values in metrics.items():
          logger.scalar('train/' + name, np.nanmean(values, dtype=np.float64))
          metrics[name].clear()
      logger.add(agent.report(batch[0]), prefix='report')
      logger.add(agent.report(next(dataset_eval)), prefix='eval')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  checkpoint.load_or_save()

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    logger.write()
    driver_eval.reset()
    driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    driver_train(policy_train, steps=args.eval_every)
    checkpoint.save()
