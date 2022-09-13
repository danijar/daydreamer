# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):

        n, c, h, w = x.size()
        assert h == w

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3

        proprio_dim = 14  # TODO: Manually set this for new envs. Please.
        self.repr_dim = 3872 + proprio_dim

        # self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
        #                              nn.ReLU())

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):

        # print('-' * 79)
        # print('encoder input:', obs['image'].shape)
        # print('-' * 79)

        image = obs['image'].permute((0, 3, 1, 2))
        image = image / 255.0 - 0.5
        h = self.convnet(image)
        h = h.reshape(h.shape[0], -1)

        # TODO: change for your env, plz, thx.
        # print(h.shape)
        # print(obs['orientations'].shape)
        h = torch.concat([
            h,
            obs['orientations'],
        ], -1)
        # print('-' * 79)
        # print(h.shape)  # (B, 3872)
        # import sys; sys.exit()
        # print('-' * 79)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.encoder = Encoder(obs_shape).to(device)
        repr_dim = self.encoder.repr_dim

        # models
        self.actor = Actor(repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation

        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):

        obs = {
            k: v.astype(np.float32) if (
                isinstance(v, np.ndarray) and v.dtype == np.float64) else v
            for k, v in obs.items()}
        obs = {
            k: torch.as_tensor(np.array(v), device=self.device)[None]
            for k, v in obs.items()}

        obs = self.encoder(obs)

        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        # obs = obs['image'].copy()

        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        # obs = obs['image'].copy()

        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = next(replay_iter)

        # obs, action, reward, discount, next_obs = utils.to_torch(
        #     batch, self.device)

        reward = torch.as_tensor(np.array(reward), device=self.device)
        action = torch.as_tensor(np.array(action), device=self.device)
        discount = torch.as_tensor(np.array(discount), device=self.device)

        obs = {
            k: v.astype(np.float32) if (
                isinstance(v, np.ndarray) and v.dtype == np.float64) else v
            for k, v in obs.items()}
        obs = {
            k: torch.as_tensor(np.array(v), device=self.device)
            for k, v in obs.items()}

        next_obs = {
            k: v.astype(np.float32) if (
                isinstance(v, np.ndarray) and v.dtype == np.float64) else v
            for k, v in next_obs.items()}
        next_obs = {
            k: torch.as_tensor(np.array(v), device=self.device)
            for k, v in next_obs.items()}

        # action, reward, discount = utils.to_torch(
        #     (action, reward, discount), self.device)
        # keys = list(obs.keys())
        # obs = utils.to_torch([obs[k] for k in keys], self.device)
        # obs = {k: v for k, v in zip(keys, obs)}
        # next_obs = utils.to_torch([next_obs[k] for k in keys], self.device)
        # next_obs = {k: v for k, v in zip(keys, next_obs)}

        # # augment
        image = obs['image'].float()
        image = image.permute((0, 3, 1, 2))
        image = self.aug(image)
        image = image.permute((0, 2, 3, 1))
        obs['image'] = image

        # TODO!!!!!! whole point of this mf algo.
        image = next_obs['image'].float()
        image = image.permute((0, 3, 1, 2))
        image = self.aug(image)
        image = image.permute((0, 2, 3, 1))
        next_obs['image'] = image

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
