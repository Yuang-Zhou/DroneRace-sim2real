# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        ratio_clip=0.2,
        value_clip=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.005,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        clip_predicted_values=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cuda:0",
        normalize_advantage_per_mini_batch=False,
        reward_scale=1.0,
        state_normalizer=None,
        value_normalizer=None,
        action_std_schedule: str = "decay",
        action_std_decay_rate: float = 0.999,
        action_std_min = 0.05,
    ):
        self.device = device
        self.clip_param = ratio_clip
        self.value_clip = ratio_clip if value_clip is None else value_clip
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_predicted_values = clip_predicted_values
        self.reward_scale = reward_scale
        self.state_normalizer = state_normalizer
        self.value_normalizer = value_normalizer
        # Action std scheduling
        self.action_std_schedule = action_std_schedule.lower() if action_std_schedule else "fixed"
        self.action_std_decay_rate = action_std_decay_rate
        self.action_std_min = action_std_min
        self._update_calls = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.state_normalizer is not None:
            obs = self.state_normalizer.normalize(obs)
            critic_obs = self.state_normalizer.normalize(critic_obs)

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self._detach_hidden_states(self.actor_critic.get_hidden_states())

        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        reward = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        time_outs = None
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            reward = reward + self.gamma * torch.squeeze(self.transition.values * time_outs, dim=1)
        # Record the transition
        self.transition.rewards = reward * self.reward_scale
        self.storage.add_transitions(self.transition)
        if self.actor_critic.is_recurrent:
            self._reset_recurrent_state(dones, time_outs)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        if self.state_normalizer is not None:
            last_critic_obs = self.state_normalizer.normalize(last_critic_obs)

        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
                observations,
                critic_observations,
                sampled_actions,
                value_targets,
                advantage_estimates,
                discounted_returns,
                prev_log_probs,
                prev_mean_actions,
                prev_action_stds,
                hidden_states,
                episode_masks,
                _,  # rnd_state_batch - not used anymore
        ) in generator:
            actor_hidden_states, critic_hidden_states = hidden_states
            masks = episode_masks

            # Forward policy network to build the current action distribution and critic values.
            self.actor_critic.act(observations, masks=masks, hidden_states=actor_hidden_states)
            predicted_values = self.actor_critic.evaluate(
                critic_observations, masks=masks, hidden_states=critic_hidden_states
            )

            if self.value_normalizer is not None:
                predicted_values = self.value_normalizer.normalize(predicted_values)

            log_probs = self.actor_critic.get_actions_log_prob(sampled_actions).reshape(-1)
            prev_log_probs_flat = prev_log_probs.reshape(-1)
            advantages = advantage_estimates.reshape(-1)

            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            ratio = torch.exp(log_probs - prev_log_probs_flat)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
            surrogate_loss = surrogate.mean()

            returns = discounted_returns.reshape(-1, 1)
            predicted_values = predicted_values.reshape(-1, 1)
            old_values = value_targets.reshape(-1, 1)

            if self.use_clipped_value_loss and self.clip_predicted_values:
                value_pred_clipped = old_values + torch.clamp(
                    predicted_values - old_values, -self.value_clip, self.value_clip
                )
                value_losses = (predicted_values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns - predicted_values).pow(2).mean()

            entropy = self.actor_critic.entropy.mean()
            loss = -surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                action_dim = sampled_actions.shape[-1]
                new_mean = self.actor_critic.action_mean.reshape(-1, action_dim)
                new_std = self.actor_critic.action_std.reshape(-1, action_dim)
                old_mean = prev_mean_actions.reshape(-1, action_dim)
                old_std = prev_action_stds.reshape(-1, action_dim)
                approx_kl = kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(-1).mean()
                if self.schedule == "adaptive":
                    if approx_kl > self.desired_kl * 2.0:
                        self.learning_rate = max(self.learning_rate / 1.5, 1.0e-6)
                    elif approx_kl < self.desired_kl * 0.5:
                        self.learning_rate = min(self.learning_rate * 1.5, 5.0e-3)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # Optionally anneal action std after each policy update
        self._update_calls += 1
        self._anneal_action_std()
        # Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy

    def _detach_hidden_states(self, hidden_states):
        if hidden_states is None:
            return None
        if isinstance(hidden_states, (list, tuple)):
            return tuple(self._detach_hidden_states(h) for h in hidden_states)
        return hidden_states.detach()

    def _reset_recurrent_state(self, dones, time_outs=None):
        done_mask = dones.squeeze(-1).bool()
        if time_outs is not None:
            done_mask = torch.logical_or(done_mask, time_outs.squeeze(-1).bool())
        if done_mask.any():
            self.actor_critic.reset(done_mask)

    def _anneal_action_std(self):
        """Multiplicative decay of action noise std; no-op when schedule is fixed."""
        if self.action_std_schedule == "fixed":
            return
        if self.action_std_decay_rate >= 1.0:
            return
        with torch.no_grad():
            # Apply decay depending on noise parameterization
            if self.actor_critic.noise_std_type == "scalar":
                new_std = self.actor_critic.std * self.action_std_decay_rate
                if self.action_std_min is not None:
                    new_std = torch.clamp(new_std, min=self.action_std_min)
                self.actor_critic.std.copy_(new_std)
            elif self.actor_critic.noise_std_type == "log":
                new_log_std = self.actor_critic.log_std + torch.log(
                    torch.tensor(self.action_std_decay_rate, device=self.actor_critic.log_std.device)
                )
                if self.action_std_min is not None:
                    min_log_std = torch.log(torch.tensor(self.action_std_min, device=self.actor_critic.log_std.device))
                    new_log_std = torch.max(new_log_std, min_log_std)
                self.actor_critic.log_std.copy_(new_log_std)
