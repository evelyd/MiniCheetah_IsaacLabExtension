# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from mini_cheetah.rsl_rl.utils import split_and_pad_trajectories
from mini_cheetah.rsl_rl.storage.rollout_storage import RolloutStorage


class KoopmanRolloutStorage(RolloutStorage):
    class NextValueTransition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.next_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):

        super().__init__(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape,
            privileged_obs_shape=privileged_obs_shape,
            actions_shape=actions_shape,
            rnd_state_shape=rnd_state_shape,
            device=device,
        )

        # for reinforcement learning
        if training_type == "rl":
            self.next_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def add_transitions(self, transition: NextValueTransition):

        # Set current step since parent class steps it
        current_step = self.step

        # Call parent class function
        super().add_transitions(transition)

        # Set the next values
        if self.training_type == "rl":
            self.next_values[current_step].copy_(transition.next_values)

    # def _save_hidden_states(self, hidden_states):
    #     if hidden_states is None or hidden_states == (None, None):
    #         return
    #     # make a tuple out of GRU hidden state sto match the LSTM format
    #     hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
    #     hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
    #     # initialize if needed
    #     if self.saved_hidden_states_a is None:
    #         self.saved_hidden_states_a = [
    #             torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
    #         ]
    #         self.saved_hidden_states_c = [
    #             torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
    #         ]
    #     # copy the states
    #     for i in range(len(hid_a)):
    #         self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
    #         self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    # def clear(self):
    #     self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        # TODO vectorize these operations if i can
        all_next_values = []
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value with the last value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else: # otherwise, use the value of the next state
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            # use Koopman value for the next value
            delta = self.rewards[step] + next_is_not_terminal * gamma * self.next_values[step] - self.values[step]
            # delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

            # Store the next values for debugging purposes in the correct order
            all_next_values.insert(0, next_values.clone())

        all_next_values = torch.stack(all_next_values, dim=0)

        # input(f"next values in compute returns: {self.next_values[0]}") # this is the Koopman value
        # input(f"next values a priori: {self.values[1]}")
        # Compute the average difference between self.next_values and self.values[step+1]
        # This is for debugging purposes to check if the Koopman value is close to the actual next value
        avg_diff = torch.mean(self.next_values - all_next_values)
        print(f"Average difference between next_values and values[step+1]: {avg_diff}")

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # # for reinforcement learning with feedforward networks
    # def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     if self.training_type != "rl":
    #         raise ValueError("This function is only available for reinforcement learning training.")
    #     batch_size = self.num_envs * self.num_transitions_per_env
    #     mini_batch_size = batch_size // num_mini_batches
    #     indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    #     # Core
    #     observations = self.observations.flatten(0, 1)
    #     if self.privileged_observations is not None:
    #         privileged_observations = self.privileged_observations.flatten(0, 1)
    #     else:
    #         privileged_observations = observations

    #     actions = self.actions.flatten(0, 1)
    #     values = self.values.flatten(0, 1)
    #     returns = self.returns.flatten(0, 1)

    #     # For PPO
    #     old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    #     advantages = self.advantages.flatten(0, 1)
    #     old_mu = self.mu.flatten(0, 1)
    #     old_sigma = self.sigma.flatten(0, 1)

    #     # For RND
    #     if self.rnd_state_shape is not None:
    #         rnd_state = self.rnd_state.flatten(0, 1)

    #     for epoch in range(num_epochs):
    #         for i in range(num_mini_batches):
    #             # Select the indices for the mini-batch
    #             start = i * mini_batch_size
    #             end = (i + 1) * mini_batch_size
    #             batch_idx = indices[start:end]

    #             # Create the mini-batch
    #             # -- Core
    #             obs_batch = observations[batch_idx]
    #             privileged_observations_batch = privileged_observations[batch_idx]
    #             actions_batch = actions[batch_idx]

    #             # -- For PPO
    #             target_values_batch = values[batch_idx]
    #             returns_batch = returns[batch_idx]
    #             old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
    #             advantages_batch = advantages[batch_idx]
    #             old_mu_batch = old_mu[batch_idx]
    #             old_sigma_batch = old_sigma[batch_idx]

    #             # -- For RND
    #             if self.rnd_state_shape is not None:
    #                 rnd_state_batch = rnd_state[batch_idx]
    #             else:
    #                 rnd_state_batch = None

    #             # yield the mini-batch
    #             yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
    #                 None,
    #                 None,
    #             ), None, rnd_state_batch

    # # for reinfrocement learning with recurrent networks
    # def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     if self.training_type != "rl":
    #         raise ValueError("This function is only available for reinforcement learning training.")
    #     padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    #     if self.privileged_observations is not None:
    #         padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
    #     else:
    #         padded_privileged_obs_trajectories = padded_obs_trajectories

    #     if self.rnd_state_shape is not None:
    #         padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
    #     else:
    #         padded_rnd_state_trajectories = None

    #     mini_batch_size = self.num_envs // num_mini_batches
    #     for ep in range(num_epochs):
    #         first_traj = 0
    #         for i in range(num_mini_batches):
    #             start = i * mini_batch_size
    #             stop = (i + 1) * mini_batch_size

    #             dones = self.dones.squeeze(-1)
    #             last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    #             last_was_done[1:] = dones[:-1]
    #             last_was_done[0] = True
    #             trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
    #             last_traj = first_traj + trajectories_batch_size

    #             masks_batch = trajectory_masks[:, first_traj:last_traj]
    #             obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
    #             privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

    #             if padded_rnd_state_trajectories is not None:
    #                 rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
    #             else:
    #                 rnd_state_batch = None

    #             actions_batch = self.actions[:, start:stop]
    #             old_mu_batch = self.mu[:, start:stop]
    #             old_sigma_batch = self.sigma[:, start:stop]
    #             returns_batch = self.returns[:, start:stop]
    #             advantages_batch = self.advantages[:, start:stop]
    #             values_batch = self.values[:, start:stop]
    #             next_values_batch = self.next_values[:, start:stop]
    #             old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

    #             # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
    #             # then take only time steps after dones (flattens num envs and time dimensions),
    #             # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
    #             last_was_done = last_was_done.permute(1, 0)
    #             hid_a_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_a
    #             ]
    #             hid_c_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_c
    #             ]
    #             # remove the tuple for GRU
    #             hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
    #             hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

    #             yield obs_batch, privileged_obs_batch, actions_batch, values_batch, next_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
    #                 hid_a_batch,
    #                 hid_c_batch,
    #             ), masks_batch, rnd_state_batch

    #             first_traj = last_traj
