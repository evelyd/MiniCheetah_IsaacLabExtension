import torch
import torch.nn as nn
from mini_cheetah.rsl_rl.modules import ActorCritic

from torch.distributions import Normal

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils

from morpho_symm.utils.robot_utils import load_symmetric_system
import numpy as np

class LatentStateActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, dae_dir, actor_hidden_dims, critic_hidden_dims, activation, **kwargs):

        self.joint_order_indices = utils.get_joint_order_indices()
        self.device = kwargs.get("device", "cpu")
        self.model_path = dae_dir
        dae_model = self._load_model()

        # Use the latent state dimensions to initialize the ActorCritic
        latent_dim = dae_model.obs_state_dim
        num_commands = 3
        super().__init__(num_actor_obs=num_actor_obs, num_critic_obs=latent_dim, num_actions=num_actions, actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims, activation=activation, **kwargs)

        # Assign the DAE model to the class
        self.dae_model = dae_model

        # Create a variable to hold the q0 for the joint offset that is needed by the symmetry groups
        self.q0 = utils.get_pybullet_q0(self.device)

        # Get the normalization info for the DAE model
        self.data_state_mean, self.data_state_std, self.data_action_mean, self.data_action_std = utils.get_stats(self.model_path, self.device)

    def update_distribution(self, observations):
        # compute mean with latent state as input
        # s = self.get_latent_state(self, observations)
        # # last_actions = observations[:, 36:48]
        # # commands = observations[:, 9:12]
        # # z = torch.cat((last_actions, commands, s), dim=-1)
        # z = torch.cat((observations, s), dim=-1)
        # mean = self.actor(z)
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        # compute mean with latent state as input
        # s = self.get_latent_state(self, observations)
        # # last_actions = observations[:, 36:48]
        # # commands = observations[:, 9:12]
        # # z = torch.cat((last_actions, commands, s), dim=-1)
        # z = torch.cat((observations, s), dim=-1)
        # actions_mean = self.actor(z)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, current_action, **kwargs):
        # compute mean with latent state as input
        # Get the normed latent state and action from the critic observations
        with torch.inference_mode():
            s, x_normed, u_normed = utils.get_latent_state(critic_observations, current_action, self.model_path, self.dae_model, self.joint_order_indices, self.q0, self.data_state_mean, self.data_state_std, self.data_action_mean, self.data_action_std)
            # Get the next latent state
            # Add a dimension in the 1 spot to u_normed
            # input(f"u before unsqueeze: {u_normed.size()}")
            u_normed = u_normed.unsqueeze(1) #.repeat(1, 5, 1)
            # input(f"u after unsqueeze: {u_normed.size()}")
            pred_x, pred_s = self.dae_model.forecast(x_normed, u_normed) #, n_steps=5)
        # assert torch.allclose(pred_s[:,0,:],s), f"Predicted latent state {pred_s[:,0,:]} does not match the input latent state {s}"
        # Concatenate the latent state with the observations
        # last_actions = critic_observations[:, 36:48]
        # commands = critic_observations[:, 9:12]
        # z = torch.cat((last_actions, commands, s), dim=-1)
        # z = torch.cat((critic_observations, pred_s[:, -1, :]), dim=-1)
        # z = torch.cat((commands, s), dim=-1)
        value = self.critic(s.clone())
        # print(f"current value 0 {value[0]}")
        return value

    def evaluate_next_value(self, critic_observations, current_action, next_critic_observations=None, **kwargs):
        # compute mean with latent state as input
        # Get the normed latent state and action from the critic observations
        with torch.inference_mode():
            s, x_normed, u_normed = utils.get_latent_state(critic_observations, current_action, self.model_path, self.dae_model, self.joint_order_indices, self.q0, self.data_state_mean, self.data_state_std, self.data_action_mean, self.data_action_std)
            # Get the next latent state
            # Add a dimension in the 1 spot to u_normed
            # input(f"u before unsqueeze: {u_normed.size()}")
            u_normed = u_normed.unsqueeze(1) #.repeat(1, 5, 1)
            # input(f"u after unsqueeze: {u_normed.size()}")
            pred_x, pred_s = self.dae_model.forecast(x_normed, u_normed) #, n_steps=5)
        assert torch.allclose(pred_s[:,0,:],s), f"Predicted latent state {pred_s[:,0,:]} does not match the input latent state {s}"

        if next_critic_observations is not None:
            with torch.inference_mode():
            #     # Get the normed latent state and action from the next critic observations
                next_s, next_x_normed, next_u_normed = utils.get_latent_state(next_critic_observations, current_action, self.model_path, self.dae_model, self.joint_order_indices, self.q0, self.data_state_mean, self.data_state_std, self.data_action_mean, self.data_action_std)
            #     # input(f"size of pred_s: {pred_s.size()}, next_s: {next_s.size()}")
                # Compute the RMSE between the predicted latent state and the actual next latent state
                rmse = torch.sqrt(torch.mean((next_s - pred_s[:,-1,:]) ** 2))
                print(f"RMSE between next_s and pred_s: {rmse}")
            #     # Concatenate the latent state with the observations
        # last_actions = critic_observations[:, 36:48]
        # commands = critic_observations[:, 9:12]
        # z = torch.cat((last_actions, commands, s), dim=-1)
        # z = torch.cat((critic_observations, pred_s[:,-1,:]), dim=-1)
        # z = torch.cat((commands, pred_s[:, 1, :]), dim=-1)
        value = self.critic(pred_s[:, 1, :])
        # print(f"next value 0 {value[0]}")
        return value

    def _load_model(self):
        # Set the dae model
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        with torch.device('cpu'):
            dae_model = utils.get_trained_dae_model(model_dir)
        dae_model.to(self.device)
        dae_model.eval()
        return dae_model