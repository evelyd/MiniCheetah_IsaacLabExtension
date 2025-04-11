import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

from torch.distributions import Normal

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils


class LatentStateActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, latent_dim, joint_order_indices, edae_dir, actor_hidden_dims, critic_hidden_dims, activation, **kwargs):

        # Use the latent state dimensions to initialize the ActorCritic
        super().__init__(num_actor_obs=latent_dim, num_critic_obs=latent_dim, num_actions=num_actions, actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims, activation=activation, **kwargs)

        self.joint_order_indices = joint_order_indices
        self.device = kwargs.get("device", "cpu")
        self.model_path = edae_dir
        self._load_model()

    def update_distribution(self, observations):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices)
            symmetric_x = self.shared_model.state_type(x)
            s = self.shared_model.obs_fn(symmetric_x).tensor
        mean = self.actor(s)
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
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices)
            symmetric_x = self.shared_model.state_type(x)
            s = self.shared_model.obs_fn(symmetric_x).tensor
        actions_mean = self.actor(s)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(critic_observations, self.joint_order_indices)
            symmetric_x = self.shared_model.state_type(x)
            s = self.shared_model.obs_fn(symmetric_x).tensor
        value = self.critic(s)
        return value

    def _load_model(self):
        # Set the eDAE model
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        with torch.device('cpu'):
            self.shared_model = utils.get_trained_eDAE_model(model_dir)
        self.shared_model.to(self.device)
        self.shared_model.eval()
