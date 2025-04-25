import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

from torch.distributions import Normal

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils

from pybullet_utils.bullet_client import BulletClient
import pybullet
from morpho_symm.utils.robot_utils import load_symmetric_system
import numpy as np

class LatentStateActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, joint_order_indices, dae_dir, actor_hidden_dims, critic_hidden_dims, activation, **kwargs):

        self.joint_order_indices = joint_order_indices
        self.device = kwargs.get("device", "cpu")
        self.model_path = dae_dir
        dae_model = self._load_model()

        # Use the latent state dimensions to initialize the ActorCritic
        latent_dim = dae_model.obs_state_dim
        super().__init__(num_actor_obs=num_actor_obs + latent_dim, num_critic_obs=num_critic_obs + latent_dim, num_actions=num_actions, actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims, activation=activation, **kwargs)

        # Assign the DAE model to the class
        self.dae_model = dae_model

        # Create a variable to hold the q0 for the joint offset that is needed by the symmetry groups
        robot, G = load_symmetric_system(robot_name="mini_cheetah")
        bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
        robot.configure_bullet_simulation(bullet_client=bullet_client)
        # Get zero reference position.
        q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
        self.q0 = torch.tensor(q0).to(self.device)

        # Get the normalization info for the DAE model
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        norm_dir = os.path.join(model_dir, "state_mean_var.npy")
        # Load state_mean and state_var from the npy file
        norm_data = np.load(norm_dir, allow_pickle=True).item()

        # Extract state_mean and state_var values
        state_mean_values = norm_data["state_mean"]
        state_var_values = norm_data["state_var"]

        # Convert to torch tensors
        self.state_mean = torch.tensor(state_mean_values, device=self.device).float()
        self.state_std = torch.sqrt(torch.tensor(state_var_values, device=self.device)).float()

    def update_distribution(self, observations):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices, self.q0)
            x_normed = (x - self.state_mean) / self.state_std
            if "E-DAE" in self.model_path:
                # E-DAE model
                symmetric_x = self.dae_model.state_type(x_normed)
                s = self.dae_model.obs_fn(symmetric_x).tensor
            else:
                # DAE model
                s = self.dae_model.obs_fn(x_normed)
        # Concatenate the latent state with the observations
        #TODO this won't work with edae if I want to use the emlp
        z = torch.cat((observations, s), dim=-1)
        mean = self.actor(z)
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
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices, self.q0)
            x_normed = (x - self.state_mean) / self.state_std
            if "E-DAE" in self.model_path:
                # E-DAE model
                symmetric_x = self.dae_model.state_type(x_normed)
                s = self.dae_model.obs_fn(symmetric_x).tensor
            else:
                # DAE model
                s = self.dae_model.obs_fn(x_normed)
        # Concatenate the latent state with the observations
        z = torch.cat((observations, s), dim=-1)
        actions_mean = self.actor(z)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(critic_observations, self.joint_order_indices, self.q0)
            x_normed = (x - self.state_mean) / self.state_std
            if "E-DAE" in self.model_path:
                # E-DAE model
                symmetric_x = self.dae_model.state_type(x_normed)
                s = self.dae_model.obs_fn(symmetric_x).tensor
            else:
                # DAE model
                s = self.dae_model.obs_fn(x_normed)
        # Concatenate the latent state with the observations
        z = torch.cat((critic_observations, s), dim=-1)
        value = self.critic(z)
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
