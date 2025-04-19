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
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, joint_order_indices, edae_dir, actor_hidden_dims, critic_hidden_dims, activation, **kwargs):

        self.joint_order_indices = joint_order_indices
        self.device = kwargs.get("device", "cpu")
        self.model_path = edae_dir
        edae_model = self._load_model()

        # Use the latent state dimensions to initialize the ActorCritic
        latent_dim = edae_model.obs_state_dim
        super().__init__(num_actor_obs=latent_dim, num_critic_obs=latent_dim, num_actions=num_actions, actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims, activation=activation, **kwargs)

        # Assign the eDAE model to the class
        self.edae_model = edae_model

        # Create a variable to hold the q0 for the joint offset that is needed by the symmetry groups
        robot, G = load_symmetric_system(robot_name="mini_cheetah")
        bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
        robot.configure_bullet_simulation(bullet_client=bullet_client)
        # Get zero reference position.
        q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))
        self.q0 = torch.tensor(q0).to(self.device)

    def update_distribution(self, observations):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices, self.q0)
            symmetric_x = self.edae_model.state_type(x)
            s = self.edae_model.obs_fn(symmetric_x).tensor
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
            x, _ = utils.get_state_action_from_obs(observations, self.joint_order_indices, self.q0)
            symmetric_x = self.edae_model.state_type(x)
            s = self.edae_model.obs_fn(symmetric_x).tensor
        actions_mean = self.actor(s)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # compute mean with latent state as input
        with torch.no_grad():  # Ensure obs_fn doesn't track gradients
            x, _ = utils.get_state_action_from_obs(critic_observations, self.joint_order_indices, self.q0)
            symmetric_x = self.edae_model.state_type(x)
            s = self.edae_model.obs_fn(symmetric_x).tensor
        value = self.critic(s)
        return value

    def _load_model(self):
        # Set the eDAE model
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        with torch.device('cpu'):
            edae_model = utils.get_trained_eDAE_model(model_dir)
        edae_model.to(self.device)
        edae_model.eval()
        return edae_model
