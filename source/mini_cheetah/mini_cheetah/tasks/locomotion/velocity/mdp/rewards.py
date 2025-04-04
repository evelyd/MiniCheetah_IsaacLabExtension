from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import mini_cheetah.tasks.locomotion.velocity.mdp as mdp
import mini_cheetah.tasks.locomotion.velocity.utils as utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def latent_penalty_l2(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """Penalizes the agent for deviations from the desired state and for taking large actions.

    This function penalizes the agent for deviating from the desired latent state and for taking large actions. It is formulated as the negative of the sum of the quadratic deviations from the desired state.

    Weighting matrices are introduced for each component of this reward.
    """

    # Create a mapping from current joint order to morphosymm order
    joint_order_indices = [env.usd_joint_order.index(joint) for joint in env.joint_order_for_morphosymm]

    # Check that the size of env.obs_buf['policy'] and env.ref_trajectories is the same
    x, u = utils.get_state_action_from_obs(env.obs_buf['policy'], joint_order_indices)

    # Get latent state
    symmetric_x = env.shared_model.state_type(x)
    s = env.shared_model.obs_fn(symmetric_x).tensor

    # Extract the reference observations at the current timestep for each environment
    script_name = os.path.basename(os.path.abspath(sys.argv[0]))
    if "play" not in script_name:
        ref_obs = env.ref_trajectories
        print("Max timestep value: ", torch.max(env.current_timesteps), " number of envs with that value: ", torch.sum(env.current_timesteps == torch.max(env.current_timesteps)))
        ref_obs_current = torch.vstack([ref_obs[env.current_timesteps[i], i, :].squeeze() for i in range(env.current_timesteps.shape[0])])

        # Get the reference latent state
        x_r, u_r = utils.get_state_action_from_obs(ref_obs_current, joint_order_indices)
        symmetric_x_r = env.shared_model.state_type(x_r)
        s_r = env.shared_model.obs_fn(symmetric_x_r).tensor

        # Get the differences
        s_diff = s - s_r
        u_diff = u - u_r

        # Define Q and R weight matrices
        state_dim = s.shape[1]
        action_dim = u.shape[1]
        Q = 0.1 * torch.eye(state_dim).to(env.device)
        R = torch.eye(action_dim).to(env.device)

        # Compute the reward (positive, the weight will be negative)
        state_rew_component = torch.einsum('bi,ij,bj->b', s_diff, Q, s_diff)
        action_rew_component = torch.einsum('bi,ij,bj->b', u_diff, R, u_diff)
        reward = state_rew_component + action_rew_component

        # Step the current timesteps
        env.update_current_timesteps(env.reset_buf)

    else:
        reward = 0

    return reward