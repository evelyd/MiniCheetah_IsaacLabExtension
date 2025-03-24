from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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

def latent_quadratic_penalty(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """Penalizes the agent for deviations from the desired state and for taking large actions.

    This function penalizes the agent for deviating from the desired latent state and for taking large actions. It is formulated as the negative of the sum of the quadratic deviations from the desired state.

    Weighting matrices are introduced for each component of this reward.
    """
    # Compose the state vector as: x = [q, \dot q, z, v, o, omega] \in \mathbb R^{46}
    # the values are taken directly from the PolicyCfg
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)).func(env)
    #TODO check the joint order bc it might be different from that of the DHA code
    cos_q_js, sin_q_js = torch.cos(joint_pos), torch.sin(joint_pos)
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    q_js_unit_circle_t = torch.stack([cos_q_js, sin_q_js], axis=2)
    joint_pos_parametrized = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)).func(env)
    base_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01)).func(env) # noise set to same as joint pos
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)).func(env)
    base_quat = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.01, n_max=0.01)).func(env) # noise set to same as joint pos
    #Convert base quat to euler angles
    base_euler_angles = utils.quat_to_euler_torch(base_quat)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)).func(env)

    # Concatenate all these states into a single state vector
    x = torch.cat([joint_pos_parametrized, joint_vel, base_z, base_lin_vel, base_euler_angles, base_ang_vel], dim=1)

    # Get latent state
    print(env.model)

    # Define the action vector
    u = ObsTerm(func=mdp.last_action).func(env)
    # Define Q and R weight matrices
    state_dim = x.shape[1]
    action_dim = u.shape[1]
    print(f"state_dim: {state_dim}, action_dim: {action_dim}")

    Q = torch.eye(state_dim).to(env.device)
    R = torch.eye(action_dim).to(env.device)

    print(f"Q dim: {Q.shape}, R dim: {R.shape}")
    print(f"x dim: {x.shape}, u dim: {u.shape}")
    # Compute the reward (positive, the weight will be negative)
    reward = torch.sum(torch.einsum('bi,ij,bj->b', x, Q, x) + torch.einsum('bi,ij,bj->b', u, R, u))
    print(reward)
    rew_test = torch.matmul(torch.matmul(x.T, Q), x) + torch.matmul(torch.matmul(u.T, R), u)
    print(rew_test)
    assert torch.allclose(reward, rew_test), "Reward and rew_test are not equal"
    input("reward computed correctly")

    return reward