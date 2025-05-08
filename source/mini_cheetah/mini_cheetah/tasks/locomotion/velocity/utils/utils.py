import torch

import os
import glob
import dha
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.DynamicsAutoEncoder import DAE
from dha.utils.mysc import class_from_name
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens

import escnn
from escnn.nn import FieldType

import re

def get_state_action_from_obs(obs, joint_order_indices, q0):
    """
    Takes an observation from the observation class and extracts the system state and action vectors.

    Puts the state in the correct form for use in the DAE model.

    State vector is composed as: $x = [q, \dot q, z, v, o, \omega] \in \mathbb R^{46}$
    """

    # Define the default joint positions in Isaaclab
    q0_isaaclab = torch.tensor([0.10000000149011612, -0.10000000149011612, 0.10000000149011612, -0.10000000149011612, -0.800000011920929, -0.800000011920929, -0.800000011920929, -0.800000011920929, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716, 1.6200000047683716], device=obs.device, dtype=obs.dtype) #TODO this is hardcoded, if the defaults change then I have to change this too

    base_vel = obs[:, :3]
    velocity_commands_xy = obs[:, 9:11] # Rep: Rd for xy, euler xyz for heading? idk
    ref_base_lin_vel = torch.hstack([velocity_commands_xy, torch.zeros((velocity_commands_xy.shape[0], 1), device=obs.device)]) # set ref lin z vel to 0
    base_vel_error = base_vel - ref_base_lin_vel

    base_ang_vel = obs[:, 3:6]
    velocity_commands_z = obs[:, 11].unsqueeze(-1) # Rep: Rd for xy, euler xyz for heading? idk
    ref_base_ang_vel = torch.hstack([torch.zeros((base_ang_vel.shape[0], 2), device=obs.device), velocity_commands_z])
    base_ang_vel_error = base_ang_vel - ref_base_ang_vel

    # Get the joint positions and velocities
    joint_pos_rel = obs[:, 12:24]
    joint_pos = joint_pos_rel + q0_isaaclab  # Compute the absolute joint positions
    # Reorder joint positions
    joint_pos_reordered = joint_pos[:, joint_order_indices]
    # Add offset to the measurements (necessary for symmetry group)
    joint_pos_reordered += q0[7:]
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format
    cos_q_js, sin_q_js = torch.cos(joint_pos_reordered), torch.sin(joint_pos_reordered)
    q_js_unit_circle_t = torch.stack([cos_q_js, sin_q_js], axis=2)
    joint_pos_parametrized = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)
    joint_vel = obs[:, 24:36]
    # Reorder joint velocities
    joint_vel = joint_vel[:, joint_order_indices]

    # Get the base pose info
    ref_base_z = 0.5 #TODO this value is hardcoded for now to avoid needing to collect the data yet again
    base_z = obs[:, 48]
    base_z_error = base_z - ref_base_z
    base_quat = obs[:, 49:53]
    #Convert base quat to euler angles
    base_ori = quat_to_euler_torch(base_quat)

    projected_gravity = obs[:, 6:9]
    ref_projected_gravity = torch.tensor([0, 0, -1.0], device=obs.device, dtype=obs.dtype)
    projected_gravity_error = projected_gravity - ref_projected_gravity

    action_joint_pos = obs[:, 36:48]
    # Reorder action joint positions to match the morphosymm order
    action_joint_pos = action_joint_pos[:, joint_order_indices] # the action joint positions are already absolute
    action_joint_pos = action_joint_pos + q0[7:]  # Add offset to the measurements
    cos_a_js, sin_a_js = torch.cos(action_joint_pos), torch.sin(action_joint_pos)  # convert from angle to unit circle parametrization
    # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
    a_js_unit_circle_t = torch.stack([cos_a_js, sin_a_js], axis=2)
    a_joint_pos_parametrized = a_js_unit_circle_t.reshape(a_js_unit_circle_t.shape[0], -1)

    # Concatenate all these states into a single state vector
    base_z_error = base_z_error.unsqueeze(-1)  # Add a dimension to match concatenation requirements
    velocity_commands_z = velocity_commands_z.unsqueeze(-1)  # Add a dimension to match concatenation requirements
    x = torch.cat([joint_pos_parametrized, joint_vel, base_z_error, base_vel_error, base_ori, base_ang_vel_error, projected_gravity_error, a_joint_pos_parametrized], dim=1).to(dtype=obs.dtype)

    return x

def quat_to_euler_torch(quaternions):
    """
    Converts quaternions to Euler angles (XYZ convention) using PyTorch.

    Args:
        quaternions (torch.Tensor): Quaternions tensor of shape (..., 4).

    Returns:
        torch.Tensor: Euler angles tensor of shape (..., 3).
    """

    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def extract_trained_model_info(state_dict) -> (int, int, bool, int):
    """Extracts model information from a state_dict."""
    layers = 0
    hidden_units = 0
    obs_state_dim = 0
    has_bias = False

    for key in state_dict.keys():
        if ".obs_fn.net" in key:
            if "model.obs_fn.net.block_" in key and "weight" in key:
                layers += 1
            if "linear_0" in key and "bias" in key:
                hidden_units = state_dict[key].shape[0]
            if 'bias' in key and not has_bias:
                has_bias = True
            if "head" in key and ".bias" in key:
                obs_state_dim = state_dict[key].shape[0]

    layers += 1  # Add one for the head layer

    return layers, hidden_units, has_bias, obs_state_dim

def remove_state_dict_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def get_trained_dae_model(model_dir):
    """
    Load the trained DAE model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        torch.nn.Module: The trained model.
    """
    ckpt_path = os.path.join(model_dir, "best.ckpt")

    # Load the model from the checkpoint
    checkpoint = torch.load(ckpt_path)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Define the state representation
    # G is the symmetry group of the system
    robot, G = load_symmetric_system(robot_name="mini_cheetah")

    # Create the state representations
    #TODO this needs to be edited if actions are added
    gspace = escnn.gspaces.no_base_space(G)
    # Extract the representations from G.representations.items()
    rep_Q_js = G.representations['Q_js']
    rep_Rd = G.representations['R3']
    rep_TqQ_js = G.representations['TqQ_js']
    rep_z = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_z.name = "base_z"
    rep_euler_xyz = G.representations['euler_xyz']

    # Define the state type using the extracted representations
    state_reps = [rep_Q_js, rep_TqQ_js, rep_z, rep_Rd, rep_euler_xyz, rep_euler_xyz, rep_Rd, rep_Q_js]
    state_type = FieldType(gspace, representations=state_reps)
    state_type.size = sum(rep.size for rep in state_reps) + rep_euler_xyz.size + rep_Rd.size + rep_Q_js.size  # Count duplicates twice
    state_type = FieldType(gspace, representations=state_reps)

    dt = 0.02
    orth_w_match = re.search(r"Orth_w:([\d\.]+)", model_dir)
    orth_w = float(orth_w_match.group(1)) if orth_w_match else 0.0
    obs_pred_w_match = re.search(r"Obs_w:([\d\.]+)", model_dir)
    obs_pred_w = float(obs_pred_w_match.group(1)) if obs_pred_w_match else 1.0
    group_avg_trick = True
    state_dependent_obs_dyn = False
    enforce_constant_fn = True
    act_match = re.search(r"Act:([\d\.]+)", model_dir)
    activation = obs_pred_w_match.group(1) if act_match else 'ELU'
    batch_norm = False

    if not "E-DAE" in model_dir:
        activation = class_from_name("torch.nn", activation)

    num_layers, num_hidden_units, bias, obs_state_dim = extract_trained_model_info(state_dict)
    obs_fn_params = {'num_layers': num_layers, 'num_hidden_units': num_hidden_units, 'activation': activation, 'bias': bias, 'batch_norm': batch_norm}

    initial_rng_state = torch.get_rng_state()

    if "E-DAE" in model_dir:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
            # reuse_input_observable=cfg.model.reuse_input_observable,
        )
    else:
        corr_w = 0.0
        model = DAE(
            state_dim=state_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            corr_w=corr_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
            # reuse_input_observable=cfg.model.reuse_input_observable,
        )

    torch.set_rng_state(initial_rng_state)
    model.load_state_dict(remove_state_dict_prefix(state_dict, "model."))

    return model

def main():
    # model_dir = "experiments/test/S=forward_minus_0_4-OS=5-G=K4xC2-H=30-EH=30_E-DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/seed=399/"
    model_dir = "experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:30-EH:30_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=776/"

    dha_dir = os.path.dirname(dha.__file__)
    model_dir = os.path.join(dha_dir, model_dir)
    try:
        model = get_trained_dae_model(model_dir)
        print("Model loaded successfully!")
        print(model)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()