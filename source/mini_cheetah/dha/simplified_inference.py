import os
import glob
import torch
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens

import escnn
from escnn.nn import FieldType

def extract_model_info(state_dict) -> (int, int, bool, int):
    """Extracts model information from a state_dict."""
    layers = 0
    hidden_units = 0
    obs_state_dim = 0
    has_bias = False

    for key in state_dict.keys():
        if ".obs_fn.net" in key:
            if "model.obs_fn.net.block_" in key and "expanded_bias" in key:
                layers += 1
            if "linear_0" in key and "bias" in key:
                hidden_units = state_dict[key].shape[0]
            if 'bias' in key and not has_bias:
                has_bias = True
            if "head" in key and "bias_expansion" in key:
                obs_state_dim = state_dict[key].shape[0]

    layers += 1  # Add one for the head layer

    return layers, hidden_units, has_bias, obs_state_dim

def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = "experiments/test/S:forward_minus_0_4-OS:5-G:K4xC2-H:30-EH:30_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/"

model_dir = os.path.join(script_dir, model_dir)

seed_dirs = [d for d in glob.glob(os.path.join(model_dir, "seed=*")) if os.path.isdir(d)]
if not seed_dirs:
    print("Directories within model_dir:", os.listdir(model_dir))
    raise FileNotFoundError(f"No directory matching 'seed=.*' found in {model_dir}")
seed_dir = seed_dirs[0]
ckpt_path = os.path.join(seed_dir, "best.ckpt")

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
state_reps = [rep_Q_js, rep_TqQ_js, rep_z, rep_Rd, rep_euler_xyz, rep_euler_xyz]
state_type = FieldType(gspace, representations=state_reps)
state_type.size = sum(rep.size for rep in state_reps) + rep_euler_xyz.size  # Count rep_euler_xyz twice
state_type = FieldType(gspace, representations=state_reps)
g = G.sample()

dt = 0.003
orth_w = 0.0
group_avg_trick = True
state_dependent_obs_dyn = False
enforce_constant_fn = True
activation = 'ELU'
batch_norm = False

num_layers, num_hidden_units, bias, obs_state_dim = extract_model_info(state_dict)
obs_fn_params = {'num_layers': num_layers, 'num_hidden_units': num_hidden_units, 'activation': activation, 'bias': bias, 'batch_norm': batch_norm}

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

model.load_state_dict(remove_prefix(state_dict, "model."))

# Handle Device Placement (GPU if available)
model.to(device)

# Set the model to evaluation mode
model.eval()

obs_fn = model.obs_fn

with torch.no_grad():
    # Example inference
    state = model.state_type(torch.randn(1, 46)).to(device)  # shape[0] is batch size, shape[1] is the size of the state vector
    print("State:", state)

    # Get the latent state by calling obs_fn
    latent_state = obs_fn(state)
    print("Latent state:", latent_state)

    # Perform decoding to get the output state
    inv_obs_fn = model.inv_obs_fn
    output_state = inv_obs_fn(latent_state)
    print("Output state:", output_state)