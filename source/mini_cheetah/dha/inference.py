import os
import glob
import torch
from dha.nn.LightningLatentMarkovDynamics import LightLatentMarkovDynamics
from dha.data.DynamicsDataModule import DynamicsDataModule

from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from typing import Any
from collections import defaultdict
from omegaconf.nodes import AnyNode
from omegaconf.listconfig import ListConfig
from dha.train_observables import get_model

import hydra
from hydra.utils import get_original_cwd
from pathlib import Path

torch.serialization.add_safe_globals([DictConfig, Metadata, Any, defaultdict, ListConfig])

@hydra.main(config_path='cfg', config_name='config', version_base='1.1')
def main(cfg: DictConfig):

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

    # Access and print the transfer_op.matrix from the state_dict
    if 'model.obs_space_dynamics.transfer_op.matrix' in state_dict:
        transfer_op_matrix = state_dict['model.obs_space_dynamics.transfer_op.matrix']
        print("Transfer Operator Matrix:", transfer_op_matrix.shape)
    else:
        print("Key 'model.obs_space_dynamics.transfer_op.matrix' not found in state_dict.")

    # Create the data module so that the model can be initialized
    # Load the dynamics dataset.
    root_path = Path(get_original_cwd()).resolve()
    data_path = root_path / "data" / cfg.system.data_path
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    datamodule = DynamicsDataModule(data_path,
                                        batch_size=cfg.model.batch_size,
                                        frames_per_step=cfg.system.frames_per_state,
                                        pred_horizon=cfg.system.pred_horizon,
                                        eval_pred_horizon=cfg.system.eval_pred_horizon,
                                        test_pred_horizon=cfg.system.test_pred_horizon,
                                        system_cfg=cfg.system,
                                        num_workers=cfg.num_workers,
                                        device=device,
                                        train_ratio=cfg.system.train_ratio,
                                        augment=cfg.model.augment,
                                        state_obs=cfg.system.get('state_obs', None),
                                        action_obs=cfg.system.get('action_obs', None),
                                        standardize=cfg.system.standardize)
    datamodule.prepare_data()

    model = get_model(cfg, datamodule)
    #TODO datamodule may not be needed, but need to recreate the get_model function then

    # Initialize the overarching markov model
    # Load lightning module handling the operations of all model variants
    epoch_metrics_fn = model.evaluate_observation_space if hasattr(model, "evaluate_observation_space") else None
    pl_model = LightLatentMarkovDynamics(lr=cfg.model.lr,
                                        batch_size=cfg.model.batch_size,
                                        run_hps=cfg.model,
                                        test_epoch_metrics_fn=epoch_metrics_fn,
                                        val_epoch_metrics_fn=epoch_metrics_fn,
                                        log_figs_every_n_epochs=10)

    pl_model.set_model(model)

    # Load the state_dict into the model
    pl_model.eval()
    pl_model.model.eval()
    pl_model.load_state_dict(state_dict, strict=False)

    # Access the obs_fn (encoder) from the model
    obs_fn = pl_model.model.obs_fn

    # TODO figure out the form of the state input to the obs_fn
    print("state type: ", datamodule.state_type)
    print("gspace: ", datamodule.gspace)
    print("symm group: ", datamodule.symm_group)
    state = model.state_type(torch.randn(1, 46)) # shape[0] is batch size, shape[1] is the size of the state vector

    # Get the latent state by calling obs_fn
    latent_state = obs_fn(state)

    print("State:", state)
    print("Latent state:", latent_state)

    # Perform decoding to get the output state
    inv_obs_fn = pl_model.model.inv_obs_fn
    output_state = inv_obs_fn(latent_state)
    print("Output state:", output_state)


if __name__ == '__main__':
    main()
    # return r