import gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
import mini_cheetah.tasks.locomotion.velocity.mini_cheetah_velocity_env_cfg as mini_cheetah_velocity_env_cfg
import mini_cheetah.tasks.locomotion.velocity.config.mini_cheetah.agents as agents #adjust import path.

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils

class MiniCheetahModelEnv(ManagerBasedRLEnv):
    model_loaded = False
    shared_model = None

    def __init__(self, env_cfg, rsl_rl_cfg):
        super().__init__(env_cfg, rsl_rl_cfg)
        self.cfg = env_cfg
        self.model_path = self.cfg.model_path

    # TODO how to get device from the cfg?
    def _load_model(self, model_path, device="cuda"):
        # Set the eDAE model
        #TODO get the model dir from the cfg
        model_dir = "experiments/test/S:forward_minus_0_4-OS:5-G:K4xC2-H:30-EH:30_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/"
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, model_dir)
        MiniCheetahModelEnv.shared_model = utils.get_trained_eDAE_model(model_dir)
        MiniCheetahModelEnv.shared_model.to(self.device)
        MiniCheetahModelEnv.shared_model.eval()

    def _pre_physics_step(self, actions):
        if not MiniCheetahModelEnv.model_loaded:
            self._load_model(self.model_path)
            MiniCheetahModelEnv.model_loaded = True
            print("Model loaded successfully (once).")

        if MiniCheetahModelEnv.shared_model is not None:
            # Use the model
            pass
        return actions