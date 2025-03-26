from isaaclab.envs import ManagerBasedRLEnv

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils

class MiniCheetahModelEnv(ManagerBasedRLEnv):
    shared_model = None
    usd_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    joint_order_for_morphosymm = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "experiments/test/S=forward_minus_0_4-OS=5-G=K4xC2-H=30-EH=30_E-DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/"
        self._load_model()
        print(f"[INFO] Created MiniCheetahModelEnv")

    # TODO how to get device from the cfg?
    def _load_model(self, device="cuda"):
        # Set the eDAE model
        #TODO get the model dir from the cfg
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        MiniCheetahModelEnv.shared_model = utils.get_trained_eDAE_model(model_dir)
        MiniCheetahModelEnv.shared_model.to(self.device)
        MiniCheetahModelEnv.shared_model.eval()

        return MiniCheetahModelEnv.shared_model