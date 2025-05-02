from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

import os
import dha
import mini_cheetah.tasks.locomotion.velocity.utils as utils
import numpy as np
import torch
import mini_cheetah
import sys

class MiniCheetahModelEnv(ManagerBasedRLEnv):
    shared_model = None
    usd_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    joint_order_for_morphosymm = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    def __init__(self, **kwargs):

        # initialize the current_times and ref trajs before the ManagerBasedRLEnv, because the attribute needs to exist before the command manager is created
        self.current_timesteps = torch.zeros(kwargs['cfg'].scene.num_envs, dtype=int)

        script_name = os.path.basename(os.path.abspath(sys.argv[0]))
        if "play" not in script_name: # TODO implement trajectory loading also in play mode
            self.package_dir = os.path.abspath(os.path.dirname(mini_cheetah.__file__))
            self.traj_savepath = "../../../logs/rsl_rl/mini_cheetah_flat/2025-03-26_21-28-50/obs_action_pairs.npy"
            self.trajectory_path = os.path.join(self.package_dir, self.traj_savepath)
            MiniCheetahModelEnv.rng = np.random.default_rng(kwargs['cfg'].seed)
            self._load_trajectories(self.trajectory_path, kwargs['cfg'].scene.num_envs, kwargs['cfg'].sim.device)
            print(f"[INFO] Loaded reference trajectories")

        super().__init__(**kwargs)
        self.model_path = "experiments/test/S=forward_minus_0_4-OS=5-G=K4xC2-H=30-EH=30_E-DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/"
        self._load_model()
        print(f"[INFO] Created MiniCheetahModelEnv")

        if "play" not in script_name: # TODO implement trajectory loading also in play mode
            from .mutable_command import MutableCommandManager  # Local import to avoid circular import
            self.command_manager: MutableCommandManager = MutableCommandManager(self.cfg.commands, self)
            print("[INFO] Set the Command Manager: ", self.command_manager)

    def _load_model(self):
        # Set the eDAE model
        #TODO get the model dir from the cfg
        dha_dir = os.path.dirname(dha.__file__)
        model_dir = os.path.join(dha_dir, self.model_path)
        MiniCheetahModelEnv.shared_model = utils.get_trained_eDAE_model(model_dir)
        MiniCheetahModelEnv.shared_model.to(self.device)
        MiniCheetahModelEnv.shared_model.eval()

        return MiniCheetahModelEnv.shared_model

    def _load_trajectories(self, traj_savepath, num_envs, device):
        MiniCheetahModelEnv.ref_trajectories =np.load(traj_savepath, allow_pickle=True)
        MiniCheetahModelEnv.trajectory_indices = torch.zeros(num_envs, dtype=int)

        # reorganize so that there is no more dict, just obs values
        MiniCheetahModelEnv.ref_trajectories = torch.tensor([traj['obs'] for traj in MiniCheetahModelEnv.ref_trajectories], device=device)

        # fix the dimensions if not the same as the number of environments
        if MiniCheetahModelEnv.ref_trajectories.shape[0] != num_envs:
            print(f"[INFO] Reshaping reference trajectories from {MiniCheetahModelEnv.ref_trajectories.shape[1]} to {num_envs}")
            # sample or repeat the reference trajectories to match the number of environments
            if MiniCheetahModelEnv.ref_trajectories.shape[1] < num_envs:
                # repeat the reference trajectories to match the number of environments
                MiniCheetahModelEnv.ref_trajectories = torch.tile(MiniCheetahModelEnv.ref_trajectories, (1, num_envs // MiniCheetahModelEnv.ref_trajectories.shape[1], 1))
        # check the dimensions of ref_trajectories
        MiniCheetahModelEnv.trajectory_indices = MiniCheetahModelEnv.rng.permutation(num_envs)

        # randomly shuffle the reference trajectories according to the seed
        MiniCheetahModelEnv.ref_trajectories = MiniCheetahModelEnv.ref_trajectories[:, MiniCheetahModelEnv.trajectory_indices, :]

    def update_current_timesteps(self, reset_buf):
        self.current_timesteps += 1
        self.current_timesteps[reset_buf] = 0
        if reset_buf.any():
            print(f"[INFO] Resetting timesteps for {reset_buf.sum()} environments")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        # This is the only line of the step() function that I modified, command manager needs access to the env in order to have the updated ref trajectories for the velocity command
        script_name = os.path.basename(os.path.abspath(sys.argv[0]))
        if "play" not in script_name: # TODO implement trajectory loading also in play mode
            self.command_manager.compute(self.step_dt, self)
        else:
            self.command_manager.compute(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras