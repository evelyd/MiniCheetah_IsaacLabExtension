# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S=forward_minus_0_4-OS=5-G=K4xC2-H=30-EH=30_E-DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/seed=399/", help="Directory path to the eDAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S=2025-04-18_09-13-49-OS=5-G=K4xC2-H=5-EH=5_DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/seed=224/", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S=forward_minus_0_4-OS=5-G=K4xC2-H=30-EH=30_DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/seed=858/", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S=2025-04-18_09-13-49-OS=5-G=K4xC2-H=5-EH=5_DAE-Obs_w=1.0-Orth_w=0.0-Act=ELU-B=True-BN=False-LR=0.001-L=5-128_system=mini_cheetah/seed=981/", help="Directory path to the DAE model file.")

# The g error ones, h=1
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:1-EH:1_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L[]:5-128_system=mini_cheetah/seed=605", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:1-EH:1_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=028", help="Directory path to the DAE model file.")

# g error, h=10
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:10-EH:10_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=581", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:10-EH:10_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=220", help="Directory path to the DAE model file.")

# g error, h=5
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:5-EH:5_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=863", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-01_22-17-31-OS:5-G:K4xC2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=193", help="Directory path to the DAE model file.")

# # g error, h=1 v0
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:1-EH:1_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=798", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:1-EH:1_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=388", help="Directory path to the DAE model file.")

# # g error, h=5 v0
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:5-EH:5_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=618", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=073", help="Directory path to the DAE model file.")

# # g error, h=10 v0
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:10-EH:10_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=369", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:10-EH:10_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=220", help="Directory path to the DAE model file.")

# # g error, h=30 v0
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:30-EH:30_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=680", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-04-18_09-13-49-OS:5-G:K4xC2-H:30-EH:30_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=785", help="Directory path to the DAE model file.")

#-------------------------------------
# imu task v0, h=1, obs state ratio 5
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:1-EH:1_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=731", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:1-EH:1_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=083", help="Directory path to the DAE model file.")
# # imu task v0, h=5, obs state ratio 5
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:5-EH:5_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=075", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=945", help="Directory path to the DAE model file.")
# # imu task v0, h=10, obs state ratio 5
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:10-EH:10_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=921", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:5-G:K4xC2-H:10-EH:10_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=061", help="Directory path to the DAE model file.")
# # imu task v0, h=1, obs state ratio 1
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:1-G:K4xC2-H:1-EH:1_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=256", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:1-G:K4xC2-H:1-EH:1_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=791", help="Directory path to the DAE model file.")
# # imu task v0, h=1, obs state ratio 3
parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:3-G:K4xC2-H:1-EH:1_DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=305", help="Directory path to the DAE model file.")
# parser.add_argument("--dae_dir", type=str, default="experiments/test/S:2025-05-09_13-19-56-OS:3-G:K4xC2-H:1-EH:1_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=930", help="Directory path to the DAE model file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# from rsl_rl.runners import OnPolicyRunner
from mini_cheetah.tasks.locomotion.velocity.utils import LatentStateOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import mini_cheetah.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env_cfg.viewer.eye = [5.0, 5.0, 0.5]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Create a mapping from current joint order to morphosymm order
    usd_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    joint_order_for_morphosymm = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    joint_order_indices = [usd_joint_order.index(joint) for joint in joint_order_for_morphosymm]

    # Define the policy parameters dictionary
    policy_params = agent_cfg.policy.to_dict()  # Start with the default policy config
    policy_params.update({
        "class_name": "LatentStateActorCritic",
        "joint_order_indices": joint_order_indices,
        "dae_dir": args_cli.dae_dir,
        "task_name": args_cli.task,
        "device": agent_cfg.device,
    })

    agent_cfg.policy = policy_params

    # create runner from rsl-rl that calls the latent state actor-critic
    runner = LatentStateOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
