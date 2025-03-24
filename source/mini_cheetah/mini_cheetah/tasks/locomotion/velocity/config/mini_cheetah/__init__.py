import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg
from mini_cheetah.tasks.locomotion.velocity.latent_model_env import MiniCheetahModelEnv

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Velocity-Flat-MiniCheetah-v0",
    entry_point="tasks.locomotion.velocity.velocity_model_env:MiniCheetahModelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MiniCheetahFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MiniCheetahFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-MiniCheetah-Play-v0",
    entry_point="tasks.locomotion.velocity.velocity_model_env:MiniCheetahModelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MiniCheetahFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MiniCheetahFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Rough-MiniCheetah-v0",
    entry_point="tasks.locomotion.velocity.velocity_model_env:MiniCheetahModelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.MiniCheetahRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MiniCheetahRoughPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Rough-MiniCheetah-Play-v0",
    entry_point="tasks.locomotion.velocity.velocity_model_env:MiniCheetahModelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.MiniCheetahRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MiniCheetahRoughPPORunnerCfg",
    },
)
