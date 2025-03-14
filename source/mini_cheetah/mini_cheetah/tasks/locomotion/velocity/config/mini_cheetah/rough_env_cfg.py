from isaaclab.utils import configclass

from mini_cheetah.tasks.locomotion.velocity.mini_cheetah_velocity_env_cfg import MiniCheetahLocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip
from mini_cheetah.assets import MINI_CHEETAH_CFG, MiniCheetahSceneCfg


@configclass
class MiniCheetahRoughEnvCfg(MiniCheetahLocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to mini cheetah
        # self.scene.robot = MINI_CHEETAH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene = MiniCheetahSceneCfg(num_envs=4096, env_spacing=4.0) #default for InteractiveSceneCfg is num_envs=4096, env_spacing=4.0

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None


@configclass
class MiniCheetahRoughEnvCfg_PLAY(MiniCheetahRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
