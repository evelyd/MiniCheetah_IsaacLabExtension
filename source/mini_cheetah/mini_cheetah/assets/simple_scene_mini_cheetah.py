from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import (
    # RayCasterCfg,
    patterns,
    ContactSensorCfg,
    FrameTransformerCfg,
    ImuCfg,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from .terrain_generator_cfg import ROUGH_TERRAINS_CFG


from .mini_cheetah import MINI_CHEETAH_CFG


@configclass
class MiniCheetahSceneCfg(InteractiveSceneCfg):
    """Configuration for the mini cheetah scene."""

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )
    # terrain.terrain_generator.difficulty_range = (0.0, 0.5)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # replace seems to be important here!
    robot: ArticulationCfg = MINI_CHEETAH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*_sole",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=True)

    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    # )

    feet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="FR_calf", prim_path="{ENV_REGEX_NS}/Robot/FR_calf"
            ),
            FrameTransformerCfg.FrameCfg(
                name="FL_calf", prim_path="{ENV_REGEX_NS}/Robot/FL_calf"
            ),
            FrameTransformerCfg.FrameCfg(
                name="RR_calf", prim_path="{ENV_REGEX_NS}/Robot/RR_calf"
            ),
            FrameTransformerCfg.FrameCfg(
                name="RL_calf", prim_path="{ENV_REGEX_NS}/Robot/RL_calf"
            ),
        ],
        debug_vis=False,
    )

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base", debug_vis=True)