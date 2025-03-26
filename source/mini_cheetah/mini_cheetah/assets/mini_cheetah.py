# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ergoCub robots.

The following configurations are available:

* :obj:`MINI_CHEETAH_CFG`: mini_cheetah robot with implicit actuator for all the joints

Reference:
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

##
# Configuration
##

class MiniCheetahCfgBuilder:
    @staticmethod
    def build_robot_cfg(
        model_path_relative: Path
    ):
        MINI_CHEETAH_CFG_PATH = Path(__file__).parent / "models"
        model_path = MINI_CHEETAH_CFG_PATH / model_path_relative
        MINI_CHEETAH_CFG = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(model_path),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.32), # Initial base position
                joint_pos={
                    # values are from https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/mini_cheetah/mini_cheetah_config.py#L13-26
                    "[FR]R_hip_joint": -0.1, # all right hip roll
                    "[FR]L_hip_joint": 0.1, # all left hip roll
                    ".*thigh_joint": -0.8, # all hip pitch
                    ".*calf_joint": 1.62, # all knee pitch
                    # "torso_to_abduct_[fh]r_j": -0.6,  # all right hip roll
                    # "torso_to_abduct_[fh]l_j": 0.6,  # all left hip roll
                    # "abduct.*": -1.0,  # all hip pitch
                    # "thigh.*": 2.7,  # all knee pitch
        },
            ),
            soft_joint_pos_limit_factor=1.8,
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness={
                        ".*": 20.0,
                    },
                    damping={
                        ".*": 0.5,
                    },
                ),
            },
        )
        return MINI_CHEETAH_CFG

MINI_CHEETAH_CFG = MiniCheetahCfgBuilder.build_robot_cfg(
    # usd generated using urdf from https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf
    Path("./mini_cheetah.usd")
)