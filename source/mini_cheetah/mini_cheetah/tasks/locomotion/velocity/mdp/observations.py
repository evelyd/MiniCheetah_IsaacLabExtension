# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg

from isaaclab.sensors import Imu

from isaaclab.assets import Articulation, RigidObject


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def imu_projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")
) -> torch.Tensor:
    """Imu sensor orientation w.r.t the env.scene.origin.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Imu sensor.

    Returns:
        Orientation quaternion (wxyz), shape of torch.tensor is (num_env,4).
    """
    asset: Imu = env.scene[asset_cfg.name]
    GRAVITY_VEC_W = (
        torch.tensor([0.0, 0.0, -1.0], device=env.device)
        .unsqueeze(0)
        .expand(env.num_envs, 3)
    )
    return math_utils.quat_rotate_inverse(asset.data.quat_w, GRAVITY_VEC_W)