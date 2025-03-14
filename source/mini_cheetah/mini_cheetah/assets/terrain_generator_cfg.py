# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,  # Fixed seed for reproducibility
    size=(8.0, 8.0),
    border_width=10.0,  # Reduced border width for smoother transitions
    num_rows=20,  # Higher resolution for finer noise detail
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.001,  # Adjust to set overall height variation amplitude
    slope_threshold=0.8,  # Slightly adjusted threshold if needed
    use_cache=True,
    sub_terrains={
        "flat_with_noise": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.0, 0.05),  # Noise range for mild irregularities
            border_width=0.2,  # Adjusted border for noise continuity
            noise_step=0.001,  # Adjusted noise step for finer detail
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "stepping_stone": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,  # Height of each stepping stone
            stone_width_range=(0.1, 0.3),  # (width, depth) of each stone
            stone_distance_range=(0.1, 0.2),  # Distance between stones
            holes_depth=-0.05,  # Depth of holes in stones
        ),
    },
)