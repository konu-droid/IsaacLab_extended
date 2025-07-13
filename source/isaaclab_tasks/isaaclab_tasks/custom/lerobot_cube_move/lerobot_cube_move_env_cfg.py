# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg

from .isaaclab_asset.lerobot_so101 import SO101_CFG


@configclass
class LerobotCubeMoveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 6
    observation_space = 22
    state_space = 0
    TABLE_HEIGHT = 0.78

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # robot
    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # A rigid table for the robot and objects to be placed on.
    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/isaaclab_asset/thor_table.usd",
            scale=(1.0, 1.0, 1.0), # Scale the table to be slightly longer
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # The table is a static, non-movable object
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT)),
    )

    # snap fit buckle
    buckle_female_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/buckle_female",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/isaaclab_asset/Female_Buckle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.1, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    buckle_male_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/buckle_male",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/isaaclab_asset/Male_Buckle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # custom parameters/scales
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    mate_reward_scale = 10.0
    action_penalty_scale = 0.05
