# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claus

from isaaclab.assets import ArticulationCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg

from .isaaclab_asset.surgical_arms import SCR_CFG


@configclass
class SurgicalChallengeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 16
    observation_space = 32
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = SCR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)

    # Object
    phantom_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rigid_Phantom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/upwork/surgical/isaac-sim-surgical-robotics-challenge/Assets/Rigid_Phantom_isaaclab.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    Needle_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Needle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/upwork/surgical/isaac-sim-surgical-robotics-challenge/Assets/Needle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.02, 0.3, 0.743)),
    )

    camera_left_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera_left",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.33, 0.86), rot=(0.7, 0.0, 0.7, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.01, 20.0)
        ),
        width=480,
        height=640,
    )

    camera_right_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera_right",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.052, 0.33, 0.86), rot=(0.7, 0.0, 0.7, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.01, 20.0)
        ),
        width=480,
        height=640,
    )
    
  

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 1.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 10.0  # reset if cart exceeds this position [m]