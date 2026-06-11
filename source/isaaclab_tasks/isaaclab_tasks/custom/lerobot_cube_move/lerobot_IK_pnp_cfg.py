# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the IK-based lerobot SO101 pick-and-place task.

Unlike :class:`LerobotCubeMoveEnvCfg` (joint-space actions), this task maps a
4-dimensional action (delta end-effector position in xyz + gripper command)
to joint targets using a differential inverse kinematics (IK) controller.
The scene, robot, and reward structure are intentionally identical to the
joint-space task so that the two action parameterizations can be compared.
"""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg

from .isaaclab_asset.lerobot_so101 import SO101_CFG


@configclass
class LerobotIKPnPEnvCfg(DirectRLEnvCfg):
    # env
    viewer: ViewerCfg = ViewerCfg(env_index=0, eye=(1.2, 0.8, 1.4), lookat=(0.1, 0.0, 0.8), origin_type="env")
    episode_length_s = 8  # 480 env steps (dt=1/120, decimation=2)
    decimation = 2
    # action: (dx, dy, dz) end-effector delta + (wrist_flex, wrist_roll) deltas + gripper command
    action_space = 6
    observation_space = 20
    state_space = 0
    TABLE_HEIGHT = 0.78

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size=2 * 1024 * 1024 * 1024  # 2gb allocated
        )
    )

    # robot
    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

    # differential IK controller: position-only command with damped least squares.
    # Position IK runs over the 3 position joints only (pan/lift/elbow, exactly
    # determined for a 3D position target); the wrist joints are owned by the
    # policy so it can orient the gripper for grasping (scripted-pick tests showed
    # the IK null-space orientation points the fingers too steeply down to ever
    # enclose the cube). DLS is robust near singularities.
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,  # the env integrates deltas into an absolute target itself
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )

    # joints driven by the IK controller (position joints only)
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    # wrist joints driven directly by the policy (orientation control)
    wrist_joint_names = ["wrist_flex", "wrist_roll"]
    # body whose position the IK controller tracks (same frame used for the reward)
    ee_body_name = "gripper_link"

    # workspace limits for the commanded end-effector target, expressed in the
    # robot's root frame. Measured: the table surface is at z ~= -0.02 in this
    # frame (cube com rests at z ~= -0.009), so allow the gripper_link target
    # slightly below 0 for low horizontal grasps.
    # Keeps the integrated target inside the reachable space so IK cannot diverge.
    ee_workspace_min = (0.08, -0.35, -0.01)
    ee_workspace_max = (0.45, 0.35, 0.35)

    # contact sensors
    contact_sensor_left_finger = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/moving_jaw_so101_v1_link",
        filter_prim_paths_expr=["/World/envs/env_.*/pick_cube"],
        history_length=3,
        debug_vis=False,
    )
    contact_sensor_right_finger = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_link",
        filter_prim_paths_expr=["/World/envs/env_.*/pick_cube"],
        history_length=3,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)

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
            usd_path=f"robots_usd/lerobot/additional_assets/thor_table.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # The table is a static, non-movable object
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT)),
    )

    # cube to pick up
    pick_cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/pick_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="robots_usd/lerobot/additional_assets/cube.usd",
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
            scale=(0.02, 0.02, 0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.1, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # visual-only marker for the place target
    target_cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="robots_usd/lerobot/additional_assets/cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.01 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # custom parameters/scales
    # max end-effector displacement (m) commanded per env step at |action| = 1
    ee_pos_action_scale = 0.02
    # wrist joint target speed (rad/s) at |action| = 1
    wrist_action_scale = 1.0
    # gripper joint target speed (rad/s) at |action| = 1.
    # Matches the gripper actuator's velocity_limit_sim (0.2 rad/s) so the
    # commanded target cannot run ahead of the physical joint.
    gripper_action_scale = 0.2

    # measured world-z at which the pick cube rests on the table. The joint-space
    # task hardcodes 0.79, but the actual rest height is ~0.771 (table surface is
    # ~2 cm below the robot root) — using the measured value removes a ~2 cm
    # dead zone at the bottom of the lift reward gradient.
    cube_rest_z_w: float = 0.7715

    # reward scales (identical to the validated joint-space task)
    # gripper-to-cube reaching shaping
    pick_reward_scale: float = 2.0
    # coarse cube-to-target shaping while carrying (or once placed)
    place_reward_scale: float = 16.0
    # sharp near-target shaping for the final few centimetres
    place_fine_reward_scale: float = 8.0
    # carry-height shaping; fades out near the target so descending is free
    lift_reward_scale: float = 5.0
    # gripper open/close/contact shaping
    gripper_reward_scale: float = 1.0
    # persistent bonus while the cube is within tolerance of the target
    success_reward_scale: float = 10.0
    # bonus for opening the gripper while the cube is at the target (release)
    release_reward_scale: float = 5.0

    pick_moved_reward_scale: float = -0.5

    action_penalty_scale: float = 0.001
