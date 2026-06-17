# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the SO101 cube-grasp task with an 8x8 Time-of-Flight sensor.

This task is derived from :class:`LerobotCubeMoveEnvCfg` (the validated joint-space
pick-and-place task). The key addition is a simulated **Pololu VL53L7CX** multizone
Time-of-Flight ranging sensor mounted on top of the gripper's fixed jaw
(``gripper_link``). The real VL53L7CX returns an 8x8 grid of per-zone distance
readings; here it is emulated with an RTX :class:`TiledCamera` rendering a depth
image that is pooled down to an 8x8 zone grid (see the env module).

Relative to the base task, the observation drops ``pick_pos_local`` and the
``picked`` heuristic flag, and instead exposes the 64 ToF zone distances so the
policy can learn to align the fixed jaw with the cube and grasp it when in range.
"""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg

from .isaaclab_asset.lerobot_so101 import SO101_CFG


@configclass
class LerobotCubeToFEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------ env --
    viewer: ViewerCfg = ViewerCfg(env_index=0, eye=(1.2, 0.8, 1.4), lookat=(0.1, 0.0, 0.8), origin_type="env")
    episode_length_s = 8  # 800 timesteps
    decimation = 2
    action_space = 6
    state_space = 0
    TABLE_HEIGHT = 0.78

    # -- Time-of-Flight sensor geometry (Pololu VL53L7CX) --
    # The VL53L7CX reports an 8x8 grid of zone distances over a ~60 deg square FoV.
    tof_zones = 8  # zones per side (8x8 = 64 zones)
    # The RTX camera renders at a higher resolution and is min-pooled into the
    # zone grid; this both avoids tiny-resolution rendering artefacts and mimics
    # how each physical zone integrates the nearest return over its solid angle.
    tof_render_upscale = 4  # render resolution per side = tof_zones * tof_render_upscale
    tof_max_range = 0.5  # [m] readings clipped to this; also the observation normaliser
    # Realistic VL53L7CX minimum ranging distance. Geometry closer than this is
    # culled by the camera near-clip plane, which both models the device and hides
    # the gripper's own jaw/fingers (~1 cm away) that would otherwise pin every
    # reading and destroy the gradient (verified with diag_tof_nearclip.py).
    tof_min_range = 0.025
    # Observation: 6 dof + pick_pos(3) + place_pos(3) + cube_to_target(3)
    #            + gripper(1) + tof grid (tof_zones**2)
    observation_space = 16 + tof_zones * tof_zones

    # ----------------------------------------------------------- simulation --
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size=2 * 1024 * 1024 * 1024  # 2gb allocated
        ),
    )

    # ---------------------------------------------------------------- robot --
    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    # ------------------------------------------------------ contact sensors --
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

    # ----------------------------------------------------- ToF (RTX camera) --
    # Mounted as a child of the fixed jaw (``gripper_link``). The offset aims the
    # optical axis (+Z in the ROS camera convention) along the grasp approach
    # direction. From diag_gripper_frame.py, gripper_link's local -X axis points
    # straight down (world -Z) = the grasp approach. The rotation below maps the
    # camera frame so +Z -> link -X (forward/down), +X -> link +Y, +Y -> link -Z.
    tof_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_link/ToF_sensor",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, -0.5, 0.5),  # (w, x, y, z): +Z -> link -X (grasp approach)
            convention="ros",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            # ~60 deg square FoV: FoV = 2*atan(aperture / (2*focal_length)).
            focal_length=5.0,
            horizontal_aperture=5.77,
            clipping_range=(tof_min_range, tof_max_range),
        ),
        depth_clipping_behavior="max",  # out-of-range zones read tof_max_range, not inf
        width=tof_zones * tof_render_upscale,
        height=tof_zones * tof_render_upscale,
    )

    # ---------------------------------------------------------------- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)

    # ---------------------------------------------------------- ground plane --
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
                kinematic_enabled=True,  # The table is a static, non-movable object
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT)),
    )

    # cube to be grasped
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

    target_cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="robots_usd/lerobot/additional_assets/cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.01 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # custom parameters/scales
    action_scale = 1.0
    dof_velocity_scale = 0.1

    # ------------------------------------------------------- reward scales --
    # gripper-to-cube coarse reaching shaping (keeps the cube inside the ToF FoV)
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
    # penalty for moving the cube while it is not grasped
    pick_moved_reward_scale: float = -0.5
    # smoothness penalty on action magnitude
    action_penalty_scale: float = 0.001

    # -- ToF-driven shaping (the new sensor-use rewards) --
    # Reward for minimising the central ToF reading. Because the sensor looks
    # straight down the grasp axis, the central reading is only small when the
    # cube is BOTH directly under the jaw (aligned for a square grip) AND close
    # (in range) -- so this single term shapes alignment + final approach.
    tof_approach_reward_scale: float = 4.0
    # reward for closing the gripper once the cube is centred and in range
    tof_grasp_reward_scale: float = 4.0

    # -- ToF reward thresholds --
    # central reading below this (m) counts the cube as "in grasp range" ahead of the jaw
    tof_grasp_dist: float = 0.08
    # ToF shaping only applies when the cube is genuinely near the gripper (m), so the
    # policy cannot farm the alignment reward off the flat table with no cube present.
    tof_cube_near_dist: float = 0.12
    # detection threshold (m) used only for logging an "object present" metric
    tof_detect_dist: float = 0.20
