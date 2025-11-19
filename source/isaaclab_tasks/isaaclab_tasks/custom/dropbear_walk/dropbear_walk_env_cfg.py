# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from isaaclab.sim import PhysxCfg

from .isaaclab_asset.dropbear import DROPBEAR_CFG


@configclass
class DropbearWalkEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0  # s * 120 STEPS
    # - spaces definition
    action_space = 18  # 28
    observation_space = 41  # 61
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=0,
            enable_ccd=True,
            gpu_collision_stack_size=2 * 1024 * 1024 * 1024  # 2gb allocated
        )
    )

    # robot
    robot_cfg: ArticulationCfg = DROPBEAR_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    # Sensors
    contact_sensor_feet: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*L_skateboard_bearing_left_2",
        update_period=0.005,
        history_length=6,
        track_air_time=True,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    actuated_joint_names = robot_cfg.joint_sdk_names  # type: ignore
    head_mesh = "head_u_joint_center__8__1"
    feet_mesh_names = ["RL_skateboard_bearing_left_2", "LL_skateboard_bearing_left_2"]
    hand_mesh_names = ["RH_shoulder_ex_al_interface_1", "LH_shoulder_ex_al_interface_1"]
    head_joint_names = ["head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
                        "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6"]
    arm_joint_names = ["LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
                       "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll"]
    should_joint_names = ["LH_yaw", "RH_yaw"]
    # - robot reset
    fall_height = 1.2  # the head is at 1.7m
    target_swing_height = 0.01
    # - action scale
    action_scale = 1.0
    # - reward scales
    rew_scale_alive = 0.25
    rew_scale_terminated = -50.0
    rew_scale_goal_dist = 20.0
    rew_scale_height_dist = -1.0
    rew_scale_foot_contact = 0.1
    rew_scale_air_time = 2.0
    rew_scale_gait_contact = 2.0
    rew_scale_upright = 0.5
    rew_scale_swing_height = -5.0
    rew_scale_not_moving = -1.0
    rew_scale_contact_vel = -0.2
    rew_scale_lin_vel_z = -0.1
    rew_scale_ang_vel_xy = -1.0
