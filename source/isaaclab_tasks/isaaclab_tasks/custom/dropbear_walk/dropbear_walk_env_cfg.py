# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Dropbear humanoid walking task.

This module defines :class:`DropbearWalkEnvCfg`, the configuration for a goal-directed
human-like locomotion task. The design follows standard Isaac Lab locomotion practice:

* The policy outputs *residual* joint targets around a fixed nominal standing pose.
* The observation is fully proprioceptive (base velocities, projected gravity, joint
  state, last action, a gait-phase clock) plus a goal-direction command.
* The reward is decomposed into well-scaled, single-purpose terms that together shape a
  natural human gait (forward velocity tracking, an alternating contact schedule, foot
  clearance, arm-swing counter-balancing the legs, upright posture, and smoothness/effort
  regularization).

The reward *scales* below are the primary knobs that get tuned from the wandb feedback
during short validation runs.
"""

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
    """Configuration for the goal-directed Dropbear humanoid walking environment."""

    # -- env --
    decimation = 2
    episode_length_s = 10.0  # 10 s episodes -> 600 control steps at 60 Hz control
    # - spaces definition
    #   action_space  = 14 actuated joints (residual targets around the nominal pose)
    #   observation_space breakdown:
    #     base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    #     + goal_dir_b_xy(2) + goal_dist(1)
    #     + joint_pos_rel(14) + joint_vel(14) + last_action(14)
    #     + gait_phase sin/cos(2)  = 56
    action_space = 14
    observation_space = 56
    state_space = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=0,
            enable_ccd=True,
            gpu_collision_stack_size=2 * 1024 * 1024 * 1024,  # 2 GB allocated
        ),
    )

    # -- robot --
    robot_cfg: ArticulationCfg = DROPBEAR_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    # -- sensors --
    # Contact sensor on both feet (the regex matches the left and right skateboard bearings).
    contact_sensor_feet: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*L_skateboard_bearing_left_2",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=4.0, replicate_physics=True
    )

    # -- body / joint name lookups --
    actuated_joint_names = robot_cfg.joint_sdk_names  # type: ignore
    head_mesh = "head_u_joint_center__8__1"
    # NOTE: feet order is [right, left] and must stay consistent with the gait clock below.
    feet_mesh_names = ["RL_skateboard_bearing_left_2", "LL_skateboard_bearing_left_2"]
    hand_mesh_names = ["RH_shoulder_ex_al_interface_1", "LH_shoulder_ex_al_interface_1"]
    # Shoulder yaw joints used for the arm swing ([left, right]).
    shoulder_joint_names = ["LH_yaw", "RH_yaw"]

    # ------------------------------------------------------------------ #
    #                       task / robot parameters                      #
    # ------------------------------------------------------------------ #
    # - goal: a point placed straight ahead of each robot's spawn (+x).
    goal_distance = 5.0  # m ahead of the spawn
    goal_reached_threshold = 0.5  # m, distance at which the goal-bonus fires

    # - locomotion targets
    target_speed = 0.6  # m/s desired forward speed toward the goal
    target_head_height = 1.55  # m, the head sits ~1.7 m when standing
    fall_height = 1.2  # m, head below this -> terminate (fallen)
    tilt_termination = -0.5  # projected-gravity z above this -> tipped over
    target_foot_clearance = 0.10  # m, desired swing-foot height above the ground
    feet_separation_threshold = 0.12  # m, feet closer than this are penalized
    contact_force_threshold = 1.0  # N, above this a foot counts as "in contact"

    # - gait clock
    gait_period = 0.9  # s for one full (left+right) step cycle
    target_air_time = 0.35  # s desired swing duration per foot

    # - action
    action_scale = 0.5  # residual targets = default_pose + action_scale * action

    # ------------------------------------------------------------------ #
    #                           reward scales                            #
    # ------------------------------------------------------------------ #
    # Positive shaping terms.
    # NOTE: forward progress must clearly out-reward simply standing upright, otherwise the
    # policy collapses to a "stand still and stay tall" local optimum (observed in run
    # v6zuyhrt). Hence progress is the dominant term and the posture/alive terms are kept
    # small so they shape but never substitute for walking.
    rew_scale_progress = 3.0      # track forward speed toward the goal (linear ramp, dominant)
    rew_scale_heading = 0.25      # face the goal
    rew_scale_alive = 0.25        # stay alive
    rew_scale_gait = 0.5          # feet match the alternating contact schedule
    rew_scale_air_time = 0.5      # encourage stepping (swing duration)
    rew_scale_foot_clearance = 0.3  # lift the swing foot to the target clearance
    rew_scale_arm_swing = 0.2     # arms swing opposite to the legs (counter-balance)
    rew_scale_upright = 0.5       # keep the torso upright
    rew_scale_head_height = 0.25  # keep the head high (tall posture)
    rew_scale_goal_bonus = 5.0    # one-off bonus for reaching the goal

    # Negative regularization terms (penalties, kept small so they shape rather than dominate)
    rew_scale_lin_vel_z = -1.0      # no vertical bouncing
    rew_scale_ang_vel_xy = -0.05    # no roll/pitch spinning
    rew_scale_action_rate = -0.01   # smooth actions
    rew_scale_joint_vel = -2.5e-4   # economical joint motion
    rew_scale_joint_torque = -2.5e-6  # economical effort
    rew_scale_feet_near = -0.5      # don't cross/scuff the feet together
    rew_scale_contact_impact = -0.1  # soft foot landings
    rew_scale_dof_limit = -1.0      # stay off the joint limits
    rew_scale_terminated = -10.0    # penalty for falling (kept moderate so walking is explored)
