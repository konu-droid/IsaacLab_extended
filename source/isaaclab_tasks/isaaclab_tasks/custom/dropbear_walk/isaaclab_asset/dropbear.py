# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Configuration for Dropbear humanoid robot.

This module defines the complete configuration for the Dropbear humanoid robot,
including physical properties, actuators, initial states, and joint mappings.
The configuration is optimized for reinforcement learning tasks in Isaac Lab.
"""

import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class DropbearArticulationCfg(ArticulationCfg):
    """Configuration class for Dropbear humanoid robot articulations.
    
    This configuration extends the base ArticulationCfg to include Dropbear-specific
    parameters and joint mappings for the humanoid robot.
    
    Attributes:
        joint_sdk_names: List of joint names for SDK integration.
        soft_joint_pos_limit_factor: Safety factor for joint position limits.
    """

    joint_sdk_names: Optional[list[str]] = None
    """List of joint names used for SDK integration and control mapping."""

    soft_joint_pos_limit_factor: float = 0.9
    """Factor to apply to joint limits for safety (0.9 = 90% of max range)."""

# Dropbear humanoid robot configuration
DROPBEAR_CFG = DropbearArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"robots_usd/hyperspawn/dropbear.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,  # Reduce depenetration velocity
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collisions for stability
            solver_position_iteration_count=255,  # Keep high for our humanoid, 100 is good but with 255(max) i get zero Nan problem!
            solver_velocity_iteration_count=2,  # Keep high for our humanoid
            fix_root_link=False,  # Disable debugging_joint to prevent coordinate explosions
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.005),
        joint_pos={
            # Left arm joints - neutral position for balance
            "LH_yaw": 0.0,
            "LH_pitch": 0.0,
            "LH_roll": 0.0,
            "LH_elbow_joint": 0.0,
            "LH_wrist_roll": 0.0,
            # Right arm joints - neutral position for balance
            "RH_yaw": 0.0,
            "RH_pitch": 0.0,
            "RH_roll": 0.0,
            "RH_elbow_joint": 0.0,
            "RH_wrist_roll": 0.0,
            # Pelvic girdle - neutral stance
            "PG_left_leg_pitch": 0.0,
            "PG_left_leg_roll": 0.0,
            "PG_right_leg_pitch": 0.0,
            "PG_right_leg_roll": 0.0,
            # Left leg - standing position
            "LL_hip_joint": 0.0,
            "LL_knee_actuator_joint": 0.0,
            "LL_Revolute67": 0.0,
            "LL_Revolute81": 0.0,
            # Right leg - standing position
            "RL_hip_joint": 0.0,
            "RL_knee_actuator_joint": 0.0,
            "RL_Revolute67": 0.0,
            "RL_Revolute81": 0.0,
            # Head/neck - neutral position
            "head_LeadScrew1": 0.0,
            "head_LeadScrew2": 0.0,
            "head_LeadScrew3": 0.0,
            "head_LeadScrew4": 0.0,
            "head_LeadScrew5": 0.0,
            "head_LeadScrew6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                # Left arm actuators
                "LH_yaw", "LH_pitch", "LH_roll", "LH_elbow_joint", "LH_wrist_roll",
                # Right arm actuators
                "RH_yaw", "RH_pitch", "RH_roll", "RH_elbow_joint", "RH_wrist_roll",
                # Pelvic girdle actuators
                "PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll",
                # Leg actuators
                "LL_hip_joint", "LL_knee_actuator_joint", "RL_hip_joint", "RL_knee_actuator_joint",
                "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",
            ],
            effort_limit_sim=200.0,
            stiffness=100.0,
            damping=5.0,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
                "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6",
            ],
            effort_limit_sim=2000.0,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    joint_sdk_names=[
        # Left arm actuators
        "LH_yaw", "LH_pitch", "LH_roll", # "LH_elbow_joint", "LH_wrist_roll",
        # Right arm actuators
        "RH_yaw", "RH_pitch", "RH_roll", # "RH_elbow_joint", "RH_wrist_roll",
        # Pelvic girdle actuators
        "PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll",
        # Leg actuators
        "LL_hip_joint", "LL_knee_actuator_joint", "RL_hip_joint", "RL_knee_actuator_joint",
        "LL_Revolute67", "LL_Revolute81", "RL_Revolute67", "RL_Revolute81",
        # # Head/neck joints
        # "head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
        # "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6",
    ],
)

# Export the main configuration
__all__ = ["DROPBEAR_CFG", "DropbearArticulationCfg"]