# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

TIAGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/tiago/tiago_omni.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        scale=(1.0, 1.0, 1.0)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7), 
        rot=(0.7, 0.7, 0.0, 0.0),
        joint_pos={
            "wheel_front_left_joint": 0.0, 
            "wheel_front_right_joint": 0.0, 
            "wheel_rear_left_joint": 0.0, 
            "wheel_rear_right_joint": 0.0,
            "arm_1_joint": 0.0,
            "arm_2_joint": 0.0,
            "arm_3_joint": 0.0,
            "arm_4_joint": 0.0,
            "arm_5_joint": 0.0,
            "arm_6_joint": 0.0,
            "arm_7_joint": 0.0,
            "torso_lift_joint": 0.0,
            "head_1_joint": 0.0,
            "head_2_joint": 0.0,
            "gripper_left_finger_joint": 0.0,
            "gripper_right_finger_joint": 0.0,
        }
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_front_left_joint", "wheel_front_right_joint", "wheel_rear_left_joint", "wheel_rear_right_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_[1-7]_joint"], 
            effort_limit=400.0, 
            velocity_limit=100.0, 
            stiffness=100.0, 
            damping=0.0
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["gripper_left_finger_joint", "gripper_right_finger_joint"], 
            effort_limit=400.0, 
            velocity_limit=100.0, 
            stiffness=100.0, 
            damping=0.0
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_lift_joint", "head_1_joint", "head_2_joint"], 
            effort_limit=400.0, 
            velocity_limit=100.0, 
            stiffness=100.0, 
            damping=0.0
        ),
    },
)
"""Configuration for a tiago robot with omni base."""
