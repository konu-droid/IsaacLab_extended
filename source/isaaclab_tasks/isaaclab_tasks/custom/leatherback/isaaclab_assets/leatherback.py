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

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/leatherback/leatherback.usd",
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
        scale=(0.01, 0.01, 0.01)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        joint_pos={
            "Wheel__Upright__Rear_Right": 0.0, 
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        }
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"], 
            effort_limit=400.0, 
            velocity_limit=100.0, 
            stiffness=100.0, 
            damping=0.0
        ),
    },
)
"""Configuration for a simple Car robot."""
