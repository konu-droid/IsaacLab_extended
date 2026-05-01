# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the surgical arms (PSM)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

SCR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/konu/Documents/upwork/surgical/isaac-sim-surgical-robotics-challenge/Assets/PSM_arms.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "PSM1_Base_Yaw": 0.0,
            "PSM1_Main_Insert_Tool_Roll": 0.0,
            "PSM1_Yaw_Pitch_End": 0.0,
            "PSM1_Pitch_End_Main_Insert": 0.0,
            "PSM1_Tool_Yaw_Tool_Pitch": 0.0,
            "PSM1_Tool_Roll_Tool_Pitch": 0.0,
            "PSM1_Tool_Yaw_Grip1": -0.04,
            "PSM1_Tool_Yaw_Grip2": 0.13,
            "PSM2_Base_Yaw": 0.0,
            "PSM2_Main_Insert_Tool_Roll": 0.0,
            "PSM2_Yaw_Pitch_End": 0.0,
            "PSM2_Pitch_End_Main_Insert": 0.0,
            "PSM2_Tool_Yaw_Tool_Pitch": 0.0,
            "PSM2_Tool_Roll_Tool_Pitch": 0.0,
            "PSM2_Tool_Yaw_Grip1": 0.13,
            "PSM2_Tool_Yaw_Grip2": -0.04,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "PSM1": ImplicitActuatorCfg(
            joint_names_expr=["PSM1.*"],
            effort_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "PSM2": ImplicitActuatorCfg(
            joint_names_expr=["PSM2.*"],
            effort_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
