# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
The following configurations are available:

* :obj:`KUKA_ROBOT_CFG`: kuka robot with hand
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

KUKA_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/kuka/kuka_w_hand.usd",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0, 
            fix_root_link=True, #robot pushes itself down without it
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "robot1_joint_a1": 0.0,
            "robot1_joint_a2": 0.0,
            "robot1_joint_a3": 0.0,
            "robot1_joint_a4": 0.0,
            "robot1_joint_a5": 0.0,
            "robot1_joint_a6": 0.0,
            "robot1_gripper_right_hand_thumb_bend_joint": 0.0,
            "robot1_gripper_right_hand_thumb_rota_joint1": 0.0,
            "robot1_gripper_right_hand_thumb_rota_joint2": 0.0,
            "robot1_gripper_right_hand_index_bend_joint": 0.0,
            "robot1_gripper_right_hand_index_joint1": 0.0,
            "robot1_gripper_right_hand_index_joint2": 0.0,
            "robot1_gripper_right_hand_mid_joint1": 0.0,
            "robot1_gripper_right_hand_mid_joint2": 0.0,
            "robot1_gripper_right_hand_ring_joint1": 0.0,
            "robot1_gripper_right_hand_ring_joint2": 0.0,
            "robot1_gripper_right_hand_pinky_joint1": 0.0,
            "robot1_gripper_right_hand_pinky_joint2": 0.0,
        },
        pos=(1.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["robot1_joint_a1", "robot1_joint_a2"],
            effort_limit=3.4e3, #maxing out
            stiffness=1e7,
            damping=1e5,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["robot1_joint_a3", "robot1_joint_a4", "robot1_joint_a5", "robot1_joint_a6"],
            effort_limit=3.4e3,
            stiffness=1e7,
            damping=1e5,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot1_gripper_right_hand_thumb_bend_joint",
                "robot1_gripper_right_hand_thumb_rota_joint1",
                "robot1_gripper_right_hand_thumb_rota_joint2",
                "robot1_gripper_right_hand_index_bend_joint",
                "robot1_gripper_right_hand_index_joint1",
                "robot1_gripper_right_hand_index_joint2",
                "robot1_gripper_right_hand_mid_joint1",
                "robot1_gripper_right_hand_mid_joint2",
                "robot1_gripper_right_hand_ring_joint1",
                "robot1_gripper_right_hand_ring_joint2",
                "robot1_gripper_right_hand_pinky_joint1",
                "robot1_gripper_right_hand_pinky_joint2"
                ],
            effort_limit=1.1,
            stiffness=1e7,
            damping=1e5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of KUKA construction robot with hand."""
