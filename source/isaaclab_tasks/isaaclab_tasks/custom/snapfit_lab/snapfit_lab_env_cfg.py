# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class SnapfitLabEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 28
    state_space = 0
    TABLE_HEIGHT = 0.78

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            # The error message suggested a size of at least 297529608
            gpu_collision_stack_size=1500000000  # Give it a bit of a buffer
        )
    )

    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, TABLE_HEIGHT),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    """Configuration of Franka Emika Panda robot."""

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)

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
            usd_path=f"/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/snapfit_lab/isaaclab_asset/thor_table.usd",
            scale=(1.0, 1.0, 1.0), # Scale the table to be slightly longer
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # The table is a static, non-movable object
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT)),
    )

    # snap fit buckle
    buckle_female_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/buckle_female",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/snapfit_lab/isaaclab_asset/Female_Buckle.usd",
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
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    buckle_male_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/buckle_male",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/konu/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/custom/snapfit_lab/isaaclab_asset/Male_Buckle_joints.usd",
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                ".*": 0.0,
            },
            pos=(0.4, 0.01, 0.1 + TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["Revolute_[1-2]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    
    # buckle_male_cfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/buckle_male",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="C:\\Users\\USER\\Documents\\upwork\\snapfit_rl\\IsaacLab\\source\\isaaclab_tasks\\isaaclab_tasks\\direct\\snapfit_lab\\isaaclab_asset\\Male_Buckle.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=50.0),
    #         scale=(0.5, 0.5, 0.5),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.1 + TABLE_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # custom parameters/scales
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    approach_weight = 2.0
    align_weight = 0.1
    grasp_weight = 1.0
    mating_weight = 1.0
    action_rate_penalty = 0.001
    harsh_movement_penalty = 0.001
    female_move_penalty = 0.001

