"""Environment config for TiagoPRO right-arm mobile pick (pill bottle)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIAGO_USD = "/home/konu/Documents/upwork/kyunki/isaac_sim/projects/tiago_pro/tiago_pro_rs.usd"
TABLE_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd"
PILL_BOTTLE_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/Props/SM_PillBottle_01t.usd"


@configclass
class PickEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    decimation = 2  # action applied every 2 physics steps
    episode_length_s = 16.667  # 1000 steps at dt*decimation = 1/60*2 => ~16.67s
    observation_space = 12
    action_space = 8  # arm(7) + gripper(1)
    state_space = 0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=5.0, replicate_physics=True
    )

    # ------------------------------------------------------------------
    # Robot — TiagoPRO
    # ------------------------------------------------------------------
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TIAGO_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # isaac_ready_to_pick pose
                "torso_lift_joint": 0.2,
                # right arm — universal_pick_ready
                "arm_right_1_joint": -2.7845,
                "arm_right_2_joint": -1.1257,
                "arm_right_3_joint": 1.7738,
                "arm_right_4_joint": -2.2121,
                "arm_right_5_joint": 1.0664,
                "arm_right_6_joint": 1.5607,
                "arm_right_7_joint": 1.7522,
                # left arm — home
                "arm_left_1_joint": 0.36,
                "arm_left_2_joint": -1.83,
                "arm_left_3_joint": 0.47,
                "arm_left_4_joint": -2.35,
                "arm_left_5_joint": 0.0,
                "arm_left_6_joint": -1.2,
                "arm_left_7_joint": 0.0,
                # gripper
                "gripper_right_finger_joint": 0.04,
                # wheels
                "wheel_front_left_joint": 0.0,
                "wheel_front_right_joint": 0.0,
                "wheel_rear_left_joint": 0.0,
                "wheel_rear_right_joint": 0.0,
            },
        ),
        actuators={
            "arm_right": ImplicitActuatorCfg(
                joint_names_expr=["arm_right_[1-7]_joint"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_lift_joint"],
                effort_limit_sim=50000.0,
                stiffness=1e7,
                damping=1e4,
            ),
            "gripper_right": ImplicitActuatorCfg(
                joint_names_expr=["gripper_right_finger_joint"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
            "gripper_right_sub": ImplicitActuatorCfg(
                joint_names_expr=[
                    "gripper_right_inner_finger_left_joint",
                    "gripper_right_inner_finger_right_joint",
                    "gripper_right_outer_finger_right_joint",
                    "gripper_right_fingertip_left_joint",
                    "gripper_right_fingertip_right_joint",
                ],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=1e5,
            ),
            "arm_left": ImplicitActuatorCfg(
                joint_names_expr=["arm_left_[1-7]_joint"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_[1-2]_joint"],
                effort_limit_sim=10000.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper_left": ImplicitActuatorCfg(
                joint_names_expr=["gripper_left_finger_joint"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
            "gripper_left_sub": ImplicitActuatorCfg(
                joint_names_expr=[
                    "gripper_left_inner_finger_left_joint",
                    "gripper_left_inner_finger_right_joint",
                    "gripper_left_outer_finger_right_joint",
                    "gripper_left_fingertip_left_joint",
                    "gripper_left_fingertip_right_joint",
                ],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ------------------------------------------------------------------
    # Object — Pill Bottle (RigidObject)
    # ------------------------------------------------------------------
    pill_bottle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PillBottle",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.812),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ------------------------------------------------------------------
    # Table (static — no physics needed beyond collision)
    # ------------------------------------------------------------------
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.9, 0.812),  # approximate table dimensions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.406),  # 1.5m forward, center of box at half height
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ------------------------------------------------------------------
    # Ground Plane
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Action / Reward parameters
    # ------------------------------------------------------------------
    # Action scale (velocity-based, frame-rate independent)
    action_scale = 7.5          # rad/s for arm joints
    gripper_speed_scale = 0.1   # gripper moves 10x slower than arm
    joint_limit_buffer = 0.2    # rad from URDF limit
    wrist_limit = 1.7           # rad for joints 6, 7

    # Gripper values
    gripper_open = 0.04
    gripper_close = 0.80

    # Reward scales
    dist_reward_scale = 2.0
    lift_reward_scale = 50.0
    fov_center_reward_scale = 0.1
    time_penalty_scale = 0.001
    collision_penalty_scale = 5.0
    action_rate_penalty_scale = 0.05
    arm_far_penalty_scale = 1.5
    object_move_penalty_scale = 5.0
    fov_lost_penalty_scale = 2.0

    # Thresholds
    lift_height = 0.10         # m above initial object Z
    arm_far_threshold = 0.5    # m — penalize arm use when farther than this
    object_move_threshold = 0.10  # m — max XY movement before penalty
    arm_joint_vel_limit = 10.0 # rad/s — terminate if any arm joint exceeds this

    # FOV visibility: camera image percentage (top 80%)
    fov_visible_fraction = 0.80
