# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import wandb
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import Camera, CameraCfg

from isaaclab_tasks.custom.tiago.isaaclab_assets.tiago import TIAGO_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

@configclass
class TiagoEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 20.0  # multiplying it with 10 since tiago need extra velocity, network output capped at -1.0 to 1.0
    action_space = 16
    observation_space = {"robot_val": 10, "depth": [480, 640, 1]} #10
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = TIAGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheels_dof_name = ["wheel_front_left_joint", "wheel_front_right_joint", "wheel_rear_left_joint", "wheel_rear_right_joint"]
    arm_dof_name = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "torso_lift_joint", "head_1_joint", "head_2_joint"]
    finger_dof_name = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
    
    # sensors
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(12.0, 0.0, 19.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/tiago/additional_assets/shop_table.usd",
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
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # in-hand object
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/tiago/additional_assets/red_cup.usd",
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
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 1.0), rot=(0.7, -0.7, 0.0, 0.0)),
    )
    
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=8.0, replicate_physics=True)

class TiagoEnv(DirectRLEnv):
    cfg: TiagoEnvCfg

    def __init__(self, cfg: TiagoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
        self._wheel_dof_idx, _ = self.tiago.find_joints(self.cfg.wheels_dof_name)
        self._arm_dof_idx, _ = self.tiago.find_joints(self.cfg.arm_dof_name)
        self._finger_dof_idx, _ = self.tiago.find_joints(self.cfg.finger_dof_name)
        self.action_scale = self.cfg.action_scale
        
        # Initialize goal states
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_orient_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_pos_threshold = 1.0
        
        # minimum distance to the goal, used to constantly progress the robot towards goal
        self.prev_min_dist_goal = torch.zeros((self.num_envs), device=self.device)

        self.joint_pos = self.tiago.data.joint_pos
        self.joint_vel = self.tiago.data.joint_vel
        
        # Initialize wandb
        wandb.init(
            project="tiago-pick",
            config={
                "num_envs": self.num_envs,
                "episode_length": self.cfg.episode_length_s,
                "action_scale": self.action_scale,
            }
        )

    def _setup_scene(self):
        self.tiago = Articulation(self.cfg.robot_cfg)
        self.camera = Camera(self.cfg.camera_cfg)
        self.table = RigidObject(self.cfg.table)
        self.cup = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["tiago"] = self.tiago
        self.scene.sensors["camera"] = self.camera
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cup"] = self.cup
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # print('actions', self.actions.shape)
        
        # Extract single values for wheels and steering
        wheel_actions = self.actions[:, 0:4] * self.action_scale # Keep dimension for broadcasting
        arm_actions = self.actions[:, 4:14]  # Keep dimension for broadcasting
        finger_actions = self.actions[:, 14:]  # Keep dimension for broadcasting
        
        wheel_actions = torch.clamp(wheel_actions, -20.0, 20.0)
        arm_actions = torch.clamp(arm_actions, -1.0, 1.0)
        finger_actions = torch.clamp(finger_actions, -1.0, 1.0)
        
        # print('wheel_actions', wheel_actions)
        # print('steering_actions', steering_actions)

        # Apply velocity control to wheels
        self.tiago.set_joint_velocity_target(wheel_actions, joint_ids=self._wheel_dof_idx)        
        # Apply position control to arm and torso
        self.tiago.set_joint_position_target(arm_actions, joint_ids=self._arm_dof_idx)
        # Apply position control to arm and torso
        self.tiago.set_joint_position_target(finger_actions, joint_ids=self._finger_dof_idx)

        # testing_wheel = torch.ones_like(wheel_actions) * 10
        # testing_wheel = torch.zeros_like(wheel_actions)
        # testing_arm = torch.zeros_like(arm_actions)
        # testing_finger = torch.zeros_like(finger_actions)
        # self.tiago.set_joint_velocity_target(testing_wheel, joint_ids=self._wheel_dof_idx)
        # self.tiago.set_joint_position_target(testing_arm, joint_ids=self._arm_dof_idx)
        # self.tiago.set_joint_position_target(testing_finger, joint_ids=self._finger_dof_idx)
        
        wandb.log({
            "action_vel": wheel_actions.mean().item(),
            "action_arm": arm_actions.mean().item(),
            "action_finger": finger_actions.mean().item(),
        }, step=self.common_step_counter)


    def _get_observations(self) -> dict:
        # print('obs', self.tiago.data.root_lin_vel_b.shape)
        # distance_vector = torch.subtract(self.goal_pos_w, self.tiago.data.root_pos_w)
        distance_vector = torch.subtract(self.cup.data.root_pos_w, self.tiago.data.root_pos_w)
        distance_to_goal = torch.linalg.norm(distance_vector, dim=1)
        distance_to_goal = distance_to_goal.unsqueeze(-1)
        
        rgb_img = self.scene["camera"].data.output["rgb"]
        depth_img = self.scene["camera"].data.output["distance_to_image_plane"]
        
        # print(distance_to_goal.shape)
        # print(self.tiago.data.root_lin_vel_b.shape)
        # print(self.tiago.data.root_ang_vel_b.shape)
        
        robot_val = torch.cat(
            (
                self.tiago.data.root_lin_vel_b,
                self.tiago.data.root_ang_vel_b,
                distance_vector,
                distance_to_goal,
            ),
            dim=-1,
        )
        
        obs = {
            "robot_val": robot_val,
            "depth": depth_img,
        }
        
        observations = {"policy": obs}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # rew_alive = 1.0 * (1.0 - self.reset_terminated.float())
        # rew_termination = -2.0 * self.reset_terminated.float()
        
        # distance_to_goal = torch.linalg.norm(self.goal_pos_w - self.tiago.data.root_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self.cup.data.root_pos_w - self.tiago.data.root_pos_w, dim=1)
        
        dist_reward = torch.where((self.prev_min_dist_goal - distance_to_goal) > 0.0, 1.0, 0.0 )
        dist_reward = 1.0/distance_to_goal + dist_reward # to encourage the robot to move towards goal but also prefer moving towards goal continously
        
        acc_penalty = torch.where(torch.any(torch.abs(self.tiago.data.root_lin_vel_w [:, :2]) > 0.8), -1.0, 0.0)
        
        # total_reward = rew_alive + rew_termination + dist_reward + acc_penalty
        total_reward = dist_reward + acc_penalty
        
        # Log metrics to wandb
        wandb.log({
            "total_reward": total_reward.mean().item(),
            # "alive_reward": rew_alive.mean().item(),
            "distance_reward": dist_reward.mean().item(),
            "acc_penalty": acc_penalty.mean().item(),
        }, step=self.common_step_counter)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.tiago.data.joint_pos
        # self.joint_vel = self.tiago.data.joint_vel
        # self.joint_acc = self.tiago.data.joint_acc
        body_acc = self.tiago.data.root_lin_vel_w
        
        # distance_to_goal = torch.linalg.norm(self.goal_pos_w - self.tiago.data.root_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self.cup.data.root_pos_w - self.tiago.data.root_pos_w, dim=1)
        
        # update prev goal min dist
        self.prev_min_dist_goal = torch.where(distance_to_goal < self.prev_min_dist_goal, distance_to_goal, self.prev_min_dist_goal)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = (torch.abs(distance_to_goal) > 5.0)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(body_acc[:, :2]) > 1.0, dim=1)
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.tiago._ALL_INDICES
        self.tiago.reset(env_ids)
        super()._reset_idx(env_ids)

        self.actions[env_ids] = 0.0
        
        # Sample new commands
        self.goal_pos_w[env_ids, 0] = torch.zeros_like(self.goal_pos_w[env_ids, 0]).uniform_(1.0, 2.0)
        self.goal_pos_w[env_ids, 1] = torch.zeros_like(self.goal_pos_w[env_ids, 1]).uniform_(-2.0, 2.0)
        self.goal_pos_w[env_ids, 2] = torch.zeros_like(self.goal_pos_w[env_ids, 2])
        self.goal_pos_w[env_ids, :2] += self.scene.env_origins[env_ids, :2]
        self.goal_orient_w[env_ids, :3] = torch.zeros_like(self.goal_orient_w[env_ids, :3])
        self.goal_orient_w[env_ids, 3] = torch.ones_like(self.goal_orient_w[env_ids, 3])
        
        # Reset robot state
        joint_pos = self.tiago.data.default_joint_pos[env_ids]
        joint_vel = self.tiago.data.default_joint_vel[env_ids]
        default_root_state = self.tiago.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.tiago.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.tiago.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.tiago.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.goal_pos_w)