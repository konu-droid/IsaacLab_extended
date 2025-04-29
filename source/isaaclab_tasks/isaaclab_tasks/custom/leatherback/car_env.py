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
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.custom.leatherback.isaaclab_assets.leatherback import LEATHERBACK_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 20.0  # multiplying it with 10 since car need extra velocity, network output capped at -1.0 to 1.0
    action_space = 2
    observation_space = 10
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)


class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
        self._wheel_dof_idx = self.car.actuators["wheels"].joint_indices
        self._steering_dof_idx = self.car.actuators["steering"].joint_indices
        self.action_scale = self.cfg.action_scale
        
        # Initialize goal states
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_orient_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_pos_threshold = 1.0
        
        # minimum distance to the goal, used to constantly progress the robot towards goal
        self.prev_min_dist_goal = torch.zeros((self.num_envs), device=self.device)

        self.joint_pos = self.car.data.joint_pos
        self.joint_vel = self.car.data.joint_vel
        
        # Initialize wandb
        wandb.init(
            project="leatherback-training",
            config={
                "num_envs": self.num_envs,
                "episode_length": self.cfg.episode_length_s,
                "action_scale": self.action_scale,
            }
        )

    def _setup_scene(self):
        self.car = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["car"] = self.car
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Extract single values for wheels and steering
        wheel_action = self.actions[:, 0:1]  # Keep dimension for broadcasting
        steering_action = self.actions[:, 1:2]  # Keep dimension for broadcasting
        
        # Duplicate the single values for both wheels/steering
        wheel_actions = wheel_action.repeat(1, 2) * self.action_scale  # Duplicate for both wheels
        steering_actions = steering_action.repeat(1, 2)  # Duplicate for both steering joints
        
        wheel_actions = torch.clamp(wheel_actions, -20.0, 20.0)
        steering_actions = torch.clamp(steering_actions, -1.0, 1.0)
        
        # print('wheel_actions', wheel_actions)
        # print('steering_actions', steering_actions)

        # Apply velocity control to wheels
        self.car.set_joint_velocity_target(wheel_actions, joint_ids=self._wheel_dof_idx)        
        # Apply position control to steering
        self.car.set_joint_position_target(steering_actions, joint_ids=self._steering_dof_idx)

        # testing_wheel = torch.ones_like(wheel_actions) * 10
        # testing_steer = torch.zeros_like(steering_actions)
        # self.car.set_joint_velocity_target(testing_wheel, joint_ids=self._wheel_dof_idx)
        # self.car.set_joint_position_target(testing_steer, joint_ids=self._steering_dof_idx)
        
        wandb.log({
            "action_vel": wheel_action.mean().item(),
            "action_steer": steering_action.mean().item(),
        }, step=self.common_step_counter)


    def _get_observations(self) -> dict:
        # print('obs', self.car.data.root_lin_vel_b.shape)
        distance_vector = torch.subtract(self.goal_pos_w, self.car.data.root_pos_w)
        distance_to_goal = torch.linalg.norm(distance_vector, dim=1)
        distance_to_goal = distance_to_goal.unsqueeze(-1)
        
        # print(distance_to_goal.shape)
        # print(self.car.data.root_lin_vel_b.shape)
        # print(self.car.data.root_ang_vel_b.shape)
        
        obs = torch.cat(
            (
                self.car.data.root_lin_vel_b,
                self.car.data.root_ang_vel_b,
                distance_vector,
                distance_to_goal,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # rew_alive = 1.0 * (1.0 - self.reset_terminated.float())
        # rew_termination = -2.0 * self.reset_terminated.float()
        
        distance_to_goal = torch.linalg.norm(self.goal_pos_w - self.car.data.root_pos_w, dim=1)
        
        dist_reward = torch.where((self.prev_min_dist_goal - distance_to_goal) > 0.0, 1.0, 0.0 )
        dist_reward = 1.0/distance_to_goal + dist_reward # to encourage the robot to move towards goal but also prefer moving towards goal continously
        
        acc_penalty = torch.where(torch.any(torch.abs(self.car.data.root_lin_vel_w [:, :2]) > 0.8), -1.0, 0.0)
        
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
        # self.joint_pos = self.car.data.joint_pos
        # self.joint_vel = self.car.data.joint_vel
        # self.joint_acc = self.car.data.joint_acc
        body_acc = self.car.data.root_lin_vel_w
        
        distance_to_goal = torch.linalg.norm(self.goal_pos_w - self.car.data.root_pos_w, dim=1)
        
        # update prev goal min dist
        self.prev_min_dist_goal = torch.where(distance_to_goal < self.prev_min_dist_goal, distance_to_goal, self.prev_min_dist_goal)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = (torch.abs(distance_to_goal) > 5.0)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(body_acc[:, :2]) > 1.0, dim=1)
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.car._ALL_INDICES
        self.car.reset(env_ids)
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
        joint_pos = self.car.data.default_joint_pos[env_ids]
        joint_vel = self.car.data.default_joint_vel[env_ids]
        default_root_state = self.car.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.car.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.car.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.car.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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