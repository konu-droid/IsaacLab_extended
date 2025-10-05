# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .dropbear_walk_env_cfg import DropbearWalkEnvCfg


class DropbearWalkEnv(DirectRLEnv):
    cfg: DropbearWalkEnvCfg

    def __init__(self, cfg: DropbearWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.head_mesh_idx, _ = self.robot.find_bodies("head_u_joint_center__8__1")
        self.actuated_joint_ids, _ = self.robot.find_joints(self.cfg.actuated_joint_names)
        self.head_joint_ids, _ = self.robot.find_joints(self.cfg.head_joint_names)
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits) 
        self.action_scale = cfg.action_scale

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.actuated_joint_ids)), device=self.device)
        self.head_dof_targets = torch.zeros((self.num_envs, len(self.head_joint_ids)), device=self.device)

        self.nan_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        targets = (self.action_scale * self.actions) * self.dt

        # handling NaNs
        bad_env_ids = torch.any(torch.isnan(targets) | torch.isinf(targets), dim=1)
        if torch.any(bad_env_ids):
            print(f"!!!--- NaN or Inf detected in actions of envs: {torch.where(bad_env_ids)[0].tolist()} ---!!!")
            # print(self.actions)
            self.nan_detected[bad_env_ids] = True
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        #testing remove later
        # self.robot_dof_target = torch.zeros_like(self.actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets, self.actuated_joint_ids)
        #setting the head joints as zero
        #head joints are very sensive and have very limited range 0 to -0.05
        #this breaks the simulation too quick so
        self.robot.set_joint_position_target(self.head_dof_targets, self.head_joint_ids)

    def _get_observations(self) -> dict:
        goal_pos = self.scene.env_origins + torch.tensor([2.0, 0.0, 0.0], device=self.device)
        # Calculate the distance to the goal in the xy-plane (ignoring height).
        dist_to_goal = torch.norm(goal_pos[:, :2] - self.robot.data.root_link_pos_w[:, :2], p=2, dim=-1).unsqueeze(dim=1)
        robot_quat = self.robot.data.root_link_quat_w
        
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.actuated_joint_ids],
                self.actions,
                dist_to_goal,
                robot_quat,
            ),
            dim=-1,
        )

        # handling NaNs
        bad_env_ids = torch.any(torch.isnan(obs) | torch.isinf(obs), dim=1)
        if torch.any(bad_env_ids):
            print(f"!!!--- NaN or Inf detected in obs of envs: {torch.where(bad_env_ids)[0].tolist()} ---!!!")
            # print(obs)
            self.nan_detected[bad_env_ids] = True
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        head_pos = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, :].squeeze()

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_goal_dist,
            self.cfg.rew_scale_height_dist,
            self.robot.data.root_link_pos_w,
            head_pos,
            self.scene.env_origins,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        fallen = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, 2].squeeze() < self.cfg.fall_height
        lower_limit = self.robot.data.joint_pos[:, self.actuated_joint_ids] < self.robot_dof_lower_limits
        upper_limit =  self.robot.data.joint_pos[:, self.actuated_joint_ids] > self.robot_dof_upper_limits

        # upper joint limit is always exceeded so ignoring it for now.
        # its the neck prismatic joints that exceed the guess it that 
        # forces pull it too far due to head weight
        # out_of_bounds = fallen | torch.any(lower_limit | upper_limit, dim=1)
        out_of_bounds = fallen
        
        # print("self.robot.data.soft_joint_pos_limits[:,:,0]")
        # print(self.robot_dof_lower_limits)
        # print("self.robot.data.soft_joint_pos_limits[:,:,1]")
        # print(self.robot_dof_upper_limits)
        # print("self.robot.data.joint_pos")
        # print(self.robot.data.joint_pos[0, self.actuated_joint_ids])
        # print("lower_limit")
        # print(lower_limit[0])
        # print("upper_limit")
        # print(upper_limit[0])
        # print("out_of_bounds")
        # print(out_of_bounds[0])

        # Any environment that is out_of_bounds OR has a NaN will be marked for reset.
        terminations = out_of_bounds | self.nan_detected

        return terminations, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset the NaN flag for the environments that are being reset
        self.nan_detected[env_ids] = False


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_goal_dist: float,
    rew_scale_height_dist: float,
    robot_root_pos: torch.Tensor,
    head_pos: torch.Tensor,
    env_origins: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """
    Computes rewards for the dropbear walking task.

    The reward function is composed of:
    - A reward for being alive.
    - A penalty for termination.
    - A reward for moving towards a goal 1 meter ahead.
    """
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # -- reward for moving forward towards a goal
    # The goal is defined as 2 meter forward from the starting position in the x-direction.
    goal_pos = env_origins + torch.tensor([2.0, 0.0, 0.0], device=robot_root_pos.device)
    # Calculate the distance to the goal in the xy-plane (ignoring height).
    dist_to_goal = torch.norm(goal_pos[:, :2] - robot_root_pos[:, :2], p=2, dim=-1)

    goal_height = env_origins + torch.tensor([0.0, 0.0, 1.5], device=robot_root_pos.device)
    dist_to_height = torch.norm(goal_height[:, 2] - head_pos[:, 2], p=2, dim=-1)

    reward_goal = rew_scale_goal_dist * torch.exp(-dist_to_goal)
    reward_height = rew_scale_height_dist * torch.exp(-dist_to_height)

    # -- total reward
    total_reward = rew_alive + rew_termination + reward_goal + reward_height
    return total_reward
