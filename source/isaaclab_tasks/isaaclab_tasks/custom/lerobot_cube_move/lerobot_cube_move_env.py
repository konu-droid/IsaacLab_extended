# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .lerobot_cube_move_env_cfg import LerobotCubeMoveEnvCfg


class LerobotCubeMoveEnv(DirectRLEnv):
    cfg: LerobotCubeMoveEnvCfg

    def __init__(self, cfg: LerobotCubeMoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.robot.find_joints("gripper")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        self.gripper_frame_link_idx = self.robot.find_bodies("gripper_link")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_dist = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_ftom_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.female_buckle = RigidObject(self.cfg.buckle_female_cfg)
        self.male_buckle = RigidObject(self.cfg.buckle_male_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["buckle_female"] = self.female_buckle
        self.scene.rigid_objects["buckle_male"] = self.male_buckle

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        self.robot_grasp_pos = self.robot.data.body_link_pos_w[:, self.gripper_frame_link_idx]
        self.robot_grasp_pos[:, 0] = self.robot_grasp_pos[:, 0] + 0.1 # since the centre of gripper is in x forward direction.
        
        female_pose = self.female_buckle.data.body_com_pose_w.squeeze(1)  # (N, 1, 7)
        male_pose = self.male_buckle.data.body_com_pose_w.squeeze(1)      # (N, 1, 7)
        # Split into rotation and translation
        female_t = female_pose[:, :3]         # shape: (N, 3)
        female_q = female_pose[:, 3:]         # shape: (N, 4)

        male_t = male_pose[:, :3]
        male_q = male_pose[:, 3:]
        
        self.buckle_grasp_pos = male_t
        # distance between the gripper and male buckle
        self.grasp_dist = self.buckle_grasp_pos - self.robot_grasp_pos

        # Invert the female pose
        female_q_inv, female_t_inv = tf_inverse(female_q, female_t)  # each is (N, 4) and (N, 3)
        # Now combine: T_relative = inv(T_female) * T_male
        relative_q, relative_t = tf_combine(female_q_inv, female_t_inv, male_q, male_t)
        # Combine back into SE(3) pose
        self.buckle_ftom_pose = torch.cat([relative_t, relative_q], dim=-1)  # shape: (N, 7)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.buckle_ftom_pose,
                self.grasp_dist
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        total_reward = compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.buckle_grasp_pos,
            self.buckle_ftom_pose,
            self.cfg.dist_reward_scale,
            self.cfg.mate_reward_scale,
            self.cfg.action_penalty_scale,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids) # type: ignore

        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints), # type: ignore
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -- BUCKLE FEMALE --
        # get default state
        buckle_female_default_state = self.female_buckle.data.default_root_state[env_ids]
        # copy to new state
        buckle_female_new_state = buckle_female_default_state.clone()

        # randomize position on xy-plane within a 10cm square
        pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 2), device=self.device) # type: ignore
        buckle_female_new_state[:, 0:2] += pos_noise

        # add environment origins to the position
        buckle_female_new_state[:, :3] += self.scene.env_origins[env_ids]
        # write the new state to the simulation
        self.female_buckle.write_root_state_to_sim(buckle_female_new_state, env_ids)

        # -- BUCKLE MALE --
        # reset the male buckle to its default state
        buckle_male_default_state = self.male_buckle.data.default_root_state[env_ids]
        buckle_male_new_state = buckle_male_default_state.clone()
        buckle_male_new_state[:, :3] += self.scene.env_origins[env_ids]
        self.male_buckle.write_root_state_to_sim(buckle_male_new_state, env_ids)


@torch.jit.script
def compute_rewards(
        actions: torch.Tensor,
        robot_grasp_pos: torch.Tensor,
        buckle_grasp_pos: torch.Tensor,
        buckle_ftom_pose: torch.Tensor,
        dist_reward_scale: float,
        mate_reward_scale: float,
        action_penalty_scale: float,
    ):

    # distance from hand to the buckle
    d = torch.norm(robot_grasp_pos - buckle_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions**2, dim=-1)

    # reward for how far the male buckle is from female
    mate_distance = torch.norm(buckle_ftom_pose[:, :3], p=2, dim=-1)  # buckle_top_joint
    mate_reward = 1.0 / (1.0 + mate_distance**2)

    rewards = (
        dist_reward_scale * dist_reward
        + mate_reward_scale * mate_reward
        - action_penalty_scale * action_penalty
    )

    return rewards
