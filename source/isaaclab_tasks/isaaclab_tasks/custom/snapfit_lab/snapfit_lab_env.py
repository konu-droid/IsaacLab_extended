# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import wandb
from collections.abc import Sequence

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .snapfit_lab_env_cfg import SnapfitLabEnvCfg


class SnapfitLabEnv(DirectRLEnv):
    cfg: SnapfitLabEnvCfg

    def __init__(self, cfg: SnapfitLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_dist = torch.zeros((self.num_envs, 3), device=self.device)
        self.buckle_ftom_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # -- NEW: Variables for reward calculation --
        # Previous actions for calculating action rate penalty
        self.prev_actions = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        # Initial position of the female buckle to penalize its movement
        self.female_buckle_initial_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize wandb
        wandb.init(
            project="snapfit_franka",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
            },
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.female_buckle = RigidObject(self.cfg.buckle_female_cfg)
        self.male_buckle = Articulation(self.cfg.buckle_male_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["buckle_female"] = self.female_buckle
        self.scene.articulations["buckle_male"] = self.male_buckle

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
        # Store previous actions
        self.prev_actions[:] = self.actions
        # Update and clamp current actions
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
        
        robot_left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        
        self.robot_grasp_pos = 0.5 * (robot_left_finger_pos + robot_right_finger_pos)
        self.robot_grasp_rot = self.robot.data.body_quat_w[:, self.right_finger_link_idx]
        
        # Split into rotation and translation
        female_t = self.female_buckle.data.body_com_pos_w.squeeze(1)  # (N, 3)
        female_q = self.female_buckle.data.body_com_quat_w.squeeze(1)      # (N, 4)

        male_t = self.male_buckle.data.root_com_pos_w.squeeze(1)      # (N, 3)
        male_q = self.male_buckle.data.root_com_quat_w.squeeze(1)      # (N, 4)
        
        self.buckle_grasp_pos = male_t
        self.buckle_grasp_rot = male_q
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
                self.buckle_ftom_pose,
                self.grasp_dist,
                self.actions
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Get all necessary data for reward calculation
        robot_left_finger_pos = self.robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self.robot.data.body_pos_w[:, self.right_finger_link_idx]
        female_buckle_pos = self.female_buckle.data.root_pos_w
        
        total_reward, wandb_log = compute_rewards(
            # State tensors
            self.actions,
            self.prev_actions,
            self.robot.data.joint_vel,
            self.robot_grasp_pos,
            self.robot_grasp_rot,
            self.buckle_grasp_pos,
            self.buckle_grasp_rot,
            female_buckle_pos,
            self.female_buckle_initial_pos_w,
            self.buckle_ftom_pose[:, :3],
            robot_left_finger_pos,
            robot_right_finger_pos,
            # Reward scales from config
            self.cfg.approach_weight,
            self.cfg.align_weight,
            self.cfg.grasp_weight,
            self.cfg.mating_weight,
            self.cfg.action_rate_penalty,
            self.cfg.harsh_movement_penalty,
            self.cfg.female_move_penalty,
        )

        wandb.log(wandb_log, step=self.common_step_counter)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
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
        pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 2), device=self.device)
        buckle_female_new_state[:, 0:2] += pos_noise

        # add environment origins to the position
        buckle_female_new_state[:, :3] += self.scene.env_origins[env_ids]
        # write the new state to the simulation
        self.female_buckle.write_root_state_to_sim(buckle_female_new_state, env_ids)
        # -- NEW: Store the initial position for reward calculation
        self.female_buckle_initial_pos_w[env_ids] = buckle_female_new_state[:, :3].clone()


        # -- BUCKLE MALE --
        # reset the male buckle to its default state
        buckle_male_default_state = self.male_buckle.data.default_root_state[env_ids]
        buckle_male_new_state = buckle_male_default_state.clone()
        buckle_male_new_state[:, :3] += self.scene.env_origins[env_ids]
        self.male_buckle.write_root_state_to_sim(buckle_male_new_state, env_ids)


@torch.jit.script
def compute_rewards(
    # State tensors
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_velocities: torch.Tensor,
    robot_grasp_pos: torch.Tensor,
    robot_grasp_rot: torch.Tensor,
    male_buckle_pos: torch.Tensor,
    male_buckle_rot: torch.Tensor,
    female_buckle_pos: torch.Tensor,
    female_buckle_initial_pos: torch.Tensor,
    male_to_female_pos: torch.Tensor,
    robot_lfinger_pos: torch.Tensor,
    robot_rfinger_pos: torch.Tensor,
    # Reward scales
    approach_weight: float,
    align_weight: float,
    grasp_weight: float,
    mating_weight: float,
    # Penalty scales
    action_rate_penalty: float,
    harsh_movement_penalty: float,
    female_move_penalty: float,
):
    # --- STAGE 1: Approach the male buckle ---
    # Reward for gripper approaching the male buckle.
    gripper_to_male_dist = torch.norm(robot_grasp_pos - male_buckle_pos, p=2, dim=-1)
    # # Using an exponential decay function for a smooth reward
    # approach_reward = torch.exp(-4.0 * gripper_to_male_dist)
    approach_reward = 1.0 / (1.0 + gripper_to_male_dist**2)
    approach_reward *= approach_reward
    approach_reward = torch.where(gripper_to_male_dist <= 0.02, approach_reward * 2, approach_reward)
    approach_reward = approach_reward + torch.exp(-4.0 * gripper_to_male_dist)  # so that there is a also a good continous reward

    # Reward for aligning the gripper's orientation with the male buckle's orientation.
    # The dot product of two quaternions measures their similarity.
    # Squaring it handles the q vs -q ambiguity and maps the reward to [0, 1].
    quat_dot = torch.abs(torch.sum(robot_grasp_rot * male_buckle_rot, dim=-1))
    align_reward = quat_dot**4  # Power of 4 to make it more peaked

    # --- STAGE 2: Grip the male buckle ---
    # Reward for closing the fingers around the buckle.
    # This is a sparse reward that should only be given when the gripper is very close.
    finger_dist = torch.norm(robot_lfinger_pos - robot_rfinger_pos, p=2, dim=-1)
    # A bonus is given for being close to the buckle AND having closed fingers.
    # The grasp is successful if finger distance is small (< 0.02m) and gripper is close to buckle (< 0.03m)
    is_gripped = (finger_dist < 0.05) & (gripper_to_male_dist < 0.03)
    grasp_reward = is_gripped.float()

    # --- STAGE 3: Take the male buckle to the female buckle ---
    # This reward is only active *after* the male buckle is gripped.
    # It rewards reducing the distance between the male and female buckles.
    male_to_female_dist = torch.norm(male_to_female_pos, p=2, dim=-1)
    mating_reward = (torch.exp(-2.0 * male_to_female_dist)) * grasp_reward  # Only apply if gripped

    # --- PENALTIES ---
    # 1. Penalty for harsh movements (high rate of action change).
    action_rate_diff = torch.norm(actions - prev_actions, p=2, dim=-1)
    action_rate_penalty_val = action_rate_diff**2

    # 2. Penalty for self-collision (proxied by high joint velocities).
    # This encourages smoother, more deliberate movements.
    harsh_movement_penalty_val = torch.sum(torch.square(joint_velocities), dim=1)

    # 3. Penalty for moving the female buckle from its initial position.
    female_move_dist = torch.norm(female_buckle_pos - female_buckle_initial_pos, p=2, dim=-1)
    female_move_penalty_val = female_move_dist**2

    # --- FINAL REWARD CALCULATION ---
    total_reward = (
        approach_reward * approach_weight
        + align_reward * align_weight
        + grasp_reward * grasp_weight
        + mating_reward * mating_weight
        - action_rate_penalty_val * action_rate_penalty
        - harsh_movement_penalty_val * harsh_movement_penalty
        - female_move_penalty_val * female_move_penalty
    )

    # --- LOGGING FOR MONITORING ---
    wandb_log = {
        "reward/total_reward": total_reward.mean().item(),
        "reward/approach_reward": approach_reward.mean().item(),
        "reward/align_reward": align_reward.mean().item(),
        "reward/grasp_reward": grasp_reward.mean().item(),
        "reward/mating_reward": mating_reward.mean().item(),
        "penalty/action_rate_penalty": action_rate_penalty_val.mean().item(),
        "penalty/harsh_movement_penalty": harsh_movement_penalty_val.mean().item(),
        "penalty/female_move_penalty": female_move_penalty_val.mean().item(),
        "state/gripper_to_male_dist": gripper_to_male_dist.mean().item(),
        "state/male_to_female_dist": male_to_female_dist.mean().item(),
        "state/finger_dist": finger_dist.mean().item(),
        "state/female_move_dist": female_move_dist.mean().item(),
    }

    return total_reward, wandb_log