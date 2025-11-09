# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import wandb
from collections.abc import Sequence

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor

from .lerobot_cube_move_env_cfg import LerobotCubeMoveEnvCfg
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


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

        self.gripper_offset = torch.tensor([0.0, 0.0, -0.01]).to(self.device)
        self.gripper_frame_link_idx = self.robot.find_bodies("gripper_link")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pick_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pick_ftom_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.picked = torch.zeros((self.num_envs, 1), device=self.device)
        
        self.gripper_dof_idx = self.robot.find_joints("gripper")[0]
        # Buffer for the pick pick's height (for lift reward)
        self.pick_pick_z_pos = torch.zeros((self.num_envs, 1), device=self.device)
        # Buffer for the gripper's joint position, normalized between [0, 1]
        self.normalized_gripper_dist = torch.zeros((self.num_envs, 1), device=self.device)
        # Extract gripper limits for normalization
        self.gripper_lower_limit = self.robot_dof_lower_limits[self.gripper_dof_idx[0]]
        self.gripper_upper_limit = self.robot_dof_upper_limits[self.gripper_dof_idx[0]]
        
        # Initialize wandb
        wandb.init(
            project="LerobotPick",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
            },
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target_cube = RigidObject(self.cfg.target_cube_cfg)
        self.pick_cube = RigidObject(self.cfg.pick_cube_cfg)
        self.contact_left_finger = ContactSensor(self.cfg.contact_sensor_left_finger)
        self.contact_right_finger = ContactSensor(self.cfg.contact_sensor_right_finger)
        
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target_cube"] = self.target_cube
        self.scene.rigid_objects["pick_cube"] = self.pick_cube
        self.scene.sensors["contact_sensor_left_finger"] = self.contact_left_finger
        self.scene.sensors["contact_sensor_right_finger"] = self.contact_right_finger

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
        self.actions = actions.clone()
        targets = self.robot_dof_targets + (self.robot_dof_speed_scales * self.dt * self.actions)
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
        self.robot_grasp_pos = self.robot.data.body_link_pos_w[:, self.gripper_frame_link_idx] + self.gripper_offset
        
        # Split into rotation and translation
        target_t = self.target_cube.data.body_com_pos_w.squeeze(1)  # (N, 3)
        target_q = self.target_cube.data.body_com_quat_w.squeeze(1)      # (N, 4)

        pick_t = self.pick_cube.data.root_com_pos_w.squeeze(1)      # (N, 3)
        pick_q = self.pick_cube.data.root_com_quat_w.squeeze(1)      # (N, 4)
        
        self.pick_pos = pick_t + torch.tensor([0.0, 0.0, 0.05], device=self.device)
        self.place_pos = target_t
        # distance 
        pick_dist_pos = self.pick_pos - self.robot_grasp_pos
        place_dist_pos = self.place_pos - self.robot_grasp_pos
        pick_dist = torch.norm(self.pick_pos - self.robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
        place_dist = torch.norm(self.place_pos - self.robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
        # check if pick is done
        self.picked = torch.where(pick_dist < 0.05, 1.0, self.picked)

        # # Invert the target pose
        # target_q_inv, target_t_inv = tf_inverse(target_q, target_t)  # each is (N, 4) and (N, 3)
        # # Now combine: T_relative = inv(T_target) * T_pick
        # relative_q, relative_t = tf_combine(target_q_inv, target_t_inv, pick_q, pick_t)
        # # Combine back into SE(3) pose
        # self.pick_ftom_pose = torch.cat([relative_t, relative_q], dim=-1)  # shape: (N, 7)
        
        # # Get the pick's Z-position for the lift reward
        # self.pick_pick_z_pos[:] = pick_t[:, 2]
        # # Get the gripper's current joint position
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_dof_idx[0]]
        # Normalize it to a [0, 1] range where 1 is closed
        self.normalized_gripper_dist[:, 0] = (self.gripper_upper_limit - gripper_pos) / (self.gripper_upper_limit - self.gripper_lower_limit)

        obs = torch.cat(
            (
                dof_pos_scaled,
                pick_dist_pos,
                place_dist_pos,
                self.picked,
                self.normalized_gripper_dist,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # We calculate the magnitude of the force vector.
        left_finger_force = torch.norm(self.contact_left_finger.data.net_forces_w, dim=-1)
        right_finger_force = torch.norm(self.contact_right_finger.data.net_forces_w, dim=-1)

        pick_cube_vel = self.pick_cube.data.root_com_lin_vel_w.squeeze(1)      # (N, 3)
        
        total_reward, wandb_log = compute_rewards(
            # -- scales
            self.cfg.pick_reward_scale,
            self.cfg.place_reward_scale,
            self.cfg.gripper_reward_scale,
            self.cfg.pick_moved_reward_scale,
            # -- tensors
            self.robot_grasp_pos,
            self.pick_pos,
            self.place_pos,
            self.picked,
            self.normalized_gripper_dist,
            pick_cube_vel,
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

        self.robot_dof_targets[env_ids] = joint_pos

        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # get default state
        target_default_state = self.target_cube.data.default_root_state[env_ids]
        # copy to new state
        target_new_state = target_default_state.clone()

        # randomize position on xy-plane within a 10cm square
        pos_noise = sample_uniform(-0.1, 0.1, (len(env_ids), 2), device=self.device) # type: ignore
        target_new_state[:, 0:2] += pos_noise

        # add environment origins to the position
        target_new_state[:, :3] += self.scene.env_origins[env_ids]
        # write the new state to the simulation
        self.target_cube.write_root_state_to_sim(target_new_state, env_ids)

        # reset picked
        self.picked[env_ids] = 0.0

        # reset the pick to its default state
        pick_default_state = self.pick_cube.data.default_root_state[env_ids]
        pick_new_state = pick_default_state.clone()
        pick_new_state[:, 0:2] += pos_noise
        pick_new_state[:, :3] += self.scene.env_origins[env_ids]
        self.pick_cube.write_root_state_to_sim(pick_new_state, env_ids)


@torch.jit.script
def compute_rewards(
    # -- scales
    pick_reward_scale: float,
    place_reward_scale: float,
    gripper_reward_scale: float,
    pick_moved_reward_scale: float,
    # -- tensors
    robot_grasp_pos: torch.Tensor,
    pick_pos: torch.Tensor,
    place_pos: torch.Tensor,
    picked: torch.Tensor,
    normalized_gripper_dist: torch.Tensor,
    pick_cube_vel: torch.Tensor,
):
    """
    Computes rewards using a staged approach for the pick insertion task.
    The stages are: reaching, grasping, lifting, and mating.
    """

    pick_dist = torch.norm(pick_pos - robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
    pick_approach_reward = pick_reward_scale * torch.exp(-pick_dist)
    pick_approach_reward = torch.where(picked < 0.5, pick_approach_reward, pick_reward_scale * 1.0)

    place_dist = torch.norm(place_pos - robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
    place_approach_reward = place_reward_scale * torch.exp(-place_dist)
    place_approach_reward = torch.where(picked > 0.5, place_approach_reward, 0.0)

    # making make false 0.0 else robot just moves quickly to place position with closed gripper for huger reward.
    gripper_open_reward = gripper_reward_scale * (1.0 - normalized_gripper_dist)
    gripper_open_reward = torch.where(picked < 0.5, gripper_open_reward, 0.0)  # gripper_reward_scale * 1.0

    gripper_close_reward = gripper_reward_scale * normalized_gripper_dist
    gripper_close_reward = torch.where(picked > 0.5, gripper_close_reward, 0.0)

    pick_cube_move_penality = pick_moved_reward_scale * (torch.norm(pick_cube_vel) > 0.01)
    
    total_reward = (
        pick_approach_reward +
        place_approach_reward +
        gripper_open_reward + 
        gripper_close_reward +
        pick_cube_move_penality
    )
    
    wandb_log = {
        "reward/total_reward": total_reward.mean().item(),
        "reward/pick_approach_reward": pick_approach_reward.mean().item(),
        "reward/place_approach_reward": place_approach_reward.mean().item(),
        "reward/gripper_open_reward": gripper_open_reward.mean().item(),
        "reward/gripper_close_reward": gripper_close_reward.mean().item(),
        "reward/pick_cube_move_penality": pick_cube_move_penality.mean().item(),
        "state/pick_dist": pick_dist.mean().item(),
        "state/place_dist": place_dist.mean().item(),
        "state/picked": picked.mean().item(),
        "state/gripper_dist": normalized_gripper_dist.mean().item(),
    }

    return total_reward, wandb_log
