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
        self.pick_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_dist = torch.zeros((self.num_envs, 3), device=self.device)
        self.pick_ftom_pose = torch.zeros((self.num_envs, 7), device=self.device)
        
        self.gripper_dof_idx = self.robot.find_joints("gripper")[0]
        # Buffer for the pick pick's height (for lift reward)
        self.pick_pick_z_pos = torch.zeros(self.num_envs, device=self.device)
        # Buffer for the gripper's joint position, normalized between [0, 1]
        self.normalized_gripper_pos = torch.zeros(self.num_envs, device=self.device)
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
        
        self.pick_grasp_pos = pick_t
        # distance between the gripper and pick pick
        self.grasp_dist = self.pick_grasp_pos - self.robot_grasp_pos

        # Invert the target pose
        target_q_inv, target_t_inv = tf_inverse(target_q, target_t)  # each is (N, 4) and (N, 3)
        # Now combine: T_relative = inv(T_target) * T_pick
        relative_q, relative_t = tf_combine(target_q_inv, target_t_inv, pick_q, pick_t)
        # Combine back into SE(3) pose
        self.pick_ftom_pose = torch.cat([relative_t, relative_q], dim=-1)  # shape: (N, 7)
        
        # Get the pick pick's Z-position for the lift reward
        self.pick_pick_z_pos[:] = pick_t[:, 2]
        # Get the gripper's current joint position
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_dof_idx[0]]
        # Normalize it to a [0, 1] range where 1 is closed
        self.normalized_gripper_pos[:] = (self.gripper_upper_limit - gripper_pos) / (self.gripper_upper_limit - self.gripper_lower_limit)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self.pick_ftom_pose,
                self.grasp_dist,
                self.actions
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # We calculate the magnitude of the force vector.
        left_finger_force = torch.norm(self.contact_left_finger.data.net_forces_w, dim=-1)
        right_finger_force = torch.norm(self.contact_right_finger.data.net_forces_w, dim=-1)
        
        total_reward, wandb_log = compute_rewards(
            self.actions,
            self.pick_ftom_pose,
            self.robot_grasp_pos,
            self.pick_grasp_pos,  # This is the pick pick position
            self.pick_pick_z_pos,
            self.normalized_gripper_pos,
            self.robot.data.joint_pos[:, self.gripper_dof_idx[0]],
            self.cfg.reach_reward_scale,
            self.cfg.alignment_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.mate_reward_scale,
            self.cfg.success_bonus,
            self.cfg.action_penalty_scale,
            self.cfg.approach_angle_reward_scale,
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

        # -- pick pick --
        # reset the pick pick to its default state
        pick_default_state = self.pick_cube.data.default_root_state[env_ids]
        pick_new_state = pick_default_state.clone()
        pick_new_state[:, 0:2] += pos_noise
        pick_new_state[:, :3] += self.scene.env_origins[env_ids]
        self.pick_cube.write_root_state_to_sim(pick_new_state, env_ids)


@torch.jit.script
def compute_rewards(
    # Core state
    actions: torch.Tensor,
    pick_ftom_pose: torch.Tensor,
    # Gripper and pick pick states
    robot_grasp_pos: torch.Tensor,
    pick_pos: torch.Tensor,
    pick_z_pos: torch.Tensor,
    normalized_gripper_pos: torch.Tensor,
    gripper_joint_angle: torch.Tensor,
    # Reward scales from config
    reach_reward_scale: float,
    aligment_reward_scale: float,
    grasp_reward_scale: float,
    lift_reward_scale: float,
    mate_reward_scale: float,
    success_bonus: float,
    action_penalty_scale: float,
    approach_angle_reward_scale: float,
):
    """
    Computes rewards using a staged approach for the pick insertion task.
    The stages are: reaching, grasping, lifting, and mating.
    """

    grasp_dist = 0.01
    cube_boundary_offset = 0.03
    # --- Stage 1: Reaching for the pick ---
    # Dense reward for decreasing the distance between the gripper and the pick.
    cube_pos = pick_pos - cube_boundary_offset
    reach_dist = torch.norm(robot_grasp_pos - cube_pos, p=2, dim=-1)
    approach_reward = 1.0 / (1.0 + reach_dist)
    approach_reward = torch.where(reach_dist <= grasp_dist, approach_reward * 2, approach_reward)
    # approach_reward = approach_reward + torch.exp(-4.0 * reach_dist)

    # 1. Positional Reward: Center the gripper over the cube.
    alignment_dist = cube_pos[:, :2] - robot_grasp_pos[:, :2]
    alignment_dist = torch.norm(alignment_dist, p=2, dim=-1)
    alignment_reward = 1.0 / (1.0 + alignment_dist)
    alignment_reward = approach_reward * alignment_reward  # scaling it  using approach reward
    
    # --- New Reward: Approach with Correct Angle (User Request) ---
    # Reward for approaching the pick , but only when the gripper joint is
    # within a specific angle range (20-30 degrees) to encourage a good posture.
    min_angle_rad = math.radians(50.0) # 20 degrees
    max_angle_rad = math.radians(90.0) # 30 degrees
    #reward only then open gripper and 5cm away cause when it gets closer to object gripper needs to close.
    is_angle_valid = ((gripper_joint_angle >= min_angle_rad) & (gripper_joint_angle <= max_angle_rad) & (reach_dist >= grasp_dist)) | ((reach_dist <= grasp_dist) & (gripper_joint_angle <= max_angle_rad))
    
    # The reward is based on proximity, active only when the angle is correct.
    approach_angle_reward = torch.where(
        is_angle_valid,
        approach_reward,  # Same decay as reach_reward
        torch.zeros_like(approach_reward)
    )

    # --- Stage 2: Grasping the pick  ---
    # --- Define a flag for when the pick is considered "grasped" ---
    # This is the key to staging the subsequent rewards.
    # We consider it grasped if the gripper is close and mostly closed.
    is_grasped = (reach_dist <= grasp_dist) & (normalized_gripper_pos > 0.5)
    # Rewards closing the gripper, but only when it's already close to the pick.
    # This encourages a deliberate grasp action at the correct time.
    # Assumes normalized_gripper_pos is 1.0 when fully closed.
    grasp_reward = torch.where(
        is_grasped,  # Only provide reward when gripper is within 3cm
        approach_reward * normalized_gripper_pos,
        torch.zeros_like(approach_reward)
    )

    # --- Stage 3: Lifting the pick ---
    # After grasping, reward the agent for lifting the pick off the surface.
    # This prevents dragging and encourages a stable pickup.
    # The reward is shaped to be proportional to the height, up to a target.
    lift_height_target = 0.1 # 10cm
    lift_reward = torch.clamp(pick_z_pos / lift_height_target, 0.0, 1.0)


    # --- Stage 4: Mating the picks ---
    # This reward is only significant after the pick is grasped.
    relative_pos = pick_ftom_pose[:, :3]
    relative_quat = pick_ftom_pose[:, 3:]  # Assuming (x, y, z, w) format

    # Reward for minimizing the distance between the two picks.
    mate_dist = torch.norm(relative_pos, p=2, dim=-1)
    mate_pos_reward = torch.exp(-10.0 * mate_dist)

    # Reward for correct alignment. The target is an identity quaternion (0,0,0,1).
    # The dot product with an identity quaternion is just the w-component.
    # We reward w^2, which is 1 for perfect alignment (w=1 or w=-1).
    mate_orient_reward = relative_quat[:, 3] ** 2

    # Combine position and orientation rewards. Orientation becomes more important
    # as the picks get closer.
    mate_reward = mate_pos_reward * (1.0 + 0.5 * mate_orient_reward)

    # --- Success Bonus ---
    # A large, sparse bonus for achieving the final goal state.
    # Condition: very close in position and well-aligned in orientation.
    is_success = (mate_dist < 0.01) & (mate_orient_reward > 0.98)
    success_reward = torch.where(
        is_grasped & is_success,
        torch.full_like(reach_dist, success_bonus),
        torch.zeros_like(reach_dist)
    )

    # --- Action Regularization ---
    # Penalize large actions to encourage smooth and efficient movements.
    action_penalty = torch.sum(actions**2, dim=-1)

    # --- Combine all reward components ---
    # Use the `is_grasped` flag to activate rewards for lifting and mating only
    # after the pick has been picked up.
    reach_reward = reach_reward_scale * approach_reward
    alignment_reward = aligment_reward_scale * alignment_reward
    grasp_reward = grasp_reward_scale * grasp_reward
    approach_angle_reward = approach_angle_reward_scale * approach_angle_reward  # Scale new reward
    lift_reward = torch.where(is_grasped, lift_reward_scale * lift_reward, torch.zeros_like(lift_reward))
    mate_reward = torch.where(is_grasped, mate_reward_scale * mate_reward, torch.zeros_like(mate_reward))
    action_penalty = action_penalty_scale * action_penalty
    
    total_reward = (
        reach_reward
        + alignment_reward
        + grasp_reward
        + approach_angle_reward
        + lift_reward
        + mate_reward
        + success_reward
        - action_penalty
    )
    
    wandb_log = {
        "reward/total_reward": total_reward.mean().item(),
        "reward/approach_reward": approach_reward.mean().item(),
        "reward/approach_angle_reward": approach_angle_reward.mean().item(),
        "reward/grasp_reward": grasp_reward.mean().item(),
        "reward/alignment_reward": alignment_reward.mean().item(),
        "reward/mate_reward": mate_reward.mean().item(),
        "penalty/action_penalty": action_penalty.mean().item(),
        "state/gripper_to_pick_dist": reach_dist.mean().item(),
        "state/alignment_dist": alignment_dist.mean().item(),
        "state/pick_to_target_dist": mate_dist.mean().item(),
    }

    return total_reward, wandb_log
