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
        # Per-episode latch: 1.0 once the cube has been held AND lifted off the table.
        # Used only for reward gating (not part of the observation) so that place /
        # success rewards cannot be earned by dragging the cube along the table.
        self.was_lifted = torch.zeros((self.num_envs, 1), device=self.device)
        
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
        
        # Gripper state
        self.robot_grasp_pos = self.robot.data.body_link_pos_w[:, self.gripper_frame_link_idx] + self.gripper_offset
        gripper_quat = self.robot.data.body_link_quat_w[:, self.gripper_frame_link_idx]
        
        # Object positions
        target_t = self.target_cube.data.body_com_pos_w.squeeze(1)  # (N, 3)
        pick_t = self.pick_cube.data.root_com_pos_w.squeeze(1)      # (N, 3)
        
        self.pick_pos = pick_t
        self.place_pos = target_t
        
        # Local position of pick cube in gripper frame
        # Local = R^T * (Pos_cube - Pos_gripper)
        rel_pos = pick_t - self.robot_grasp_pos
        # Invert the quaternion to transform from world to local frame
        # Isaac Sim quaternions are usually (w, x, y, z). Inverse is (w, -x, -y, -z)
        inv_gripper_quat = torch.cat([gripper_quat[:, :1], -gripper_quat[:, 1:]], dim=-1)
        pick_pos_local = tf_vector(inv_gripper_quat, rel_pos)
        
        # Distances for logic
        pick_dist = torch.norm(self.pick_pos - self.robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
        
        # Update 'picked' status: consider it picked if distance is small AND gripper is relatively closed
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_dof_idx[0]]
        norm_gripper = (self.gripper_upper_limit - gripper_pos) / (self.gripper_upper_limit - self.gripper_lower_limit)
        norm_gripper_2d = norm_gripper.unsqueeze(-1)  # (N,) -> (N, 1) to match pick_dist shape and avoid (N,N) broadcast

        # Update picked status: if we are close and gripper is closed, we mark as picked.
        self.picked = torch.where((pick_dist < 0.05) & (norm_gripper_2d > 0.2), 1.0, self.picked)
        # If the cube is too far from gripper, it's dropped
        self.picked = torch.where(pick_dist > 0.1, 0.0, self.picked)

        self.normalized_gripper_dist = norm_gripper.unsqueeze(dim=-1)
        
        # Vector from cube to target
        cube_to_target = self.place_pos - self.pick_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self.pick_pos,
                self.place_pos,
                cube_to_target,
                pick_pos_local,
                self.picked.unsqueeze(dim=-1) if self.picked.dim() == 1 else self.picked,
                self.normalized_gripper_dist,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # We calculate the magnitude of the force vector.
        # Shape of net_forces_w is (N, B, 3), we want (N, 1)
        left_finger_force = torch.norm(self.contact_left_finger.data.net_forces_w, dim=-1).max(dim=-1, keepdim=True)[0]
        right_finger_force = torch.norm(self.contact_right_finger.data.net_forces_w, dim=-1).max(dim=-1, keepdim=True)[0]

        pick_cube_vel = self.pick_cube.data.root_com_lin_vel_w.squeeze(1)      # (N, 3)

        # Latch "was lifted" once the cube is held more than 3 cm above the table.
        # The latch persists for the rest of the episode (cleared in _reset_idx).
        lift_height_now = (self.pick_pos[:, 2].unsqueeze(dim=-1) - 0.79).clamp(min=0.0)
        self.was_lifted = torch.maximum(
            self.was_lifted, ((self.picked > 0.5) & (lift_height_now > 0.03)).float()
        )

        total_reward, wandb_log = compute_rewards(
            # -- scales
            self.cfg.pick_reward_scale,
            self.cfg.place_reward_scale,
            self.cfg.place_fine_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.gripper_reward_scale,
            self.cfg.success_reward_scale,
            self.cfg.release_reward_scale,
            self.cfg.pick_moved_reward_scale,
            getattr(self.cfg, "action_penalty_scale", 0.001),
            # -- tensors
            self.robot_grasp_pos,
            self.pick_pos,
            self.place_pos,
            self.picked,
            self.was_lifted,
            self.normalized_gripper_dist,
            pick_cube_vel,
            self.actions,
            left_finger_force,
            right_finger_force,
        )
        
        wandb.log(wandb_log, step=self.common_step_counter)
        
        return total_reward.squeeze(-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Terminate if the cube drops off the table (TABLE_HEIGHT is ~0.78, so below 0.7 is off)
        cube_z = self.pick_cube.data.root_com_pos_w[:, 2]
        dropped = cube_z < 0.7
        
        died = dropped
        
        return died, time_out

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
        target_pos_noise = sample_uniform(-0.1, 0.1, (len(env_ids), 2), device=self.device) # type: ignore
        target_new_state[:, 0:2] += target_pos_noise

        # add environment origins to the position
        target_new_state[:, :3] += self.scene.env_origins[env_ids]

        # reset picked and the per-episode lift latch
        self.picked[env_ids] = 0.0
        self.was_lifted[env_ids] = 0.0

        # reset the pick to its default state
        pick_default_state = self.pick_cube.data.default_root_state[env_ids]
        pick_new_state = pick_default_state.clone()
        pick_pos_noise = sample_uniform(-0.1, 0.1, (len(env_ids), 2), device=self.device) # type: ignore
        pick_new_state[:, 0:2] += pick_pos_noise
        pick_new_state[:, :3] += self.scene.env_origins[env_ids]

        # Enforce a minimum xy separation between the pick cube and the target so
        # an episode never starts already "at target" (tolerance is 3 cm; use 8 cm
        # margin). Overlapping targets are pushed radially away from the pick cube.
        spawn_offset_xy = target_new_state[:, 0:2] - pick_new_state[:, 0:2]
        spawn_dist_xy = torch.norm(spawn_offset_xy, p=2, dim=-1, keepdim=True)
        # Fall back to +x for (nearly) coincident spawns to avoid division by zero
        fallback_dir = torch.zeros_like(spawn_offset_xy)
        fallback_dir[:, 0] = 1.0
        push_dir = torch.where(spawn_dist_xy > 1e-4, spawn_offset_xy / spawn_dist_xy.clamp(min=1e-6), fallback_dir)
        pushed_target_xy = pick_new_state[:, 0:2] + push_dir * 0.08
        target_new_state[:, 0:2] = torch.where(spawn_dist_xy < 0.08, pushed_target_xy, target_new_state[:, 0:2])

        self.target_cube.write_root_state_to_sim(target_new_state, env_ids)
        self.pick_cube.write_root_state_to_sim(pick_new_state, env_ids)


@torch.jit.script
def compute_rewards(
    # -- scales
    pick_reward_scale: float,
    place_reward_scale: float,
    place_fine_reward_scale: float,
    lift_reward_scale: float,
    gripper_reward_scale: float,
    success_reward_scale: float,
    release_reward_scale: float,
    pick_moved_reward_scale: float,
    action_penalty_scale: float,
    # -- tensors
    robot_grasp_pos: torch.Tensor,
    pick_pos: torch.Tensor,
    place_pos: torch.Tensor,
    picked: torch.Tensor,
    was_lifted: torch.Tensor,
    normalized_gripper_dist: torch.Tensor,
    pick_cube_vel: torch.Tensor,
    actions: torch.Tensor,
    left_finger_force: torch.Tensor,
    right_finger_force: torch.Tensor,
):
    """
    Computes staged rewards for the full pick-and-place-and-release task.

    Stages: 1. Reach Cube -> 2. Grasp -> 3. Lift & Carry -> 4. Descend at Target
            -> 5. Release (open gripper, cube resting at target).

    The reward landscape is designed so that "cube resting at the target with the
    gripper open" is the global maximum. Key properties:
      - Place rewards stay active after release (gated on `picked OR at_target`),
        so the policy is not punished for letting go of the cube at the target.
      - The lift reward fades to zero as the cube approaches the target in xy,
        so descending to the table-level target costs nothing.
      - A persistent success bonus plus a release bonus (open gripper at target)
        make completing the task strictly better than hovering with the cube.

    Args:
        pick_reward_scale:       Scale for the gripper-to-cube reaching reward.
        place_reward_scale:      Scale for the coarse cube-to-target reward.
        place_fine_reward_scale: Scale for the sharp near-target place reward.
        lift_reward_scale:       Scale for the carry-height reward.
        gripper_reward_scale:    Scale for gripper open/close/contact shaping.
        success_reward_scale:    Persistent bonus while the cube is at the target.
        release_reward_scale:    Bonus for opening the gripper while cube is at target.
        pick_moved_reward_scale: (Negative) scale penalizing cube motion when not picked.
        action_penalty_scale:    Scale penalizing action magnitude.
        robot_grasp_pos:         (N, 3) world position of the grasp frame.
        pick_pos:                (N, 3) world position of the pick cube.
        place_pos:               (N, 3) world position of the target location.
        picked:                  (N, 1) flag, 1.0 when the cube is held.
        was_lifted:              (N, 1) per-episode latch, 1.0 once the cube has been
                                 lifted off the table while held. Gates all place /
                                 success / release rewards so dragging earns nothing.
        normalized_gripper_dist: (N, 1) gripper closure in [0, 1]; 1 = closed, 0 = open.
        pick_cube_vel:           (N, 3) linear velocity of the pick cube.
        actions:                 (N, A) last applied actions.
        left_finger_force:       (N, 1) contact force magnitude on the left finger.
        right_finger_force:      (N, 1) contact force magnitude on the right finger.

    Returns:
        Tuple of (total_reward (N, 1) tensor, dict of mean metrics for wandb logging).
    """
    # -- distances used by several terms
    pick_dist = torch.norm(pick_pos - robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
    cube_to_target_dist = torch.norm(place_pos - pick_pos, p=2, dim=-1).unsqueeze(dim=-1)
    cube_to_target_xy = torch.norm(place_pos[:, :2] - pick_pos[:, :2], p=2, dim=-1).unsqueeze(dim=-1)
    # Cube counts as delivered when within 3 cm of the target (3D, target sits on the table)
    at_target = (cube_to_target_dist < 0.03).float()

    # 1. Reaching Reward: keep the gripper close to the cube at all times
    reach_reward = pick_reward_scale * torch.exp(-10.0 * pick_dist)

    # 2. Gripper & Contact Reward
    # Reward for being open before pick and closed with contact after pick
    left_contact = torch.clamp(left_finger_force, max=5.0) / 5.0
    right_contact = torch.clamp(right_finger_force, max=5.0) / 5.0
    contact_val = left_contact * right_contact  # Enforce both fingers touching

    # If far from cube, reward opening. If close to cube, reward closing.
    gripper_reward = torch.where(
        pick_dist < 0.05,
        gripper_reward_scale * normalized_gripper_dist,  # Reward closing when close
        gripper_reward_scale * (1.0 - normalized_gripper_dist),  # Reward opening when far
    )
    # If picked, reward contact
    gripper_reward = torch.where(
        picked > 0.5,
        gripper_reward_scale * contact_val,
        gripper_reward,
    )
    # Once the cube has been delivered (lifted there, now at the target), stop
    # rewarding closing/contact so the release bonus (below) can pull the gripper
    # open without competition. If the cube only got there by being pushed
    # (was_lifted == 0), keep the normal shaping so the policy still grasps it.
    delivered = (at_target > 0.5) & (was_lifted > 0.5)
    gripper_reward = torch.where(delivered, torch.zeros_like(gripper_reward), gripper_reward)

    # 3. Lift & Carry Reward
    # Reward height above the table while carrying (cube rests at z=0.79).
    cube_z = pick_pos[:, 2].unsqueeze(dim=-1)
    lift_height = (cube_z - 0.79).clamp(min=0.0)
    # A 5 cm carry height is enough to clear the table; don't reward pulling higher.
    capped_lift_height = lift_height.clamp(max=0.05)
    # Fade the lift reward out within 10 cm (xy) of the target so that descending
    # to the table-level target does not cost reward (fixes the hover-at-cap exploit).
    carry_fade = torch.clamp(cube_to_target_xy / 0.10, min=0.0, max=1.0)
    lift_reward = torch.where(
        picked > 0.5,
        lift_reward_scale * (1.0 - torch.exp(-20.0 * capped_lift_height)) * carry_fade,
        torch.zeros_like(lift_height),
    )

    # 4. Place Reward
    # Active while carrying, or once the cube has been delivered (at target after a
    # real carry) so that releasing the cube at the target does not zero the reward.
    place_active = (picked > 0.5) | ((at_target > 0.5) & (was_lifted > 0.5))
    # Continuous lift gate: zero while the cube slides on the table (no reward for
    # dragging/pushing), scaling up to full place reward at >= 3 cm carry height.
    # This product is the main gradient that teaches the policy to lift the cube.
    lift_gate = torch.clamp(lift_height / 0.03, min=0.0, max=1.0)
    # After a real carry (latch set), keep the gate fully open near the target so
    # descending to the table and releasing the cube does not lose reward.
    endgame = (cube_to_target_dist < 0.05) & (was_lifted > 0.5)
    lift_gate = torch.where(endgame, torch.ones_like(lift_gate), lift_gate)
    # Coarse term guides the carry; fine term gives a sharp gradient for the final cm.
    place_coarse = place_reward_scale * torch.exp(-5.0 * cube_to_target_dist) * lift_gate
    place_fine = place_fine_reward_scale * torch.exp(-25.0 * cube_to_target_dist) * lift_gate
    place_reward = torch.where(place_active, place_coarse + place_fine, torch.zeros_like(place_coarse))

    # 5. Success & Release Rewards
    # Persistent bonus while the cube sits at the target (independent of grasp state),
    # plus a bonus for opening the gripper there to encourage actually letting go.
    # Both require the cube to have been lifted this episode (no reward for pushing).
    success_reward = success_reward_scale * at_target * was_lifted
    release_reward = release_reward_scale * at_target * was_lifted * (1.0 - normalized_gripper_dist)

    # 6. Movement Penalty (Reduced impact, only if not picked to avoid kicking)
    cube_vel_norm = torch.norm(pick_cube_vel, p=2, dim=-1).unsqueeze(dim=-1)
    move_penalty = torch.where(picked < 0.5, pick_moved_reward_scale * cube_vel_norm, torch.zeros_like(cube_vel_norm))

    # 7. Smoothness Penalty (Action penalty)
    action_penalty = action_penalty_scale * torch.norm(actions, p=2, dim=-1).unsqueeze(dim=-1)

    total_reward = (
        reach_reward
        + gripper_reward
        + lift_reward
        + place_reward
        + success_reward
        + release_reward
        + move_penalty
        - action_penalty
    )

    wandb_log = {
        "reward/total_reward": total_reward.mean().item(),
        "reward/reach": reach_reward.mean().item(),
        "reward/gripper": gripper_reward.mean().item(),
        "reward/lift": lift_reward.mean().item(),
        "reward/place": place_reward.mean().item(),
        "reward/success": success_reward.mean().item(),
        "reward/release": release_reward.mean().item(),
        "reward/move_penalty": move_penalty.mean().item(),
        "reward/action_penalty": action_penalty.mean().item(),
        "state/pick_dist": pick_dist.mean().item(),
        "state/place_dist": cube_to_target_dist.mean().item(),
        "state/picked": picked.mean().item(),
        "state/was_lifted": was_lifted.mean().item(),
        "state/at_target": at_target.mean().item(),
        "state/gripper_dist": normalized_gripper_dist.mean().item(),
        "state/lift_height": lift_height.mean().item(),
        "state/contact_val": contact_val.mean().item(),
    }

    return total_reward, wandb_log
