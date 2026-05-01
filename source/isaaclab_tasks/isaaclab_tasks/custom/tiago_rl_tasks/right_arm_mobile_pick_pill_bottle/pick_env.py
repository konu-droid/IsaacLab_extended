"""TiagoPRO right-arm mobile pick environment for Isaac Lab.
EE delta action space with Differential IK. Observation 5D, Action 7D.
"""

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
# IMPORT FIX: Added combine_frame_transforms
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, combine_frame_transforms, quat_from_euler_xyz, quat_mul
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from .pick_env_cfg import PickEnvCfg


class PickEnv(DirectRLEnv):
    """Pick a pill bottle using EE delta actions + Differential IK."""

    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # --- Joint indices ---
        self.arm_joint_ids = self._robot.find_joints("arm_right_[1-7]_joint")[0]
        self.torso_joint_id = self._robot.find_joints("torso_lift_joint")[0][0]
        self.gripper_joint_id = self._robot.find_joints("gripper_right_finger_joint")[0][0]
        self.gripper_sub_ids = [
            self._robot.find_joints(n)[0][0] for n in [
                "gripper_right_inner_finger_left_joint",
                "gripper_right_inner_finger_right_joint",
                "gripper_right_outer_finger_right_joint",
                "gripper_right_fingertip_left_joint",
                "gripper_right_fingertip_right_joint",
            ]
        ]

        # --- Body indices ---
        self.fingertip_left_id = self._robot.find_bodies("gripper_right_fingertip_left_link")[0][0]
        self.fingertip_right_id = self._robot.find_bodies("gripper_right_fingertip_right_link")[0][0]
        self.ee_body_id = self._robot.find_bodies("arm_right_tool_link")[0][0]

        # --- Differential IK Controller ---
        # We control both the torso and arm joints to reach the EE target (same cspace as before)
        self._ik_joint_ids = [self.torso_joint_id] + list(self.arm_joint_ids)
        # Fix: IsaacLab expects "pose" for command_type (use_relative_mode=False is default for absolute control)
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)

        # --- Joint limits ---
        self.dof_lower = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.dof_upper = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # --- Action targets ---
        self.dof_targets = self._robot.data.joint_pos.clone()

        # --- Episode bookkeeping ---
        self.object_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._pill_bottle = RigidObject(self.cfg.pill_bottle)
        self._table = RigidObject(self.cfg.table)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["pill_bottle"] = self._pill_bottle
        self.scene.rigid_objects["table"] = self._table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Pre-physics: process actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Action: [0:3] ee_pos_delta, [3:6] ee_rot_delta, [6] gripper
        ee_pos_actions = self.actions[:, 0:3] * self.cfg.ee_action_scale
        ee_rot_actions = self.actions[:, 3:6] * self.cfg.ee_rot_action_scale
        gripper_actions = self.actions[:, 6]

        # Get current EE and Root pose in World frame
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, self.ee_body_id]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w

        # Transform EE pose to base frame.
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # --- Apply Position Actions ---
        target_pos_b = ee_pos_b + ee_pos_actions
        # --- Apply Rotation Actions ---
        rot_delta_quat = quat_from_euler_xyz(ee_rot_actions[:, 0], ee_rot_actions[:, 1], ee_rot_actions[:, 2])
        target_quat_b = quat_mul(rot_delta_quat, ee_quat_b)

        # Formulate absolute pose command for IK controller
        ee_pose_commands = torch.cat([target_pos_b, target_quat_b], dim=1)
        self.ik_controller.set_command(ee_pose_commands, ee_pos_b, ee_quat_b)

        # --- Get Jacobian ---
        ik_jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_body_id - 1, :, self._ik_joint_ids]
        ik_joint_pos = self._robot.data.joint_pos[:, self._ik_joint_ids]

        # Compute desired joint positions
        desired_ik_joint_pos = self.ik_controller.compute(ee_pos_b, ee_quat_b, ik_jacobian, ik_joint_pos)

        # Apply to targets
        self.dof_targets[:, self._ik_joint_ids] = desired_ik_joint_pos

        # --- Gripper ---
        gripper_delta = gripper_actions * 0.1 * 7.5 * self.dt
        prev_gripper = self.dof_targets[:, self.gripper_joint_id]
        new_gripper = (prev_gripper + gripper_delta).clamp(
            self.cfg.gripper_open, self.cfg.gripper_close
        )
        self.dof_targets[:, self.gripper_joint_id] = new_gripper

    def _apply_action(self):
        self._robot.set_joint_position_target(self.dof_targets)

    # ------------------------------------------------------------------
    # Observations (5D)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        grasp_pos = self._get_grasp_point()
        target_pos = self._pill_bottle.data.root_pos_w
        target_dist = target_pos - grasp_pos  # (N, 3) xyz distance

        torso_pos = self._robot.data.joint_pos[:, self.torso_joint_id].unsqueeze(-1)
        gripper_pos = self._robot.data.joint_pos[:, self.gripper_joint_id].unsqueeze(-1)

        obs = torch.cat([
            torso_pos,      # 1
            gripper_pos,    # 1
            target_dist,    # 3
        ], dim=-1)  # total = 5

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        grasp_pos = self._get_grasp_point()
        target_pos = self._pill_bottle.data.root_pos_w
        dist = torch.norm(target_pos - grasp_pos, p=2, dim=-1)

        # 1. Distance reward
        dist_reward = torch.exp(-2.0 * dist)

        # 2. Lift success
        object_z = self._pill_bottle.data.root_pos_w[:, 2]
        lifted = object_z > (self.object_init_pos[:, 2] + self.cfg.lift_height)
        lift_reward = lifted.float()

        # 3. Object movement penalty
        obj_xy = self._pill_bottle.data.root_pos_w[:, :2]
        init_xy = self.object_init_pos[:, :2]
        obj_moved = torch.norm(obj_xy - init_xy, p=2, dim=-1)
        not_lifted = ~lifted
        object_move_penalty = (
            (obj_moved > self.cfg.object_move_threshold) & not_lifted
        ).float()

        # 4. Time penalty
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # 5. Collision penalty (joint velocity spike)
        arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]
        collision_penalty = (arm_vel.abs() > self.cfg.arm_joint_vel_limit).any(dim=-1).float()

        # 6. Action rate penalty
        action_diff = self.actions - self.prev_actions
        action_rate_penalty = torch.sum(action_diff ** 2, dim=-1)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.lift_reward_scale * lift_reward
            - self.cfg.time_penalty_scale * time_penalty
            - self.cfg.collision_penalty_scale * collision_penalty
            - self.cfg.action_rate_penalty_scale * action_rate_penalty
            - self.cfg.object_move_penalty_scale * object_move_penalty
        )

        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            "lift_reward": (self.cfg.lift_reward_scale * lift_reward).mean(),
            "action_rate_penalty": (-self.cfg.action_rate_penalty_scale * action_rate_penalty).mean(),
            "object_move_penalty": (-self.cfg.object_move_penalty_scale * object_move_penalty).mean(),
            "mean_dist": dist.mean(),
            "lift_success_rate": lift_reward.mean(),
        }

        self.prev_actions[:] = self.actions
        return rewards

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Success: object lifted
        object_z = self._pill_bottle.data.root_pos_w[:, 2]
        lifted = object_z > (self.object_init_pos[:, 2] + self.cfg.lift_height)
        success = lifted

        # Failure: object pushed
        obj_xy = self._pill_bottle.data.root_pos_w[:, :2]
        init_xy = self.object_init_pos[:, :2]
        obj_moved = torch.norm(obj_xy - init_xy, p=2, dim=-1)
        object_pushed = (obj_moved > self.cfg.object_move_threshold) & ~lifted

        # Failure: arm flailing
        arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]
        arm_flailing = (arm_vel.abs() > self.cfg.arm_joint_vel_limit).any(dim=-1)

        # Failure: fell over
        root_z = self._robot.data.root_pos_w[:, 2]
        root_quat = self._robot.data.root_quat_w
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        fell_over = (root_z < -0.2) | (up_z < 0.7)

        # Failure: base too close to table
        root_pos = self._robot.data.root_pos_w
        table_pos = self._table.data.root_pos_w
        base_table_dist = torch.norm(root_pos[:, :2] - table_pos[:, :2], p=2, dim=-1)
        base_collision = base_table_dist < 0.6

        terminated = success | object_pushed | arm_flailing | fell_over | base_collision
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Robot joint reset
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # Randomize torso
        torso_vals = torch.tensor([0.175, 0.263, 0.35], device=self.device)
        rand_torso = torso_vals[torch.randint(0, 3, (n,), device=self.device)]
        joint_pos[:, self.torso_joint_id] = rand_torso

        # Small noise on arm joints
        arm_noise = sample_uniform(-0.087, 0.087, (n, len(self.arm_joint_ids)), self.device)
        for i, jid in enumerate(self.arm_joint_ids):
            joint_pos[:, jid] += arm_noise[:, i]
        joint_pos = torch.clamp(joint_pos, self.dof_lower, self.dof_upper)
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Table at 0.8m
        env_origins = self.scene.env_origins[env_ids]
        table_state = self._table.data.default_root_state[env_ids].clone()
        table_state[:, 0] = env_origins[:, 0] + 0.8
        table_state[:, 1] = env_origins[:, 1]
        table_z_offset = sample_uniform(-0.1, 0.1, (n,), self.device)
        table_state[:, 2] = 0.406 + table_z_offset
        self._table.write_root_state_to_sim(table_state, env_ids=env_ids)
        table_pos = table_state[:, :3]

        # Object on table
        obj_pos = env_origins.clone()
        obj_pos[:, 0] = table_pos[:, 0] + sample_uniform(-0.15, 0.15, (n,), self.device)
        obj_pos[:, 1] = table_pos[:, 1] + sample_uniform(-0.25, 0.25, (n,), self.device)
        obj_pos[:, 2] = table_pos[:, 2] + 0.406 + 0.04

        obj_state = self._pill_bottle.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos
        self._pill_bottle.write_root_state_to_sim(obj_state, env_ids=env_ids)
        self.object_init_pos[env_ids] = obj_pos

        # Robot at origin, facing forward
        root_pos = env_origins.clone()
        root_quat = torch.zeros((n, 4), device=self.device)
        root_quat[:, 0] = 1.0
        root_vel = torch.zeros((n, 6), device=self.device)
        self._robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_vel], dim=-1),
            env_ids=env_ids,
        )

        self.prev_actions[env_ids] = 0.0
        self.dof_targets[env_ids] = joint_pos

    # ------------------------------------------------------------------
    # Helper: grasp point
    # ------------------------------------------------------------------
    def _get_grasp_point(self) -> torch.Tensor:
        left_pos = self._robot.data.body_pos_w[:, self.fingertip_left_id]
        right_pos = self._robot.data.body_pos_w[:, self.fingertip_right_id]
        return (left_pos + right_pos) * 0.5