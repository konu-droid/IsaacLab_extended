"""TiagoPRO right-arm mobile pick environment for Isaac Lab."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .pick_env_cfg import PickEnvCfg


class PickEnv(DirectRLEnv):
    """Pick a pill bottle off a table using the right arm + omnidirectional base."""

    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # --- Joint indices (resolved once) ---
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
        self.wheel_joint_ids = self._robot.find_joints("wheel_.*_joint")[0]

        # --- Body indices ---
        self.fingertip_left_id = self._robot.find_bodies("gripper_right_fingertip_left_link")[0][0]
        self.fingertip_right_id = self._robot.find_bodies("gripper_right_fingertip_right_link")[0][0]

        # Wrist camera — actual camera prim is not an articulation body,
        # so we use arm_right_tool_link + fixed rotation offset.
        # USD Camera convention: -Z forward, X right, Y up.
        # Offset from tool_link to camera: quat wxyz = [0, 0.707107, 0.707107, 0]
        self.wrist_camera_id = self._robot.find_bodies("arm_right_tool_link")[0][0]
        self.cam_offset_pos = torch.tensor([0.05, 0.0115, 0.03], device=self.device)
        self.cam_offset_quat = torch.tensor([0.0, 0.707107, 0.707107, 0.0], device=self.device)  # wxyz

        # Arm body indices for geometric collision check with table
        self._arm_body_ids = [
            self._robot.find_bodies(f"arm_right_{i}_link")[0][0] for i in range(1, 8)
        ]


        # --- Joint limits ---
        self.dof_lower = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.dof_upper = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # --- Action targets ---
        self.dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # --- Episode bookkeeping ---
        self.object_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )

        # --- Camera intrinsics for FOV projection (RealSense D455) ---
        # Horizontal FOV ~87°, image 640x480
        self.cam_hfov = math.radians(87.0)
        self.cam_vfov = math.radians(58.0)
        self.cam_img_h = 480

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

        # Split action vector: [0:7] arm_right, [7] gripper

        # --- Arm target (velocity-based, dt-independent) ---
        arm_delta = self.actions[:, 0:7] * self.cfg.action_scale * self.dt
        prev_arm = torch.stack([self.dof_targets[:, jid] for jid in self.arm_joint_ids], dim=-1)
        new_arm = prev_arm + arm_delta
        new_arm[:, 5] = new_arm[:, 5].clamp(-self.cfg.wrist_limit, self.cfg.wrist_limit)
        new_arm[:, 6] = new_arm[:, 6].clamp(-self.cfg.wrist_limit, self.cfg.wrist_limit)
        arm_lower = self.dof_lower[self.arm_joint_ids] + self.cfg.joint_limit_buffer
        arm_upper = self.dof_upper[self.arm_joint_ids] - self.cfg.joint_limit_buffer
        new_arm = torch.clamp(new_arm, arm_lower, arm_upper)
        for i, jid in enumerate(self.arm_joint_ids):
            self.dof_targets[:, jid] = new_arm[:, i]

        # --- Gripper (continuous accumulation with speed scale) ---
        gripper_delta = self.actions[:, 7] * self.cfg.gripper_speed_scale * self.cfg.action_scale * self.dt
        prev_gripper = self.dof_targets[:, self.gripper_joint_id]
        new_gripper = (prev_gripper + gripper_delta).clamp(self.cfg.gripper_open, self.cfg.gripper_close)
        self.dof_targets[:, self.gripper_joint_id] = new_gripper
        # Lock gripper sub-joints to 0
        for sid in self.gripper_sub_ids:
            self.dof_targets[:, sid] = 0.0

    def _apply_base_command(self, base_lin: torch.Tensor, base_ang: torch.Tensor):
        """Apply base vx, vy, omega via root body velocity.

        Sets linear and angular velocity on the articulation root directly.
        This works for omnidirectional movement and preserves physics
        so held objects move with the robot.
        """
        # Convert delta per step to velocity
        vx = base_lin[:, 0] / self.dt   # m/s
        vy = base_lin[:, 1] / self.dt   # m/s
        omega = base_ang.squeeze(-1) / self.dt  # rad/s

        # Transform local velocity to world frame using current yaw
        root_quat = self._robot.data.root_quat_w  # (N, 4) wxyz
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        vx_world = vx * torch.cos(yaw) - vy * torch.sin(yaw)
        vy_world = vx * torch.sin(yaw) + vy * torch.cos(yaw)

        # Set root velocity: [lin_vel(3), ang_vel(3)]
        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0] = vx_world
        root_vel[:, 1] = vy_world
        root_vel[:, 5] = omega  # angular velocity around Z

        # Only write velocity when non-zero to avoid overriding physics
        has_command = (root_vel.abs().sum(dim=-1) > 1e-4)
        if has_command.any():
            self._robot.write_root_velocity_to_sim(root_vel)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.dof_targets)

    # ------------------------------------------------------------------
    # Observations (16D)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        grasp_pos = self._get_grasp_point()
        target_pos = self._pill_bottle.data.root_pos_w
        dist_vec = target_pos - grasp_pos  # (N, 3)

        arm_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]  # (N, 7)
        torso_pos = self._robot.data.joint_pos[:, self.torso_joint_id].unsqueeze(-1)  # (N, 1)
        gripper_pos = self._robot.data.joint_pos[:, self.gripper_joint_id].unsqueeze(-1)  # (N, 1)

        # Base pose (world X, Y, yaw)
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        base_pose = torch.stack([root_pos[:, 0], root_pos[:, 1], yaw], dim=-1)  # (N, 3)

        # Table height (Z of table root)
        table_z = self._table.data.root_pos_w[:, 2:3]  # (N, 1)

        obs = torch.cat([
            dist_vec,       # 3
            arm_pos,        # 7
            torso_pos,      # 1
            gripper_pos,    # 1
            base_pose,      # 3
            table_z,        # 1
        ], dim=-1)  # total = 16

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        grasp_pos = self._get_grasp_point()
        target_pos = self._pill_bottle.data.root_pos_w
        dist = torch.norm(target_pos - grasp_pos, p=2, dim=-1)

        # 1. Distance reward — closer is better
        dist_reward = torch.exp(-2.0 * dist)

        # 2. Lift success — object Z above initial + threshold
        object_z = self._pill_bottle.data.root_pos_w[:, 2]
        lifted = object_z > (self.object_init_pos[:, 2] + self.cfg.lift_height)
        lift_reward = lifted.float()

        # 3. Object movement penalty (XY only, before lift)
        obj_xy = self._pill_bottle.data.root_pos_w[:, :2]
        init_xy = self.object_init_pos[:, :2]
        obj_moved = torch.norm(obj_xy - init_xy, p=2, dim=-1)
        not_lifted = ~lifted
        object_move_penalty = (
            (obj_moved > self.cfg.object_move_threshold) & not_lifted
        ).float()

        # 4. FOV center reward — target closer to camera center
        fov_center_rew = self._compute_fov_center_reward(target_pos)

        # 5. Time penalty
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # 6. Collision penalty — contact sensor on arm
        net_forces = self._contact_sensor.data.net_forces_w_history
        arm_contact = torch.max(
            torch.norm(net_forces[:, :, self._arm_contact_ids], dim=-1), dim=1
        )[0]
        collision_penalty = torch.any(arm_contact > 1.0, dim=1).float()

        # 7. FOV lost penalty — penalize when target leaves camera view
        fov_lost = self._check_fov_lost(target_pos)
        fov_lost_penalty = fov_lost.float()

        # 8. Action rate penalty — penalize change from previous action
        action_diff = self.actions - self.prev_actions
        action_rate_penalty = torch.sum(action_diff ** 2, dim=-1)

        # 9. Arm-far penalty — penalize arm movement when grasp > 0.5m
        arm_actions = self.actions[:, 0:7]
        arm_action_magnitude = torch.sum(arm_actions ** 2, dim=-1)
        arm_far_penalty = (dist > self.cfg.arm_far_threshold).float() * arm_action_magnitude

        # Combine rewards
        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.lift_reward_scale * lift_reward
            + self.cfg.fov_center_reward_scale * fov_center_rew
            - self.cfg.time_penalty_scale * time_penalty
            - self.cfg.collision_penalty_scale * collision_penalty
            - self.cfg.action_rate_penalty_scale * action_rate_penalty
            - self.cfg.arm_far_penalty_scale * arm_far_penalty
            - self.cfg.object_move_penalty_scale * object_move_penalty
            - self.cfg.fov_lost_penalty_scale * fov_lost_penalty
        )

        # Store for logging
        self.extras["log"] = {
            "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            "lift_reward": (self.cfg.lift_reward_scale * lift_reward).mean(),
            "fov_center_reward": (self.cfg.fov_center_reward_scale * fov_center_rew).mean(),
            "action_rate_penalty": (-self.cfg.action_rate_penalty_scale * action_rate_penalty).mean(),
            "arm_far_penalty": (-self.cfg.arm_far_penalty_scale * arm_far_penalty).mean(),
            "object_move_penalty": (-self.cfg.object_move_penalty_scale * object_move_penalty).mean(),
            "fov_lost_penalty": (-self.cfg.fov_lost_penalty_scale * fov_lost_penalty).mean(),
            "mean_dist": dist.mean(),
            "lift_success_rate": lift_reward.mean(),
        }

        # Update previous actions
        self.prev_actions[:] = self.actions

        return rewards

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_pos = self._get_grasp_point()
        target_pos = self._pill_bottle.data.root_pos_w

        # --- Success: object lifted ---
        object_z = self._pill_bottle.data.root_pos_w[:, 2]
        lifted = object_z > (self.object_init_pos[:, 2] + self.cfg.lift_height)
        success = lifted

        # --- Failure: object moved too much (before lift) ---
        obj_xy = self._pill_bottle.data.root_pos_w[:, :2]
        init_xy = self.object_init_pos[:, :2]
        obj_moved = torch.norm(obj_xy - init_xy, p=2, dim=-1)
        object_pushed = (obj_moved > self.cfg.object_move_threshold) & ~lifted

        # --- Failure: arm collision (contact sensor) ---
        net_forces = self._contact_sensor.data.net_forces_w_history
        arm_force = torch.max(
            torch.norm(net_forces[:, :, self._arm_contact_ids], dim=-1), dim=1
        )[0]
        arm_collision = torch.any(arm_force > 1.0, dim=1)

        # --- Failure: base too close to table (geometric, base_link lacks direct CollisionAPI) ---
        root_pos = self._robot.data.root_pos_w
        table_pos = self._table.data.root_pos_w
        base_table_dist = torch.norm(root_pos[:, :2] - table_pos[:, :2], p=2, dim=-1)
        base_collision = base_table_dist < 0.6

        # --- Failure: arm joint velocity too high (post-collision flailing) ---
        arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]
        arm_flailing = (arm_vel.abs() > self.cfg.arm_joint_vel_limit).any(dim=-1)

        # --- Combined termination (FOV is penalty only, not termination) ---
        terminated = success | object_pushed | arm_collision | arm_flailing | base_collision

        # --- Truncated: timeout ---
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # --- Robot joint reset with randomization ---
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # Randomize arm joints ±0.175 rad (±10°)
        arm_noise = sample_uniform(-0.175, 0.175, (n, len(self.arm_joint_ids)), self.device)
        for i, jid in enumerate(self.arm_joint_ids):
            joint_pos[:, jid] += arm_noise[:, i]
        joint_pos = torch.clamp(joint_pos, self.dof_lower, self.dof_upper)
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- Table height randomization ---
        env_origins = self.scene.env_origins[env_ids]
        table_state = self._table.data.default_root_state[env_ids].clone()
        table_state[:, 0] = env_origins[:, 0] + 1.5   # table X = 1.5m forward from env origin
        table_state[:, 1] = env_origins[:, 1]
        table_z_offset = sample_uniform(-0.1, 0.1, (n,), self.device)
        table_state[:, 2] = 0.406 + table_z_offset  # cuboid center height
        self._table.write_root_state_to_sim(table_state, env_ids=env_ids)
        table_pos = table_state[:, :3]

        # --- Object position randomization (before robot, so we can aim at it) ---
        root_pos = env_origins.clone()
        root_pos[:, 0] += sample_uniform(-0.2, 0.2, (n,), self.device)
        root_pos[:, 1] += sample_uniform(-0.2, 0.2, (n,), self.device)

        obj_pos = env_origins.clone()
        obj_pos[:, 0] += sample_uniform(1.34, 1.66, (n,), self.device) + (root_pos[:, 0] - env_origins[:, 0])
        obj_pos[:, 1] += sample_uniform(-0.41, 0.41, (n,), self.device) + (root_pos[:, 1] - env_origins[:, 1])
        obj_pos[:, 2] = table_pos[:, 2] + 0.406 + 0.04  # table top (center + half height) + half cylinder

        obj_state = self._pill_bottle.data.default_root_state[env_ids].clone()
        obj_state[:, :3] = obj_pos
        self._pill_bottle.write_root_state_to_sim(obj_state, env_ids=env_ids)
        self.object_init_pos[env_ids] = obj_pos

        # --- Robot root pose: face the target + small noise ---
        dx = obj_pos[:, 0] - root_pos[:, 0]
        dy = obj_pos[:, 1] - root_pos[:, 1]
        target_yaw = torch.atan2(dy, dx)
        yaw_noise = sample_uniform(-math.radians(5), math.radians(5), (n,), self.device)
        yaw = target_yaw + yaw_noise

        half_yaw = yaw * 0.5
        root_quat = torch.zeros((n, 4), device=self.device)
        root_quat[:, 0] = torch.cos(half_yaw)
        root_quat[:, 3] = torch.sin(half_yaw)
        root_vel = torch.zeros((n, 6), device=self.device)

        self._robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_vel], dim=-1),
            env_ids=env_ids,
        )

        # Reset action history
        self.prev_actions[env_ids] = 0.0

        # Reset targets
        self.dof_targets[env_ids] = joint_pos

    # ------------------------------------------------------------------
    # Helper: grasp point (average of two fingertips)
    # ------------------------------------------------------------------
    def _get_grasp_point(self) -> torch.Tensor:
        left_pos = self._robot.data.body_pos_w[:, self.fingertip_left_id]
        right_pos = self._robot.data.body_pos_w[:, self.fingertip_right_id]
        return (left_pos + right_pos) * 0.5

    # ------------------------------------------------------------------
    # Helper: geometric arm-table collision check
    # ------------------------------------------------------------------
    def _check_arm_table_collision(self) -> torch.Tensor:
        """Check if any arm link position is inside the table bounding box.

        Table is a cuboid: size (0.6, 0.9, 0.812), so half-extents (0.3, 0.45, 0.406).
        Returns True if any arm link is inside.
        """
        table_pos = self._table.data.root_pos_w  # (N, 3) center of cuboid
        half_x, half_y, half_z = 0.3, 0.45, 0.406

        any_inside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for body_id in self._arm_body_ids:
            arm_pos = self._robot.data.body_pos_w[:, body_id]  # (N, 3)
            dx = (arm_pos[:, 0] - table_pos[:, 0]).abs()
            dy = (arm_pos[:, 1] - table_pos[:, 1]).abs()
            dz = (arm_pos[:, 2] - table_pos[:, 2]).abs()

            inside = (dx < half_x) & (dy < half_y) & (dz < half_z)
            any_inside = any_inside | inside

        return any_inside

    # ------------------------------------------------------------------
    # Helper: camera pose from tool_link + fixed offset
    # ------------------------------------------------------------------
    def _get_camera_pose(self):
        """Get camera world pose from tool_link + fixed offset."""
        tool_pos = self._robot.data.body_pos_w[:, self.wrist_camera_id]
        tool_quat = self._robot.data.body_quat_w[:, self.wrist_camera_id]
        cam_pos = tool_pos + self._quat_rotate(
            tool_quat, self.cam_offset_pos.unsqueeze(0).expand(self.num_envs, -1)
        )
        cam_quat = self._quat_multiply(
            tool_quat, self.cam_offset_quat.unsqueeze(0).expand(self.num_envs, -1)
        )
        return cam_pos, cam_quat

    # ------------------------------------------------------------------
    # Helper: FOV center reward
    # ------------------------------------------------------------------
    def _compute_fov_center_reward(self, target_pos: torch.Tensor) -> torch.Tensor:
        """Small reward when target is near the center of wrist camera FOV.

        USD Camera convention: -Z = optical axis, X = right, Y = up.
        """
        cam_pos, cam_quat = self._get_camera_pose()

        to_target = target_pos - cam_pos
        local_vec = self._quat_rotate_inverse(cam_quat, to_target)

        # -Z is optical axis; project onto image plane
        depth = -local_vec[:, 2]  # positive when in front
        px = local_vec[:, 0] / (depth + 1e-6)   # image X (right +)
        py = -local_vec[:, 1] / (depth + 1e-6)  # image Y (down +, flip Y since USD Y=up)

        ang_x = torch.atan(px)
        ang_y = torch.atan(py)

        norm_x = 1.0 - (ang_x.abs() / (self.cam_hfov * 0.5)).clamp(0.0, 1.0)
        norm_y = 1.0 - (ang_y.abs() / (self.cam_vfov * 0.5)).clamp(0.0, 1.0)

        in_front = (depth > 0.01).float()
        return norm_x * norm_y * in_front

    # ------------------------------------------------------------------
    # Helper: FOV visibility check (5-point spine projection)
    # ------------------------------------------------------------------
    def _check_fov_lost(self, target_pos: torch.Tensor) -> torch.Tensor:
        """Check if bbox spine points 2,3,4 are all outside camera upper 80%.

        Returns True (lost) if none of points 2,3,4 are visible.
        """
        cam_pos, cam_quat = self._get_camera_pose()

        obj_height = 0.12
        obj_center = target_pos.clone()

        any_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        vfov_half_tan = math.tan(self.cam_vfov * 0.5)
        hfov_half_tan = math.tan(self.cam_hfov * 0.5)

        for frac in [0.25, 0.5, 0.75]:  # points 2, 3, 4
            point = obj_center.clone()
            point[:, 2] += obj_height * (frac - 0.5)

            to_point = point - cam_pos
            local = self._quat_rotate_inverse(cam_quat, to_point)

            # USD Camera: -Z is optical axis
            depth = -local[:, 2]
            in_front = depth > 0.01

            # Project to image plane
            px = local[:, 0] / (depth + 1e-6)    # image X (right +)
            py = -local[:, 1] / (depth + 1e-6)   # image Y (down +, flip USD Y=up)

            in_hfov = px.abs() < hfov_half_tan
            in_vfov = py.abs() < vfov_half_tan

            # Upper 80%: gripper at bottom (py large positive)
            gripper_threshold = vfov_half_tan * (2.0 * self.cfg.fov_visible_fraction - 1.0)
            not_in_gripper_zone = py < gripper_threshold

            visible = in_front & in_hfov & in_vfov & not_in_gripper_zone
            any_visible = any_visible | visible

        return ~any_visible

    # ------------------------------------------------------------------
    # Helper: inverse quaternion rotation
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by inverse of quaternion q (wxyz format)."""
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
        return a - b + c

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q (wxyz format)."""
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
        return a + b + c

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (wxyz format): q1 * q2."""
        w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
        w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]
        return torch.cat([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)
