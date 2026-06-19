# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 cube-grasp task driven by an 8x8 Time-of-Flight sensor.

This environment derives from the validated joint-space pick-and-place task
(:mod:`lerobot_cube_move_env`) and adds a simulated **Pololu VL53L7CX** multizone
Time-of-Flight ranging sensor on top of the gripper's fixed jaw. The sensor is
emulated with an RTX :class:`TiledCamera` whose depth image is min-pooled into an
8x8 grid of per-zone distances (matching the real device's output).

Differences from the base task:
  * The observation drops ``pick_pos_local`` and the ``picked`` heuristic flag.
  * The 64 ToF zone distances plus 3 derived ToF features (central distance,
    detection-centroid offset, detect flag) are appended to the observation.
  * New ToF-driven reward terms encourage the policy to centre the cube in the
    sensor field of view and close the gripper once the cube is in grasp range,
    which naturally aligns the fixed jaw for a proper grip. The grasp/lift/place
    machinery is gated on a reward-side geometric ``is_grasped`` signal (cube
    close + gripper closed) instead of the removed ``picked`` flag, and a dense
    open/close gripper term keeps the jaws open on empty air to avoid the fingers
    closing through each other.
"""

from __future__ import annotations

import torch
import wandb
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor, TiledCamera

from .lerobot_cube_tof_env_cfg import LerobotCubeToFEnvCfg
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


class LerobotCubeToFEnv(DirectRLEnv):
    cfg: LerobotCubeToFEnvCfg

    def __init__(self, cfg: LerobotCubeToFEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # joint limits / speed scaling
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.robot.find_joints("gripper")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # grasp frame: a small downward offset from the gripper_link origin
        self.gripper_offset = torch.tensor([0.0, 0.0, -0.01]).to(self.device)
        self.gripper_frame_link_idx = self.robot.find_bodies("gripper_link")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pick_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Per-episode latch: 1.0 once the cube has been held AND lifted off the table.
        # Used only for reward gating (not part of the observation) so that place /
        # success rewards cannot be earned by dragging the cube along the table.
        self.was_lifted = torch.zeros((self.num_envs, 1), device=self.device)

        self.gripper_dof_idx = self.robot.find_joints("gripper")[0]
        # gripper closure normalised to [0, 1] (1 = closed)
        self.normalized_gripper_dist = torch.zeros((self.num_envs, 1), device=self.device)
        self.gripper_lower_limit = self.robot_dof_lower_limits[self.gripper_dof_idx[0]]
        self.gripper_upper_limit = self.robot_dof_upper_limits[self.gripper_dof_idx[0]]

        # -- ToF (8x8) buffers and pooling helpers --
        self.tof_zones = self.cfg.tof_zones
        self.tof_upscale = self.cfg.tof_render_upscale
        self.tof_max_range = self.cfg.tof_max_range
        # normalised zone distances exposed to the policy (1.0 = nothing in range)
        self.tof_grid = torch.ones((self.num_envs, self.tof_zones, self.tof_zones), device=self.device)

        # Initialize wandb (separate project so ToF runs are easy to find)
        wandb.init(
            project="LerobotToF",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
                "tof_zones": cfg.tof_zones,
            },
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target_cube = RigidObject(self.cfg.target_cube_cfg)
        self.pick_cube = RigidObject(self.cfg.pick_cube_cfg)
        self.contact_left_finger = ContactSensor(self.cfg.contact_sensor_left_finger)
        self.contact_right_finger = ContactSensor(self.cfg.contact_sensor_right_finger)
        self.tof_camera = TiledCamera(self.cfg.tof_camera)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target_cube"] = self.target_cube
        self.scene.rigid_objects["pick_cube"] = self.pick_cube
        self.scene.sensors["contact_sensor_left_finger"] = self.contact_left_finger
        self.scene.sensors["contact_sensor_right_finger"] = self.contact_right_finger
        self.scene.sensors["tof_camera"] = self.tof_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
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

    def _read_tof_grid(self) -> torch.Tensor:
        """
        Read the RTX depth image and pool it into the 8x8 ToF zone grid.

        The render is min-pooled per zone so each zone reports the *nearest* surface
        in its solid angle, mirroring how the VL53L7CX reports the closest return per
        zone. Non-finite pixels (no hit) are treated as the max range.

        Returns:
            (N, tof_zones, tof_zones) tensor of zone distances in metres, clipped to
            ``[0, tof_max_range]``.
        """
        # distance_to_camera: (N, H, W, 1) radial distance from the optical centre
        depth = self.tof_camera.data.output["distance_to_camera"]
        depth = depth.squeeze(-1)  # (N, H, W)
        # sanitise: inf/nan -> max range (out-of-range / no return)
        depth = torch.nan_to_num(depth, nan=self.tof_max_range, posinf=self.tof_max_range, neginf=self.tof_max_range)
        depth = depth.clamp(min=0.0, max=self.tof_max_range)

        # min-pool render pixels into zones: (N, Z, up, Z, up) -> min over the up dims
        n = depth.shape[0]
        z, up = self.tof_zones, self.tof_upscale
        depth = depth.view(n, z, up, z, up)
        zone_dist = depth.amin(dim=(2, 4))  # (N, Z, Z)
        return zone_dist

    def _compute_tof_features(self, zone_dist: torch.Tensor):
        """
        Derive grasp-relevant scalar features from the 8x8 zone grid.

        Only central zones are used: the outer ring sees the gripper's own jaw edges
        sitting at the near-clip plane, so a whole-grid centroid would track the jaw,
        not the cube. The central 2x2 reading is the robust workhorse signal -- with
        the sensor aimed straight down the grasp axis it is small only when the cube
        is directly under the jaw (aligned) AND close (in range).

        Args:
            zone_dist: (N, Z, Z) per-zone distances in metres.

        Returns:
            Tuple of:
              * ``tof_center_dist`` (N, 1): nearest reading in the central 2x2 zones.
              * ``tof_centroid_offset`` (N, 1): normalised offset of the closeness-
                weighted detection centroid within the central 4x4 region; logged for
                monitoring only (not used in the reward).
              * ``tof_detect`` (N, 1): 1.0 if any central-4x4 zone sees an object
                nearer than the detection threshold; logged for monitoring only.
        """
        z = self.tof_zones
        # central 2x2 = straight ahead of the jaw (cleanest, jaw-free)
        c2 = z // 2 - 1
        center_dist = zone_dist[:, c2 : c2 + 2, c2 : c2 + 2].amin(dim=(1, 2))  # (N,)

        # central 4x4 sub-grid for the (logging-only) detection centroid
        c4 = z // 2 - 2
        sub = zone_dist[:, c4 : c4 + 4, c4 : c4 + 4]  # (N, 4, 4)
        detect_dist = self.cfg.tof_detect_dist
        weights = torch.clamp(detect_dist - sub, min=0.0)  # (N, 4, 4)
        weight_sum = weights.sum(dim=(1, 2))  # (N,)
        detect = (weight_sum > 0.0).float()

        idx = torch.arange(4, device=zone_dist.device, dtype=torch.float32)
        rows = idx.view(1, 4, 1)
        cols = idx.view(1, 1, 4)
        safe_sum = weight_sum.clamp(min=1e-6)
        centroid_row = (weights * rows).sum(dim=(1, 2)) / safe_sum
        centroid_col = (weights * cols).sum(dim=(1, 2)) / safe_sum
        offset = torch.sqrt(((centroid_row - 1.5) / 1.5) ** 2 + ((centroid_col - 1.5) / 1.5) ** 2)
        centroid_offset = torch.where(detect > 0.5, offset, torch.ones_like(offset))

        return center_dist.unsqueeze(-1), centroid_offset.unsqueeze(-1), detect.unsqueeze(-1)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        # grasp frame pose
        self.robot_grasp_pos = self.robot.data.body_link_pos_w[:, self.gripper_frame_link_idx] + self.gripper_offset

        # object positions
        target_t = self.target_cube.data.body_com_pos_w.squeeze(1)  # (N, 3)
        pick_t = self.pick_cube.data.root_com_pos_w.squeeze(1)  # (N, 3)
        self.pick_pos = pick_t
        self.place_pos = target_t

        # gripper closure in [0, 1] (1 = closed)
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_dof_idx[0]]
        norm_gripper = (self.gripper_upper_limit - gripper_pos) / (self.gripper_upper_limit - self.gripper_lower_limit)
        self.normalized_gripper_dist = norm_gripper.unsqueeze(dim=-1)

        # ToF zone grid (metres) and its normalised form for the policy
        self.tof_grid = self._read_tof_grid()
        tof_obs = (self.tof_grid / self.tof_max_range).reshape(self.num_envs, -1)  # (N, Z*Z) in [0, 1]

        # Derived ToF grasp features. Computed here (before the reward) and cached so
        # both the observation and the reward use the same readings. Exposing these
        # summarised scalars alongside the raw grid gives the MLP a direct "is the
        # cube centred and in range?" signal, which it needs to learn to align and
        # close the jaw on the cube rather than just hover in front of it.
        self.tof_center_dist, self.tof_centroid_offset, self.tof_detect = self._compute_tof_features(self.tof_grid)
        tof_features = torch.cat(
            (self.tof_center_dist / self.tof_max_range, self.tof_centroid_offset, self.tof_detect), dim=-1
        )  # (N, 3)

        # vector from cube to target
        cube_to_target = self.place_pos - self.pick_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self.pick_pos,
                self.place_pos,
                cube_to_target,
                self.normalized_gripper_dist,
                tof_obs,
                tof_features,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # contact force magnitudes per finger -> normalised contact value
        left_finger_force = torch.norm(self.contact_left_finger.data.net_forces_w, dim=-1).max(dim=-1, keepdim=True)[0]
        right_finger_force = torch.norm(self.contact_right_finger.data.net_forces_w, dim=-1).max(dim=-1, keepdim=True)[0]

        pick_cube_vel = self.pick_cube.data.root_com_lin_vel_w.squeeze(1)  # (N, 3)

        # ToF-derived grasp features (cached from `_get_observations`, same readings)
        tof_center_dist, tof_centroid_offset, tof_detect = (
            self.tof_center_dist,
            self.tof_centroid_offset,
            self.tof_detect,
        )

        # Per-finger contact magnitudes, normalised to [0, 1] and combined so both
        # fingers must touch the cube for a non-trivial contact value.
        left_contact = torch.clamp(left_finger_force, max=5.0) / 5.0
        right_contact = torch.clamp(right_finger_force, max=5.0) / 5.0
        contact_val = left_contact * right_contact
        pick_dist = torch.norm(self.pick_pos - self.robot_grasp_pos, p=2, dim=-1, keepdim=True)

        # Geometric grasp signal (mirrors the validated base task): the cube counts as
        # grasped the instant the jaw is close to it AND the gripper is closed past 20%.
        # This is reward-side only (not part of the observation), so it does not violate
        # the removal of the `picked` observation. It is deliberately NOT contact-gated:
        # requiring real bilateral contact force makes the whole lift/place/success chain
        # unreachable until a near-perfect grip is already learned, which left the policy
        # stuck hovering in front of the cube. The downstream lift/place rewards still
        # require the cube to physically rise/move, so a "closed but empty" jaw earns
        # nothing -- only the availability of those rewards is unlocked here.
        is_grasped = ((pick_dist < 0.05) & (self.normalized_gripper_dist > 0.2)).float()
        # ToF shaping only applies when the cube is genuinely near the gripper, so the
        # alignment reward cannot be farmed off the flat table with no cube present.
        cube_near = (pick_dist < self.cfg.tof_cube_near_dist).float()
        lift_height_now = (self.pick_pos[:, 2].unsqueeze(dim=-1) - 0.79).clamp(min=0.0)
        self.was_lifted = torch.maximum(self.was_lifted, ((is_grasped > 0.5) & (lift_height_now > 0.03)).float())

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
            self.cfg.action_penalty_scale,
            self.cfg.tof_approach_reward_scale,
            self.cfg.tof_grasp_reward_scale,
            self.cfg.tof_grasp_dist,
            # -- tensors
            self.robot_grasp_pos,
            self.pick_pos,
            self.place_pos,
            is_grasped,
            self.was_lifted,
            self.normalized_gripper_dist,
            pick_cube_vel,
            self.actions,
            contact_val,
            cube_near,
            tof_center_dist,
            tof_centroid_offset,
            tof_detect,
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
            env_ids = self.robot._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore

        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),  # type: ignore
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

        # target cube: randomize position on xy-plane within a 10cm square
        target_default_state = self.target_cube.data.default_root_state[env_ids]
        target_new_state = target_default_state.clone()
        target_pos_noise = sample_uniform(-0.1, 0.1, (len(env_ids), 2), device=self.device)  # type: ignore
        target_new_state[:, 0:2] += target_pos_noise
        target_new_state[:, :3] += self.scene.env_origins[env_ids]

        # reset the per-episode lift latch
        self.was_lifted[env_ids] = 0.0

        # pick cube: randomize within a 10cm square
        pick_default_state = self.pick_cube.data.default_root_state[env_ids]
        pick_new_state = pick_default_state.clone()
        pick_pos_noise = sample_uniform(-0.1, 0.1, (len(env_ids), 2), device=self.device)  # type: ignore
        pick_new_state[:, 0:2] += pick_pos_noise
        pick_new_state[:, :3] += self.scene.env_origins[env_ids]

        # Enforce a minimum xy separation between the pick cube and the target so an
        # episode never starts already "at target" (tolerance 3 cm; use 8 cm margin).
        spawn_offset_xy = target_new_state[:, 0:2] - pick_new_state[:, 0:2]
        spawn_dist_xy = torch.norm(spawn_offset_xy, p=2, dim=-1, keepdim=True)
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
    tof_approach_reward_scale: float,
    tof_grasp_reward_scale: float,
    tof_grasp_dist: float,
    # -- tensors
    robot_grasp_pos: torch.Tensor,
    pick_pos: torch.Tensor,
    place_pos: torch.Tensor,
    is_grasped: torch.Tensor,
    was_lifted: torch.Tensor,
    normalized_gripper_dist: torch.Tensor,
    pick_cube_vel: torch.Tensor,
    actions: torch.Tensor,
    contact_val: torch.Tensor,
    cube_near: torch.Tensor,
    tof_center_dist: torch.Tensor,
    tof_centroid_offset: torch.Tensor,
    tof_detect: torch.Tensor,
):
    """
    Computes staged rewards for the ToF-driven cube grasp-and-place task.

    Stages: 1. Reach (coarse) -> 2. ToF align & approach -> 3. Grasp (close when the
            cube is centred and in range) -> 4. Lift & carry -> 5. Descend at target
            -> 6. Release.

    Relative to the base task, the ``picked`` observation flag is dropped and replaced
    by a reward-side geometric ``is_grasped`` signal (cube close + gripper closed), and
    ToF terms (approach/align and grasp) reward the policy for using the sensor to
    centre the cube in the field of view and close the gripper when it is in range. A
    dense open/close gripper term keeps the jaws open on empty air (no interpenetration)
    and rewards closing only once the cube is within grasp distance.

    Args:
        pick_reward_scale:       Scale for the coarse gripper-to-cube reaching reward.
        place_reward_scale:      Scale for the coarse cube-to-target reward.
        place_fine_reward_scale: Scale for the sharp near-target place reward.
        lift_reward_scale:       Scale for the carry-height reward.
        gripper_reward_scale:    Scale for contact shaping once grasped.
        success_reward_scale:    Persistent bonus while the cube is at the target.
        release_reward_scale:    Bonus for opening the gripper while cube is at target.
        pick_moved_reward_scale: (Negative) scale penalizing cube motion when not grasped.
        action_penalty_scale:    Scale penalizing action magnitude.
        tof_approach_reward_scale: Scale for minimising the central ToF reading
                                 (shapes alignment + final approach together).
        tof_grasp_reward_scale:  Scale for closing the gripper when centred and in range.
        tof_grasp_dist:          Central ToF reading (m) below which the cube is "in range".
        robot_grasp_pos:         (N, 3) world position of the grasp frame.
        pick_pos:                (N, 3) world position of the pick cube.
        place_pos:               (N, 3) world position of the target location.
        is_grasped:              (N, 1) flag, 1.0 when the cube is held (contact-based).
        was_lifted:              (N, 1) per-episode latch, 1.0 once the cube has been
                                 lifted off the table while grasped.
        normalized_gripper_dist: (N, 1) gripper closure in [0, 1]; 1 = closed, 0 = open.
        pick_cube_vel:           (N, 3) linear velocity of the pick cube.
        actions:                 (N, A) last applied actions.
        contact_val:             (N, 1) product of normalised left/right finger contact.
        cube_near:               (N, 1) 1.0 when the cube is within reach of the gripper;
                                 gates the ToF terms so the table cannot farm them.
        tof_center_dist:         (N, 1) nearest central-zone distance (straight ahead).
        tof_centroid_offset:     (N, 1) central-region detection-centroid offset (logging).
        tof_detect:              (N, 1) 1.0 if a central zone detects an object (logging).

    Returns:
        Tuple of (total_reward (N, 1) tensor, dict of mean metrics for wandb logging).
    """
    # -- distances --
    pick_dist = torch.norm(pick_pos - robot_grasp_pos, p=2, dim=-1).unsqueeze(dim=-1)
    cube_to_target_dist = torch.norm(place_pos - pick_pos, p=2, dim=-1).unsqueeze(dim=-1)
    cube_to_target_xy = torch.norm(place_pos[:, :2] - pick_pos[:, :2], p=2, dim=-1).unsqueeze(dim=-1)
    at_target = (cube_to_target_dist < 0.03).float()

    # 1. Coarse reach: keep the cube near the gripper so it lands inside the ToF FoV
    reach_reward = pick_reward_scale * torch.exp(-10.0 * pick_dist)

    # 2. ToF approach + alignment (single term): the down-looking central reading is
    # small only when the cube is directly under the jaw (aligned) AND close (in
    # range), so minimising it shapes both squaring up and the final approach.
    tof_approach_reward = tof_approach_reward_scale * cube_near * torch.exp(-12.0 * tof_center_dist)

    # 3. ToF grasp: once the cube is centred and within grasp range, reward closing
    tof_ready = cube_near * (tof_center_dist < tof_grasp_dist).float()
    tof_grasp_reward = tof_grasp_reward_scale * tof_ready * normalized_gripper_dist

    # 4. Gripper open/close shaping (dense). This is the key bootstrap that teaches
    # the policy to grasp and -- just as importantly -- keeps the jaws OPEN whenever
    # there is nothing to grasp, so the two fingers do not close through each other on
    # empty air (the reported interpenetration). Far from the cube reward opening;
    # near the cube reward closing; once grasped reward firm bilateral contact.
    gripper_reward = torch.where(
        pick_dist < 0.05,
        gripper_reward_scale * normalized_gripper_dist,  # reward closing when near
        gripper_reward_scale * (1.0 - normalized_gripper_dist),  # reward opening when far
    )
    gripper_reward = torch.where(
        is_grasped > 0.5,
        gripper_reward_scale * contact_val,
        gripper_reward,
    )

    # Once delivered (lifted there, now at target) stop rewarding closing/contact so the
    # release bonus can pull the gripper open without competition.
    delivered = (at_target > 0.5) & (was_lifted > 0.5)
    gripper_reward = torch.where(delivered, torch.zeros_like(gripper_reward), gripper_reward)
    tof_grasp_reward = torch.where(delivered, torch.zeros_like(tof_grasp_reward), tof_grasp_reward)

    # 6. Lift & carry: reward height above the table while grasping (cube rests at z=0.79)
    cube_z = pick_pos[:, 2].unsqueeze(dim=-1)
    lift_height = (cube_z - 0.79).clamp(min=0.0)
    capped_lift_height = lift_height.clamp(max=0.05)
    carry_fade = torch.clamp(cube_to_target_xy / 0.10, min=0.0, max=1.0)
    lift_reward = torch.where(
        is_grasped > 0.5,
        lift_reward_scale * (1.0 - torch.exp(-20.0 * capped_lift_height)) * carry_fade,
        torch.zeros_like(lift_height),
    )

    # 7. Place: active while grasping, or once delivered at the target after a real carry
    place_active = (is_grasped > 0.5) | ((at_target > 0.5) & (was_lifted > 0.5))
    lift_gate = torch.clamp(lift_height / 0.03, min=0.0, max=1.0)
    endgame = (cube_to_target_dist < 0.05) & (was_lifted > 0.5)
    lift_gate = torch.where(endgame, torch.ones_like(lift_gate), lift_gate)
    place_coarse = place_reward_scale * torch.exp(-5.0 * cube_to_target_dist) * lift_gate
    place_fine = place_fine_reward_scale * torch.exp(-25.0 * cube_to_target_dist) * lift_gate
    place_reward = torch.where(place_active, place_coarse + place_fine, torch.zeros_like(place_coarse))

    # 8. Success & release: persistent bonus at target plus a bonus for opening there
    success_reward = success_reward_scale * at_target * was_lifted
    release_reward = release_reward_scale * at_target * was_lifted * (1.0 - normalized_gripper_dist)

    # 9. Movement penalty when not grasped (discourages knocking the cube around)
    cube_vel_norm = torch.norm(pick_cube_vel, p=2, dim=-1).unsqueeze(dim=-1)
    move_penalty = torch.where(is_grasped < 0.5, pick_moved_reward_scale * cube_vel_norm, torch.zeros_like(cube_vel_norm))

    # 10. Action smoothness penalty
    action_penalty = action_penalty_scale * torch.norm(actions, p=2, dim=-1).unsqueeze(dim=-1)

    total_reward = (
        reach_reward
        + tof_approach_reward
        + tof_grasp_reward
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
        "reward/tof_approach": tof_approach_reward.mean().item(),
        "reward/tof_grasp": tof_grasp_reward.mean().item(),
        "reward/gripper": gripper_reward.mean().item(),
        "reward/lift": lift_reward.mean().item(),
        "reward/place": place_reward.mean().item(),
        "reward/success": success_reward.mean().item(),
        "reward/release": release_reward.mean().item(),
        "reward/move_penalty": move_penalty.mean().item(),
        "reward/action_penalty": action_penalty.mean().item(),
        "state/pick_dist": pick_dist.mean().item(),
        "state/place_dist": cube_to_target_dist.mean().item(),
        "state/is_grasped": is_grasped.mean().item(),
        "state/was_lifted": was_lifted.mean().item(),
        "state/at_target": at_target.mean().item(),
        "state/gripper_dist": normalized_gripper_dist.mean().item(),
        "state/lift_height": lift_height.mean().item(),
        "state/contact_val": contact_val.mean().item(),
        "tof/center_dist": tof_center_dist.mean().item(),
        "tof/centroid_offset": tof_centroid_offset.mean().item(),
        "tof/detect": tof_detect.mean().item(),
        "tof/ready": tof_ready.mean().item(),
        "tof/cube_near": cube_near.mean().item(),
    }

    return total_reward, wandb_log
