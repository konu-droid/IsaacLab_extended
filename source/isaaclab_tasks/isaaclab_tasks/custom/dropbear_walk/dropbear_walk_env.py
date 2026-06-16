# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-directed human-like walking environment for the Dropbear humanoid.

The environment rewards the robot for walking toward a goal placed straight ahead while
maintaining a natural, human-like gait: an upright torso, an alternating left/right
stepping pattern, swing-foot clearance, and arm swing that counter-balances the legs.

See :class:`~.dropbear_walk_env_cfg.DropbearWalkEnvCfg` for the tunable parameters and the
module docstring there for the high-level design rationale.
"""

from __future__ import annotations

import math
import wandb
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .dropbear_walk_env_cfg import DropbearWalkEnvCfg


class DropbearWalkEnv(DirectRLEnv):
    """Direct RL environment implementing goal-directed human-like locomotion."""

    cfg: DropbearWalkEnvCfg

    def __init__(self, cfg: DropbearWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # control timestep (sim dt * decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # -- resolve body / joint indices --
        self.head_mesh_idx, _ = self.robot.find_bodies(self.cfg.head_mesh)
        self.feet_mesh_idx, _ = self.robot.find_bodies(self.cfg.feet_mesh_names)
        self.hand_mesh_idx, _ = self.robot.find_bodies(self.cfg.hand_mesh_names)
        self.actuated_joint_ids, _ = self.robot.find_joints(self.cfg.actuated_joint_names)
        self.shoulder_joint_ids, _ = self.robot.find_joints(self.cfg.shoulder_joint_names)

        # -- joint limits for the actuated joints (used for clamping & limit penalties) --
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 0].to(self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 1].to(self.device)

        # nominal standing pose the policy adds residuals to
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.actuated_joint_ids].clone()

        self.action_scale = cfg.action_scale

        # -- persistent buffers --
        num_act = len(self.actuated_joint_ids)
        self.actions = torch.zeros((self.num_envs, num_act), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, num_act), device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, num_act), device=self.device)

        # per-env goal position in world coordinates (filled on reset)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # alternating contact schedule target: column 0 -> first foot, 1 -> second foot
        self.contact_target = torch.zeros((self.num_envs, len(self.feet_mesh_idx)), device=self.device, dtype=torch.bool)

        # NaN bookkeeping (a blown-up env is reset rather than crashing the whole run)
        self.nan_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # initialize goals for all envs
        self._resample_goals(self.robot._ALL_INDICES)  # type: ignore

        # weights & biases logging (the env owns its own run, matching the previous setup)
        wandb.init(
            project="dropbear_walk",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
                "target_speed": cfg.target_speed,
                "gait_period": cfg.gait_period,
                "action_scale": cfg.action_scale,
            },
        )

    # ------------------------------------------------------------------ #
    #                              scene                                 #
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        """Create the robot, foot contact sensor, ground plane and lighting."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor_f = ContactSensor(self.cfg.contact_sensor_feet)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # register entities
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensors"] = self.contact_sensor_f
        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    #                            actions                                #
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert the policy output into clamped, residual joint position targets.

        Args:
            actions: Raw policy output of shape (num_envs, num_actuated_joints). It is
                interpreted as an offset (scaled by ``action_scale``) around the nominal
                standing pose.
        """
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # NaN/Inf guard on the raw actions
        bad_env_ids = torch.any(torch.isnan(self.actions) | torch.isinf(self.actions), dim=1)
        if torch.any(bad_env_ids):
            self.nan_detected[bad_env_ids] = True
            self.actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)

        # residual targets around the default pose, clamped to the soft joint limits
        targets = self.default_dof_pos + self.action_scale * self.actions
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        """Send the position targets to the actuated joints (head joints are left untouched)."""
        self.robot.set_joint_position_target(self.robot_dof_targets, self.actuated_joint_ids)

    # ------------------------------------------------------------------ #
    #                          observations                             #
    # ------------------------------------------------------------------ #
    def _compute_phase(self) -> torch.Tensor:
        """Return the normalized gait phase in ``[0, 1)`` for every environment.

        Returns:
            A tensor of shape (num_envs,) giving the fraction of the current step cycle.
        """
        phase = (self.episode_length_buf.float() * self.dt) % self.cfg.gait_period
        return phase / self.cfg.gait_period

    def _goal_direction_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the goal direction in the base frame and the planar distance to goal.

        Returns:
            A tuple ``(goal_dir_b_xy, goal_dist)`` where ``goal_dir_b_xy`` has shape
            (num_envs, 2) (unit direction toward the goal expressed in the robot's base
            frame, xy only) and ``goal_dist`` has shape (num_envs, 1).
        """
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w

        goal_vec_w = self.goal_pos_w - root_pos
        goal_vec_w[:, 2] = 0.0  # planar goal
        goal_dist = torch.norm(goal_vec_w[:, :2], p=2, dim=-1, keepdim=True)

        goal_dir_b = quat_apply_inverse(root_quat, goal_vec_w)
        goal_dir_b_xy = goal_dir_b[:, :2]
        goal_dir_b_xy = goal_dir_b_xy / (torch.norm(goal_dir_b_xy, p=2, dim=-1, keepdim=True) + 1e-6)
        return goal_dir_b_xy, goal_dist

    def _get_observations(self) -> dict:
        """Assemble the proprioceptive + goal observation vector."""
        goal_dir_b_xy, goal_dist = self._goal_direction_base()
        phase = self._compute_phase()

        joint_pos_rel = self.robot.data.joint_pos[:, self.actuated_joint_ids] - self.default_dof_pos
        joint_vel = self.robot.data.joint_vel[:, self.actuated_joint_ids]

        obs = torch.cat(
            (
                self.robot.data.root_link_lin_vel_b,            # (3)
                self.robot.data.root_link_ang_vel_b,            # (3)
                self.robot.data.projected_gravity_b,            # (3)
                goal_dir_b_xy,                                  # (2)
                torch.clamp(goal_dist, max=self.cfg.goal_distance),  # (1)
                joint_pos_rel,                                  # (14)
                joint_vel,                                      # (14)
                self.actions,                                   # (14) last action
                torch.sin(2.0 * math.pi * phase).unsqueeze(-1),  # (1)
                torch.cos(2.0 * math.pi * phase).unsqueeze(-1),  # (1)
            ),
            dim=-1,
        )

        # robustness: clamp & sanitize before handing to the policy
        obs = torch.clamp(obs, min=-100.0, max=100.0)
        bad_env_ids = torch.any(torch.isnan(obs) | torch.isinf(obs), dim=1)
        if torch.any(bad_env_ids):
            self.nan_detected[bad_env_ids] = True
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return {"policy": obs}

    # ------------------------------------------------------------------ #
    #                            rewards                                #
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> torch.Tensor:
        """Gather the state needed for the reward and delegate to :func:`compute_rewards`."""
        root_quat = self.robot.data.root_link_quat_w
        proj_gravity = self.robot.data.projected_gravity_b
        root_lin_vel_w = self.robot.data.root_link_lin_vel_w
        root_lin_vel_b = self.robot.data.root_link_lin_vel_b
        root_ang_vel_b = self.robot.data.root_link_ang_vel_b
        head_z = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, 2].squeeze(-1)

        feet_z = self.robot.data.body_link_pos_w[:, self.feet_mesh_idx, 2]
        feet_pos_xy = self.robot.data.body_link_pos_w[:, self.feet_mesh_idx, :2]
        feet_vel_z = self.robot.data.body_link_lin_vel_w[:, self.feet_mesh_idx, 2]

        joint_vel = self.robot.data.joint_vel[:, self.actuated_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self.actuated_joint_ids]
        applied_torque = self.robot.data.applied_torque[:, self.actuated_joint_ids]
        shoulder_ang = self.robot.data.joint_pos[:, self.shoulder_joint_ids]

        # foot contact from the sensor (magnitude over the last history step)
        net_contact_f = self.contact_sensor_f.data.net_forces_w
        current_air_time = self.contact_sensor_f.data.current_air_time

        # gait clock -> desired contact schedule (feet alternate every half cycle)
        phase = self._compute_phase()
        self.contact_target[:, 0] = phase < 0.5
        self.contact_target[:, 1] = phase >= 0.5

        # goal-relative quantities (world frame, planar)
        goal_vec_w = self.goal_pos_w - self.robot.data.root_link_pos_w
        goal_dist = torch.norm(goal_vec_w[:, :2], p=2, dim=-1)
        goal_dir_w = goal_vec_w[:, :2] / (goal_dist.unsqueeze(-1) + 1e-6)
        vel_to_goal = torch.sum(root_lin_vel_w[:, :2] * goal_dir_w, dim=-1)

        # base forward axis projected onto the goal direction (heading alignment)
        forward_w = quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        forward_w_xy = forward_w[:, :2] / (torch.norm(forward_w[:, :2], p=2, dim=-1, keepdim=True) + 1e-6)
        heading_dot = torch.sum(forward_w_xy * goal_dir_w, dim=-1)

        total_reward, log = compute_rewards(
            # scales
            self.cfg.rew_scale_progress,
            self.cfg.rew_scale_heading,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_gait,
            self.cfg.rew_scale_air_time,
            self.cfg.rew_scale_foot_clearance,
            self.cfg.rew_scale_arm_swing,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_head_height,
            self.cfg.rew_scale_goal_bonus,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_joint_torque,
            self.cfg.rew_scale_feet_near,
            self.cfg.rew_scale_contact_impact,
            self.cfg.rew_scale_dof_limit,
            self.cfg.rew_scale_terminated,
            # params
            self.cfg.target_speed,
            self.cfg.target_head_height,
            self.cfg.target_foot_clearance,
            self.cfg.target_air_time,
            self.cfg.feet_separation_threshold,
            self.cfg.contact_force_threshold,
            self.cfg.goal_reached_threshold,
            # tensors
            vel_to_goal,
            heading_dot,
            goal_dist,
            proj_gravity,
            head_z,
            root_lin_vel_b,
            root_ang_vel_b,
            feet_z,
            feet_pos_xy,
            feet_vel_z,
            net_contact_f,
            current_air_time,
            self.contact_target,
            shoulder_ang,
            self.actions,
            self.prev_actions,
            joint_vel,
            joint_pos,
            applied_torque,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
            self.reset_terminated,
        )

        wandb.log(log, step=self.common_step_counter)
        return total_reward

    # ------------------------------------------------------------------ #
    #                          terminations                             #
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episodes end on timeout, on a fall (head too low / torso tipped over) or on NaN."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        head_z = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, 2].squeeze(-1)
        fallen = head_z < self.cfg.fall_height

        # projected gravity z is ~-1 when upright; rising above the threshold means tipped over
        tipped = self.robot.data.projected_gravity_b[:, 2] > self.cfg.tilt_termination

        terminations = fallen | tipped | self.nan_detected
        return terminations, time_out

    # ------------------------------------------------------------------ #
    #                             reset                                 #
    # ------------------------------------------------------------------ #
    def _resample_goals(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Place each env's goal a fixed distance straight ahead of its spawn origin.

        Args:
            env_ids: Indices of the environments whose goals should be (re)placed.
        """
        offset = torch.tensor([self.cfg.goal_distance, 0.0, 0.0], device=self.device)
        self.goal_pos_w[env_ids] = self.scene.env_origins[env_ids] + offset

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the selected environments to the default pose and re-place their goals."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # clear action history and NaN flags, refresh goals
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.nan_detected[env_ids] = False
        self._resample_goals(env_ids)


@torch.jit.script
def compute_rewards(
    # -- scales --
    rew_scale_progress: float,
    rew_scale_heading: float,
    rew_scale_alive: float,
    rew_scale_gait: float,
    rew_scale_air_time: float,
    rew_scale_foot_clearance: float,
    rew_scale_arm_swing: float,
    rew_scale_upright: float,
    rew_scale_head_height: float,
    rew_scale_goal_bonus: float,
    rew_scale_lin_vel_z: float,
    rew_scale_ang_vel_xy: float,
    rew_scale_action_rate: float,
    rew_scale_joint_vel: float,
    rew_scale_joint_torque: float,
    rew_scale_feet_near: float,
    rew_scale_contact_impact: float,
    rew_scale_dof_limit: float,
    rew_scale_terminated: float,
    # -- params --
    target_speed: float,
    target_head_height: float,
    target_foot_clearance: float,
    target_air_time: float,
    feet_separation_threshold: float,
    contact_force_threshold: float,
    goal_reached_threshold: float,
    # -- tensors --
    vel_to_goal: torch.Tensor,
    heading_dot: torch.Tensor,
    goal_dist: torch.Tensor,
    proj_gravity: torch.Tensor,
    head_z: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    feet_z: torch.Tensor,
    feet_pos_xy: torch.Tensor,
    feet_vel_z: torch.Tensor,
    net_contact_f: torch.Tensor,
    current_air_time: torch.Tensor,
    contact_target: torch.Tensor,
    shoulder_ang: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    applied_torque: torch.Tensor,
    dof_lower: torch.Tensor,
    dof_upper: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """Compute the per-environment walking reward and a dict of component means for logging.

    The reward is a sum of single-purpose terms (each documented inline). Positive terms
    shape the desired human gait; negative terms regularize for smoothness, effort and
    stability. Returns the total reward (num_envs,) and a dict of scalar means for wandb.
    """
    terminated = reset_terminated.float()

    # -- contact state of the two feet --
    contact = torch.norm(net_contact_f, dim=-1) > contact_force_threshold  # (num_envs, 2) bool
    swing = ~contact

    # ---------------- positive shaping terms ---------------- #
    # progress: linear ramp in the forward speed toward the goal. Standing still earns 0,
    # reaching the target speed earns the full scale, and faster is not extra-rewarded
    # (capped at 1). Moving away from the goal is penalized down to -0.5 * scale. This makes
    # walking strictly better than standing, removing the "stand still" local optimum.
    rew_progress = rew_scale_progress * torch.clamp(vel_to_goal / target_speed, min=-0.5, max=1.0)

    # heading: face the goal (dot of forward axis with goal direction, in [-1, 1])
    rew_heading = rew_scale_heading * heading_dot

    # alive: small constant bonus while not terminated
    rew_alive = rew_scale_alive * (1.0 - terminated)

    # gait: feet match the alternating contact schedule (matches in {0,1,2} -> centered)
    contact_matches = torch.sum((contact == contact_target).float(), dim=1)
    rew_gait = rew_scale_gait * (contact_matches - 1.0)

    # air time: reward swing duration up to the target, discourage dragging beyond it
    air_time_shaping = torch.where(
        current_air_time > target_air_time,
        2.0 * target_air_time - current_air_time,
        current_air_time,
    )
    rew_air_time = rew_scale_air_time * torch.sum(air_time_shaping, dim=-1)

    # foot clearance: swing foot should be near the target height above the ground
    clearance = torch.exp(-torch.square((feet_z - target_foot_clearance) / 0.05))
    rew_foot_clearance = rew_scale_foot_clearance * torch.sum(clearance * swing.float(), dim=1)

    # arm swing: each shoulder should swing opposite to the contralateral leg's stance phase.
    # contact_target[:, 0] is the stance flag of foot-0; the left arm (shoulder col 0) should
    # be forward when that foot is in stance. Shoulder sign convention is mirrored L/R.
    shoulder_forward = shoulder_ang > 0.0
    shoulder_forward[:, 0] = ~shoulder_forward[:, 0]  # mirror the left-side sign convention
    arm_matches = torch.sum((shoulder_forward == contact_target).float(), dim=1)
    rew_arm_swing = rew_scale_arm_swing * (arm_matches - 1.0)

    # upright: projected gravity z is -1 when perfectly upright
    rew_upright = rew_scale_upright * torch.exp(-torch.square(proj_gravity[:, 2] + 1.0) / 0.1)

    # head height: keep the head tall (only penalize being below target)
    head_deficit = torch.clamp(target_head_height - head_z, min=0.0)
    rew_head_height = rew_scale_head_height * torch.exp(-torch.square(head_deficit / 0.2))

    # goal bonus: one-off reward for being within the goal threshold
    rew_goal_bonus = rew_scale_goal_bonus * (goal_dist < goal_reached_threshold).float()

    # ---------------- negative regularization terms ---------------- #
    # vertical bounce of the base
    rew_lin_vel_z = rew_scale_lin_vel_z * torch.square(root_lin_vel_b[:, 2])

    # roll/pitch angular velocity of the base
    rew_ang_vel_xy = rew_scale_ang_vel_xy * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)

    # action smoothness (rate of change of the policy output)
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(actions - prev_actions), dim=1)

    # economical joint motion and effort
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
    rew_joint_torque = rew_scale_joint_torque * torch.sum(torch.square(applied_torque), dim=1)

    # feet should not cross / scuff together
    feet_too_near = torch.norm(feet_pos_xy[:, 0] - feet_pos_xy[:, 1], p=2, dim=-1) < feet_separation_threshold
    rew_feet_near = rew_scale_feet_near * feet_too_near.float()

    # soft landings: penalize high vertical foot speed while in contact
    rew_contact_impact = rew_scale_contact_impact * torch.sum(torch.abs(feet_vel_z) * contact.float(), dim=1)

    # stay off the joint limits
    out_of_lower = torch.clamp(dof_lower - joint_pos, min=0.0)
    out_of_upper = torch.clamp(joint_pos - dof_upper, min=0.0)
    rew_dof_limit = rew_scale_dof_limit * torch.sum(out_of_lower + out_of_upper, dim=1)

    # termination penalty (falling)
    rew_termination = rew_scale_terminated * terminated

    # ---------------- total ---------------- #
    total_reward = (
        rew_progress
        + rew_heading
        + rew_alive
        + rew_gait
        + rew_air_time
        + rew_foot_clearance
        + rew_arm_swing
        + rew_upright
        + rew_head_height
        + rew_goal_bonus
        + rew_lin_vel_z
        + rew_ang_vel_xy
        + rew_action_rate
        + rew_joint_vel
        + rew_joint_torque
        + rew_feet_near
        + rew_contact_impact
        + rew_dof_limit
        + rew_termination
    )

    log = {
        "reward/progress": rew_progress.mean().item(),
        "reward/heading": rew_heading.mean().item(),
        "reward/alive": rew_alive.mean().item(),
        "reward/gait": rew_gait.mean().item(),
        "reward/air_time": rew_air_time.mean().item(),
        "reward/foot_clearance": rew_foot_clearance.mean().item(),
        "reward/arm_swing": rew_arm_swing.mean().item(),
        "reward/upright": rew_upright.mean().item(),
        "reward/head_height": rew_head_height.mean().item(),
        "reward/goal_bonus": rew_goal_bonus.mean().item(),
        "reward/pen_lin_vel_z": rew_lin_vel_z.mean().item(),
        "reward/pen_ang_vel_xy": rew_ang_vel_xy.mean().item(),
        "reward/pen_action_rate": rew_action_rate.mean().item(),
        "reward/pen_joint_vel": rew_joint_vel.mean().item(),
        "reward/pen_joint_torque": rew_joint_torque.mean().item(),
        "reward/pen_feet_near": rew_feet_near.mean().item(),
        "reward/pen_contact_impact": rew_contact_impact.mean().item(),
        "reward/pen_dof_limit": rew_dof_limit.mean().item(),
        "reward/termination": rew_termination.mean().item(),
        "reward/total": total_reward.mean().item(),
        # diagnostics
        "state/vel_to_goal": vel_to_goal.mean().item(),
        "state/goal_dist": goal_dist.mean().item(),
        "state/head_z": head_z.mean().item(),
        "state/upright_z": proj_gravity[:, 2].mean().item(),
        "state/contact_matches": contact_matches.mean().item(),
        "state/terminated_frac": terminated.mean().item(),
    }

    return total_reward, log
