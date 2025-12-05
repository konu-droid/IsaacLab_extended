# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import wandb
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.utils.math import quat_apply

from .dropbear_walk_env_cfg import DropbearWalkEnvCfg


class DropbearWalkEnv(DirectRLEnv):
    cfg: DropbearWalkEnvCfg

    def __init__(self, cfg: DropbearWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.head_mesh_idx, _ = self.robot.find_bodies(self.cfg.head_mesh)
        self.feet_mesh_idx, _ = self.robot.find_bodies(self.cfg.feet_mesh_names)
        self.hand_mesh_idx, _ = self.robot.find_bodies(self.cfg.hand_mesh_names)
        self.actuated_joint_ids, _ = self.robot.find_joints(self.cfg.actuated_joint_names)
        self.head_joint_ids, _ = self.robot.find_joints(self.cfg.head_joint_names)
        self.arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self.shoulder_joint_ids, _ = self.robot.find_joints(self.cfg.should_joint_names)
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.actuated_joint_ids, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.action_scale = cfg.action_scale

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.actuated_joint_ids)), device=self.device)
        self.head_dof_targets = torch.zeros((self.num_envs, len(self.head_joint_ids)), device=self.device)
        self.arm_dof_targets = torch.zeros((self.num_envs, len(self.arm_joint_ids)), device=self.device)

        self.leg_phase = torch.zeros((self.num_envs, len(self.feet_mesh_idx)), device=self.device, dtype=torch.bool)
        self.prev_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.up_vec = torch.tensor([-0.2, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.up_vec = torch.nn.functional.normalize(self.up_vec, dim=0)

        self.nan_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Initialize wandb
        wandb.init(
            project="dropbear_walk",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
            },
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor_f = ContactSensor(self.cfg.contact_sensor_feet)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensors"] = self.contact_sensor_f
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        targets = (self.action_scale * self.actions)

        # handling NaNs
        bad_env_ids = torch.any(torch.isnan(targets) | torch.isinf(targets), dim=1)
        if torch.any(bad_env_ids):
            print(f"!!!--- NaN or Inf detected in actions of envs: {torch.where(bad_env_ids)[0].tolist()} ---!!!")
            # print(self.actions)
            self.nan_detected[bad_env_ids] = True
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets, self.actuated_joint_ids)
        # setting the head joints as zero
        # head joints are very sensive and have very limited range 0 to -0.05
        # this breaks the simulation too quick so
        # self.robot.set_joint_position_target(self.head_dof_targets, self.head_joint_ids)
        # self.robot.set_joint_position_target(self.arm_dof_targets, self.arm_joint_ids)

    def _get_observations(self) -> dict:
        goal_pos = self.scene.env_origins + torch.tensor([2.0, 0.0, 0.0], device=self.device)

        robot_pos = self.robot.data.root_link_pos_w
        robot_quat = self.robot.data.root_link_quat_w
        # Calculate the distance to the goal in the xy-plane (ignoring height).
        dist_to_goal = torch.norm(goal_pos[:, :2] - robot_pos[:, :2], p=2, dim=-1).unsqueeze(dim=1)

        period = 1.0  # seconds for a full step cycle
        phase = (self.episode_length_buf * self.dt) % period
        self.leg_phase[:, 0] = phase < 0.5  # 0.5 is for half cycle, irrespective of period
        self.leg_phase[:, 1] = phase >= 0.5  # on leg should eb true other false

        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.actuated_joint_ids],
                self.actions,
                dist_to_goal,
                robot_quat,
                self.leg_phase,
            ),
            dim=-1,
        )

        # testing if this works
        obs = torch.clamp(obs, min=-5000.0, max=5000.0)
        obs = torch.nan_to_num(obs, nan=0.0)

        # handling NaNs
        bad_env_ids = torch.any(torch.isnan(obs) | torch.isinf(obs), dim=1)
        if torch.any(bad_env_ids):
            print(f"!!!--- NaN or Inf detected in obs of envs: {torch.where(bad_env_ids)[0].tolist()} ---!!!")
            # print(obs)
            self.nan_detected[bad_env_ids] = True
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self.prev_root_pos = torch.where(robot_pos[:, 0:1] > self.prev_root_pos[:, 0:1], self.robot.data.root_link_pos_w, self.prev_root_pos)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w
        root_lin_vel = self.robot.data.root_link_lin_vel_b
        root_ang_vel = self.robot.data.root_link_ang_vel_b
        head_pos = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, :].squeeze()
        net_contact_F = self.contact_sensor_f.data.net_forces_w
        current_air_time = self.contact_sensor_f.data.current_air_time
        feet_pos = self.robot.data.body_link_pos_w[:, self.feet_mesh_idx, :]
        feet_vel = self.robot.data.body_link_lin_vel_w[:, self.feet_mesh_idx, :]
        shoulder_ang = self.robot.data.joint_pos[:, self.shoulder_joint_ids]

        period = 1.0  # seconds for a full step cycle
        phase = (self.episode_length_buf * self.dt) % period
        self.leg_phase[:, 0] = phase < 0.5  # 0.5 is for half cycle, irrespective of period
        self.leg_phase[:, 1] = phase >= 0.5  # on leg should eb true other false

        # Clamping values to prevent NaNs
        feet_vel = torch.clamp(feet_vel, min=-1.0, max=5.0)
        root_lin_vel = torch.clamp(root_lin_vel, min=-1.0, max=5.0)
        root_ang_vel = torch.clamp(root_ang_vel, min=-1.0, max=5.0)
        feet_pos[:, :, :2] = torch.clamp(feet_pos[:, :, :2], min=-5000.0, max=5000.0)
        head_pos[:, :2] = torch.clamp(head_pos[:, :2], min=-5000.0, max=5000.0)
        root_pos[:, :2] = torch.clamp(root_pos[:, :2], min=-5000.0, max=5000.0)
        feet_pos[:, :, 2] = torch.clamp(feet_pos[:, :, 2], min=-0.01, max=1.0)
        head_pos[:, 2] = torch.clamp(head_pos[:, 2], min=-0.01, max=2.0)
        root_pos[:, 2] = torch.clamp(root_pos[:, 2], min=-0.01, max=1.0)
        root_quat = torch.clamp(root_quat, min=-1.0, max=1.0)

        # Filtering NaNs
        feet_vel = torch.nan_to_num(feet_vel, nan=0.0)
        root_lin_vel = torch.nan_to_num(root_lin_vel, nan=0.0)
        root_ang_vel = torch.nan_to_num(root_ang_vel, nan=0.0)
        feet_pos = torch.nan_to_num(feet_pos, nan=0.0)
        head_pos = torch.nan_to_num(head_pos, nan=0.0)
        root_pos = torch.nan_to_num(root_pos, nan=0.0)
        root_quat[:, :3] = torch.nan_to_num(root_quat[:, :3], nan=0.0)

        total_reward, wandb_log = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_goal_dist,
            self.cfg.rew_scale_not_moving,
            self.cfg.rew_scale_height_dist,
            self.cfg.rew_scale_foot_contact,
            self.cfg.rew_scale_air_time,
            self.cfg.rew_scale_gait_contact,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_swing_height,
            self.cfg.rew_scale_feet_near,
            self.cfg.rew_scale_contact_vel,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.target_swing_height,
            self.cfg.feet_seperation_threshold,
            head_pos,
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            net_contact_F,
            current_air_time,
            shoulder_ang,
            feet_pos,
            feet_vel,
            period,
            self.leg_phase,
            self.prev_root_pos,
            self.up_vec,
            self.scene.env_origins,
            self.reset_terminated,
        )

        wandb.log(wandb_log, step=self.common_step_counter)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        fallen = self.robot.data.body_link_pos_w[:, self.head_mesh_idx, 2].squeeze() < self.cfg.fall_height
        # lower_limit = self.robot.data.joint_pos[:, self.actuated_joint_ids] < self.robot_dof_lower_limits
        # upper_limit = self.robot.data.joint_pos[:, self.actuated_joint_ids] > self.robot_dof_upper_limits

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

        # adding this because once nan show in one env everything becomes nan in all envs,
        # so its better exit
        if torch.any(self.nan_detected):
            sys.exit()

        return terminations, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore

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
    # -- scales
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_goal_dist: float,
    rew_scale_not_moving: float,
    rew_scale_height_dist: float,
    rew_scale_foot_contact: float,
    rew_scale_air_time: float,
    rew_scale_gait_contact: float,
    rew_scale_upright: float,
    rew_scale_swing_height: float,
    rew_scale_feet_near: float,
    rew_scale_contact_vel: float,
    rew_scale_lin_vel_z: float,
    rew_scale_ang_vel_xy: float,
    target_swing_height: float,
    feet_seperation_threshold: float,
    # -- tensors
    head_pos: torch.Tensor,
    robot_root_pos: torch.Tensor,
    robot_root_quat: torch.Tensor,
    robot_root_lin_vel: torch.Tensor,
    robot_root_ang_vel: torch.Tensor,
    net_contact_F: torch.Tensor,
    current_air_time: torch.Tensor,
    shoulder_ang: torch.Tensor,
    feet_pos: torch.Tensor,
    feet_vel: torch.Tensor,
    period: float,
    leg_phase: torch.Tensor,
    prev_root_pos: torch.Tensor,
    up_vec: torch.Tensor,
    env_origins: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """
    Computes rewards for the dropbear walking task.

    The reward function is composed of:
    - A reward for being alive.
    - A penalty for termination.
    - A reward for moving towards a goal 1 meter ahead.
    - A reward for maintaining a forward velocity.
    - A reward for keeping the head high.
    - A reward for solid foot contact with the ground.
    - A reward for desired foot air time to encourage a walking gait.
    - A penalty for vertical, roll, and pitch velocity to encourage stability.
    - Penalties for deviating from target base height, swing height, and hip position.
    - A penalty for high foot velocity on impact.
    - A reward for matching a desired gait pattern (contact schedule).
    """

    # -- reward for being alive and penalty for termination
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # -- reward for moving towards a goal
    goal_pos = env_origins + torch.tensor([4.0, 0.0, 0.0], device=robot_root_pos.device)
    dist_to_goal = torch.norm(goal_pos[:, :2] - robot_root_pos[:, :2], p=2, dim=-1)
    reward_goal = rew_scale_goal_dist * torch.exp(-dist_to_goal)

    # -- penalize for falling
    goal_height = env_origins + torch.tensor([0.0, 0.0, 1.6], device=robot_root_pos.device)
    dist_to_height = goal_height[:, 2] - head_pos[:, 2]
    reward_height = rew_scale_height_dist * torch.where(dist_to_height >= 0.2, 1.0, 0.0)

    # -- reward for keeping the feet on the ground --
    contact_force_magnitudes = torch.norm(net_contact_F, dim=-1) > 1.0
    # feet_contact_reward = rew_scale_foot_contact * torch.sum(contact_force_magnitudes, dim=1)  # sum can be 0, 1 or 2

    # -- reward for desired air time (gait reward) --
    air_time_shaping = torch.where(current_air_time > (period/4), -current_air_time, current_air_time)  # should be 1/4 of the period (one walk cycle). 0.5
    rew_air_time = rew_scale_air_time * torch.sum(air_time_shaping, dim=-1)

    # -- reward for matching a desired contact schedule based on a gait phase clock.
    contact_schedule_matches = torch.sum(~(contact_force_magnitudes ^ leg_phase), dim=1)  # sum can be 0, 1 or 2
    rew_gait_contact = rew_scale_gait_contact * (contact_schedule_matches - 1.0)

    # -- reward for lifting opposite arm to the leg on a gait phase clock. leg lifts when gait is 0.0
    shoulder_ang_bool = shoulder_ang > 0.0
    # this is a work around, as for right arm +ve angle means arm going backwards
    # and for left arm it means going forward
    shoulder_ang_bool[:, 0] = ~shoulder_ang_bool[:, 0]
    shoulder_schedule_matches = torch.sum((shoulder_ang_bool ^ leg_phase), dim=1)  # sum can be 0, 1 or 2
    rew_gait_shoulder = rew_scale_gait_contact * (shoulder_schedule_matches - 1.0)

    # -- reward for up right position --
    upright_z = quat_apply(robot_root_quat, up_vec)
    rew_upright = rew_scale_upright * (upright_z[:, 2] > 0.98)

    # -- penalize feet for being too low during the swing phase.
    swing_height_error = torch.sum(torch.abs(feet_pos[:, :, 2] - target_swing_height) * ~contact_force_magnitudes, dim=1)
    rew_swing_height = rew_scale_swing_height * swing_height_error

    # -- Penalize if the feet get too close to each other.
    feet_too_near = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], p=2, dim=-1) < feet_seperation_threshold
    rew_feet_near = rew_scale_feet_near * feet_too_near

    # -- penalize robot not moving. 1 cm per frame
    movement_magnitude = 0.01 > torch.norm(robot_root_pos[:, :1] - prev_root_pos[:, :1], p=2, dim=-1)
    rew_no_move = rew_scale_not_moving * movement_magnitude

    # -- penalize feet for having high velocity at the moment of contact.
    contact_vel_error = torch.sum((torch.abs(feet_vel[:, :, 2]) * contact_force_magnitudes), dim=1)
    rew_contact_vel = rew_scale_contact_vel * contact_vel_error

    # -- penalize vertical base velocity to reduce bouncing.
    rew_lin_vel_z = rew_scale_lin_vel_z * torch.square(robot_root_lin_vel[:, 2])

    # -- penalize roll and pitch angular velocities to encourage stability.
    rew_ang_vel_xy = rew_scale_ang_vel_xy * torch.sum(torch.square(robot_root_ang_vel[:, :2]), dim=1)

    # -- total reward --
    total_reward = (
        rew_alive
        + rew_termination
        + reward_goal
        + reward_height
        # + feet_contact_reward
        + rew_air_time
        + rew_gait_contact
        + rew_gait_shoulder
        + rew_upright
        + rew_swing_height
        + rew_feet_near
        + rew_no_move
        + rew_contact_vel
        + rew_lin_vel_z
        + rew_ang_vel_xy
    )

    wandb_log = {
        # Original logs
        "reward/rew_alive": rew_alive.mean().item(),
        "reward/rew_termination": rew_termination.mean().item(),
        "reward/reward_goal": reward_goal.mean().item(),
        "reward/reward_height": reward_height.mean().item(),
        # "reward/feet_contact_reward": feet_contact_reward.mean().item(),
        "reward/rew_air_time": rew_air_time.mean().item(),
        "reward/rew_gait_contact": rew_gait_contact.mean().item(),
        "reward/rew_gait_shoulder": rew_gait_shoulder.mean().item(),
        "reward/rew_upright": rew_upright.mean().item(),
        "reward/rew_swing_height": rew_swing_height.mean().item(),
        "reward/rew_feet_near": rew_feet_near.mean().item(),
        "reward/rew_no_move": rew_no_move.mean().item(),
        "reward/rew_contact_vel": rew_contact_vel.mean().item(),
        "reward/rew_lin_vel_z": rew_lin_vel_z.mean().item(),
        "reward/rew_ang_vel_xy": rew_ang_vel_xy.mean().item(),
        # Total
        "reward/total_reward": total_reward.mean().item(),
        # State logs
        "state/dist_to_goal": dist_to_goal.mean().item(),
        "state/head_height_dist": dist_to_height.mean().item(),
        "state/forward_velocity_lin": robot_root_lin_vel[:, 0].mean().item(),
        "state/forward_velocity_ang": robot_root_ang_vel[:, 0].mean().item(),
    }

    return total_reward, wandb_log
