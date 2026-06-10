# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import wandb
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from isaaclab_tasks.custom.kuka_hand.isaaclab_assets.kuka_hand import KUKA_ROBOT_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

@configclass
class KukaHandEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 20.0  # multiplying it with 10 since car need extra velocity, network output capped at -1.0 to 1.0
    action_space = 18
    observation_space = 85 #{"robot_val": 10, "depth": [480, 640, 1]}
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # increasing the gpu buffer
    sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
    
    # robot
    robot_cfg: ArticulationCfg = KUKA_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # change viewer settings
    # viewer = ViewerCfg(eye=(3.0, 3.0, 5.0),resolution=(1920,1080))
    # viewer = ViewerCfg(eye=(7.5, 7.5, 7.5),resolution=(3840,2160))

    # contact sensors
    contact_sensor_thumb = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot1_gripper_right_hand_thumb_rota_link2",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Brick.*"],
    )

    contact_sensor_index = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot1_gripper_right_hand_index_rota_link2",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Brick.*"],
    )

    contact_sensor_pinky = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot1_gripper_right_hand_pinky_link2",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Brick.*"],
    )

    # Set Cube as object
    brick = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Brick",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/konu/Documents/IsaacLab/robots_usd/kuka/additional_assets/brick.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.01, 0.005), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Add target cube configuration, disabled collision and gravity
    # target_cube = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/TargetCube",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=UsdFileCfg(
    #         usd_path=f"/home/konu/Documents/upwork/Construction_arm_rl/USD/brick.usd",
    #         scale=(0.01, 0.01, 0.01),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             kinematic_enabled=True,
    #         ),
    #         collision_props=CollisionPropertiesCfg(
    #             collision_enabled=False,
    #         ),
    #     ),
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)

    # Add wall configuration parameters
    wall_height_range = (0.1, 1.0)  # Z-axis range for the wall
    wall_width_range = (0.0, 0.5)  # X-axis range for the wall
    wall_distance = -0.7  # Fixed Y distance where wall will be built

class KukaHandEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KukaHandEnvCfg


    def __init__(self, cfg: KukaHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self.action_scale = self.cfg.action_scale

        # buffers for position targets
        # self.hand_dof_targets = torch.zeros((self.num_envs, self.action_space), dtype=torch.float, device=self.device)
        # self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # self.cur_targets = torch.zeros((self.num_envs, self.ac), dtype=torch.float, device=self.device)
        
        # list of actuated joints
        self.shoulder_idx = self._robot.actuators["shoulder"].joint_indices
        self.forearm_idx = self._robot.actuators["forearm"].joint_indices
        self.fingers_idx = self._robot.actuators["fingers"].joint_indices
        
        self.actuated_dof_idx = torch.cat((self.shoulder_idx, self.forearm_idx, self.fingers_idx), dim=-1,)

        # joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        
        # Store previous joint velocities for acceleration computation
        self.prev_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        
        # Fingertip indices
        self.fingertip_idx, self.fingertip_joint_names = self._robot.find_bodies([
            "robot1_gripper_right_hand_thumb_rota_link2",
            "robot1_gripper_right_hand_index_rota_link2",
            "robot1_gripper_right_hand_mid_link2",
            "robot1_gripper_right_hand_ring_link2",
            "robot1_gripper_right_hand_pinky_link2",
        ])

        # Add desired position for cube target
        self.desired_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.desired_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired_positions_w = torch.zeros((self.num_envs, 3), device=self.device) #world poses
        
        # Store wall configuration
        self.wall_height_range = torch.tensor(self.cfg.wall_height_range, device=self.device)
        self.wall_width_range = torch.tensor(self.cfg.wall_width_range, device=self.device)
        self.wall_distance = self.cfg.wall_distance

        # Initialize wandb
        wandb.init(
            project="KukaHand",
            config={
                "episode_length_s": cfg.episode_length_s,
                "action_space": cfg.action_space,
                "observation_space": cfg.observation_space,
                "num_envs": self.num_envs,
                "wall_height_range": cfg.wall_height_range,
                "wall_width_range": cfg.wall_width_range,
                "wall_distance": cfg.wall_distance,
            },
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor_thumb = ContactSensor(self.cfg.contact_sensor_thumb)
        self._contact_sensor_index = ContactSensor(self.cfg.contact_sensor_index)
        self._contact_sensor_pinky = ContactSensor(self.cfg.contact_sensor_pinky)
        self._brick = RigidObject(self.cfg.brick)
        # self._target_cube = RigidObject(self.cfg.target_cube)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor_thumb"] = self._contact_sensor_thumb
        self.scene.sensors["contact_sensor_index"] = self._contact_sensor_index
        self.scene.sensors["contact_sensor_pinky"] = self._contact_sensor_pinky
        self.scene.rigid_objects["brick"] = self._brick
        # self.scene.rigid_objects["target_cube"] = self._target_cube
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions[:, self.actuated_dof_idx], joint_ids=self.actuated_dof_idx)

    # post-physics step calls

    def _get_observations(self) -> dict:
        #changing the coordinate frame to base_link
        cur_brick_pos = self._brick.data.root_pos_w - self.scene.env_origins
        cur_fingertip_position = self._robot.data.body_state_w[:, self.fingertip_idx, :7]
        cur_fingertip_position[:, :, :3] = cur_fingertip_position[:, :, :3] - self.scene.env_origins.unsqueeze(1).repeat(1, len(self.fingertip_idx), 1)
        cur_fingertip_position = cur_fingertip_position.reshape(self.num_envs, -1)
        
        obs = torch.cat(
            (
                self._robot.data.joint_pos,
                cur_fingertip_position,
                cur_brick_pos,
                self._brick.data.root_quat_w,
                self.desired_positions,
                self.desired_orientations,
                self.actions,
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards for the task.
        Returns:
            torch.Tensor: Reward tensor with shape (num_envs,)
        """

        rewards = torch.zeros(self.num_envs, device=self.device)
        reach_reward_scale = 1.0
        contact_reward_scale = 1.0
        contact_threshold = 1.0 
        grasp_force_scale = 0.1
        grasp_threshold = 1.0 #printing out the forces showed range of -250 to 250 for fingers
        vel_lin_punishment = 0.0001 #tested 0.0001
        vel_ang_punishment = 0.001 #tested 0.001
        acc_punishment = 0.0000001

        # Positions and velocities
        self._robot.root_physx_view.get_link_incoming_joint_force()
        fingertip_positions = self._robot.data.body_pos_w[:, self.fingertip_idx, :]
        fingertip_velocities = self._robot.data.body_vel_w[:, self.fingertip_idx, :]
        fingertip_accelerations = self._robot.data.body_acc_w[:, self.fingertip_idx, :] #shape is num_env, num_bodies, 6 the 6 is for x,y,z linear and rpy roll
        brick_positions = self._brick.data.root_pos_w
        brick_velocities = self._brick.data.root_vel_w

        fingertip_direction = fingertip_positions - brick_positions.unsqueeze(1)
        fingertip_to_brick = torch.linalg.vector_norm(fingertip_positions - brick_positions.unsqueeze(1), ord=2, dim=-1)
        fingertip_distance = torch.max(fingertip_to_brick, dim=-1).values
        
        # 1. Distance Reward: Encourage reaching closer to the brick
        # reach_reward = self.exponential_reward(fingertip_distance, scale=reach_reward_scale, beta=2.0)
        reach_reward = reach_reward_scale * 1/fingertip_distance

        # 2. Contact Reward: Emphasize physical contact with the brick
        thumb_force = torch.sum(self._contact_sensor_thumb.data.force_matrix_w, dim=3)
        index_force = torch.sum(self._contact_sensor_index.data.force_matrix_w, dim=3)
        pinky_force = torch.sum(self._contact_sensor_pinky.data.force_matrix_w, dim=3)
        contact_reward_thumb = torch.where(thumb_force > abs(contact_threshold), contact_reward_scale, 0.0)
        contact_reward_index = torch.where(index_force > abs(contact_threshold), contact_reward_scale, 0.0)
        contact_reward_pinky = torch.where(pinky_force > abs(contact_threshold), contact_reward_scale, 0.0)
        
        contact_reward = (contact_reward_thumb + contact_reward_index + contact_reward_pinky).flatten()
        
        # 3. Smoothing punishment to make the joint move in a smooth motion
        # try to reduce jitter at zero crossing point.
        finger_vel_punish = self.big_diff_punishment(fingertip_velocities, vel_lin_punishment, vel_ang_punishment)
        finger_acc_punish = self.big_diff_punishment(fingertip_accelerations, acc_punishment, acc_punishment)
        
        # 4. grasp force reward, more the grasp force better the reward
        grasp_force_reward = (((thumb_force + index_force + pinky_force) / grasp_threshold) * grasp_force_scale).flatten()

        # Combine all rewards
        rewards = (
            reach_reward +
            contact_reward +
            grasp_force_reward +
            finger_vel_punish 
        )
        
        # preventing the value going too low
        rewards = torch.where((rewards < -100), -100, rewards)    

        wandb.log({
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reach_reward": reach_reward.mean().item(),
            "contact_reward": contact_reward.mean().item(),
            "grasp_force_reward": grasp_force_reward.mean().item(),
            "finger_vel_punish": finger_vel_punish.mean().item(),
            "finger_acc_punish": finger_acc_punish.mean().item(),
            "fingertip_distance": fingertip_distance.mean().item(),
        }, step=self.common_step_counter)
                
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # """
        # Determine terminal states for the pick-and-place task.
        # Terminal conditions:
        # 1. Success: Brick reaches target position and orientation within thresholds
        # 2. Truncation: Episode length exceeded
        
        # Returns:
        #     terminated (torch.Tensor): Boolean tensor indicating if episode should end
        #     truncated (torch.Tensor): Boolean tensor indicating if episode was cut short
        # """
        # # Get current brick state
        # brick_pos = self._brick.data.root_pos_w
        # brick_rot = self._brick.data.root_quat_w
        
        # # Define success thresholds
        # pos_threshold = 0.02  # 2cm position threshold
        # orient_threshold = 0.05  # Approximately 5.7 degrees in quaternion distance
        
        # # Check position error
        # pos_error = torch.norm(brick_pos - self.desired_positions_w, dim=-1)
        # pos_success = pos_error < pos_threshold
        
        # # Check orientation error (quaternion dot product)
        # orient_error = 1.0 - torch.abs(torch.sum(brick_rot * self.desired_orientations, dim=-1))
        # orient_success = orient_error < orient_threshold
        
        # # Success requires both position and orientation to be correct
        # success = pos_success & orient_success
        
        # # Terminal conditions
        # max_episode_steps = self.max_episode_length - 1
        # timeout = self.episode_length_buf >= max_episode_steps
        
        # # Terminated if:
        # # 1. Success achieved (position + orientation)
        # # 2. Maximum episode length reached
        # terminated = success | timeout
        
        # # Truncated only if maximum episode length reached without success
        # truncated = timeout & ~success
        
        # return terminated, truncated
    
        # self.joint_pos = self.car.data.joint_pos
        # self.joint_vel = self.car.data.joint_vel
        # self.joint_acc = self.car.data.joint_acc
        body_acc = self._robot.data.root_lin_vel_w
        body_pos = self._robot.data.root_pos_w
        
        distance_to_goal = torch.linalg.norm(self.desired_positions_w - body_pos, dim=1)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = (torch.abs(distance_to_goal) > 5.0)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(body_pos[:, 2]) > 0.5) #checking if the robot is off the ground
        out_of_bounds = out_of_bounds | torch.any(torch.abs(body_acc[:, :2]) > 1.0, dim=1) #checking if the robot is moving too fast
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.actions[env_ids] = 0.0

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset root state
        root_state = self._brick.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] += sample_uniform(0.3, 0.8, (len(env_ids), 2), self.device)
        root_state[:, 7:] = 0.0

        # write root state to simulation
        self._brick.write_root_state_to_sim(root_state)
        self._brick.reset()

        # # Generate new desired positions and orientations for reset environments
        # positions, orientations = self._generate_wall_positions(env_ids)
        
        # # Update the desired positions and orientations for reset environments
        # self.desired_positions[env_ids] = positions
        # self.desired_orientations[env_ids] = orientations

        # # Set positions (convert to world coordinates)
        # self.desired_positions_w[env_ids] = positions + self.scene.env_origins[env_ids]

        # # Create root state for target cubes
        # target_state = self._target_cube.data.default_root_state.clone()
        # target_state[:, :3] = self.desired_positions_w[env_ids]
        # target_state[:, 3:7] = self.desired_orientations[env_ids]
        # # Set rest of state (linear/angular velocities) to zero
        # target_state[:, 7:] = 0.0

        # # Update target cubes visualization
        # self._target_cube.write_root_state_to_sim(target_state, env_ids)
        
        # Sample new commands
        self.desired_positions_w[env_ids, 0] = torch.zeros_like(self.desired_positions_w[env_ids, 0]).uniform_(0.5, 1.0)
        self.desired_positions_w[env_ids, 1] = torch.zeros_like(self.desired_positions_w[env_ids, 1]).uniform_(-1.0, 1.0)
        self.desired_positions_w[env_ids, 2] = torch.zeros_like(self.desired_positions_w[env_ids, 2])
        self.desired_positions_w[env_ids, :2] += self.scene.env_origins[env_ids, :2]

    # auxiliary methods
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.desired_positions_w)

    def _generate_wall_positions(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random positions and orientations for the wall placement
        Returns:
            positions: (N, 3) tensor of positions
            orientations: (N, 4) tensor of quaternions
        """
        num_resets = len(env_ids)
        
        # Generate random positions along the wall plane
        positions = torch.zeros((num_resets, 3), device=self.device)
        positions[:, 0] = torch.rand(num_resets, device=self.device) * (
            self.wall_width_range[1] - self.wall_width_range[0]
        ) + self.wall_width_range[0]  # Random X within range
        positions[:, 1] = self.wall_distance  # Fixed Y distance
        positions[:, 2] = torch.rand(num_resets, device=self.device) * (
            self.wall_height_range[1] - self.wall_height_range[0]
        ) + self.wall_height_range[0]  # Random Z within range

        # Generate orientations (for now, keeping blocks aligned with axes)
        # Can be modified to add random rotations if needed
        orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)

        return positions, orientations
    
    def potential_based_reward(self, current_dist, prev_dist, scale=1.0, clamp_max=None):
        """
        Calculates potential-based reward for distance reduction.
        Reward is positive if distance decreases.
        """
        reward = scale * (prev_dist - current_dist)
        if clamp_max is not None:
            reward = torch.clamp(reward, max=clamp_max)
        return reward

    def exponential_reward(self, dist, scale=1.0, beta=1.0):
        """
        Exponential reward, higher for smaller distances.
        Returns reward in range [0, scale].
        """
        return scale * torch.exp(-beta * dist)

    def gaussian_reward(self, value, target, scale=1.0, sigma=0.1):
        """
        Gaussian reward, peaks when value is close to target.
        Returns reward in range [0, scale].
        """
        error_sq = torch.square(value - target)
        return scale * torch.exp(-0.5 * error_sq / torch.square(sigma))
    
    def big_diff_punishment(self, value, scale_lin=1.0, scale_ang=1.0):
        """
        Vectorized: Penalizes squared L2 norm of velocities/accelerations,
        with potentially different scales for linear and angular components.
        Input shape: (num_env, num_bodies, 6)
        Output shape: (num_env,)
        """
        assert scale_lin >= 0.0 and scale_ang >= 0.0

        # Separate linear (first 3) and angular (last 3) components
        linear = value[..., 0:3]
        angular = value[..., 3:6]

        # # Calculate squared L2 norm for linear part (sum over bodies and 3D vec)
        # linear_norm_sq = torch.sum(torch.square(linear), dim=(-1, -2)) # Sum over last two dims

        # # Calculate squared L2 norm for angular part (sum over bodies and 3D vec)
        # angular_norm_sq = torch.sum(torch.square(angular), dim=(-1, -2)) # Sum over last two dims
        
        # Calculate L2 norm for linear part (norm over bodies and 3D vec)
        # dim=(-1, -2) computes vector norms treating the last two dimensions as the vectors
        # Square the norm to get the squared L2 norm
        linear_norm_sq = torch.square(torch.linalg.norm(linear, ord=2, dim=(-1, -2)))

        # Calculate L2 norm for angular part
        angular_norm_sq = torch.square(torch.linalg.norm(angular, ord=2, dim=(-1, -2)))

        # Apply separate scales and sum penalties
        punishment = -scale_lin * linear_norm_sq - scale_ang * angular_norm_sq
        return punishment 
    
    def punish_action_difference_vec(self, action_t, action_t_minus_1, scale=1.0):
        """
        Vectorized: Penalizes squared L2 norm of action differences.
        Input shape: (num_env, num_features...) e.g., (num_env, num_bodies, 1)
        Output shape: (num_env,)
        """
        assert scale >= 0.0
        action_diff = action_t - action_t_minus_1
        # Sum over all feature dimensions (starting from dim=1)
        dims_to_sum = tuple(range(1, action_diff.dim())) # e.g., (1, 2) for (num_env, num_bodies, 1)
        action_diff_norm_sq = torch.sum(torch.square(action_diff), dim=dims_to_sum)
        punishment = -scale * action_diff_norm_sq
        return punishment # Shape: (num_env,)
