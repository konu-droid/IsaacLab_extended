# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.sim import PhysxCfg

from .isaaclab_asset.dropbear import DROPBEAR_CFG

@configclass
class DropbearWalkEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 3.0 # 300 STEPS
    # - spaces definition
    action_space = 28
    observation_space = 61
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type = 0,
            enable_ccd = True,
            gpu_collision_stack_size=2 * 1024 * 1024 * 1024  # 2gb allocated
        )
    )

    # robot(s)
    robot_cfg: ArticulationCfg = DROPBEAR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    actuated_joint_names = robot_cfg.joint_sdk_names
    head_joint_names = ["head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
                        "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6"]
    # - robot reset 
    fall_height = 1.0 # the head is at 1.7m 
    # - action scale
    action_scale = 10.0
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_goal_dist = 1.0
    rew_scale_height_dist = 0.2