# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
One-shot diagnostic: print the gripper_link world orientation for the SO101 arm.

Purpose: the ToF sensor is mounted as a child of ``gripper_link`` (the fixed
jaw). To aim its optical axis along the grasp approach direction we need to know
which *local* axis of ``gripper_link`` maps to the world grasp direction (which
the joint-space task treats as world -Z, i.e. straight down, via the world-frame
grasp offset ``[0, 0, -0.01]``).

This script builds the existing joint-space env (no camera needed), resets it,
and prints the gripper_link world quaternion plus the world directions of the
link's local +X/+Y/+Z axes so the correct camera offset rotation can be derived.

Run from the repository root (relative USD paths):
    python source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/diag_gripper_frame.py
"""

import argparse
import os

os.environ.setdefault("WANDB_MODE", "offline")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose SO101 gripper_link frame orientation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_tasks.custom.lerobot_cube_move.lerobot_cube_move_env import LerobotCubeMoveEnv
from isaaclab_tasks.custom.lerobot_cube_move.lerobot_cube_move_env_cfg import LerobotCubeMoveEnvCfg


def main():
    """Build the env, reset, and report the gripper_link world frame axes."""
    cfg = LerobotCubeMoveEnvCfg()
    cfg.scene.num_envs = 1
    env = LerobotCubeMoveEnv(cfg)
    env.reset()
    # step once so kinematics settle to the default joint configuration
    zero_action = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
    for _ in range(5):
        env.step(zero_action)

    link_idx = env.gripper_frame_link_idx
    quat = env.robot.data.body_link_quat_w[:, link_idx]  # (N, 4) wxyz
    pos = env.robot.data.body_link_pos_w[:, link_idx]
    rot = matrix_from_quat(quat)  # (N, 3, 3); columns are the local axes in world

    print(f"[DIAG] body names      : {env.robot.body_names}")
    print(f"[DIAG] gripper_link idx : {link_idx}")
    print(f"[DIAG] gripper_link pos : {pos[0].tolist()}")
    print(f"[DIAG] gripper_link quat: {quat[0].tolist()}  (w, x, y, z)")
    print(f"[DIAG] local +X in world: {rot[0, :, 0].tolist()}")
    print(f"[DIAG] local +Y in world: {rot[0, :, 1].tolist()}")
    print(f"[DIAG] local +Z in world: {rot[0, :, 2].tolist()}")
    print(f"[DIAG] cube pos (world) : {env.pick_cube.data.root_com_pos_w.squeeze(1)[0].tolist()}")
    print("[DIAG] grasp approach in world is -Z (down). The local axis whose world")
    print("[DIAG] vector is closest to (0,0,-1) is the camera optical-axis direction.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
