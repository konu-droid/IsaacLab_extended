# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Focused diagnostic: does a realistic ToF minimum range remove the gripper
self-occlusion at the centre-axis mount?

The sweep (diag_tof_sensor.py) showed the centre-axis mount keeps the cube on
the optical axis (aligned cube -> centred reading) but the gripper's own jaw
occludes the view when it moves/closes. A real VL53L7CX cannot range closer than
~2-3 cm, so culling everything nearer than the camera near-clip plane both models
the device and hides the jaw.

This builds the ToF env with the near-clip raised to 0.025 m and reports the 8x8
zone grid for the failure mode that broke the first run -- the gripper CLOSED --
as well as the cube at several approach distances. A good result: the closed-jaw
background stays far (jaw culled, sees table ~0.2 m) and the cube is clearly
detected and centred at grasp distances.

Run from the repo root:
    python source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/diag_tof_nearclip.py
"""

import os

os.environ.setdefault("WANDB_MODE", "offline")

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Test ToF near-clip self-occlusion removal.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.utils.math import quat_mul, quat_apply

from isaaclab_tasks.custom.lerobot_cube_move.lerobot_cube_tof_env import LerobotCubeToFEnv
from isaaclab_tasks.custom.lerobot_cube_move.lerobot_cube_tof_env_cfg import LerobotCubeToFEnvCfg


def render_grid(env: LerobotCubeToFEnv) -> torch.Tensor:
    """Force a render + sensor update and return the (8, 8) zone grid for env 0."""
    env.sim.render()
    env.tof_camera.update(dt=0.0, force_recompute=True)
    return env._read_tof_grid()[0]


def summarize(grid: torch.Tensor) -> str:
    """One-line summary: overall min, central 2x2 min, max, fraction of zones in range."""
    z = grid.shape[0]
    c0 = z // 2 - 1
    center = grid[c0 : c0 + 2, c0 : c0 + 2].min().item()
    frac_close = (grid < 0.15).float().mean().item()
    return f"min={grid.min().item():.3f} center={center:.3f} max={grid.max().item():.3f} frac<0.15={frac_close:.2f}"


def settle(env, action, steps=6):
    """Step the env with a fixed action a few times to let kinematics settle."""
    for _ in range(steps):
        env.step(action)


def main():
    """Build env with near-clip 0.025 m and probe open/closed gripper + cube distances."""
    cfg = LerobotCubeToFEnvCfg()
    cfg.scene.num_envs = 1
    # Realistic ToF minimum range: cull geometry (the jaw) closer than 2.5 cm.
    cfg.tof_camera.spawn.clipping_range = (0.025, cfg.tof_max_range)
    env = LerobotCubeToFEnv(cfg)
    env.reset()

    zero_action = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
    settle(env, zero_action, steps=8)

    link_idx = env.gripper_frame_link_idx
    link_pos = env.robot.data.body_link_pos_w[:, link_idx]
    link_quat = env.robot.data.body_link_quat_w[:, link_idx]
    offset_rot = torch.tensor([[0.5, -0.5, -0.5, 0.5]], device=env.device)
    cam_quat = quat_mul(link_quat, offset_rot)
    forward_w = quat_apply(cam_quat, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))
    env0 = torch.tensor([0], device=env.device)

    far_away = env.scene.env_origins[0:1] + torch.tensor([[0.0, 0.0, 5.0]], device=env.device)

    def park_cube_far():
        st = env.pick_cube.data.default_root_state[0:1].clone()
        st[:, :3] = far_away
        env.pick_cube.write_root_state_to_sim(st, env0)

    def place_cube_ahead(dist):
        cam_pos = env.robot.data.body_link_pos_w[:, link_idx]
        st = env.pick_cube.data.default_root_state[0:1].clone()
        st[:, :3] = cam_pos + forward_w * dist
        env.pick_cube.write_root_state_to_sim(st, env0)

    print(f"[DIAG] near-clip = {cfg.tof_camera.spawn.clipping_range}")
    print(f"[DIAG] optical axis (w) : {forward_w[0].tolist()}\n")

    # --- gripper OPEN (default), cube far: background ---
    park_cube_far()
    print(f"[OPEN gripper] background : {summarize(render_grid(env))}")
    for d in (0.03, 0.05, 0.08):
        place_cube_ahead(d)
        env.sim.render()
        print(f"[OPEN gripper] cube@{int(d*100)}cm : {summarize(render_grid(env))}")

    # --- gripper CLOSED (the failure mode from run 1) ---
    gripper_idx = env.gripper_dof_idx[0]
    closed = env.robot_dof_targets.clone()
    closed[:, gripper_idx] = env.gripper_upper_limit
    env.robot.set_joint_position_target(closed)
    park_cube_far()
    settle(env, zero_action, steps=10)
    print(f"\n[CLOSED gripper] background : {summarize(render_grid(env))}")
    for d in (0.03, 0.05, 0.08):
        place_cube_ahead(d)
        env.sim.render()
        print(f"[CLOSED gripper] cube@{int(d*100)}cm : {summarize(render_grid(env))}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
