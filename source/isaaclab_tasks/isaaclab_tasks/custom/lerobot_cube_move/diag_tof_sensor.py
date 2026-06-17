# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sweep diagnostic for the ToF sensor mount on the SO101 fixed jaw.

The first ToF training run showed the sensor pinned at ~1 cm in every zone
regardless of the cube position -- i.e. the camera mounted at the gripper_link
origin is self-occluded by the gripper's own jaw/finger mesh.

This script builds the ToF env once, holds the arm static in its default pose,
and sweeps a set of candidate *local* mount offsets (relative to gripper_link).
For each offset it renders the 8x8 zone grid twice:
  * with the pick cube teleported far away (the "background" reading), and
  * with the pick cube placed on the optical axis a fixed distance ahead.

A good mount is one where the background reading is ~max range (no self-occlusion)
and the cube is clearly detected when placed ahead. Run from the repo root:

    python source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/diag_tof_sensor.py
"""

import os

os.environ.setdefault("WANDB_MODE", "offline")

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Sweep ToF sensor mount offsets on the SO101 fixed jaw.")
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


# ROS-convention offset rotation that aims camera +Z along gripper_link -X (grasp approach).
OFFSET_ROT = torch.tensor([0.5, -0.5, -0.5, 0.5])

# Candidate local mount offsets (x, y, z) in the gripper_link frame [m].
# +X = world up (away from approach), -X = down the approach, Y/Z are lateral.
CANDIDATE_OFFSETS = [
    (0.0, 0.0, 0.0),
    (0.03, 0.0, 0.0),
    (0.05, 0.0, 0.0),
    (0.0, 0.0, 0.03),
    (0.0, 0.0, 0.05),
    (0.03, 0.0, 0.03),
    (0.05, 0.0, 0.05),
    (0.0, 0.0, -0.03),
    (0.0, 0.0, -0.05),
    (0.03, 0.0, -0.03),
    (0.05, 0.0, -0.05),
    (0.0, 0.03, 0.0),
    (0.0, -0.03, 0.0),
]


def render_grid(env: LerobotCubeToFEnv) -> torch.Tensor:
    """Force a render + sensor update and return the (8, 8) zone grid for env 0."""
    env.sim.render()
    env.tof_camera.update(dt=0.0, force_recompute=True)
    return env._read_tof_grid()[0]  # (Z, Z)


def summarize(grid: torch.Tensor) -> str:
    """One-line summary: overall min, central 2x2 min, fraction of zones in range."""
    z = grid.shape[0]
    c0 = z // 2 - 1
    center = grid[c0 : c0 + 2, c0 : c0 + 2].min().item()
    frac_close = (grid < 0.15).float().mean().item()
    return f"min={grid.min().item():.3f} center={center:.3f} max={grid.max().item():.3f} frac<0.15={frac_close:.2f}"


def main():
    """Build the env, hold static, and sweep candidate mount offsets."""
    cfg = LerobotCubeToFEnvCfg()
    cfg.scene.num_envs = 1
    env = LerobotCubeToFEnv(cfg)
    env.reset()

    zero_action = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
    for _ in range(8):
        env.step(zero_action)

    link_idx = env.gripper_frame_link_idx
    link_pos = env.robot.data.body_link_pos_w[:, link_idx]  # (1, 3)
    link_quat = env.robot.data.body_link_quat_w[:, link_idx]  # (1, 4) wxyz
    offset_rot = OFFSET_ROT.to(env.device).unsqueeze(0)  # (1, 4)
    cam_world_quat = quat_mul(link_quat, offset_rot)  # (1, 4)
    # optical axis (camera +Z) expressed in world coordinates
    forward_w = quat_apply(cam_world_quat, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))

    print(f"[DIAG] gripper_link pos : {link_pos[0].tolist()}")
    print(f"[DIAG] optical axis (w) : {forward_w[0].tolist()}")
    print(f"[DIAG] sweeping {len(CANDIDATE_OFFSETS)} candidate offsets (local gripper_link frame)\n")

    far_away = env.scene.env_origins[0:1] + torch.tensor([[0.0, 0.0, 5.0]], device=env.device)

    for off in CANDIDATE_OFFSETS:
        offset_local = torch.tensor([list(off)], device=env.device)  # (1, 3)
        cam_pos_w = link_pos + quat_apply(link_quat, offset_local)
        env.tof_camera.set_world_poses(cam_pos_w, cam_world_quat, convention="ros")

        # 1) background reading: cube parked far away
        pick_state = env.pick_cube.data.default_root_state[0:1].clone()
        pick_state[:, :3] = far_away
        env.pick_cube.write_root_state_to_sim(pick_state, torch.tensor([0], device=env.device))
        bg = render_grid(env)

        # 2) cube placed on the optical axis, 8 cm ahead of the sensor
        cube_pos = cam_pos_w + forward_w * 0.08
        pick_state[:, :3] = cube_pos
        env.pick_cube.write_root_state_to_sim(pick_state, torch.tensor([0], device=env.device))
        ahead = render_grid(env)

        print(f"[OFFSET {off}]")
        print(f"    background : {summarize(bg)}")
        print(f"    cube@8cm   : {summarize(ahead)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
