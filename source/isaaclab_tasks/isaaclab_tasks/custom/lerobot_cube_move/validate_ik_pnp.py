# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Validation script for the IK action pipeline of ``LerobotIKPnPEnv``.

It drives the end-effector to a sequence of Cartesian goals using a scripted
proportional controller that only acts through the environment's own action
interface (delta EE position actions). If the differential IK is wired
correctly, the end-effector must converge to each goal with a small error.

Run from the repository root (relative USD paths):
    python source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/validate_ik_pnp.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

# do not pollute the wandb project with validation runs
os.environ.setdefault("WANDB_MODE", "offline")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the differential IK pipeline of LerobotIKPnPEnv.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--steps_per_goal", type=int, default=80, help="Env steps allowed to reach each goal.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab_tasks.custom.lerobot_cube_move.lerobot_IK_pnp import LerobotIKPnPEnv
from isaaclab_tasks.custom.lerobot_cube_move.lerobot_IK_pnp_cfg import LerobotIKPnPEnvCfg

# goals for the end-effector in the robot root frame (x forward, z up, table at z=0).
# These cover the task-relevant workspace: hovering, and low grasp/place poses over
# the cube spawn region (pick x in [0.2, 0.4], target x in [0.1, 0.3]).
# NOTE: poses with x <= 0.15 and z <= 0.03 (close to the base, low) are kinematically
# marginal for the 5-DOF arm (~38 mm residual in earlier tests) — avoid relying on them.
GOALS = [
    (0.25, 0.00, 0.10),
    (0.30, 0.10, 0.05),
    (0.20, -0.10, 0.15),
    (0.20, 0.05, 0.03),
    (0.35, 0.00, 0.04),
]
# maximum acceptable final distance to each goal (m)
GOAL_TOLERANCE = 0.02


def run_validation(env: LerobotIKPnPEnv, steps_per_goal: int) -> bool:
    """
    Drive the EE through the goal sequence with a scripted P-controller.

    Args:
        env:            The IK pick-and-place environment instance.
        steps_per_goal: Number of env steps allowed per goal.

    Returns:
        True if every goal was reached within GOAL_TOLERANCE on all envs.
    """
    device = env.device
    num_envs = env.num_envs

    # print the robot facts the IK setup depends on
    print(f"[VAL] is_fixed_base      : {env.robot.is_fixed_base}")
    print(f"[VAL] body names         : {env.robot.body_names}")
    print(f"[VAL] joint names        : {env.robot.joint_names}")
    print(f"[VAL] arm joint ids      : {env.arm_joint_ids}")
    print(f"[VAL] ee body idx        : {env.ee_body_idx} (jacobian idx {env.ee_jacobi_idx})")
    print(f"[VAL] gripper limits     : [{env.gripper_lower_limit.item():.3f}, {env.gripper_upper_limit.item():.3f}]")

    env.reset()
    all_ok = True

    for goal_xyz in GOALS:
        goal = torch.tensor(goal_xyz, device=device).repeat(num_envs, 1)
        tracking_errors = []
        for _ in range(steps_per_goal):
            ee_pos_b = env._compute_ee_pos_b()
            # P-controller expressed in the env's own action space (wrist actions stay 0)
            delta = (goal - ee_pos_b) / env.cfg.ee_pos_action_scale
            actions = torch.zeros((num_envs, env.cfg.action_space), device=device)
            actions[:, :3] = torch.clamp(delta, -1.0, 1.0)
            env.step(actions)
            tracking_errors.append(torch.norm(env.ee_target_pos_b - env._compute_ee_pos_b(), dim=-1).mean().item())

        final_err = torch.norm(goal - env._compute_ee_pos_b(), dim=-1)
        ok = bool((final_err < GOAL_TOLERANCE).all())
        all_ok &= ok
        print(
            f"[VAL] goal {goal_xyz}: final err mean={final_err.mean().item()*1000:.1f} mm "
            f"max={final_err.max().item()*1000:.1f} mm | steady-state IK tracking err "
            f"{tracking_errors[-1]*1000:.1f} mm | {'PASS' if ok else 'FAIL'}"
        )

    # gripper sanity check: command full-close, then full-open
    # action +1 drives the joint toward its upper limit (open); -1 closes it
    for direction, name in ((-1.0, "close"), (1.0, "open")):
        before = env.normalized_gripper_dist.mean().item()
        actions = torch.zeros((num_envs, env.cfg.action_space), device=device)
        actions[:, 5] = direction
        for _ in range(60):
            env.step(actions)
        after = env.normalized_gripper_dist.mean().item()
        print(f"[VAL] gripper {name}: normalized closure {before:.3f} -> {after:.3f}")

    return all_ok


def main():
    """Create the environment and run the IK validation."""
    cfg = LerobotIKPnPEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = LerobotIKPnPEnv(cfg)
    ok = run_validation(env, args_cli.steps_per_goal)
    print(f"[VAL] RESULT: {'ALL GOALS REACHED' if ok else 'IK VALIDATION FAILED'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
