"""Test if gripper can grasp the pill bottle in the Isaac Lab environment.
Moves EE to object, then closes gripper, then lifts.

Usage:
    cd /home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks
    /home/kkim13/isaac_sim_venv/bin/python test_grasp.py
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle as right_arm_mobile_pick_pill_bottle  # noqa: F401
from isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle import PickEnvCfg

env_cfg = PickEnvCfg()
env_cfg.scene.num_envs = 1

env = gym.make("Isaac-TiagoPro-Pick-PillBottle-Direct-v0", cfg=env_cfg)
obs, info = env.reset()

raw_env = env.unwrapped

print(f"Obs: {obs['policy'][0]}")
print(f"  torso={obs['policy'][0][0]:.3f}")
print(f"  gripper={obs['policy'][0][1]:.3f}")
print(f"  target_xyz=({obs['policy'][0][2]:.3f}, {obs['policy'][0][3]:.3f}, {obs['policy'][0][4]:.3f})")

target_dist = obs['policy'][0][2:5]
print(f"\nTarget distance: {target_dist}")
print(f"Target dist norm: {torch.norm(target_dist):.3f}m")

# Phase 1: Approach — move EE toward target
print("\n=== Phase 1: Approach ===")
for step in range(300):
    # Action 7D: ee_delta(6) + gripper(1)
    # Get target direction from obs
    obs_t = obs['policy'][0]
    dx, dy, dz = obs_t[2].item(), obs_t[3].item(), obs_t[4].item()
    dist = (dx**2 + dy**2 + dz**2)**0.5

    if dist < 0.03:
        print(f"Step {step}: Close enough! dist={dist:.4f}")
        break

    # Normalize direction and scale
    scale = min(1.0, dist * 10)  # slow down near target
    action = torch.zeros(1, 7)
    action[0, 0] = dx / dist * scale  # ee_dx
    action[0, 1] = dy / dist * scale  # ee_dy
    action[0, 2] = dz / dist * scale  # ee_dz
    action[0, 6] = -1.0  # gripper open

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 50 == 0:
        d = obs['policy'][0][2:5]
        print(f"Step {step}: dist={torch.norm(d):.3f}, reward={reward[0]:.2f}")

    if terminated[0] or truncated[0]:
        print(f"Step {step}: Episode ended! terminated={terminated[0]}, truncated={truncated[0]}")
        obs, info = env.reset()
        break

# Phase 2: Close gripper gradually
print("\n=== Phase 2: Grasp ===")
for step in range(30):
    action = torch.zeros(1, 7)
    action[0, 6] = 1.0  # gripper close

    obs, reward, terminated, truncated, info = env.step(action)

    gripper_val = obs['policy'][0][1].item()
    print(f"Grasp step {step}: gripper={gripper_val:.3f}")

    if terminated[0] or truncated[0]:
        print(f"TERMINATED during grasp! terminated={terminated[0]}")
        break

# Phase 3: Lift
print("\n=== Phase 3: Lift ===")
for step in range(100):
    action = torch.zeros(1, 7)
    action[0, 2] = 1.0   # ee_dz up
    action[0, 6] = 1.0   # gripper closed

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0:
        print(f"Lift step {step}: reward={reward[0]:.2f}, lift_success={raw_env.extras['log'].get('lift_success_rate', 0)}")

    if terminated[0] or truncated[0]:
        print(f"Episode ended! terminated={terminated[0]}")
        break

print("\nDone!")
env.close()
simulation_app.close()
