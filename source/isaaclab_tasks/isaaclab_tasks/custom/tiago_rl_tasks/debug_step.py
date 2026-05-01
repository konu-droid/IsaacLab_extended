"""Step-by-step debug script for TiagoPRO pick environment.

Usage:
    cd /home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks
    /home/kkim13/isaac_sim_venv/bin/python debug_step.py
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

import isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle as right_arm_mobile_pick_pill_bottle  # noqa: F401
from isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle import PickEnvCfg

env_cfg = PickEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = 42

env = gym.make("Isaac-TiagoPro-Pick-PillBottle-Direct-v0", cfg=env_cfg)

obs, info = env.reset()

# Move viewport camera close to gripper
try:
    from pxr import UsdGeom, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cam = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if cam.IsValid():
        xform = UsdGeom.Xformable(cam)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(1.2, -0.8, 1.0))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-30, 0, 30))
        print("[DEBUG] Camera moved close to robot")
except Exception as e:
    print(f"[DEBUG] Camera move failed: {e}")
# Print all joint positions after reset
raw_env = env.unwrapped
all_pos = raw_env._robot.data.joint_pos[0]
dof_names = raw_env._robot.joint_names
print("\n=== All joints after reset ===")
for i, name in enumerate(dof_names):
    print(f"  {i:2d}: {name:45s} pos={all_pos[i].item():.4f}  target={raw_env.dof_targets[0, i].item():.4f}")

print(f"\nEnvironment loaded. Observation shape: {obs['policy'].shape}")
print(f"Action space: {env.action_space}")

step = 0
while True:
    cmd = input(f"\n[Step {step}] Enter=random step, 'a'=arm only, 'b'=base only, 'g'=gripper toggle, 'q'=quit: ").strip()

    if cmd == 'q':
        break

    action = torch.zeros(1, 12)  # all zeros = no movement

    if cmd == '' or cmd == 'r':
        # Random action
        action = torch.rand(1, 12) * 2 - 1  # [-1, 1]
    elif cmd == 'a':
        # Arm only - small random
        action[0, 0:7] = torch.rand(7) * 0.5 - 0.25
    elif cmd == 'b':
        # Base forward
        action[0, 8] = 0.5  # forward
    elif cmd == 'g':
        # Gripper close
        action[0, 11] = 1.0
    elif cmd == 'o':
        # Gripper open
        action[0, 11] = -1.0

    # Print action breakdown
    arm_act = action[0, 0:7].tolist()
    torso_act = action[0, 7].item()
    base_act = action[0, 8:11].tolist()
    grip_act = action[0, 11].item()
    print(f"  Action: arm={[f'{a:.2f}' for a in arm_act]} torso={torso_act:.2f} base={[f'{a:.2f}' for a in base_act]} grip={grip_act:.2f}")

    obs, reward, terminated, truncated, info = env.step(action)

    grasp_dist = torch.norm(obs['policy'][0, 0:3]).item()
    arm_pos = obs['policy'][0, 3:10].tolist()
    gripper_pos = obs['policy'][0, 10].item()
    base_pose = obs['policy'][0, 11:14].tolist()

    print(f"  Reward: {reward.item():.3f}")
    print(f"  Grasp dist: {grasp_dist:.3f}m")
    print(f"  Arm joints: {[f'{j:.2f}' for j in arm_pos]}")
    print(f"  Gripper: {gripper_pos:.3f}")
    print(f"  Base pose: {[f'{p:.2f}' for p in base_pose]}")
    print(f"  Terminated: {terminated.item()}, Truncated: {truncated.item()}")

    # Gripper sub-joint values
    raw_env = env.unwrapped
    all_pos = raw_env._robot.data.joint_pos[0]
    gripper_main = all_pos[raw_env.gripper_joint_id].item()
    # Find sub-joint indices
    sub_names = ["gripper_right_inner_finger_left_joint", "gripper_right_inner_finger_right_joint",
                 "gripper_right_outer_finger_right_joint", "gripper_right_fingertip_left_joint",
                 "gripper_right_fingertip_right_joint"]
    sub_vals = []
    for name in sub_names:
        idx = raw_env._robot.find_joints(name)[0][0]
        sub_vals.append(f"{all_pos[idx].item():.3f}")
    print(f"  Gripper main: {gripper_main:.3f}, subs: {sub_vals}")

    if terminated or truncated:
        print("  >> Episode ended, resetting...")
        obs, info = env.reset()

    step += 1

env.close()
simulation_app.close()
