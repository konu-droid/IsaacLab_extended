"""Keyboard teleop for TiagoPRO pick environment (Isaac Lab).
Control EE delta with keyboard, test grasp behavior.

Controls:
  W/S: EE forward/backward (X)
  A/D: EE left/right (Y)
  Q/E: EE up/down (Z)
  G:   toggle gripper open/close
  R:   reset environment
  ESC: quit

Usage:
    cd /home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks
    /home/kkim13/isaac_sim_venv/bin/python teleop_test.py
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
import threading

import isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle as right_arm_mobile_pick_pill_bottle  # noqa: F401
from isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle import PickEnvCfg

# Keyboard listener
import carb.input

class KeyboardController:
    def __init__(self):
        self.keys_pressed = set()
        self.gripper_closed = False
        self.reset_flag = False
        self.quit_flag = False

        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()
        self._sub = input_iface.subscribe_to_keyboard_events(self._keyboard, self._on_key)

    def _on_key(self, event, *args):
        key = event.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys_pressed.add(key)
            if key == carb.input.KeyboardInput.G:
                self.gripper_closed = not self.gripper_closed
            if key == carb.input.KeyboardInput.R:
                self.reset_flag = True
            if key == carb.input.KeyboardInput.ESCAPE:
                self.quit_flag = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys_pressed.discard(key)
        return True

    def get_action(self):
        """Return 7D action: ee_delta(6) + gripper(1)"""
        action = torch.zeros(1, 7)
        speed = 0.8  # action scale

        K = carb.input.KeyboardInput
        if K.W in self.keys_pressed:
            action[0, 0] = speed   # ee forward
        if K.S in self.keys_pressed:
            action[0, 0] = -speed  # ee backward
        if K.A in self.keys_pressed:
            action[0, 1] = speed   # ee left
        if K.D in self.keys_pressed:
            action[0, 1] = -speed  # ee right
        if K.Q in self.keys_pressed:
            action[0, 2] = speed   # ee up
        if K.E in self.keys_pressed:
            action[0, 2] = -speed  # ee down

        # Gripper
        if self.gripper_closed:
            action[0, 6] = 1.0
        else:
            action[0, 6] = -1.0

        return action


env_cfg = PickEnvCfg()
env_cfg.scene.num_envs = 1

env = gym.make("Isaac-TiagoPro-Pick-PillBottle-Direct-v0", cfg=env_cfg)
obs, info = env.reset()

kb = KeyboardController()

print("\n=== Keyboard Teleop ===")
print("  W/S: forward/back")
print("  A/D: left/right")
print("  Q/E: up/down")
print("  G:   toggle gripper")
print("  R:   reset")
print("  ESC: quit")
print("========================\n")

step = 0
while not kb.quit_flag:
    if kb.reset_flag:
        obs, info = env.reset()
        kb.reset_flag = False
        print("[RESET]")
        step = 0
        continue

    action = kb.get_action()

    # Only step when there's actual input
    has_input = (action[0, :6].abs().sum() > 0.01) or kb.gripper_closed
    if not has_input:
        # Just render, don't step physics
        simulation_app.update()
        continue

    obs, reward, terminated, truncated, info = env.step(action)

    o = obs['policy'][0]
    dist = torch.norm(o[2:5])
    print(f"Step {step}: dist={dist:.3f} torso={o[0]:.3f} grip={o[1]:.3f} grip_cmd={'CLOSE' if kb.gripper_closed else 'OPEN'}")

    if terminated[0] or truncated[0]:
        print(f"[TERMINATED] step={step}")
        obs, info = env.reset()
        step = 0
        continue

    step += 1

env.close()
simulation_app.close()
