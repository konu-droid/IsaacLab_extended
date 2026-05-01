"""Train TiagoPRO pick task with RSL-RL PPO.

Usage:
    cd /home/kkim13/workspace_holland_2026_01/IsaacLab
    python /home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks/train.py \
        --task Isaac-TiagoPro-Pick-PillBottle-Direct-v0 \
        --num_envs 512 \
        --max_iterations 3000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TiagoPRO pick task.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="Isaac-TiagoPro-Pick-PillBottle-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=3000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- After sim launch ---

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our custom task
import isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle as right_arm_mobile_pick_pill_bottle  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    from isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle import PickEnvCfg
    from isaaclab_tasks.custom.tiago_rl_tasks.right_arm_mobile_pick_pill_bottle.config import PickPPORunnerCfg

    env_cfg = PickEnvCfg()
    agent_cfg = PickPPORunnerCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations

    # Logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to: {log_dir}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if agent_cfg.max_iterations == 0:
        print("[INFO] max_iterations=0, pausing for inspection. Press Enter to exit.")
        input()
        env.close()
        return

    # Video recording
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Save configs
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    if hasattr(runner, "logger") and hasattr(runner.logger, "log_data"):
        for key, val in runner.logger.log_data.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
    print(f"  Total timesteps: {runner.tot_timesteps}")
    print(f"  Log dir: {log_dir}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
