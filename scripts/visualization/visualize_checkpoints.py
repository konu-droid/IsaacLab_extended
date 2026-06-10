import argparse
import os
import sys
import glob
import re
import cv2
import imageio
import numpy as np
import subprocess
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record videos of all checkpoints for a given task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--run_dir", type=str, required=True, help="Path to the experiment run directory.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"])
parser.add_argument("--fps", type=int, default=60, help="FPS for the recorded videos.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import skrl
from skrl.utils.runner.torch import Runner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

def create_video_collage(video_paths, output_path, grid_size=None):
    if not video_paths:
        return
    n = len(video_paths)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        cols, rows = grid_size
        
    readers = [imageio.get_reader(p) for p in video_paths]
    fps = readers[0].get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)
    
    # Read first frame to get shape
    first_frame = readers[0].get_data(0)
    h, w, c = first_frame.shape
    blank_frame = np.zeros((h, w, c), dtype=np.uint8)
    
    # Reset first reader
    readers[0].set_image_index(0)
    
    while True:
        frames = []
        has_frames = False
        for reader in readers:
            try:
                frame = reader.get_next_data()
                frames.append(frame)
                has_frames = True
            except (IndexError, EOFError):
                frames.append(blank_frame)
                
        if not has_frames:
            break
            
        # Pad with blank frames
        while len(frames) < rows * cols:
            frames.append(blank_frame)
            
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_frames = frames[r*cols:(r+1)*cols]
            grid_rows.append(np.hstack(row_frames))
        
        collage_frame = np.vstack(grid_rows)
        writer.append_data(collage_frame)
        
    for reader in readers:
        reader.close()
    writer.close()

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    task_module = gym.envs.registry[args_cli.task].entry_point.split(':')[0]
    import importlib
    mod = importlib.import_module(task_module)
    task_dir = os.path.dirname(mod.__file__)
    debug_viz_dir = os.path.join(task_dir, "debug_viz")
    os.makedirs(debug_viz_dir, exist_ok=True)
    print(f"[INFO] Saving videos to: {debug_viz_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        max_ep_len = env.unwrapped.max_episode_length
    except AttributeError:
        max_ep_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    total_steps = max_ep_len * 3
    print(f"[INFO] Recording for {total_steps} steps (3 episodes)")

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    ckpt_dir = os.path.join(args_cli.run_dir, "checkpoints")
    checkpoints = glob.glob(os.path.join(ckpt_dir, "agent_*.pt"))
    checkpoints.sort(key=lambda x: int(re.findall(r'agent_(\d+).pt', os.path.basename(x))[0]) if re.findall(r'agent_(\d+).pt', os.path.basename(x)) else 0)

    video_paths = []
    
    for ckpt in checkpoints:
        step_num = re.findall(r'agent_(\d+).pt', os.path.basename(ckpt))
        step_num = step_num[0] if step_num else "best"
        print(f"[INFO] Evaluating checkpoint at step {step_num}")
        
        runner.agent.load(ckpt)
        runner.agent.set_running_mode("eval")
        
        obs, _ = env.reset()
        
        video_path = os.path.join(debug_viz_dir, f"checkpoint_{step_num}.mp4")
        writer = imageio.get_writer(video_path, fps=args_cli.fps)
        
        for _ in range(total_steps):
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
            
            frame = env.unwrapped.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if isinstance(frame, list):
                frame = frame[0]
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255)
                frame = frame.astype(np.uint8)
            frame = np.ascontiguousarray(frame)
            cv2.putText(frame, f"Step: {step_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            writer.append_data(frame)
        
        writer.close()
        video_paths.append(video_path)
    
    env.close()

    # Collage of ALL videos
    collage_all_path = os.path.join(debug_viz_dir, "collage_all.mp4")
    print(f"[INFO] Creating all-checkpoints collage at {collage_all_path}")
    create_video_collage(video_paths, collage_all_path)

    # Collage of selected videos
    if len(video_paths) >= 4:
        idx = [0, len(video_paths)//3, 2*len(video_paths)//3, len(video_paths)-1]
        selected_paths = [video_paths[i] for i in idx]
        collage_selected_path = os.path.join(debug_viz_dir, "collage_first_last_inbetween.mp4")
        print(f"[INFO] Creating 4-checkpoints collage at {collage_selected_path}")
        create_video_collage(selected_paths, collage_selected_path, grid_size=(2, 2))

if __name__ == "__main__":
    main()
    simulation_app.close()
