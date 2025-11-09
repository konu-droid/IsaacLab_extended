# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script visualizes SMPL-X AMASS motion data in Isaac Lab alongside a robot model.

It loads an AMASS .npz file, computes the 3D joint positions for each frame,
and displays these joints as sphere markers.

It also loads a specified Articulation (robot model) and places it statically
in the scene next to the animation for comparison.

.. code-block:: bash

    # Usage
    # 1. Make sure you have smplx installed in your python environment:
    #    pip install smplx
    #
    # 2. Make sure your paths in the 'main' function are correct
    #    (amass_file_path, smplx_model_path)
    #
    # 3. Run the script to play the animation with the robot:
    #    ./isaaclab.sh -p retarget_smplx/smplx_viz_with_robot.py
    #
    # 4. Run the script to show the static T-pose with the robot:
    #    ./isaaclab.sh -p retarget_smplx/smplx_viz_with_robot.py --tpose

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# Import smplx and data loading libraries
import os
import numpy as np
import smplx
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="This script visualizes SMPL-X AMASS motion data in Isaac Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Add custom flag for T-pose
parser.add_argument(
    "--tpose",
    action="store_true",
    help="Display the model in a static T-pose instead of playing the animation."
)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg  
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext

from isaaclab_asset.dropbear import DROPBEAR_CFG


def load_smplx_data(amass_file_path, smplx_model_path, device="cpu"):
    """
    Loads the AMASS file and initializes the SMPL-X model.
    
    This function is based on the user's provided smplx_visualizer.py script.
    """
    
    # --- 1. LOAD AMASS DATA ---
    try:
        if not os.path.exists(amass_file_path):
            raise FileNotFoundError(f"AMASS file not found at: {amass_file_path}")
        
        bdata = np.load(amass_file_path)
        print(f"Successfully loaded AMASS file: {amass_file_path}")
        print(f"Available keys in the file: {list(bdata.keys())}")
        
        gender = bdata['gender']
        if isinstance(gender, np.ndarray):
            gender = gender.item()
            if isinstance(gender, bytes):
                gender = gender.decode('utf-8')

        if 'male' in gender:
            gender_str = 'male'
        elif 'female' in gender:
            gender_str = 'female'
        else:
            gender_str = 'neutral'
            
        print(f"Gender: {gender_str}")
        mocap_frame_rate = bdata['mocap_frame_rate']
        print(f"Framerate (fps): {mocap_frame_rate}")

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        simulation_app.close()
        exit()

    # --- 2. INITIALIZE SMPL-X MODEL ---
    try:
        if not os.path.exists(smplx_model_path):
             raise FileNotFoundError(f"SMPL-X model path not found: {smplx_model_path}")

        model = smplx.create(
            model_path=smplx_model_path,
            model_type='smplx',
            gender=gender_str,
            use_pca=False,
            num_betas=16,
            ext='npz'
        ).to(device)
        print("SMPL-X model created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the SMPL-X model: {e}")
        simulation_app.close()
        exit()

    # --- 3. PREPARE DATA TENSORS ---
    # We load all frames, the visualization loop will cycle through them
    num_frames = len(bdata['poses'])
    print(f"Processing {num_frames} frames of motion.")
    
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    betas = torch.from_numpy(bdata['betas'][:16][np.newaxis]).float().repeat(body_pose.shape[0], 1).to(device)
    transl = torch.from_numpy(bdata['trans']).float().to(device)

    return model, global_orient, body_pose, betas, transl, mocap_frame_rate, num_frames


def define_joint_markers() -> VisualizationMarkers:
    """Define markers to represent the SMPL-X joints."""
    # We will use a small red sphere for each joint
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SMPLX_Joints",
        markers={
            "joint_sphere": sim_utils.SphereCfg(
                radius=0.01, # 1cm radius for each joint marker
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # Red
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def main():
    """Main function."""
    
    # --- 1. SET YOUR PATHS HERE ---
    amass_file_path = 'retarget_smplx/ACCAD/Male2Walking_c3d/B4_-_Stand_to_Walk_backwards_stageii.npz'
    smplx_model_path = 'retarget_smplx/model'

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.0166, device=args_cli.device) # ~60 Hz
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 1.0]) # Adjusted camera to see both models

    # Spawn things into stage
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    # --- 2. ADD ROBOT LOADING ---
    print("[INFO]: Loading robot model...")
    robot_cfg: ArticulationCfg = DROPBEAR_CFG.replace(prim_path="/World/Robot")
    
    # Spawn the robot *next to* the SMPL-X animation (which is at origin)
    # We set its initial position to (0, 1.0, 0.4)
    robot_cfg.init_state.pos = (0.0, 0.0, 0.05) # x, y, z
    robot_cfg.init_state.rot = (0.7071, 0.0, 0.0, 0.7071) # w, x, y, z
    
    # Create the robot articulation asset
    # This robot will be static, as we are not setting joint targets for it.
    robot = Articulation(robot_cfg)
    # Note: In a simple script, we don't need to add it to `self.scene`
    # like in an RL env. It will be added to the stage on `sim.reset()`.

    # --- 3. LOAD SMPL-X DATA ---
    model, global_orient, body_pose, betas, transl, mocap_frame_rate, num_frames = load_smplx_data(
        amass_file_path, smplx_model_path, device=sim.device
    )

    # --- 4. CREATE MARKERS ---
    joint_visualizer = define_joint_markers()

    # --- 5. SETUP VISUALIZATION MODE ---
    sim.reset() # This will spawn the robot and markers

    if args_cli.tpose:
        # --- 5a. T-POSE LOGIC ---
        print("[INFO]: T-pose mode selected.")
        # Create neutral pose tensors for T-Pose
        # Rotate 90 degrees (pi/2) around the X-axis to make it stand up (Z-up)
        pi_2 = 1.57079632679
        tpose_global_orient = torch.tensor([[pi_2, 0.0, 0.0]], device=sim.device)
        tpose_body_pose = torch.zeros(1, 63, device=sim.device) # 21 joints * 3 DoF
        tpose_betas = betas[0:1] # Use the shape from the loaded file
        # Set translation to origin temporarily. We will auto-adjust height.
        tpose_transl = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device) 
        
        # Get the number of joints and their T-pose positions (first pass to find lowest point)
        with torch.no_grad():
            initial_output = model(
                global_orient=tpose_global_orient,
                body_pose=tpose_body_pose,
                betas=tpose_betas,
                transl=tpose_transl,
                return_verts=False,
                return_full_pose=True # This is needed to compute joints
            )
        num_joints = initial_output.joints.shape[1]
        
        # Auto-adjust height to stand on the ground plane (Z=0)
        temp_joint_positions = initial_output.joints.squeeze(0).float()
        min_z = torch.min(temp_joint_positions[:, 2])
        # Apply the offset to the translation
        tpose_transl[0, 2] = -min_z
        
        # Get final T-pose joint positions (second pass with correct height)
        with torch.no_grad():
            final_output = model(
                global_orient=tpose_global_orient,
                body_pose=tpose_body_pose,
                betas=tpose_betas,
                transl=tpose_transl, # Use the height-adjusted translation
                return_verts=False,
                return_full_pose=True 
            )
        
        # Get T-pose joint positions: shape (1, num_joints, 3) -> (num_joints, 3)
        tpose_joint_positions = final_output.joints.squeeze(0).float()
        print(f"Visualizing {num_joints} joints in T-Pose (standing).")
        
        # Create a default orientation tensor for the markers (spheres are rotation-invariant)
        # Shape: (num_joints, 4) -> (w, x, y, z)
        default_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device).repeat(num_joints, 1)

        print("[INFO]: Setup complete... Starting visualization.")
        
        # --- 6a. T-POSE SIMULATION LOOP ---
        while simulation_app.is_running():
            
            # Continuously visualize the static T-pose
            joint_visualizer.visualize(tpose_joint_positions, default_orientations)
            
            # perform simulation step
            sim.step()
    
    else:
        # --- 5b. ANIMATION LOGIC ---
        print("[INFO]: Animation mode selected.")
        
        # Get the number of joints from an initial run
        with torch.no_grad():
            initial_output = model(
                global_orient=global_orient[0:1],
                body_pose=body_pose[0:1],
                betas=betas[0:1],
                transl=transl[0:1],
                return_verts=False,
                return_full_pose=True # This is needed to compute joints
            )
        num_joints = initial_output.joints.shape[1]
        print(f"Visualizing {num_joints} joints.")
        
        # Create a default orientation tensor for the markers (spheres are rotation-invariant)
        default_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device).repeat(num_joints, 1)

        # --- 6b. SETUP ANIMATION TIMING ---
        sim_dt = sim.get_physics_dt()
        sim_steps_per_mocap_frame = int(round((1.0 / mocap_frame_rate) / sim_dt))
        if sim_steps_per_mocap_frame == 0:
            sim_steps_per_mocap_frame = 1 # Update every step if mocap is faster than sim
            
        print(f"Mocap FPS: {mocap_frame_rate} | Sim FPS: {1.0/sim_dt:.2f}")
        print(f"Updating SMPL-X frame every {sim_steps_per_mocap_frame} simulation steps.")

        frame_index = 0
        step_count = 0
        
        print("[INFO]: Setup complete... Starting visualization.")
        
        # --- 7b. ANIMATION SIMULATION LOOP ---
        while simulation_app.is_running():
            
            # Check if it's time to update the mocap frame
            if step_count % sim_steps_per_mocap_frame == 0:
                
                # --- Get current frame's joint data ---
                with torch.no_grad():
                    model_output = model(
                        global_orient=global_orient[frame_index:frame_index+1],
                        body_pose=body_pose[frame_index:frame_index+1],
                        betas=betas[frame_index:frame_index+1],
                        transl=transl[frame_index:frame_index+1],
                        return_verts=False,      # Don't need skin
                        return_full_pose=True    # Need joints
                    )
                
                # Get joint positions: shape (1, num_joints, 3) -> (num_joints, 3)
                joint_positions = model_output.joints.squeeze(0).float()
                
                # --- Visualize markers ---
                # This visualizes the red spheres
                joint_visualizer.visualize(joint_positions, default_orientations)
                
                # Move to the next frame
                frame_index = (frame_index + 1) % num_frames
            
            # perform simulation step
            sim.step()
            step_count += 1


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        # close sim app
        simulation_app.close()
