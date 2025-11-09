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

SMPLX_JOINT_IDX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "jaw": 22,
    "left_eye_smplhf": 23,
    "right_eye_smplhf": 24,
    "left_index1": 25,
    "left_index2": 26,
    "left_index3": 27,
    "left_middle1": 28,
    "left_middle2": 29,
    "left_middle3": 30,
    "left_pinky1": 31,
    "left_pinky2": 32,
    "left_pinky3": 33,
    "left_ring1": 34,
    "left_ring2": 35,
    "left_ring3": 36,
    "left_thumb1": 37,
    "left_thumb2": 38,
    "left_thumb3": 39,
    "right_index1": 40,
    "right_index2": 41,
    "right_index3": 42,
    "right_middle1": 43,
    "right_middle2": 44,
    "right_middle3": 45,
    "right_pinky1": 46,
    "right_pinky2": 47,
    "right_pinky3": 48,
    "right_ring1": 49,
    "right_ring2": 50,
    "right_ring3": 51,
    "right_thumb1": 52,
    "right_thumb2": 53,
    "right_thumb3": 54,
    "nose": 55,
    "right_eye": 56,
    "left_eye": 57,
    "right_ear": 58,
    "left_ear": 59,
    "left_big_toe": 60,
    "left_small_toe": 61,
    "left_heel": 62,
    "right_big_toe": 63,
    "right_small_toe": 64,
    "right_heel": 65,
    "left_thumb": 66,
    "left_index": 67,
    "left_middle": 68,
    "left_ring": 69,
    "left_pinky": 70,
    "right_thumb": 71,
    "right_index": 72,
    "right_middle": 73,
    "right_ring": 74,
    "right_pinky": 75,
    "right_eye_brow1": 76,
    "right_eye_brow2": 77,
    "right_eye_brow3": 78,
    "right_eye_brow4": 79,
    "right_eye_brow5": 80,
    "left_eye_brow5": 81,
    "left_eye_brow4": 82,
    "left_eye_brow3": 83,
    "left_eye_brow2": 84,
    "left_eye_brow1": 85,
    "nose1": 86,
    "nose2": 87,
    "nose3": 88,
    "nose4": 89,
    "right_nose_2": 90,
    "right_nose_1": 91,
    "nose_middle": 92,
    "left_nose_1": 93,
    "left_nose_2": 94,
    "right_eye1": 95,
    "right_eye2": 96,
    "right_eye3": 97,
    "right_eye4": 98,
    "right_eye5": 99,
    "right_eye6": 100,
    "left_eye4": 101,
    "left_eye3": 102,
    "left_eye2": 103,
    "left_eye1": 104,
    "left_eye6": 105,
    "left_eye5": 106,
    "right_mouth_1": 107,
    "right_mouth_2": 108,
    "right_mouth_3": 109,
    "mouth_top": 110,
    "left_mouth_3": 111,
    "left_mouth_2": 112,
    "left_mouth_1": 113,
    "left_mouth_5": 114,  # 59 in OpenPose output
    "left_mouth_4": 115,  # 58 in OpenPose output
    "mouth_bottom": 116,
    "right_mouth_4": 117,
    "right_mouth_5": 118,
    "right_lip_1": 119,
    "right_lip_2": 120,
    "lip_top": 121,
    "left_lip_2": 122,
    "left_lip_1": 123,
    "left_lip_3": 124,
    "lip_bottom": 125,
    "right_lip_3": 126,
    # Face contour
    "right_contour_1": 127,
    "right_contour_2": 128,
    "right_contour_3": 129,
    "right_contour_4": 130,
    "right_contour_5": 131,
    "right_contour_6": 132,
    "right_contour_7": 133,
    "right_contour_8": 134,
    "contour_middle": 135,
    "left_contour_8": 136,
    "left_contour_7": 137,
    "left_contour_6": 138,
    "left_contour_5": 139,
    "left_contour_4": 140,
    "left_contour_3": 141,
    "left_contour_2": 142,
    "left_contour_1": 143,
}

JOINT_MAP = [
    ("left_hip", "PG_RMD_X10_S2_MIR4__2_Rotor_1"),
    ("right_hip", "PG_RMD_X10_S2_MIR4__1_Rotor_1"),
    ("left_knee", "LL_skateboard_bearing__20__1"),
    ("right_knee", "RL_skateboard_bearing__20__1"),
    ("left_ankle", "LL_basis_left_1"),
    ("right_ankle", "RL_basis_left_1"),
    ("left_shoulder", "LH_RMD_X8_Pro_MIR8_MIR1__1__1"),
    ("right_shoulder", "RH_RMD_X8_Pro_MIR8_MIR1__1__1"),
    ("left_elbow", "LH_6mm_bearing__9__1"),
    ("right_elbow", "RH_6mm_bearing__9__1"),
    ("left_wrist", "LH_shoulder_ex_al_interface_1"),
    ("right_wrist", "RH_shoulder_ex_al_interface_1"),
]


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

def load_tpose(model, betas, device="cpu"):
    # --- 5a. T-POSE LOGIC ---
    print("[INFO]: T-pose mode selected.")
    # Create neutral pose tensors for T-Pose
    # Rotate 90 degrees (pi/2) around the X-axis to make it stand up (Z-up)
    pi_2 = 1.57079632679
    tpose_global_orient = torch.tensor([[pi_2, 0.0, 0.0]], device=device)
    tpose_body_pose = torch.zeros(1, 63, device=device) # 21 joints * 3 DoF
    tpose_betas = betas[0:1] # Use the shape from the loaded file
    # Set translation to origin temporarily. We will auto-adjust height.
    tpose_transl = torch.tensor([[0.0, 0.0, 0.0]], device=device) 
    
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
    default_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_joints, 1)

    return tpose_joint_positions, default_orientations


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


def joint_scaling(robot, smplx_joint_pos, smplx_joint_map, joint_map):

    print("starting joint scaling")
    smplx_joint_scale = {}
    smplx_joint_scale1 = []

    prev_smplx_link = joint_map[0][0]
    prev_robot_link = joint_map[0][1]

    for smplx_link, robot_link in joint_map:
        # 1. Get the 3D positions for the joints you want to measure
        smplx_pose = smplx_joint_pos[smplx_joint_map[smplx_link]]
        robot_pose = robot.data.body_link_pos_w[:, robot.find_bodies(robot_link)[0], :]
        prev_smplx_pose = smplx_joint_pos[smplx_joint_map[prev_smplx_link]]
        prev_robot_pose = robot.data.body_link_pos_w[:, robot.find_bodies(prev_robot_link)[0], :]

        # 2. Calculate the distance (Euclidean norm)
        smplx_dist = torch.norm(prev_smplx_pose - smplx_pose)
        robot_dist = torch.norm(prev_robot_pose - robot_pose)

        # 3. Print the distances
        print("--- Limb Lengths (in meters) ---")
        print(f"  smplx_dist: {smplx_dist.item():.4f} m")
        print(f"  robot_dist:     {robot_dist.item():.4f} m")

        # Add a small epsilon to avoid division by zero, just in case
        epsilon = 1e-8
        joint_scale = robot_dist / (smplx_dist + epsilon)

        smplx_joint_scale[smplx_link] = {joint_scale.item()}
        smplx_joint_scale1.append(joint_scale.item())

        prev_smplx_pose = smplx_pose
        prev_robot_pose = robot_pose
    
    print(smplx_joint_scale)
    print(smplx_joint_scale1)


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

    # --- 3. LOAD SMPL-X DATA ---
    model, global_orient, body_pose, betas, transl, mocap_frame_rate, num_frames = load_smplx_data(
        amass_file_path, smplx_model_path, device=sim.device
    )

    # --- 4. CREATE MARKERS ---
    joint_visualizer = define_joint_markers()

    # --- 5. SETUP VISUALIZATION MODE ---
    sim.reset() # This will spawn the robot and markers

    # --- Get current frame's joint data ---
    with torch.no_grad():
        model_output = model(
            return_verts=False,      # Don't need skin
            return_full_pose=True    # Need joints
        )
    
    # Get joint positions: shape (1, num_joints, 3) -> (num_joints, 3)
    joint_positions = model_output.joints.squeeze(0).float()

    joint_scaling(robot, joint_positions, SMPLX_JOINT_IDX, JOINT_MAP)

    if args_cli.tpose:
        
        tpose_joint_positions, default_orientations = load_tpose(model, betas, device=sim.device)

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
