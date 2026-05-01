#!/usr/bin/env python3
"""Generate pick trajectories using Lula (IK + PathPlanner + FK). No sim needed.

Run inside Isaac Sim script editor or via MCP.

Combinations:
- Torso: 5 (0.0 ~ 0.35m)
- Joint 5: 6 (50° ~ 90°)
- Object XY: 10 (2X × 5Y on table at 0.8m)
- Object height: 3 (-10cm, 0, +10cm from table top)
- Approach angle: 5 (0°, 30°, 45°, 60°, 90°)
- Noise: 4 repeats (±5° on joints 1,2,3,4,6,7)
Total: 5 × 6 × 30 × 5 × 4 = 18,000

Each trajectory: approach → grasp → lift
All pure calculation. No simulation stepping.

Output: planner_trajectories/trajectories.npz
  - states: (N, 12) = torso + arm_7 + gripper + target_xyz
  - actions: (N, 10) = GR00T format [base(3), ee_delta(6), gripper(1)]
  - episode_ids: (N,)
"""

from omni.isaac.motion_generation import lula
import numpy as np
import math
import itertools
import os
import time

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
URDF = "/home/kkim13/workspace_holland_2026_01/robot_neuron/RoboNeuron/urdf/tiago_pro.urdf"
ROBOT_DESC = "/home/kkim13/workspace_holland_2026_01/isaac_sim/motion_configs/tiago_pro/tiago_pro_robot_description.yaml"

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
TABLE_DIST = 0.8
TABLE_HEIGHT = 0.812
LIFT_HEIGHT = 0.10
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.80
EE_FRAME = "arm_right_tool_link"

BASE_ARM = np.array([-2.7845, -1.1257, 1.7738, -2.2121, 1.0664, 1.5607, 1.7522])

TORSO_VALUES = np.linspace(0.0, 0.35, 5)
JOINT5_DEGREES = [50, 58, 66, 74, 82, 90]
JOINT5_RADS = [math.radians(d) for d in JOINT5_DEGREES]

OBJ_X = [-0.15, 0.15]
OBJ_Y = [-0.25, -0.125, 0.0, 0.125, 0.25]
OBJ_XY = list(itertools.product(OBJ_X, OBJ_Y))  # 10

OBJ_Z_OFFSETS = [-0.10, 0.0, 0.10]

APPROACH_QUATS = {
    0:  lula.Rotation3(0.0000, 0.7071, 0.0000, 0.7071),
    30: lula.Rotation3(0.0000, 0.8660, 0.0000, 0.5000),
    45: lula.Rotation3(0.0000, 0.9239, 0.0000, 0.3827),
    60: lula.Rotation3(0.0000, 0.9659, 0.0000, 0.2588),
    90: lula.Rotation3(0.0000, 1.0000, 0.0000, 0.0000),
}

NOISE_STD = math.radians(5)
NUM_REPEATS = 4

# -------------------------------------------------------
# Helper: quaternion to euler (for EE delta rotation)
# -------------------------------------------------------
def rot3_to_euler(rot):
    """Extract roll, pitch, yaw from lula.Rotation3."""
    # Convert to quaternion wxyz
    w, x, y, z = rot.w(), rot.x(), rot.y(), rot.z()
    # Roll
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1, min(1, sinp)))
    # Yaw
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw])

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    t_start = time.time()

    # Load Lula
    robot_desc = lula.load_robot(ROBOT_DESC, URDF)
    kin = robot_desc.kinematics()
    world = lula.create_world()
    world_view = world.add_world_view()
    planner = lula.create_motion_planner(robot_desc, world_view)
    ik_config = lula.CyclicCoordDescentIkConfig()

    # All combinations
    combos = list(itertools.product(
        TORSO_VALUES, JOINT5_RADS, OBJ_XY, OBJ_Z_OFFSETS, APPROACH_QUATS.keys()
    ))
    print(f"Combinations: {len(combos)} × {NUM_REPEATS} repeats = {len(combos) * NUM_REPEATS}")

    all_states = []     # (N, 12): torso + arm_7 + gripper + target_xyz
    all_actions = []    # (N, 10): GR00T format
    all_episode_ids = []
    episode_idx = 0
    failed = 0

    for combo_idx, (torso, j5, (ox, oy), oz, angle_deg) in enumerate(combos):
        # Object position in robot base frame
        obj_pos = np.array([TABLE_DIST + ox, oy, TABLE_HEIGHT + oz])
        # Lift position
        lift_pos = obj_pos + np.array([0, 0, LIFT_HEIGHT])
        # Approach rotation
        approach_rot = APPROACH_QUATS[angle_deg]

        for noise_idx in range(NUM_REPEATS):
            # Start cspace: [torso, arm1-7]
            arm = BASE_ARM.copy()
            arm[4] = j5
            if noise_idx > 0:
                noise = np.random.randn(7) * NOISE_STD
                noise[4] = 0  # no noise on joint 5
                arm += noise
            start_q = np.array([torso] + arm.tolist(), dtype=np.float64)

            # --- Phase 1: IK for approach target ---
            approach_pose = lula.Pose3(approach_rot, obj_pos)
            ik_result = lula.compute_ik_ccd(kin, approach_pose, EE_FRAME, ik_config)
            if not ik_result.success or ik_result.position_error > 0.02:
                failed += 1
                continue

            approach_q = np.array(ik_result.cspace_position, dtype=np.float64)

            # --- Phase 2: Path plan start → approach ---
            path_result = planner.plan_to_cspace_target(
                start_q.reshape(-1, 1), approach_q.reshape(-1, 1), True
            )
            if not path_result.path_found:
                failed += 1
                continue

            approach_path = path_result.interpolated_path

            # --- Phase 3: IK for lift target ---
            lift_pose = lula.Pose3(approach_rot, lift_pos)
            ik_lift = lula.compute_ik_ccd(kin, lift_pose, EE_FRAME, ik_config)
            if not ik_lift.success or ik_lift.position_error > 0.02:
                failed += 1
                continue

            lift_q = np.array(ik_lift.cspace_position, dtype=np.float64)

            # --- Phase 4: Path plan approach → lift ---
            lift_result = planner.plan_to_cspace_target(
                approach_q.reshape(-1, 1), lift_q.reshape(-1, 1), True
            )
            if not lift_result.path_found:
                failed += 1
                continue

            lift_path = lift_result.interpolated_path

            # --- Record trajectory ---
            # Phase 1: approach (gripper open)
            prev_ee_pos = None
            prev_ee_rot = None

            for wp in approach_path:
                q = np.array(wp).flatten()
                pose = kin.pose(q, EE_FRAME)
                ee_pos = np.array(pose.translation)
                ee_rot = rot3_to_euler(pose.rotation)

                if prev_ee_pos is not None:
                    ee_delta_pos = ee_pos - prev_ee_pos
                    ee_delta_rot = ee_rot - prev_ee_rot
                else:
                    ee_delta_pos = np.zeros(3)
                    ee_delta_rot = np.zeros(3)

                # State: torso + arm joints + gripper + target_xyz (12D)
                state = np.concatenate([q, [GRIPPER_OPEN], obj_pos]).astype(np.float32)
                # Action: [base(3)=0, ee_delta_pos(3), ee_delta_rot(3), gripper=0]
                action = np.concatenate([
                    np.zeros(3),           # base delta
                    ee_delta_pos,          # ee position delta
                    ee_delta_rot,          # ee rotation delta
                    [0.0],                 # gripper open
                ]).astype(np.float32)

                all_states.append(state)
                all_actions.append(action)
                all_episode_ids.append(episode_idx)

                prev_ee_pos = ee_pos
                prev_ee_rot = ee_rot

            # Phase 2: grasp (10 steps, gripper closing gradually)
            grasp_steps = 10
            for gi in range(grasp_steps):
                grip_val = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (gi + 1) / grasp_steps
                state = np.concatenate([approach_q, [grip_val], obj_pos]).astype(np.float32)
                action = np.concatenate([np.zeros(9), [1.0]]).astype(np.float32)
                all_states.append(state)
                all_actions.append(action)
                all_episode_ids.append(episode_idx)

            # Phase 3: lift (gripper closed)
            prev_ee_pos = np.array(kin.pose(approach_q, EE_FRAME).translation)
            prev_ee_rot = rot3_to_euler(kin.pose(approach_q, EE_FRAME).rotation)

            for wp in lift_path:
                q = np.array(wp).flatten()
                pose = kin.pose(q, EE_FRAME)
                ee_pos = np.array(pose.translation)
                ee_rot = rot3_to_euler(pose.rotation)

                ee_delta_pos = ee_pos - prev_ee_pos
                ee_delta_rot = ee_rot - prev_ee_rot

                state = np.concatenate([q, [GRIPPER_CLOSE], obj_pos]).astype(np.float32)
                action = np.concatenate([
                    np.zeros(3),
                    ee_delta_pos,
                    ee_delta_rot,
                    [1.0],
                ]).astype(np.float32)

                all_states.append(state)
                all_actions.append(action)
                all_episode_ids.append(episode_idx)

                prev_ee_pos = ee_pos
                prev_ee_rot = ee_rot

            episode_idx += 1

        if (combo_idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  {combo_idx + 1}/{len(combos)} combos, {episode_idx} episodes, {failed} failed, {elapsed:.1f}s")

    # Save
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_episode_ids = np.array(all_episode_ids, dtype=np.int32)

    out_dir = "/home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks/planner_trajectories"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trajectories.npz")

    np.savez_compressed(
        out_path,
        states=all_states,
        actions=all_actions,
        episode_ids=all_episode_ids,
        num_episodes=episode_idx,
        num_failed=failed,
    )

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed:.1f}s")
    print(f"  Episodes: {episode_idx} (failed: {failed})")
    print(f"  Total steps: {len(all_states)}")
    print(f"  States: {all_states.shape}")
    print(f"  Actions: {all_actions.shape}")
    print(f"  Saved: {out_path}")

main()
