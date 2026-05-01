"""Playback a trajectory episode on robot in Isaac Sim.
Run via MCP execute_isaac_code:
  exec(open(".../playback_trajectory.py").read())
"""
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.app

EPISODE = 5000  # change this to test different episodes

app = omni.kit.app.get_app()
robot = Articulation("/World/tiago_pro_rs")
robot.initialize()

dof_names = list(robot.dof_names)
torso_idx = dof_names.index("torso_lift_joint")
arm_indices = [dof_names.index(f"arm_right_{i}_joint") for i in range(1, 8)]
gripper_idx = dof_names.index("gripper_right_finger_joint")

# Load data (states: torso + arm7 + gripper = 9D)
data = np.load("/home/kkim13/workspace_holland_2026_01/isaac_sim/rl_tasks/planner_trajectories/trajectories.npz")
states = data["states"]
episode_ids = data["episode_ids"]

mask = episode_ids == EPISODE
ep_states = states[mask]

# Find grasp start (gripper value changes)
grip_vals = ep_states[:, 8]
grip_change = np.diff(grip_vals)
grasp_indices = np.where(grip_change > 0.01)[0]
grasp_start = grasp_indices[0] if len(grasp_indices) > 0 else len(ep_states)

print(f"Episode {EPISODE}: {len(ep_states)} steps")
print(f"  Approach: 0-{grasp_start-1}")
print(f"  Grasp: {grasp_start}-{grasp_start+9}")
print(f"  Lift: {grasp_start+10}-{len(ep_states)-1}")

# Reset to start pose (arm only)
jp = robot.get_joint_positions()
jp[torso_idx] = ep_states[0, 0]
for i, idx in enumerate(arm_indices):
    jp[idx] = ep_states[0, 1 + i]
robot.set_joint_positions(jp)
for _ in range(30):
    app.update()

# Phase 1: Approach (arm + torso only, no gripper)
print("Playing approach...")
for step in range(grasp_start):
    state = ep_states[step]
    jp = robot.get_joint_positions()
    jp[torso_idx] = state[0]
    for i, idx in enumerate(arm_indices):
        jp[idx] = state[1 + i]
    # Only set arm+torso, leave gripper alone
    action = ArticulationAction(
        joint_positions=np.array([state[0]] + [state[1+i] for i in range(7)]),
        joint_indices=np.array([torso_idx] + arm_indices),
    )
    robot.apply_action(action)
    for _ in range(5):
        app.update()

# Phase 2: Grasp (gripper only)
print("Closing gripper...")
for step in range(grasp_start, min(grasp_start + 10, len(ep_states))):
    grip_val = ep_states[step, 8]
    action = ArticulationAction(
        joint_positions=np.array([grip_val]),
        joint_indices=np.array([gripper_idx]),
    )
    robot.apply_action(action)
    for _ in range(10):
        app.update()

# Phase 3: Lift (arm + torso only)
print("Lifting...")
for step in range(grasp_start + 10, len(ep_states)):
    state = ep_states[step]
    action = ArticulationAction(
        joint_positions=np.array([state[0]] + [state[1+i] for i in range(7)]),
        joint_indices=np.array([torso_idx] + arm_indices),
    )
    robot.apply_action(action)
    for _ in range(5):
        app.update()

print("Done!")
