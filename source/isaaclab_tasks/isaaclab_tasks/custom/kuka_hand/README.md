## Initial step for developer
1. Setup isaac sim on your local machine.
2. Download the Github repo for Isaac lab and run the setup script.
3. place the construction_rl folder at IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/construction_rl
4. Please select a suitable path for the USD folder provided and update the path in construction_rl/assets/kuka_hand.py file.

## To run the training
To train the network, run the following command in the root folder of isaac lab

```bash
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-kuka-Direct-v0 --num_envs 1024 
```

To check the performance of the trained agents run 
```bash
./isaaclab.sh -p source/standalone/workflows/skrl/play.py --task Isaac-kuka-Direct-v0 --checkpoint "path to checkpoint, should be in log folder"
```

### robot joints
* Articulation_root: base_link
* Shoulder_joints: robot1_joint_a1, robot1_joint_a2
* forearm_joints: robot1_joint_a3, robot1_joint_a4, robot1_joint_a5, robot1_joint_a6
* hand_joints: robot1_gripper_right_hand_thumb_bend_joint, robot1_gripper_right_hand_thumb_rota_joint1, robot1_gripper_right_hand_thumb_rota_joint2,
             robot1_gripper_right_hand_index_bend_joint, robot1_gripper_right_hand_index_joint1, robot1_gripper_right_hand_index_joint2,
             robot1_gripper_right_hand_mid_joint1, robot1_gripper_right_hand_mid_joint2,
             robot1_gripper_right_hand_ring_joint1, robot1_gripper_right_hand_ring_joint2,
             robot1_gripper_right_hand_pinky_joint1, robot1_gripper_right_hand_pinky_joint2

