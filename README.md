# 🚀 IsaacLab Extended - Custom RL Environments

Welcome to **IsaacLab Extended**! This repository serves as a collection of specialized, high-fidelity reinforcement learning (RL) environments built on top of [IsaacLab](https://github.com/NVIDIA-Omniverse/IsaacLab). 🤖✨

Here you will find a variety of custom robot tasks, ranging from humanoid locomotion to precision surgical challenges, designed to push the boundaries of robotic learning.

---

## 🛠️ How to Use These Tasks

To get started with these custom environments, ensure you have IsaacLab installed and configured.

### 1. Installation
Clone this repository into your IsaacLab workspace or follow your specific project's installation guide.

### 2. Running a Task
You can typically run these environments using the standard IsaacLab entry points. Navigate to your IsaacLab root and execute:

```bash
# Example for running a custom task (replace <task_name> with the actual task ID)
python source/isaaclab_tasks/isaaclab_tasks/custom/<task_name>.py
```

### 3. Configuration
Each task folder contains its own configuration files where you can adjust reward functions, observation spaces, and hyperparameters.

---

## 🌟 Custom RL Task Gallery

Explore the available custom environments below. Each task includes a dedicated folder with implementation details and a configuration guide.

| Task Name | Documentation | Preview |
| :--- | :---: | :---: |
| **Tiago RL Tasks** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/tiago_rl_tasks/README.md) | ![Tiago RL](images/tiago_rl_tasks.png) |
| **Dropbear Walk** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/dropbear_walk/README.md) | ![Dropbear](images/dropbear_walk.png) |
| **Kuka Hand** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/kuka_hand/README.md) | ![Kuka Hand](images/kuka_hand.png) |
| **Snapfit Lab** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/snapfit_lab/README.md) | ![Snapfit](images/snapfit_lab.png) |
| **Leatherback** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/leatherback/README.md) | ![Leatherback](images/leatherback.png) |
| **Humanoid Walk** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/humanoid_walk/README.md) | ![Humanoid](images/humanoid_walk.png) |
| **Tiago** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/tiago/README.md) | ![Tiago](images/tiago.png) |
| **Surgical Challenge** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/surgical_challenge/README.md) | ![Surgical](images/surgical_challenge.png) |
| **LeRobot Cube Move** | [📖 Read More](source/isaaclab_tasks/isaaclab_tasks/custom/lerobot_cube_move/README.md) | ![LeRobot](images/lerobot_cube_move.png) |

---

## 📂 Project Structure

```text
IsaacLab_extended/
├── images/                 # 🖼️ Visual previews for each task
└── source/
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── custom/     # 🚀 Custom RL Environments reside here
```

---
*Developed with ❤️ for the Robotics & RL Community.*
