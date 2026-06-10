# ROS2 Clean Architecture Project

This project is set up with a comprehensive set of **Claude Skills** designed to facilitate ROS2 development following **Clean Architecture** principles.

## Available Skills

The following skills are available in `.claude/skills` and can be used to guide development:

| Skill Name              | Description                               | Key Components                                                                        |
| ----------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------- |
| **ros2_node_creation**  | Create Clean Architecture compliant Nodes | `BaseNode` template, Dependency Injection, QoS profiles (Python & C++)                |
| **ros2_launch_config**  | Modular Launch files                      | Composition, `IncludeLaunchDescription`, Parameter management, C++ executable support |
| **ros2_service_action** | Services and Actions                      | Server/Client wrappers, Domain Use Case integration (Python & C++)                    |
| **ros2_messaging**      | Pub/Sub Patterns                          | Domain-driven publishers, Generic subscribers, Thread-safe buffers, Synchronization   |
| **ros2_testing**        | Testing Strategy                          | Unit (Domain), Integration (Node), E2E (Launch), GTest/GMock support                  |
| **ros2_lifecycle**      | Managed Nodes                             | Lifecycle Node templates, State transition management, Lifecycle Clients              |
| **ros2_transforms**     | TF2 Management                            | TF2 Wrappers avoiding domain dependency on `geometry_msgs`                            |
| **ros2_diagnostics**    | Health Monitoring                         | `diagnostic_updater` integration, Health entities, Frequency monitoring               |
| **ros2_bag**            | Data Recording                            | Programmatic bag recording and replay utilities (rosbag2)                             |

### Nav2 (Navigation2) Skills

| Skill Name              | Description                               | Key Components                                                                        |
| ----------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------- |
| **nav2_core_interfaces**| Nav2 Plugin Base Classes                  | Controller, GlobalPlanner, Smoother, GoalChecker, ProgressChecker, Behavior interfaces |
| **nav2_controllers**    | Controller Plugins                        | DWB (13 critics), MPPI (11 critics), RPP, Graceful, Rotation Shim                    |
| **nav2_planners**       | Planner Plugins                           | NavFn, SMAC (2D/Hybrid-A*/Lattice), Theta*, Route Server                             |
| **nav2_behavior_tree**  | BT Nodes & Recovery Behaviors             | 82 BT nodes (47 action, 22 condition, 6 control, 7 decorator), 15 default BT XMLs    |
| **nav2_costmap**        | Costmap2D Layers                          | Static, Obstacle, Voxel, Inflation, Range, Denoise layers + Keepout/Speed filters     |
| **nav2_localization**   | AMCL & Map Server                         | Particle filter, motion/sensor models, map serving/saving                             |
| **nav2_servers**        | Nav2 Server Nodes                         | Lifecycle manager, Velocity smoother, Collision monitor, Docking, Following, Python API|

## Nav2 Rules & References

| Rule File                | Description                               |
| ------------------------ | ----------------------------------------- |
| `nav2_architecture.md`   | Overall Nav2 system architecture (38 packages), data flow, design patterns |
| `nav2_parameters.md`     | Complete parameter reference for all Nav2 servers and plugins |
| `nav2_msgs_reference.md` | All 30 messages, 18 services, 19 actions with fields and error codes |

## Nav2 Source Code

Nav2 source is at `~/nav2_ws/src/navigation2/`. See `.claude/commands/nav2.md` for build, launch, and debugging commands.

## Project Structure

The project follows a strict separation of concerns:

- **src/domain/**: Pure business logic, entities, and use cases. No ROS2 dependencies.
- **src/application/**: Application services and interfaces. Orchestrates logic.
- **src/infrastructure/**: ROS2 specific implementations (Nodes, Publishers, Subscribers).

## Getting Started

To use a skill, reference the skill file (e.g., `.claude/skills/ros2_node_creation/SKILL.md`) for templates and best practices.

## Common Commands

For a comprehensive list of ROS2 commands, build instructions, and debugging tools, please refer to:

- **[ROS2 Commands Reference](.claude/commands/ros2.md)**: `colcon`, `ros2`, `rqt`, etc.
- **[Nav2 Commands Reference](.claude/commands/nav2.md)**: Nav2 launch, navigation, costmap, localization, docking, debugging commands

### Quick Reference

- **Build**: `colcon build --symlink-install`
- **Test**: `colcon test`
- **Source**: `source install/setup.bash`
