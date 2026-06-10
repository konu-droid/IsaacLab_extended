### Please install isaaclab following the guide 
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-

**Note:**  Please ensure that skrl is install with isaaclab

### Setting up the tiago RL task
1. Copy the custom folder to "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
2. Please change the file location of the tiago usd file in "custom/tiago/isaaclab_assets" on line 17 usd_path to match the path of your usd file.
3. To start the training run the following in the root of the isaaclab folder
    ```shell
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task=Isaac-Tiago-Direct-v0 --headless
    ```
    this should start the training of the robot for the provided task ( navigation for now ). Remove the --headless argument if you want to see the robot training.
4. After the network is trained you can use the following command to records a small video fo the trained policy. 
    ```shell
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task=Isaac-Tiago-Nav-Direct-v0 --video --video_length 600 --num_envs 64
    ```
5. The file provided directly controls the wheels of the robot to try to move it to the red cube. 
6. Please checkout the _get_observation and _get_reward functions in tiago_nav.py files are thee are the functions which basically refine what the robot see and what are the reward funnctionns give the robot so that it learns.
7. the tiago.py file in isaaclab_assets is to load and configure the tiago robot joints for use with isaaclab libraries. 
8. the .yaml files in agents folder define the hyperparameters for our ppo network.
9. init.py is the file where we register our environment as isaaclab task.