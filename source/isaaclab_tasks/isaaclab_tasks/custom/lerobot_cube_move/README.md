## To run the lerobot trianing task

### To train
```shell
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task=Isaac-Lerobot-Cube-Move-Direct-v0 --headless --num_envs 1024
```

### To play
```shell
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task=Isaac-Lerobot-Cube-Move-Direct-v0
```