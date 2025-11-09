# TODO: 
SMPL Targets -> lulaIK -> Target Joint Angles -> RL Controller (GPU)
1. dockerfile update to install isaaclab
2. test smplx viz in isaaclab (already done get the files from other repo)
3. How to scale the joint length to robot length? scale the joint distances to humanoids
3. add pinocchio for IK
4. use IK library inside isaacsim and isaaclab to move the robot to the desired positions.
5. load SMPLX data and visulize it too. 

### To make the docker container

building
```bash
docker build -t hyperspawn:retarget . 
```

For development

```bash
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /dev:/dev --gpus=all -v /home/konu/Documents/upwork/hyperspawn_humanoid:/home/konu/Documents/upwork/hyperspawn_humanoid --name hyperspawn osrf/ros:jazzy-desktop-full
```

### To visualize SMPLX dataset
```bash
python3 visualize_smplx.py --points --skin
```

## chosing lulaIK
Lula can handle closed loop articulation where as curobo can not since it also uses URDF.
curobo is much faster since it uses gpu but it will need for us to load everyhitng using urdf.
Lula can use the articulation from our USD and since we already tested that everything works 
fine with closed loop in isaacsim adn isaaclab this should be our approach. 

### To use pyroki
1. load dropbear with IK in pyroki
```bash
cd pyroki/examples
python3 dropbear_ik.py
```
pyroki can not handle closed loop articulation even though it has pinocchio IK library as backend, since pyroki uses URDF to load the robot.

<!-- ```bash
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /dev:/dev --runtime=nvidia --gpus=all -v /home/konu/Documents/upwork/hyperspawn_humanoid:/home/konu/Documents/upwork/hyperspawn_humanoid --name hyperspawn -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:5.0.0
``` -->