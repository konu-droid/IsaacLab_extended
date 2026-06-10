# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
leatherback car environment for navigation and obstacle avoidance.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Leatherback-Direct-v0",
    entry_point=f"{__name__}.car_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env:LeatherbackEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-Camera-Direct-v0",
    entry_point=f"{__name__}.car_camera_nav:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_camera_nav:LeatherbackEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cam_cfg.yaml",
    },
)