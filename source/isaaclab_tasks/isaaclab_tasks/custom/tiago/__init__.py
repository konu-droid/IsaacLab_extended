# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tiago robot environment for navigation and pick-place operation.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Tiago-Nav-Direct-v0",
    entry_point=f"{__name__}.tiago_nav:TiagoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_nav:TiagoEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_nav_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Tiago-Pick-Direct-v0",
    entry_point=f"{__name__}.tiago_pick:TiagoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_pick:TiagoEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pick_cfg.yaml",
    },
)