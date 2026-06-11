# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Lerobot-Cube-Move-Direct-v0",
    entry_point=f"{__name__}.lerobot_cube_move_env:LerobotCubeMoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobot_cube_move_env_cfg:LerobotCubeMoveEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lerobot-IK-PnP-Direct-v0",
    entry_point=f"{__name__}.lerobot_IK_pnp:LerobotIKPnPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobot_IK_pnp_cfg:LerobotIKPnPEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_ik_pnp_cfg.yaml",
    },
)