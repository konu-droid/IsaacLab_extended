"""TiagoPRO right-arm mobile pick (pill bottle) — Isaac Lab RL task."""

import gymnasium as gym
from . import agents

from .pick_env import PickEnv
from .pick_env_cfg import PickEnvCfg

gym.register(
    id="Isaac-TiagoPro-Pick-PillBottle-Direct-v0",
    entry_point=f"{__name__}.pick_env:PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_env_cfg:PickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config.ppo_cfg:PickPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pick_cfg.yaml",
    },
)

__all__ = ["PickEnv", "PickEnvCfg"]
