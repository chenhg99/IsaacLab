# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rsl_rl_yaml_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg.yaml",
        "rsl_rl_play_yaml_cfg_entry_point": f"{agents.__name__}:rsl_rl_play_cfg.yaml",
        "rsl_rl_oderl_cfg_entry_point": "oderl.configs:rsl_rl_oderl_ant.yaml",
        "rsl_rl_oderl_play_cfg_entry_point": "oderl.configs:rsl_rl_oderl_ant_play.yaml",
        "rsl_rl_resnet_cfg_entry_point": "oderl.configs:rsl_rl_resnet_ant.yaml",
        "rsl_rl_resnet_play_cfg_entry_point": "oderl.configs:rsl_rl_resnet_ant_play.yaml",
        "rsl_rl_gru_cfg_entry_point": "oderl.configs:rsl_rl_gru_ant.yaml",
        "rsl_rl_gru_play_cfg_entry_point": "oderl.configs:rsl_rl_gru_ant_play.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
