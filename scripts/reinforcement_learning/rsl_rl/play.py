# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from typing import Any, cast

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--metrics_steps",
    type=int,
    default=1000,
    help="Number of environment steps for play metrics. Set <= 0 to run continuously.",
)
parser.add_argument(
    "--metrics_report_interval",
    type=int,
    default=200,
    help="Print intermediate metrics every N environment steps.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata
import statistics

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
from collections import deque

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlMLPModelCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _ensure_model_cfg_containers(agent_cfg: Any, yaml_agent_cfg: dict) -> None:
    """Create typed model config containers before merging YAML actor/critic overrides."""
    for model_name in ("actor", "critic", "student", "teacher"):
        model_data = yaml_agent_cfg.get(model_name)
        if not isinstance(model_data, dict):
            continue
        if isinstance(getattr(agent_cfg, model_name, None), type(__import__("dataclasses").MISSING)):
            model_cfg = RslRlMLPModelCfg(
                hidden_dims=model_data.get("hidden_dims", []),
                activation=model_data.get("activation", "elu"),
                obs_normalization=model_data.get("obs_normalization", False),
            )
            model_cfg.use_ode = False
            model_cfg.ode_layer_index = 0
            model_cfg.ode_time = 0.1
            model_cfg.ode_method = "rk4"
            model_cfg.ode_rtol = 1.0e-3
            model_cfg.ode_atol = 1.0e-3
            dist_data = model_data.get("distribution_cfg")
            if isinstance(dist_data, dict):
                dist_class_name = dist_data.get("class_name", "GaussianDistribution")
                if dist_class_name == "GaussianDistribution":
                    model_cfg.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
                        init_std=dist_data.get("init_std", 1.0),
                        std_type=dist_data.get("std_type", "scalar"),
                    )
                elif dist_class_name == "HeteroscedasticGaussianDistribution":
                    model_cfg.distribution_cfg = RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg(
                        init_std=dist_data.get("init_std", 1.0),
                        std_type=dist_data.get("std_type", "scalar"),
                    )
            setattr(agent_cfg, model_name, model_cfg)


def _prune_ode_cfg_for_builtin_models(agent_cfg_dict: dict) -> dict:
    """Drop external ODE fields before constructing built-in RSL-RL models."""
    ode_keys = ("use_ode", "ode_layer_index", "ode_time", "ode_method", "ode_rtol", "ode_atol")
    builtin_models = {"MLPModel", "RNNModel", "CNNModel"}
    for model_name in ("actor", "critic", "student", "teacher"):
        model_cfg = agent_cfg_dict.get(model_name)
        if isinstance(model_cfg, dict) and model_cfg.get("class_name") in builtin_models:
            for key in ode_keys:
                model_cfg.pop(key, None)
    return agent_cfg_dict


def _init_play_metrics(num_envs: int, device: str | torch.device) -> dict[str, Any]:
    return {
        "rewbuffer": deque(maxlen=100),
        "lenbuffer": deque(maxlen=100),
        "cur_reward_sum": torch.zeros(num_envs, dtype=torch.float, device=device),
        "cur_episode_length": torch.zeros(num_envs, dtype=torch.float, device=device),
        "ep_extras": [],
    }


def _update_play_metrics(metrics: dict[str, Any], rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> torch.Tensor:
    if "episode" in extras:
        cast(list[dict], metrics["ep_extras"]).append(extras["episode"])
    elif "log" in extras:
        cast(list[dict], metrics["ep_extras"]).append(extras["log"])

    if rewards.ndim > 1:
        rewards = rewards.squeeze(-1)
    if dones.ndim > 1:
        dones = dones.squeeze(-1)

    cur_reward_sum = cast(torch.Tensor, metrics["cur_reward_sum"])
    cur_episode_length = cast(torch.Tensor, metrics["cur_episode_length"])
    rewbuffer = cast(deque, metrics["rewbuffer"])
    lenbuffer = cast(deque, metrics["lenbuffer"])

    cur_reward_sum += rewards
    cur_episode_length += 1

    dones = dones.reshape(-1)
    new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    if new_ids.numel() > 0:
        rewbuffer.extend(cur_reward_sum[new_ids].detach().cpu().tolist())
        lenbuffer.extend(cur_episode_length[new_ids].detach().cpu().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0

    return dones


def _compute_episode_extras_mean(ep_extras: list[dict], device: str | torch.device) -> dict[str, float]:
    if not ep_extras:
        return {}

    extras_mean: dict[str, float] = {}
    for key in ep_extras[0]:
        infotensor = torch.tensor([], device=device)
        for ep_info in ep_extras:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([value])
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
            infotensor = torch.cat((infotensor, value.to(device)))
        if infotensor.numel() > 0:
            extras_mean[key] = torch.mean(infotensor).item()
    return extras_mean


def _print_train_like_metrics(metrics: dict[str, Any], step: int, device: str | torch.device, clear_extras: bool = True):
    rewbuffer = cast(deque, metrics["rewbuffer"])
    lenbuffer = cast(deque, metrics["lenbuffer"])
    ep_extras = cast(list[dict], metrics["ep_extras"])

    print(f"[METRICS][step={step}]")
    if len(rewbuffer) > 0:
        print(
            f"[METRICS] train_like_mean_reward={statistics.mean(rewbuffer):.6f}, "
            f"train_like_mean_episode_length={statistics.mean(lenbuffer):.6f}, "
            f"train_like_completed_episodes_in_window={len(rewbuffer)}"
        )
    else:
        print("[METRICS] train_like_mean_reward=nan, train_like_mean_episode_length=nan, no completed episodes yet")

    extras_mean = _compute_episode_extras_mean(ep_extras, device)
    for key in sorted(extras_mean.keys()):
        if "/" in key:
            print(f"[METRICS] {key}={extras_mean[key]:.6f}")
        else:
            print(f"[METRICS] mean_episode_{key}={extras_mean[key]:.6f}")

    if clear_extras:
        ep_extras.clear()


def _print_play_metrics_summary(metrics: dict[str, Any], total_steps: int, device: str | torch.device):
    print("[METRICS][summary]")
    _print_train_like_metrics(metrics, total_steps, device, clear_extras=False)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # Support YAML agent configs by merging parsed dict into the default object config.
    if isinstance(agent_cfg, dict):
        yaml_agent_cfg = agent_cfg
        yaml_env_cfg = yaml_agent_cfg.pop("env", None)
        if yaml_env_cfg is not None:
            cast(Any, env_cfg).from_dict(yaml_env_cfg)
        default_agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        if isinstance(default_agent_cfg, dict):
            raise TypeError("Expected object-based RSL-RL default config for conversion from YAML overrides.")
        merged_agent_cfg = cast(Any, default_agent_cfg)
        _ensure_model_cfg_containers(merged_agent_cfg, yaml_agent_cfg)
        merged_agent_cfg.from_dict(yaml_agent_cfg)
        agent_cfg = cast(RslRlBaseRunnerCfg, merged_agent_cfg)

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_cfg_dict = _prune_ode_cfg_for_builtin_models(agent_cfg.to_dict())
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # Use non-strict loading in play mode to tolerate checkpoint/model shape drift
    # (e.g., extra legacy keys while keeping current inference graph).
    runner.load(resume_path, strict=False)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    policy_nn = None
    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        try:
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        except Exception as exc:
            print(f"[WARNING] Failed to export policy to JIT/ONNX. Continuing play without export. Error: {exc}")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    play_metrics = _init_play_metrics(env.num_envs, env.unwrapped.device)

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            dones = _update_play_metrics(play_metrics, rewards, dones, extras)

            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                assert policy_nn is not None
                policy_nn.reset(dones)

        timestep += 1

        if args_cli.metrics_steps > 0 and args_cli.metrics_report_interval > 0:
            if timestep % args_cli.metrics_report_interval == 0:
                _print_train_like_metrics(play_metrics, timestep, env.unwrapped.device)

        if args_cli.metrics_steps > 0 and timestep >= args_cli.metrics_steps:
            break

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    _print_play_metrics_summary(play_metrics, timestep, env.unwrapped.device)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
