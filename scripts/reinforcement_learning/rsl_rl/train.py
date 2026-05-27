# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import copy
import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlMLPModelCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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
            model_cfg.ode_steps = 4
            model_cfg.derivative_scale = 0.1
            model_cfg.learn_derivative_scale = False
            model_cfg.derivative_scale_min = 0.05
            model_cfg.derivative_scale_max = 1.0
            model_cfg.zero_init_derivative = True
            model_cfg.use_gate = False
            model_cfg.gate_bias = -1.0
            model_cfg.reset_gate_bias = 0.0
            model_cfg.update_gate_bias = -1.0
            model_cfg.augment_dim = 64
            model_cfg.adapter_scale = 1.0
            model_cfg.residual_layer_index = 0
            model_cfg.rnn_hidden_dim = 128
            model_cfg.rnn_num_layers = 1
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


def _prune_unused_model_cfg(agent_cfg_dict: dict) -> dict:
    """Drop research-model fields that do not apply to the selected model class."""
    research_keys = {
        "use_ode",
        "ode_layer_index",
        "ode_time",
        "ode_method",
        "ode_rtol",
        "ode_atol",
        "ode_steps",
        "derivative_scale",
        "learn_derivative_scale",
        "derivative_scale_min",
        "derivative_scale_max",
        "zero_init_derivative",
        "use_gate",
        "gate_bias",
        "augment_dim",
        "adapter_scale",
        "residual_layer_index",
        "rnn_hidden_dim",
        "rnn_num_layers",
        "reset_gate_bias",
        "update_gate_bias",
    }
    keep_by_class = {
        "oderl.models:ODEMLPModel": {"ode_time", "ode_method", "ode_rtol", "ode_atol"},
        "oderl.models:ODERecurrentModel": {"ode_time", "ode_method", "ode_rtol", "ode_atol", "use_gate", "gate_bias"},
        "oderl.models:GatedODERecurrentModel": {"ode_time", "ode_method", "derivative_scale", "learn_derivative_scale", "derivative_scale_min", "derivative_scale_max", "zero_init_derivative", "reset_gate_bias", "update_gate_bias"},
        "oderl.models:RK4ODEMLPModel": {"ode_time", "ode_steps", "derivative_scale", "zero_init_derivative", "use_gate", "gate_bias"},
        "oderl.models:AugmentedRK4ODEMLPModel": {"ode_time", "ode_steps", "derivative_scale", "zero_init_derivative", "augment_dim", "use_gate", "gate_bias"},
        "oderl.models:ODEAdapterMLPModel": {"ode_time", "ode_steps", "derivative_scale", "zero_init_derivative", "adapter_scale", "use_gate", "gate_bias"},
    }
    builtin_models = {"MLPModel", "RNNModel", "CNNModel"}
    for model_name in ("actor", "critic", "student", "teacher"):
        model_cfg = agent_cfg_dict.get(model_name)
        if not isinstance(model_cfg, dict):
            continue
        class_name = model_cfg.get("class_name")
        keep_keys = keep_by_class.get(class_name, set())
        if class_name in builtin_models or isinstance(class_name, str):
            for key in research_keys - keep_keys:
                model_cfg.pop(key, None)
    return agent_cfg_dict


def _shape_repr(value: Any) -> str:
    """Return a compact shape/type representation for hook inputs and outputs."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, TensorDict):
        items = ", ".join(f"{key}: {tuple(tensor.shape)}" for key, tensor in value.items())
        return f"TensorDict(batch_size={list(value.batch_size)}, {items})"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_shape_repr(item)}" for key, item in value.items()) + "}"
    if value is None:
        return "None"
    return type(value).__name__


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _small_observation_batch(obs: TensorDict, max_batch: int = 2) -> TensorDict:
    batch = int(obs.batch_size[0]) if obs.batch_size else max_batch
    return obs[: min(max_batch, batch)].clone()


def _trace_model_shapes(model: nn.Module, obs: TensorDict, name: str) -> list[str]:
    lines: list[str] = []
    handles = []
    old_training = model.training
    root_id = id(model)

    def register_hooks() -> None:
        for module_name, module in model.named_modules():
            label = module_name if module_name else "<root>"
            if id(module) != root_id and any(module.children()):
                continue

            def hook(mod: nn.Module, inputs: tuple[Any, ...], output: Any, label: str = label) -> None:
                lines.append(
                    f"{label:<45s} {mod.__class__.__name__:<28s} "
                    f"in={_shape_repr(inputs)} -> out={_shape_repr(output)}"
                )

            handles.append(module.register_forward_hook(hook))

    try:
        model.eval()
        register_hooks()
        with torch.no_grad():
            output = model(obs, stochastic_output=False)
        lines.insert(0, f"{name} final_output: {_shape_repr(output)}")
    except Exception as exc:
        lines.append(f"{name} shape trace failed: {type(exc).__name__}: {exc}")
    finally:
        for handle in handles:
            handle.remove()
        model.train(old_training)
        reset = getattr(model, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
    return lines


def _save_model_debug_report(runner: Any, env: Any, log_dir: str) -> None:
    """Save model repr, parameter counts, and one forward-pass shape trace for the current run."""
    result_dir = os.path.join(log_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    alg = getattr(runner, "alg", None)
    models = [("actor", getattr(alg, "actor", None)), ("critic", getattr(alg, "critic", None))]

    arch_lines = ["# Model Architecture", ""]
    trace_lines = ["# Model Shape Trace", "", "This trace uses a small batch from env.get_observations() before training.", ""]

    try:
        obs = _small_observation_batch(env.get_observations())
        trace_lines.append(f"input_observation: {_shape_repr(obs)}")
        trace_lines.append("")
    except Exception as exc:
        obs = None
        trace_lines.append(f"failed_to_get_observations: {type(exc).__name__}: {exc}")
        trace_lines.append("")

    for name, model in models:
        if not isinstance(model, nn.Module):
            continue
        total, trainable = _count_parameters(model)
        arch_lines.extend(
            [
                f"## {name}",
                f"class: {model.__class__.__module__}.{model.__class__.__name__}",
                f"parameters_total: {total}",
                f"parameters_trainable: {trainable}",
                "",
                "```text",
                repr(model),
                "```",
                "",
            ]
        )
        trace_lines.extend([f"## {name}", f"parameters_total: {total}", ""])
        if obs is not None:
            trace_lines.extend(_trace_model_shapes(model, obs, name))
        trace_lines.append("")

    arch_path = os.path.join(result_dir, "model_architecture.txt")
    trace_path = os.path.join(result_dir, "model_shape_trace.txt")
    with open(arch_path, "w", encoding="utf-8") as f:
        f.write("\n".join(arch_lines).rstrip() + "\n")
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines).rstrip() + "\n")
    print(f"Saved model architecture to: {arch_path}")
    print(f"Saved model shape trace to: {trace_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # Support YAML agent configs by merging the parsed dict into the default object config.
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

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {prefix}_{time-stamp}_{run_name}
    log_dir_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_prefix_parts = []
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    algorithm_class_name = getattr(algorithm_cfg, "class_name", None)
    if algorithm_class_name:
        log_dir_prefix_parts.append(str(algorithm_class_name))
    env_decimation = getattr(env_cfg, "decimation", None)
    if env_decimation is not None:
        log_dir_prefix_parts.append(f"dec{env_decimation}")
    log_dir = log_dir_timestamp
    if log_dir_prefix_parts:
        log_dir = "_".join(log_dir_prefix_parts + [log_dir_timestamp])
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    agent_cfg_dict = _prune_unused_model_cfg(agent_cfg.to_dict())
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, copy.deepcopy(agent_cfg_dict), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, copy.deepcopy(agent_cfg_dict), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
    _save_model_debug_report(runner, env, log_dir)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
