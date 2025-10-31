"""Script to play a controller to explore the dynamics."""
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, cast

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from mjlab.viewer.base import EnvProtocol
from mjlab.dynamics.exploration import TrajectoryBatch, explore

ViewerChoice = Literal["auto", "native", "viser"]
ResolvedViewer = Literal["native", "viser"]

@dataclass(frozen=True)
class ExploreConfig:
  agent: Literal["zero", "random", "trained", "pre-defined"] = "trained" # pre-defined are controllers does not require training
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  save_trajectory_file: str | None = None

  
def run_exploration(task: str, cfg: ExploreConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[INFO]: Using device: {device}")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )   

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = cfg.agent == "trained"
  PRE_DEFINED_MODE = cfg.agent == "pre-defined"
  if(PRE_DEFINED_MODE):
    raise NotImplementedError("Pre-defined controller mode is not implemented yet.")
  
  if TRAINED_MODE:
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
    )
  # TODO: implement the ControllerCfg
  # elif PRE_DEFINED_MODE: 
  #   agent_cfg = cast(
  #       ControllerCfg, load_cfg_from_registry(task, "ctrl_cfg_entry_point")
  #   )

  ###################################################
  ### specify motion file path for tracking if needed
  ###################################################
  if isinstance(env_cfg, TrackingEnvCfg):
    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      env_cfg.commands.motion.motion_file = str(
        Path(artifact.download()) / "motion.npz"
      )
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        env_cfg.commands.motion.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

  ###########################################
  ### specify path for loading trained policy
  ###########################################
  log_dir: Optional[Path] = None
  resume_path: Optional[Path] = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    print(f"[INFO]: Loading experiment from: {log_root_path}")
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
    print(f"[INFO]: Loading checkpoint: {resume_path}")
    log_dir = resume_path.parent

  ##############
  ### env setup
  ##############
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = gym.make(task, cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None)

  if (TRAINED_MODE or PRE_DEFINED_MODE) and cfg.video:
    print("[INFO] Recording videos during exploration")
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=str(Path(log_dir) / "videos" / "play"),  # type: ignore[arg-type]
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  elif TRAINED_MODE:
    if isinstance(env_cfg, TrackingEnvCfg):
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
  # TODO: implement the controller for
  # elif PRE_DEFINED_MODE:
  #   pass
  
  runner.add_git_repo_to_log(__file__)
  policy = runner.get_inference_policy(device=device)

  traj_batch = explore(policy, env)
  traj_batch.save(cfg.save_trajectory_file) if cfg.save_trajectory_file else None

  env.close()



def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    ExploreConfig,
    args=remaining_args,
    default=ExploreConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_exploration(chosen_task, args)


if __name__ == "__main__":
  main()
