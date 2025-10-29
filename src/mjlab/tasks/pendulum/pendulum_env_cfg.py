"""Pendulum task environment configuration."""

import math
from dataclasses import dataclass, field
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    term,
)
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.asset_zoo.robots.pendulum.pendulum_constants import PENDULUM_ROBOT_CFG
from mjlab.rl import RslRlOnPolicyRunnerCfg
# from mjlab.envs import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.tasks.pendulum import mdp

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
      terrain_type="plane",
    ),
    num_envs=64,
    extent=1.0,
    entities={"robot": PENDULUM_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="tip",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)

@dataclass
class ActionCfg:
    joint_pos: mdp.JointPositionActionCfg = term(
        mdp.JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )

# velocity tracking command
@dataclass
class CommandsCfg:
  motion: mdp.VelocityCommandCfg = term(
    mdp.VelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    ranges=mdp.VelocityCommandCfg.Ranges(
      joint_vel = (-0.5, 0.5),
    ),
  )
# position tracking command
# @dataclass
# class CommandsCfg:
#   motion: mdp.PositionCommandCfg = term(
#     mdp.PositionCommandCfg,
#     asset_name="robot",
#     resampling_time_range=(1.0e9, 1.0e9),
#     debug_vis=True,
#     ranges=mdp.PositionCommandCfg.Ranges(
#       joint_pos = (-2.0 * 3.14, 2.0 * 3.14),
#     ),
#   )
    
@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    angle: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qpos[:, 0:1])
    ang_vel: ObsTerm = term(ObsTerm, func=lambda env: env.sim.data.qvel[:, 0:1])
    command: ObsTerm = term(ObsTerm, func=mdp.generated_commands, params={"command_name": "motion"})
    def __post_init__(self):
      self.enable_corruption = True
  @dataclass
  class CriticCfg(PolicyCfg):
    pass

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)

# velocity tracking reward
def compute_tracking_reward(
    env: ManagerBasedRlEnvCfg, 
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
  """Reward tracking of joint velocity command."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ang_vel_err = env.sim.data.qvel[:, 0] - command[:, 0]
  tracking_reward =  - ang_vel_err ** 2
  return tracking_reward

# position tracking reward
# def compute_tracking_reward(
#     env: ManagerBasedRlEnvCfg, 
#     command_name: str = "motion",
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
#   """Reward tracking of joint position command."""
#   asset: Entity = env.scene[asset_cfg.name]
#   command = env.command_manager.get_command(command_name)
#   assert command is not None, f"Command '{command_name}' not found."
#   ang_pos_err = env.sim.data.qpos[:, 0] - command[:, 0]
#   tracking_reward =  - ang_pos_err ** 2
#   return tracking_reward



def compute_effort_penalty(env):
  return -0.01 * (env.sim.data.ctrl[:, 0] ** 2)

@dataclass
class RewardCfg:
  tracking_velocity: RewardTerm = term(
      RewardTerm, 
      func=compute_tracking_reward, 
      weight=10.0, 
      params={"command_name": "motion"})
  effort: RewardTerm = term(RewardTerm, func=compute_effort_penalty, weight=1.0)

# apply random push forces  
# def random_push_cart(env, env_ids, force_range=(-5, 5)):
#   n = len(env_ids)
#   random_forces = (
#     torch.rand(n, device=env.device) *
#     (force_range[1] - force_range[0]) +
#     force_range[0]
#   )
#   env.sim.data.qfrc_applied[env_ids, 0] = random_forces

@dataclass
class EventCfg:
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "position_range": (-0.1, 0.1),
      "velocity_range": (0.0, 0.0),
    },
  )
  
def check_aggressive_behavior(env):
  return env.sim.data.qpos[:, 0].abs() > math.radians(90)

@dataclass
class TerminationCfg:
  timeout: DoneTerm = term(DoneTerm, func=lambda env: False, time_out=True)
  # timeout: DoneTerm = term(
  #   DoneTerm,
  #   func=lambda env: env.sim.data.time >= 10,
  #   time_out=True,
  # )
  # tipped: DoneTerm = term(DoneTerm, func=check_aggressive_behavior, time_out=False)

SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.02,
    iterations=1,
  ),
)

@dataclass
class PendulumEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 1
  episode_length_s: float = 10.0
#   is_finite_horizon: bool = True