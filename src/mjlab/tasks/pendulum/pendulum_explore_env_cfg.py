"""Pendulum task environment configuration for exploration."""

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

from mjlab.tasks.pendulum.pendulum_env_cfg import (
    SCENE_CFG, SIM_CFG, VIEWER_CONFIG, ActionCfg,
    CommandsCfg, ObservationCfg, RewardCfg, TerminationCfg, 
    SimulationCfg
)

@dataclass
class EventCfg:
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "position_range": (-0.1, 0.1),
      "velocity_range": (0.0, 0.0),
    },
  )
  body_mass_randomization: EventTerm = term(
    EventTerm,
    mode="startup", 
    func=mdp.randomize_field,
    params={ 
      "asset_cfg": SceneEntityCfg("robot", body_names=["tip"]),
      "operation": "add", 
      "field": "body_mass", 
      "ranges": ( -0.02, 0.02),    
    }, 
  )
    


@dataclass
class PendulumExploreEnvCfg(ManagerBasedRlEnvCfg):
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
  episode_length_s: float = 100.0
#   is_finite_horizon: bool = True