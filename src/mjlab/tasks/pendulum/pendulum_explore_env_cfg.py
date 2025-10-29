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
    EventCfg, SimulationCfg
)

# apply random push torque to the joint
def random_push_cart(env, env_ids, force_range=(-1, 1)):
  n = len(env_ids)
  random_torques = (
    torch.rand(n, device=env.device) *
    (force_range[1] - force_range[0]) +
    force_range[0]
  )
  env.sim.data.qfrc_applied[env_ids, 0] = random_torques

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