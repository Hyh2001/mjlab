"""Pendulum task environment configuration for steering."""

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
from mjlab.dynamics.steering import RslRlVecEnvWrapper, ReferenceGenerator

from mjlab.tasks.pendulum.pendulum_env_cfg import (
    SCENE_CFG, SIM_CFG, VIEWER_CONFIG, ActionCfg,
    CommandsCfg, ObservationCfg, RewardCfg, TerminationCfg, 
    SimulationCfg
)

from dynamics_training.controllers.MPPI import MPPI, MPPIConfig
from dynamics_training.dynamics_model.pendulum.pendulum_model import PendulumModel, PendulumModelConfig

class PendulumReferenceGenerator(ReferenceGenerator):
    def __init__(self, env: RslRlVecEnvWrapper, horizon: int):
        super().__init__(env, horizon)
        self.dt = self.env.unwrapped.step_dt
        
    def generate_reference(self) -> torch.Tensor:
        # forward velocity command: cos wave
        t = torch.arange(0, self.horizon * self.dt, self.dt, device=self.env.device).unsqueeze(0)  # (1, horizon)
        vel_command = torch.cos(t).repeat(self.env.num_envs, 1)  # (num_envs, horizon)
        return vel_command.unsqueeze(-1)  # (num_envs, horizon, 1)
    
mppi_cfg = MPPIConfig(
    horizon = 20,
    dt = 0.02,
    num_samples = 500,
    act_dim = 1,
    obs_dim = 5 # obs_dim * his_length
)
input_horizon = 5
output_horizon = 1
save_path = "/home/yuhao/Packages/Online_Dynamics_Finetuning/dynamics_training/test/models/test_model_perfect_data_good.npz"
model_cfg = PendulumModelConfig(
    input_dim = 1 * input_horizon + 1 * output_horizon,
    output_dim = 1 * output_horizon,
    hidden_sizes=[64, 256, 64],
    activation="relu",
    load_path=save_path
)

model = PendulumModel(model_cfg)
mppi = MPPI(model, mppi_cfg)

@dataclass 
class SteerEnvCfg():
    mppi: MPPI = field(default_factory=lambda: mppi)
    model: PendulumModel = field(default_factory=lambda: model)
    reference_generator: PendulumReferenceGenerator = field(default_factory=lambda: PendulumReferenceGenerator)
    command_indices: list[int] = field(default_factory=lambda: [2])  # index of velocity command in observation
    key: int = 42
    
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
      "operation": "abs", 
      "field": "body_mass", 
      "ranges": (0.5, 0.5), # a fixed value    
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