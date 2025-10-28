from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

### velocity command
class VelocityCommand(CommandTerm):
  cfg: VelocityCommandCfg

  def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]

    self.vel_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_vel"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel"] += (
      torch.norm(
        self.vel_command[:, :] - self.robot.data.data.qvel[0], dim=-1
      )
      / max_command_step
    )
    
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass 
  
  def _update_command(self) -> None:
    # r = torch.empty(self.num_envs, device=self.device)
    # self.vel_command[:, 0] = r.uniform_(*self.cfg.ranges.joint_vel)
    t = torch.as_tensor(self._env.sim.data.time, device=self.device, dtype=self.vel_command.dtype)
    self.vel_command[:] = torch.cos(t).unsqueeze(-1)
    # print(self.vel_command[0,:])
    
@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    asset_name : str 
    # init_velocity_prob: float = 0.0
    class_type: type[CommandTerm] = VelocityCommand
    
    
    @dataclass
    class Ranges:
        joint_vel: tuple[float, float]
        
    ranges: Ranges
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.2
        scale: float = 0.75

    viz: VizCfg = field(default_factory=VizCfg)
    

### position command
class PositionCommand(CommandTerm):
  cfg: PositionCommandCfg

  def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]

    self.pos_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.pos_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos"] += (
      torch.norm(
        self.pos_command[:, :] - self.robot.data.data.qpos[0], dim=-1
      )
      / max_command_step
    )
    # print(self.pos_command[0,:])
    
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.pos_command[env_ids, 0] = 3.14
    
  def _update_command(self) -> None:
    # self.pos_command.fill_(3.14)  
    t = torch.as_tensor(self._env.sim.data.time, device=self.device, dtype=self.pos_command.dtype)
    self.pos_command[:] = torch.sin(t).unsqueeze(-1)
     
    
@dataclass(kw_only=True)
class PositionCommandCfg(CommandTermCfg):
    asset_name : str 
    class_type: type[CommandTerm] = PositionCommand
    
    
    @dataclass
    class Ranges:
        joint_pos: tuple[float, float]
        
    ranges: Ranges
    
    @dataclass
    class VizCfg:
        z_offset: float = 0.2
        scale: float = 0.75

    viz: VizCfg = field(default_factory=VizCfg)