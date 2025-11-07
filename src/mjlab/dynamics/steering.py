""" 
    This file contains the implementation for using trained closed-loop dynamics model with 
    MPPI to steer a policy
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import sys
from typing import Any, Optional, Tuple, Union, List
import numpy as np
import torch
from tensordict import TensorDict
import jax
import jax.numpy as jnp
from abc import abstractmethod

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.managers.manager_term_config import ManagerTermBaseCfg
from mjlab.dynamics.exploration import TrajectoryBatch

from dynamics_training.controllers.MPPI import Model, MPPI
from dynamics_training.utils.tensor_utils import torch_to_jax, jax_to_torch

class ReferenceGenerator(ABC):
    """Base class for generating reference commands for different morphologies."""
    
    def __init__(self, env: RslRlVecEnvWrapper, horizon: int):
        self.env = env
    
    @abstractmethod
    def generate_reference(self) -> torch.Tensor:
        """Generate a sequence of reference commands for the given horizon.
        
        Args:
            env: The RL environment wrapper
            horizon: The horizon length for which to generate reference commands
            
        Returns:
            reference: (n_envs, horizon, command_dim) array of reference commands
        """
        pass


def steer(policy: Any, mppi: MPPI, env: RslRlVecEnvWrapper, reference_generator: ReferenceGenerator, key: int, command_indices: List[int]) -> TrajectoryBatch:
    """ 
        Use the trained dynamics model with MPPI to steer a policy in the given environment.
        1. Instantiate policy, MPPI controller with the dynamics model and environment
        2. Initialize obs, actions, rewards and done buffers
        3. For each timestep:
            a. Use MPPI to take in the current observation, distill the commands from the observations, 
               and predict the new command
            b. Apply the steered observation to the policy and output an action
            c. Step the environment with the new action
            c. Store the observation, action, reward, and done flag
            
        @return: TrajectoryBatch containing the collected trajectory execution data
    """ 
    n_envs = env.num_envs
    max_steps = env.max_episode_length # in steps
    key = jax.random.PRNGKey(key)
    
    obs_t, _ = env.reset()
    obs_policy = obs_t["policy"]
    obs_shape = obs_policy.shape[1]
    action_shape = env.unwrapped.action_manager.total_action_dim

    obs_before = torch.zeros((n_envs, 1, obs_shape), dtype=torch.float32, device=env.unwrapped.device)
    obs_after = torch.zeros((n_envs, 1, obs_shape), dtype=torch.float32, device=env.unwrapped.device)
    actions = torch.zeros((n_envs, 0, action_shape), dtype=torch.float32, device=env.unwrapped.device)
    dones = torch.zeros((n_envs, 0), dtype=torch.bool, device=env.unwrapped.device)

    obs_before[:, 0] = obs_policy.to(device=env.unwrapped.device)
    ## TODO: we need to take the command manager out for building a sequence of reference velocities
    reference = torch_to_jax(reference_generator.generate_reference())  # (n_envs, horizon, command_dim)
    new_command, new_key, previous_solution = mppi.act(obs_policy, reference, key, previous_solution)    
    new_command = jax_to_torch(new_command)
    obs_after[:, 0] = _apply_command_to_obs(obs_policy, new_command, command_indices)
    
    done_flags = torch.zeros((n_envs,), dtype=torch.bool, device=env.unwrapped.device)
    t = 0
    
    while t <= max_steps - 1:
        with torch.no_grad():
            obs_t = env.get_observations()
            
            # done in Jax
            obs_t_jax = torch_to_jax(obs_t["policy"])
            reference = torch_to_jax(reference_generator.generate_reference())  # (n_envs, horizon, command_dim)
            new_command, new_key, previous_solution = mppi.act(obs_t_jax, reference, new_key, previous_solution)    
            new_command = jax_to_torch(new_command)

            # back to torch
            obs_t_new = _apply_command_to_obs(obs_t["policy"], new_command, command_indices)
            action_t = policy(obs_t_new)
            obs_t, reward_t, done_t, _= env.step(action_t)
        # breakpoint()
        actions = torch.cat([actions, action_t.unsqueeze(1)], dim=1)
        dones = torch.cat([dones, done_t.unsqueeze(1)], dim=1)
        obs_before = torch.cat([obs_before, obs_t["policy"].unsqueeze(1)], dim=1)
        obs_after = torch.cat([obs_after, obs_t_new["policy"].unsqueeze(1)], dim=1)
        t += 1
        print(f"Exploration step: {t}/{max_steps}", end="\r")
    # breakpoint()
    return TrajectoryBatch(obs=obs_before, actions=actions, dones=dones, infos=obs_after) # use infos to store the obs_after
    
def _apply_command_to_obs(obs: torch.Tensor, command: torch.Tensor, command_index: list[int]) -> torch.Tensor:
    """ Apply the given command to the observation at the specified indices.
    
    Args:
        obs: (n_envs, obs_dim) tensor of observations
        command: (n_envs, command_dim) tensor of commands to apply
        command_index: List of indices in the observation where the command should be applied
    
    Returns:
        new_obs: (n_envs, obs_dim) tensor of observations with applied commands
    """
    new_obs = obs.clone()
    for i, idx in enumerate(command_index):
        new_obs[:, idx] = command[:, i]
    return new_obs