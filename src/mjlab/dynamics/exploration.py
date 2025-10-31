# ...existing code...
from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper

TensorLike = Union[torch.Tensor, np.ndarray]

@dataclass
class TrajectoryBatch:
    """
    Batched trajectories for n_envs environments (PyTorch-first).

    Fields:
      - obs:          (n_envs, T+1, obs_shape)         torch.Tensor
      - actions:      (n_envs, T,   action_shape)      torch.Tensor
      - dones:        (n_envs, T)                      torch.BoolTensor
      - states: (n_envs, T+1, state_shape)             torch.Tensor, (generalized coordinate, generalized velocities)
      - masks:        Optional[(n_envs, T)]            torch.Tensor (float mask)
      - lengths:      Optional[(n_envs,)]              torch.LongTensor
      - infos:        Optional[Any]
    """
    obs: torch.Tensor
    actions: torch.Tensor
    dones: torch.Tensor
    states: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    infos: Optional[Any] = None

    def __post_init__(self) -> None:
        n_envs = int(self.obs.shape[0])
        T_plus_1 = int(self.obs.shape[1])
        assert tuple(self.actions.shape[:2]) == (n_envs, T_plus_1 - 1), "actions must be (n_envs, T, ...)"
        assert tuple(self.dones.shape[:2]) == (n_envs, T_plus_1 - 1), "dones must be (n_envs, T)"
        if self.states is not None:
            assert tuple(self.states.shape[:2]) == (n_envs, T_plus_1), "states must be (n_envs, T+1, ...)"
        if self.masks is not None:
            assert tuple(self.masks.shape[:2]) == (n_envs, T_plus_1 - 1), "masks must be (n_envs, T)"
        if self.lengths is not None:
            assert int(self.lengths.shape[0]) == n_envs, "lengths must be (n_envs,)"

    @property
    def n_envs(self) -> int:
        return int(self.obs.shape[0])

    @property
    def T(self) -> int:
        return int(self.obs.shape[1] - 1)

    @classmethod
    def zeros(
        cls,
        n_envs: int,
        T: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        state_shape: Optional[Tuple[int, ...]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "TrajectoryBatch":
        dev = torch.device(device)
        obs = torch.zeros((n_envs, T + 1, *obs_shape), dtype=dtype, device=dev)
        actions = torch.zeros((n_envs, T, *action_shape), dtype=dtype, device=dev)
        dones = torch.zeros((n_envs, T), dtype=torch.bool, device=dev)
        masks = torch.ones((n_envs, T), dtype=dtype, device=dev)
        lengths = torch.zeros((n_envs,), dtype=torch.long, device=dev)
        states = None
        if state_shape is not None:
            states = torch.zeros((n_envs, T + 1, *state_shape), dtype=dtype, device=dev)
        return cls(obs=obs, actions=actions, dones=dones, states=states, masks=masks, lengths=lengths)

    def slice(self, start: int = 0, end: Optional[int] = None) -> "TrajectoryBatch":
        """Return a time-slice [start:end) (obs -> start..end+1)."""
        if end is None:
            end = self.T
        return TrajectoryBatch(
            obs=self.obs[:, start : end + 1].clone(),
            actions=self.actions[:, start:end].clone(),
            dones=self.dones[:, start:end].clone(),
            states=None if self.states is None else self.states[:, start : end + 1].clone(),
            masks=None if self.masks is None else self.masks[:, start:end].clone(),
            lengths=None if self.lengths is None else torch.clamp(self.lengths - start, min=0, max=end - start),
            infos=self.infos,
        )

    def to_torch(self, device: Union[str, torch.device] = "cpu") -> "TrajectoryBatch":
        """Ensure all array fields are torch.Tensor on `device`."""
        dev = torch.device(device)
        def _to_t(x: Optional[TensorLike]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            return torch.from_numpy(np.asarray(x)).to(dev)
        return TrajectoryBatch(
            obs=_to_t(self.obs),
            actions=_to_t(self.actions),
            dones=_to_t(self.dones),
            states=_to_t(self.states),
            masks=_to_t(self.masks),
            lengths=_to_t(self.lengths),
            infos=self.infos,
        )

    def to_numpy(self) -> "TrajectoryBatch":
        """Return a copy where torch tensors are converted to NumPy (CPU)."""
        def _to_np(x: Optional[TensorLike]) -> Optional[np.ndarray]:
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        return TrajectoryBatch(
            obs=torch.as_tensor(_to_np(self.obs)),
            actions=torch.as_tensor(_to_np(self.actions)),
            dones=torch.as_tensor(_to_np(self.dones)).to(torch.bool),
            states=None if self.states is None else torch.as_tensor(_to_np(self.states)),
            masks=None if self.masks is None else torch.as_tensor(_to_np(self.masks)),
            lengths=None if self.lengths is None else torch.as_tensor(_to_np(self.lengths)).to(torch.long),
            infos=self.infos,
        )

    def set_states_at(self, time_idx: int, states: TensorLike) -> "TrajectoryBatch":
        """
        Return a new batch with states updated at time index `time_idx`.
        `states` shape: (n_envs, *state_shape) and will be converted to torch.Tensor on same device.
        """
        if self.states is None:
            raise ValueError("states is not initialized; create batch with state_shape via TrajectoryBatch.zeros(..., state_shape=...)")
        dev = self.states.device
        if not isinstance(states, torch.Tensor):
            states_t = torch.from_numpy(np.asarray(states)).to(dev)
        else:
            states_t = states.to(dev)
        new_rs = self.states.clone()
        new_rs[:, time_idx] = states_t
        return TrajectoryBatch(
            obs=self.obs,
            actions=self.actions,
            dones=self.dones,
            states=new_rs,
            masks=self.masks,
            lengths=self.lengths,
            infos=self.infos,
        )

    def save(self, filepath: str) -> None:
        """Save the trajectory batch to a file using torch.save.

        The file will contain a dictionary of tensors/objects so it is robust
        to minor class changes and easy to load on a specific device.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        def _prepare(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            # try to convert numpy/other array-likes to tensor on CPU
            return torch.as_tensor(np.asarray(x))

        payload = {
            "obs": _prepare(self.obs),
            "actions": _prepare(self.actions),
            "dones": _prepare(self.dones),
        }
        if self.states is not None:
            payload["states"] = _prepare(self.states)
        if self.masks is not None:
            payload["masks"] = _prepare(self.masks)
        if self.lengths is not None:
            payload["lengths"] = _prepare(self.lengths)
        if self.infos is not None:
            payload["infos"] = self.infos

        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[Union[str, torch.device]] = "cpu") -> "TrajectoryBatch":
        """Load a TrajectoryBatch saved with save(). Moves tensors to `device` if provided."""
        payload = torch.load(filepath, map_location="cpu")
        def _move(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device)
            # fallback: convert numpy/array-like to tensor on device
            return torch.as_tensor(np.asarray(x)).to(device)

        return cls(
            obs=_move(payload.get("obs")),
            actions=_move(payload.get("actions")),
            dones=_move(payload.get("dones")),
            states=_move(payload.get("states")),
            masks=_move(payload.get("masks")),
            lengths=_move(payload.get("lengths")),
            infos=payload.get("infos"),
        )
    
def explore(policy: Any, env: RslRlVecEnvWrapper, )-> TrajectoryBatch:
    """
    Explore the environment using the given policy and collect a trajectory batch.

    Args:
      - policy: A callable that takes observations and returns actions.
      - env: An instance of RslRlVecEnvWrapper.

    Returns:
      - A TrajectoryBatch containing the collected trajectories.
    """
    n_envs = env.num_envs
    max_steps = env.max_episode_length # in steps
    obs_t, _ = env.reset()
    obs_policy = obs_t["policy"]
    obs_shape = obs_policy.shape[1]
    action_shape = env.unwrapped.action_manager.total_action_dim

    obs = torch.zeros((n_envs, 1, obs_shape), dtype=torch.float32, device=env.unwrapped.device)
    actions = torch.zeros((n_envs, 0, action_shape), dtype=torch.float32, device=env.unwrapped.device)
    dones = torch.zeros((n_envs, 0), dtype=torch.bool, device=env.unwrapped.device)

    obs[:, 0] = obs_policy.to(device=env.unwrapped.device)

    done_flags = torch.zeros((n_envs,), dtype=torch.bool, device=env.unwrapped.device)
    t = 0

    while t <= max_steps - 1:
        with torch.no_grad():
            obs_t = env.get_observations()
            action_t = policy(obs_t)
            obs_t, reward_t, done_t, _= env.step(action_t)
        # breakpoint()
        actions = torch.cat([actions, action_t.unsqueeze(1)], dim=1)
        dones = torch.cat([dones, done_t.unsqueeze(1)], dim=1)
        obs = torch.cat([obs, obs_t["policy"].unsqueeze(1)], dim=1)
        t += 1
        print(f"Exploration step: {t}/{max_steps}", end="\r")
    # breakpoint()
    return TrajectoryBatch(obs=obs, actions=actions, dones=dones)
