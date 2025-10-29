import gymnasium as gym

gym.register(
  id="Mjlab-Pendulum",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.pendulum_env_cfg:PendulumEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.pendulum_env_cfg:RslRlOnPolicyRunnerCfg",
  },
)

gym.register(
  id="Mjlab-Pendulum-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.pendulum_env_cfg:PendulumEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.pendulum_env_cfg:RslRlOnPolicyRunnerCfg",
  },
)

gym.register(
  id="Mjlab-Pendulum-Explore",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.pendulum_explore_env_cfg:PendulumExploreEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.pendulum_explore_env_cfg:RslRlOnPolicyRunnerCfg",
  },
)