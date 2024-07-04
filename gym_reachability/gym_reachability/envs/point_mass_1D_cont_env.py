import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
import os
from .utils import signed_dist_fn_rectangle

class PointMass1DContEnv(gym.Env):

  def __init__(self, device, cfg=None):

    self.env_cfg = cfg

    self.n = 2
    self.m = 1
    self.dt = cfg.dt
    self.mode = cfg.mode
    
    self.task = cfg.task
    self.goal = (self.env_cfg.target_set.high[0] + self.env_cfg.target_set.low[0])/2
    self.goal_rad = (self.env_cfg.target_set.high[0] - self.env_cfg.target_set.low[0])/2

    self.N_x = cfg.N_x  # Number of grid points per dimension
    self.bounds = np.array([cfg.state_ranges.low, cfg.state_ranges.high]).T

    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = cfg.sample_inside_obs
    self.device = device

    # Set random seed.
    self.set_costParam()

    X = [np.linspace(self.low[i], self.high[i], self.N_x[i]) for i in range(self.n)]
    self.grid_x = np.stack(np.meshgrid(*X, indexing='ij'), axis=-1)
    self.grid_x_flat = torch.from_numpy(self.grid_x.reshape(-1, self.grid_x.shape[-1])).float().to(self.device)
    self.target_T = self.target_margin(self.grid_x)
    self.obstacle_T = self.safety_margin(self.grid_x)
    
    # Time-step Parameters.
    self.time_step = cfg.dt

    # Gym variables.
    self.action_space = gym.spaces.Box(np.array(cfg.control_low), np.array(cfg.control_high))
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval / 2),
        np.float32(self.midpoint + self.interval / 2)
    )
    self.viewer = None


    self.state = np.zeros(self.n)
    self.doneType = cfg.doneType

    # Visualization Parameters
    self.visual_initial_states = [
      np.array([-1.75, 0.]),
      np.array([1.5, 0.]),
    ]

    self.device = device

  def reset(self, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state the environment has been reset to.
    """
    if start is None:
      self.state = self.sample_random_state(
          sample_inside_obs=self.sample_inside_obs
      )
    else:
      self.state = start

    return np.copy(self.state)

  def sample_random_state(self, sample_inside_obs=False):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
        the obstacles if True. Defaults to False.

    Returns:
        np.ndarray: sampled initial state.
    """
    inside_obs = True
    # Repeat sampling until outside obstacle if needed.
    while inside_obs:
      xy_sample = np.random.uniform(low=self.low, high=self.high)
      g_x = self.safety_margin(xy_sample)
      inside_obs = (g_x > 0)
      if sample_inside_obs:
        break

    return xy_sample
  
  # == Dynamics ==
  def step(self, action):

    ut = np.array(action[0])
    state, [l_x, g_x] = self.integrate_forward(self.state, ut)
    self.state = state

    fail = g_x > 0
    success = l_x <= 0

    # = `cost` signal
    if self.mode == 'RA':
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.
    else:
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        if self.costType == 'dense_ell':
          cost = l_x
        elif self.costType == 'dense':
          cost = l_x + g_x
        elif self.costType == 'sparse':
          cost = 0. * self.scaling
        elif self.costType == 'max_ell_g':
          cost = max(l_x, g_x)
        else:
          raise ValueError("invalid cost type!")

    # = `done` signal
    if self.doneType == 'toEnd':
      done = self.check_within_env(self.state)
    elif self.doneType == 'fail':
      done = fail
    elif self.doneType == 'TF':
      done = fail or success
    elif self.doneType == 'all':
      done = fail or success or self.check_within_env(self.state)
    else:
      raise ValueError("invalid done type!")

    # = `info`
    if done and self.doneType == 'fail':
      info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    
    if 'reward' in self.return_type:
      cost = -1.*cost
    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, ut):
    """Integrates the dynamics forward by one step.

    Returns:
        np.ndarray: next state.
    """
    xtp1 = state.copy()
    xtp1[0] = state[0] + state[1]*self.dt # x_tp1 = xt + xdot_t *  dt
    xtp1[1] = state[1] + ut*self.dt # xdot_tp1 = xdot_t + u *  dt

    l_x = self.target_margin(xtp1.reshape(1,self.n))
    g_x = self.safety_margin(xtp1.reshape(1,self.n))

    info = np.array([l_x[0], g_x[0]])
    return xtp1, info

  # == Setting Hyper-Parameters ==
  def set_costParam(self):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.

    Args:
        penalty (float, optional): cost when entering the obstacles or crossing
            the environment boundary. Defaults to 1.0.
        reward (float, optional): cost when reaching the targets.
            Defaults to -1.0.
        costType (str, optional): providing extra information when in
            neither the failure set nor the target set. Defaults to 'sparse'.
        scaling (float, optional): scaling factor of the cost. Defaults to 1.0.
    """
    self.penalty = self.env_cfg.penalty
    self.reward = self.env_cfg.reward
    self.costType = self.env_cfg.costType
    self.scaling = self.env_cfg.scaling
    self.return_type = self.env_cfg.return_type



  def set_bounds(self, bounds):
    """Sets the boundary and the observation_space of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

    # Double the range in each state dimension for Gym interface.
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2), np.float32(midpoint + interval/2)
    )

  def set_sample_type(self, sample_inside_obs=False, verbose=False):
    """Sets the type of the sampling method.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.sample_inside_obs = sample_inside_obs
    if verbose:
      print("sample_inside_obs-{}".format(self.sample_inside_obs))

  # == Getting Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent. shape (batch, n)

    Returns:
      float: postive numbers indicate being inside the failure set (safety
          violation).
    """
    gx = signed_dist_fn_rectangle(
      s,
      np.array(self.env_cfg.obstacle_set.low), 
      np.array(self.env_cfg.obstacle_set.high), 
      obstacle=True)
    return self.scaling * gx
  

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent. shape (batch, n)

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """

    # l(x)<0 is target
    lx = signed_dist_fn_rectangle(
      s,
      np.array(self.env_cfg.target_set.low), 
      np.array(self.env_cfg.target_set.high),)

    return self.scaling * lx

  # == Getting Information ==
  def check_within_env(self, state):
    """Checks if the robot is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: True if the agent is not in the environment.
    """
    outsideLeft = (state[0] <= self.bounds[0, 0])
    outsideRight = (state[0] >= self.bounds[0, 1])
    return outsideLeft or outsideRight

  def get_warmup_examples(self, num_warmup_samples=100):
    """Gets the warmup samples.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values, here we used max{ell, g}.
    """

    xy_samples = np.random.uniform(low=self.low, high=self.high, size=(num_warmup_samples, 4))

    l_x = self.target_margin(xy_samples)
    g_x = self.safety_margin(xy_samples)

    heuristic_v = np.stack([l_x, g_x],axis=1)

    return xy_samples, heuristic_v

  def plot_value_fn(self, value_fn, trajs=[], save_dir='', name=''):
      
    save_plot_name = os.path.join(save_dir, f'Value_fn_{name}.png')
    # Plot contour plot
    plt.figure(figsize=(8, 6))

    max_V = np.max(np.abs(value_fn))
    levels=np.arange(-max_V, max_V, 0.01)
    levels=np.linspace(-max_V, max_V, 11) if len(levels) < 11 else levels
    plt.contourf(self.grid_x[...,0], self.grid_x[...,1], value_fn, levels=levels, cmap='seismic')
    
    plt.colorbar(label='Value fn')
    plt.xlabel('X')
    plt.ylabel('X dot')

    plt.contour(self.grid_x[...,0], self.grid_x[...,1], value_fn, levels=[0], colors='black', linewidths=2)

    targ = plt.contour(self.grid_x[...,0], self.grid_x[...,1], self.target_T, levels=[0], colors='green', linestyles='dashed')
    plt.clabel(targ, fontsize=12, inline=1, fmt='target')
    obst = plt.contour(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T, levels=[0], colors='darkred', linestyles='dashed')
    plt.clabel(obst, fontsize=12, inline=1, fmt='obstacle')

    for traj in trajs:
      plt.plot(traj[:,0],traj[:,1])

    plt.savefig(save_plot_name)
    plt.close()

  def plot_env(self, save_dir=''):
      
    save_plot_name = os.path.join(save_dir, f'target_and_obstacle_set_PointMass1DContEnv.png')

    fig, axes = plt.subplots(3, figsize=(12, 12))

    max_V = np.max(np.abs(self.target_T))
    levels=np.arange(-max_V, max_V, 0.01)
    levels=np.linspace(-max_V, max_V, 11) if len(levels) < 11 else levels
        
    ctr_t = axes[0].contourf(self.grid_x[...,0], self.grid_x[...,1], self.target_T, levels=levels, cmap='seismic')
    targ = axes[0].contour(self.grid_x[...,0], self.grid_x[...,1], self.target_T, levels=[0], colors='black')
    axes[0].set_title(f'Target set l(x)')
    axes[0].set_xlabel('X')
    axes[0].set_xlabel('X dot')
    axes[0].clabel(targ, fontsize=12, inline=1, fmt='target')
    fig.colorbar(ctr_t, ax=axes[0])


    max_V = np.max(np.abs(self.obstacle_T))
    levels=np.arange(-max_V, max_V, 0.01)
    levels=np.linspace(-max_V, max_V, 11) if len(levels) < 11 else levels
    ctr_o = axes[1].contourf(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T, levels=levels, cmap='seismic')
    obst = axes[1].contour(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T, levels=[0], colors='black')
    axes[1].set_title(f'Obstacle set g(x)')
    axes[1].set_xlabel('X')
    axes[1].set_xlabel('X dot')
    # axes[1].clabel(obst, fontsize=12, inline=1, fmt='obstacle')
    fig.colorbar(ctr_o, ax=axes[1])

    ra = np.maximum(self.obstacle_T, self.target_T)

    ctr_ra = axes[2].contourf(self.grid_x[...,0], self.grid_x[...,1], ra, levels=np.arange(-2, 2, 0.1), cmap='seismic')
    axes[2].contour(self.grid_x[...,0], self.grid_x[...,1], ra, levels=[0], colors='black')
    axes[2].set_title(f'Terminal Value fn')
    axes[2].set_xlabel('X')
    axes[2].set_xlabel('X dot')
    fig.colorbar(ctr_ra, ax=axes[2])

    plt.savefig(save_plot_name)
    plt.close()