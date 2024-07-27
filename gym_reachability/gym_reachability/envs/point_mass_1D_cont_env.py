import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
import os
import wandb
from .utils import signed_dist_fn_rectangle, create_grid

class PointMass1DContEnv(gym.Env):

  def __init__(self, device, cfg=None):

    self.env_cfg = cfg

    self.n = 2
    self.m = 1
    self.dt = cfg.dt
    self.mode = cfg.mode
    self.sample_inside_obs = cfg.sample_inside_obs

    self.task = cfg.task
    self.goal = (self.env_cfg.target_set.high[0] + self.env_cfg.target_set.low[0])/2
    self.goal_rad = (self.env_cfg.target_set.high[0] - self.env_cfg.target_set.low[0])/2

    self.N_x = cfg.N_x  # Number of grid points per dimension
    self.bounds = np.array([cfg.state_ranges.low, cfg.state_ranges.high]).T

    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.u_min = np.array(cfg.control_low)
    self.u_max = np.array(cfg.control_high)

    self.device = device

    # Set random seed.
    self.set_costParam()

    wall_pixels = 2
    self.wall_thickness = (self.high[0]-self.low[0])/self.N_x[0]*wall_pixels

    self.grid_x = create_grid(self.low, self.high, self.N_x)
    self.grid_x_flat = torch.from_numpy(self.grid_x.reshape(-1, self.grid_x.shape[-1])).float().to(self.device)
    self.target_T = self.target_margin(self.grid_x)
    self.obstacle_T = self.safety_margin(self.grid_x)
    
    # Time-step Parameters.
    self.time_step = cfg.dt

    # Gym variables.
    self.action_space = gym.spaces.Box(self.u_min, self.u_max)
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
      inside_obs, g_x = self.check_failure(xy_sample)
      if sample_inside_obs:
        break

    return xy_sample
  
  def get_cost(self, l_x, g_x, success, fail):
    if self.costType == 'dense_ell':
      cost = l_x
    elif self.costType == 'dense':
      cost = l_x + g_x
    elif self.costType == 'sparse':
      cost = 0.
    elif self.costType == 'max_ell_g':
      if 'reward' in self.return_type:
        cost = np.minimum(l_x, g_x)
      else:
        cost = np.maximum(l_x, g_x)
    else:
        raise ValueError("invalid cost type!")
    
    if 'reward' in self.return_type:
      cost[success] = -1.*self.reward
      cost[fail] = -1.*self.penalty
    else:
      cost[success] = self.reward
      cost[fail] = self.penalty
    return cost

  def get_done(self, state, success, fail):
    # state: shape(batch,n)
    if self.doneType == 'toEnd':
      done = self.check_within_env(state)
    elif self.doneType == 'fail':
      done = fail
    elif self.doneType == 'TF':
      done = np.logical_or(fail, success)
    elif self.doneType == 'all':
      done = np.logical_or(np.logical_or(fail, success), self.check_within_env(state))
    else:
      raise ValueError("invalid done type!")
    return done
  
  # == Dynamics ==
  def step(self, action):

    ut = np.array(action[0])
    xtp1 = self.integrate_forward(self.state, ut)

    fail, g_x = self.check_failure(self.state.reshape(1,self.n))
    success, l_x = self.check_success(self.state.reshape(1,self.n))
    done = self.get_done(self.state.reshape(1,self.n), success, fail)[0]
    cost = self.get_cost(l_x, g_x, success, fail)[0]

    self.state = xtp1
    info = {"g_x": g_x[0], "l_x": l_x[0]}
    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, ut):
    """Integrates the dynamics forward by one step.

    Returns:
        np.ndarray: next state.
    """
    xtp1 = state.copy()
    xtp1[0] = state[0] + state[1]*self.dt # x_tp1 = xt + xdot_t *  dt
    xtp1[1] = state[1] + ut*self.dt # xdot_tp1 = xdot_t + u *  dt
    return xtp1

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
  
  def find_boundary_value_fn(self, s):
    # s: shape (batch, n)
    wall_left = self.low[0] - s[...,0] + self.wall_thickness
    wall_right = s[...,0] - self.high[0] + self.wall_thickness
    return np.maximum(wall_left, wall_right)
  
  # == Getting Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent. shape (batch, n)

    Returns:
      float: postive numbers indicate being inside the failure set (safety
          violation).
    """
    # g(x)>0 is obstacle
    obstacle = signed_dist_fn_rectangle(
      s,
      np.array(self.env_cfg.obstacle_set.low), 
      np.array(self.env_cfg.obstacle_set.high), 
      obstacle=True)

    boundary = self.find_boundary_value_fn(s)
    
    gx = self.scaling * np.maximum(obstacle, boundary)

    if 'reward' in self.return_type: # g(x)<0 is obstacle
      gx = -1.*gx
    return gx
  
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

    lx = self.scaling * lx

    if 'reward' in self.return_type: # l(x)>0 is target
      lx = -1.*lx

    return lx

  def check_failure(self, state):
    g_x = self.safety_margin(state)
    if 'reward' in self.return_type: 
      return g_x<0, g_x
    else:
      return g_x>0, g_x # g(x)>0 is failure
  
  def check_success(self, state):
    l_x = self.target_margin(state)
    if 'reward' in self.return_type: 
      return l_x>0, l_x
    else:
      return l_x<0, l_x # l(x)<0 is target
    
  # == Getting Information ==
  def check_within_env(self, state):
    """Checks if the robot is still in the environment.

    Args:
        state (np.ndarray): the state of the agent. shape = (batch, n)

    Returns:
        bool: True if the agent is not in the environment.
    """
    outsideLeft = (state[...,0] <= self.bounds[0, 0])
    outsideRight = (state[...,0] >= self.bounds[0, 1])
    return np.logical_or(outsideLeft, outsideRight)

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

  def plot_value_fn(
      self, 
      value_fn, 
      grid_x, 
      target_T=None, 
      obstacle_T=None, 
      trajs=[], 
      save_dir='', 
      name='',
      debug=True
    ):
      
    save_plot_name = os.path.join(save_dir, f'Value_fn_{name}.png')
    # Plot contour plot
    plt.figure(figsize=(8, 6))

    max_V = np.max(np.abs(value_fn))
    levels=np.arange(-max_V, max_V, 0.01)
    levels=np.linspace(-max_V, max_V, 5) if len(levels) < 5 else levels
    plt.contourf(grid_x[...,0], grid_x[...,1], value_fn, levels=levels, cmap='seismic')
    
    plt.colorbar(label='Value fn')
    plt.xlabel('X')
    plt.ylabel('X dot')

    plt.contour(grid_x[...,0], grid_x[...,1], value_fn, levels=[0], colors='black', linewidths=2)

    if target_T is not None:
      targ = plt.contour(grid_x[...,0], grid_x[...,1], target_T, levels=[0], colors='green', linestyles='dashed')
      plt.clabel(targ, fontsize=12, inline=1, fmt='target')
    if obstacle_T is not None:
      obst = plt.contour(grid_x[...,0], grid_x[...,1], obstacle_T, levels=[0], colors='darkred', linestyles='dashed')
      plt.clabel(obst, fontsize=12, inline=1, fmt='obstacle')

    for traj in trajs:
      plt.plot(traj[:,0],traj[:,1])

    plt.savefig(save_plot_name)
    plt.close()
    if not debug:
      wandb.log({f"Value_fn_{name}": wandb.Image(save_plot_name)})

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

  def get_GT_value_fn(self):
    outputs = np.load("/home/saumyas/Projects/safe_control/HJR_manip/outputs/DoubleIntegrator/goto_goal_inf_horizon/value_fn_inf_horizon_DoubleIntegrator_goto_goal_grid_Nx_101_101_Nu_51_dt_0.05_T_5.00.npz")

    value_fn = outputs['value_fn']
    min_u_idx = outputs['min_u_idx']
    grid_x = outputs['grid_x']
    grid_x_u = outputs['grid_x_u']
    grid_x_tp1 = outputs['grid_x_tp1']
    target_T = outputs['target_T']
    obstacle_T = outputs['obstacle_T']
    t_series = outputs['t_series']

    return value_fn, grid_x, target_T, obstacle_T
    
    

    

    