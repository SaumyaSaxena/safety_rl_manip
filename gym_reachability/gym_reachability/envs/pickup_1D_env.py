"""
This module implements an environment considering the 1D point object dynamics.
This environemnt shows comparison between reach-avoid Q-learning and Sum
(Lagrange) Q-learning.
envType:
    'basic': corresponds to Fig. 1 and 2 in the paper.
    'show': corresponds to Fig. 3 in the paper.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
import os


class Pickup1DEnv(gym.Env):

  def __init__(self, device, cfg):

    self.env_cfg = cfg

    self.n = 4
    self.m = 1
    self.N_O = 2
    
    self.thresh = cfg.thresh
    self.task = cfg.task
    self.goal = cfg.goal

    self.N_x = cfg.N_x  # Number of grid points per dimension
    self.bounds = np.array([cfg.state_ranges.low, cfg.state_ranges.high]).T

    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.device = device

    self.set_costParam()

    X = [np.linspace(self.low[i], self.high[i], self.N_x[i]) for i in range(4)]
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

    if self.doneType == 'toEnd':
        self.sample_inside_obs = True
    elif self.doneType == 'TF' or self.doneType == 'fail':
        self.sample_inside_obs = False

    # Visualization Parameters
    self.visual_initial_states = [
      np.array([-0.5, 0., 0., 0.]),
      np.array([0.5, 0., 0., 0.]),
    ]

    # Define dynamics parameters
    Apart = np.array(([1., cfg.dt],
                      [0., 1.]))
    self._A = np.zeros((4,4))
    self._A[:2, :2] = Apart
    self._A[2:4, 2:4] = Apart
        
    self._B1 = np.array(([0.],
      [cfg.dt/cfg.gripper_mass],
      [0.], 
      [0.]))
    self._B2 = np.array(([0.],
      [cfg.dt/(cfg.gripper_mass+cfg.object_mass)],
      [0.],
      [cfg.dt/(cfg.gripper_mass+cfg.object_mass)]))
    self._C = np.array(([1., 0., 0., 0.],
      [0., cfg.gripper_mass/(cfg.gripper_mass+cfg.object_mass), 0., cfg.object_mass/(cfg.gripper_mass+cfg.object_mass)], 
      [0., 0., 1., 0.], 
      [0., cfg.gripper_mass/(cfg.gripper_mass+cfg.object_mass), 0., cfg.object_mass/(cfg.gripper_mass+cfg.object_mass)]))
    

  def reset(self, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state the environment has been reset to.
    """
    self._current_timestep = 0
    if start is None:
      self.state = self.sample_random_state(
          sample_inside_obs=self.sample_inside_obs
      )
    else:
      self.state = start

    self._modetm1 = 0
    self._modet = self._find_mode(self.state[0], self.state[2])
    return np.copy(self.state)

  def sample_random_state(self, sample_inside_obs=False):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
        the obstacles if True. Defaults to False.

    Returns:
        np.ndarray: sampled initial state.
    """
    inside_obs_or_colliding = True
    # Repeat sampling until outside obstacle if needed.
    while inside_obs_or_colliding:
      xy_sample = np.random.uniform(low=self.low, high=self.high)
      g_x = self.safety_margin(xy_sample)
      colliding = self._is_in_collision(xy_sample)
      inside_obs_or_colliding = (g_x > 0) or colliding
      if sample_inside_obs and not colliding: # should not be colliding but can be inside obstacle
        break

    return xy_sample

  def _is_in_collision(self, state):
    dist = np.linalg.norm(state[0] - state[2])
    return dist < self.thresh

  def _find_mode(self, gripper_pos, block_pos):
    if np.linalg.norm(gripper_pos - block_pos) < self.thresh:
      return 1
    else:
      return 0
  
  # == Dynamics ==
  def step(self, action):
    """Evolves the environment one step forward under given action.
    Args:
        action (float)
    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """
    self._current_timestep += 1
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

  def integrate_forward(self, state, ut):
    """Integrates the dynamics forward by one step.

    Returns:
        np.ndarray: next state.
    """
    xt = state.reshape(4,1)
    if self._modet == self._modetm1:
      self.At = self._A.copy()
      if self._modet == 0:
          xtp1 = self._A@xt + self._B1@ut
          self.Bt = self._B1.copy()
      else:
          xtp1 = self._A@xt + self._B2@ut
          self.Bt = self._B2.copy()
    else:
      self.At = self._A@self._C
      if self._modet == 0:
          xtp1 = self._A@self._C@xt + self._B1@ut
          self.Bt = self._B1.copy()
      else:
          xtp1 = self._A@self._C@xt + self._B2@ut
          self.Bt = self._B2.copy()

    xtp1 = xtp1.flatten()
    modetp1 = self._find_mode(xtp1[0], xtp1[2])
    self._modetm1 = self._modet + 0
    self._modet = modetp1 + 0

    l_x = self.target_margin(xtp1.reshape(1,self.n))
    g_x = self.safety_margin(xtp1.reshape(1,self.n))

    info = np.array([l_x[0], g_x[0]])
    return xtp1, info

  # == Getting Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent. shape (batch, n)

    Returns:
      float: postive numbers indicate being inside the failure set (safety
          violation).
    """
    gx = -np.inf*np.ones(s.shape[:-1]) # g(x) > 0 is obstacle
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
    gripper_to_obj = np.linalg.norm(s[:,0] - s[:,2]) - self.thresh
    object_to_goal = np.linalg.norm(s[:,2]-self.goal[0]) - self.thresh

    if 'pick_object' in self.task:
      lx = np.maximum(gripper_to_obj, object_to_goal)
    elif 'object_to_goal' in self.task:
      lx = object_to_goal
    else:
        raise NotImplementedError('Task not defined')

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
    _state_names = ['x_g', 'x_g_dot', 'x_o', 'x_o_dot']

    _plot_state = [
        [0,2], # x_g, x_o # keep first plot state as [0,2] always to plot trajectory on it
        [0,2], # x_g_dot, x_o_dot
        [1,3], # x_g, x_g_dot
        [1,3], # x_g, x_g_dot
        [0,1], # x_o, x_o_dot
        [2,3], # x_o, x_o_dot
    ]
    _slice_loc = [
        [None, 0.0, None, 0.], # velocities
        [None, 1.0, None, 0.],
        [0., None, 0., None],
        [1., None, 0., None],
        [None, None, 0., 0.0],
        [0.0, 0., None, None],
    ]
    
    # Plot contour plot
    plt_shape = [np.ceil(len(_plot_state)/2).astype(int), 2]
    fig, axes = plt.subplots(plt_shape[0], plt_shape[1], figsize=(12, 6*plt_shape[0]))
    save_plot_name =  os.path.join(save_dir, f'Value_fn_{name}.png')

    for i in range(len(_plot_state)):
      _slice = [None]*len(value_fn.shape)
      title_str = ''
      for j in range(len(value_fn.shape)):
        if j in _plot_state[i]:
          _slice[j] = slice(None) # keep all
        else:
          _slice[j] = np.rint(
              (_slice_loc[i][j] - np.min(self.grid_x[...,j]))/(np.max(self.grid_x[...,j])-np.min(self.grid_x[...,j])) * (value_fn.shape[0]-1)
          ).astype(int)
          title_str += f'{_state_names[j]}={_slice_loc[i][j]}  '

      max_V = np.max(np.abs(value_fn[tuple(_slice)]))
      axes_idx = [int(i%plt_shape[0]), int(i/plt_shape[0])]

      levels=np.arange(-max_V, max_V, 0.01)
      if len(levels) < 11:
        levels=np.linspace(-max_V, max_V, 11)

      ctr = axes[axes_idx[0],axes_idx[1]].contourf(
        self.grid_x[(*_slice,_plot_state[i][0])], 
        self.grid_x[(*_slice,_plot_state[i][1])], 
        value_fn[tuple(_slice)],
        levels=levels, cmap='seismic')

      fig.colorbar(ctr, ax=axes[axes_idx[0],axes_idx[1]])
      axes[axes_idx[0],axes_idx[1]].set_xlabel(f'{_state_names[_plot_state[i][0]]}')
      axes[axes_idx[0],axes_idx[1]].set_ylabel(f'{_state_names[_plot_state[i][1]]}')
      axes[axes_idx[0],axes_idx[1]].set_title(f'{title_str}')

      axes[axes_idx[0],axes_idx[1]].contour(
        self.grid_x[(*_slice,_plot_state[i][0])], 
        self.grid_x[(*_slice,_plot_state[i][1])], 
        value_fn[tuple(_slice)],
        levels=[0], colors='black', linewidths=2)

      targ = axes[axes_idx[0],axes_idx[1]].contour(
        self.grid_x[(*_slice,_plot_state[i][0])], 
        self.grid_x[(*_slice,_plot_state[i][1])], 
        self.target_T[tuple(_slice)],
        levels=[0], colors='green', linestyles='dashed')
      axes[axes_idx[0],axes_idx[1]].clabel(targ, fontsize=12, inline=1, fmt='target')

      obst = axes[axes_idx[0],axes_idx[1]].contour(
        self.grid_x[(*_slice,_plot_state[i][0])], 
        self.grid_x[(*_slice,_plot_state[i][1])], 
        self.obstacle_T[tuple(_slice)],
        levels=[0], colors='darkred', linestyles='dashed')
      axes[axes_idx[0],axes_idx[1]].clabel(obst, fontsize=12, inline=1, fmt='obstacle')
        
    for traj in trajs:
      # assumes that (0,0) will always be x_g vs x_o
      axes[0,0].plot(traj[:,0], traj[:,2])

    plt.savefig(save_plot_name)
    plt.close()
  
  def plot_env(self, save_dir=''):
    save_plot_name = os.path.join(save_dir, f'target_and_obstacle_set_Pickup1DEnv.png')

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
    outputs = np.load("/home/saumyas/Projects/safe_control/HJR_manip/outputs/Pickup1D/pick_object/value_fn_Pickup1D_pick_object_grid_61_61_dt_0.05_T_10.0.npz")

    value_fn = outputs['value_fn']
    min_u_idx = outputs['min_u_idx']
    grid_x = outputs['grid_x']
    grid_x_u = outputs['grid_x_u']
    grid_x_tp1 = outputs['grid_x_tp1']
    target_T = outputs['target_T']
    obstacle_T = outputs['obstacle_T']
    t_series = outputs['t_series']

    