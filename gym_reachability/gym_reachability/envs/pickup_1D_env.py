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
    self.sample_inside_obs = cfg.sample_inside_obs

    X = [np.linspace(self.low[i], self.high[i], self.N_x[i]) for i in range(4)]
    self.grid_x = np.stack(np.meshgrid(*X, indexing='ij'), axis=-1)
    self.grid_x_flat = torch.from_numpy(self.grid_x.reshape(-1, self.grid_x.shape[-1])).float().to(self.device)

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

    # Set random seed.
    self.seed_val = 0
    self.set_seed(self.seed_val)

    # Cost Parameters
    self.penalty = 1.
    self.reward = -1.
    self.costType = 'sparse'
    self.scaling = 1.
    self.return_type = cfg.return_type

    self.state = np.zeros(self.n)
    self.doneType = cfg.doneType

    # Visualization Parameters
    self.visual_initial_states = [
      np.array([-0.5, 0., 0., 0.]),
      np.array([0.5, 0., 0., 0.]),
    ]

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )

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
    # for torch
    self.device = device

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

    l_x = self.target_margin(xtp1.reshape(1,4))
    g_x = self.safety_margin(xtp1.reshape(1,4))

    info = np.array([l_x, g_x])
    return xtp1, info

  # == Setting Hyper-Parameters ==
  def set_costParam(
      self, penalty=1., reward=-1., costType='sparse', scaling=1.
  ):
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
    self.penalty = penalty
    self.reward = reward
    self.costType = costType
    self.scaling = scaling

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

  def set_doneType(self, doneType):
    """Sets the condition to terminate the episode.

    Args:
        doneType (str): conditions to raise `done flag in training.
    """
    self.doneType = doneType

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

  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    x_span = self.bounds[0, 1] - self.bounds[0, 0]
    y_span = self.bounds[1, 1] - self.bounds[1, 0]
    aspect_ratio = x_span / y_span
    axes = np.array([
        self.bounds[0, 0] - .05, self.bounds[0, 1] + .05,
        self.bounds[1, 0] - .05, self.bounds[1, 1] + .05
    ])
    return [axes, aspect_ratio]

  def get_value(self, q_func, nx=41, ny=121, addBias=False):
    """Gets the state values given the Q-network.

    Args:
        q_func (object): agent's Q-network.
        nx (int, optional): # points in x-axis. Defaults to 41.
        ny (int, optional): # points in y-axis. Defaults to 121.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.

    Returns:
        np.ndarray: x-position of states
        np.ndarray: y-position of states
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index

      x = xs[idx[0]]
      y = ys[idx[1]]
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))

      if self.mode == 'normal' or self.mode == 'RA':
        state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
      else:
        z = max([l_x, g_x])
        state = torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)

      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return xs, ys, v

  # == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=250, state=None, keepOutOf=False, toEnd=False
  ):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        state (np.ndarray, optional): if provided, set the initial state to its
            value. Defaults to None.
        keepOutOf (bool, optional): smaple states inside obstacles if False.
            Defaults to False.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary if True. Defaults to False.

    Returns:
        np.ndarray: x-positions of the trajectory.
        np.ndarray: y-positions of the trajectory.
        int: the binary reach-avoid outcome.
    """
    if state is None:
      state = self.sample_random_state(sample_inside_obs=not keepOutOf)
    x, y = state[:2]
    traj_x = [x]
    traj_y = [y]
    result = 0  # not finished

    for _ in range(T):
      if toEnd:
        outsideTop = (state[1] > self.bounds[1, 1])
        outsideLeft = (state[0] < self.bounds[0, 0])
        outsideRight = (state[0] > self.bounds[0, 1])
        done = outsideTop or outsideLeft or outsideRight
        if done:
          result = 1
          break
      else:
        if self.safety_margin(state[:2]) > 0:
          result = -1  # failed
          break
        elif self.target_margin(state[:2]) <= 0:
          result = 1  # succeeded
          break

      state_tensor = torch.FloatTensor(state)
      state_tensor = state_tensor.to(self.device).unsqueeze(0)
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      u = self.discrete_controls[action_index]

      state, _ = self.integrate_forward(state, u)
      traj_x.append(state[0])
      traj_y.append(state[1])

    return traj_x, traj_y, result

  def simulate_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot crosses
            the boundary if True. Defaults to False.

    Returns:
        list of np.ndarrays: each element is a tuple consisting of x and y
            positions along the trajectory.
        np.ndarray: the binary reach-avoid outcomes.
    """

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      if self.envType == 'basic' or self.envType == 'easy':
        nx = 21
        ny = 61
      else:
        nx = 41
        ny = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      it = np.nditer(results, flags=['multi_index'])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]
        state = np.array([x, y])
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result
        it.iternext()
      results = results.reshape(-1)
    else:
      results = np.empty(shape=(len(states),), dtype=int)
      for idx, state in enumerate(states):
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result

    return trajectories, results

  # == Visualizing ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, labels=None,
      boolPlot=False, addBias=False, cmap='seismic'
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form if True.
            Defaults to False.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
        cmap (str, optional): color map. Defaults to 'seismic'.
    """

    _state_names = ['x_g', 'x_g_dot', 'x_o', 'x_o_dot']

    _plot_state = [
      [0,2], # x_g, x_o # keep first plot state as [0,2] always to plot trajectory on it
      [0,2], # x_g_dot, x_o_dot
      [0,2], # x_g, x_g_dot
    ]
    _slice_loc = [
      [None, 0.,None, 0.], # velocities
      [None, -0.5,None, 0.],
      [None, 0.5,None, 0.],
    ]

    # Plot contour plot
    plt_shape = [np.rint(len(_plot_state)/2).astype(int), 2]
    fig, axes = plt.subplots(plt_shape[0], plt_shape[1], figsize=(12, 12))

    for i in range(len(_plot_state)):
        _slice = [None]*len(value_fn.shape)
        title_str = ''
        for j in range(len(value_fn.shape)):
            if j in _plot_state[i]:
                _slice[j] = slice(None) # keep all
            else:
                _slice[j] = np.rint(
                    (_slice_loc[i][j] - np.min(grid_x[...,j]))/(np.max(grid_x[...,j])-np.min(grid_x[...,j])) * (value_fn.shape[0]-1)
                ).astype(int)
                title_str += f'{_state_names[j]}={_slice_loc[i][j]}  '

        max_V = np.max(np.abs(value_fn[tuple(_slice)]))
        axes_idx = [int(i%plt_shape[0]), int(i/plt_shape[0])]

        levels=np.arange(-max_V, max_V, 0.01)
        if len(levels) < 11:
            levels=np.linspace(-max_V, max_V, 11)

        ctr = axes[axes_idx[0],axes_idx[1]].contourf(
            grid_x[(*_slice,_plot_state[i][0])], 
            grid_x[(*_slice,_plot_state[i][1])], 
            value_fn[tuple(_slice)],
            levels=levels, cmap='seismic')

        fig.colorbar(ctr, ax=axes[axes_idx[0],axes_idx[1]])
        axes[axes_idx[0],axes_idx[1]].set_xlabel(f'{_state_names[_plot_state[i][0]]}')
        axes[axes_idx[0],axes_idx[1]].set_ylabel(f'{_state_names[_plot_state[i][1]]}')
        axes[axes_idx[0],axes_idx[1]].set_title(f'{title_str}')

        axes[axes_idx[0],axes_idx[1]].contour(
            grid_x[(*_slice,_plot_state[i][0])], 
            grid_x[(*_slice,_plot_state[i][1])], 
            value_fn[tuple(_slice)],
            levels=[0], colors='black', linewidths=2)

        targ = axes[axes_idx[0],axes_idx[1]].contour(
            grid_x[(*_slice,_plot_state[i][0])], 
            grid_x[(*_slice,_plot_state[i][1])], 
            target_T[tuple(_slice)],
            levels=[0], colors='green', linestyles='dashed')
        axes[axes_idx[0],axes_idx[1]].clabel(targ, fontsize=12, inline=1, fmt='target')

        obst = axes[axes_idx[0],axes_idx[1]].contour(
            grid_x[(*_slice,_plot_state[i][0])], 
            grid_x[(*_slice,_plot_state[i][1])], 
            obstacle_T[tuple(_slice)],
            levels=[0], colors='darkred', linestyles='dashed')
        axes[axes_idx[0],axes_idx[1]].clabel(obst, fontsize=12, inline=1, fmt='obstacle')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cbarPlot = True

    # == Plot V ==
    self.plot_v_values(
        q_func, ax=ax, fig=fig, vmin=vmin, vmax=vmax, cmap=cmap,
        boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias
    )

    # == Plot Trajectories ==
    self.plot_trajectories(
        q_func, states=self.visual_initial_states, toEnd=False, ax=ax
    )

    # == Formatting ==
    self.plot_formatting(ax=ax, labels=labels)
    fig.tight_layout()

  def plot_v_values(
      self, q_func, ax=None, fig=None, vmin=-1, vmax=1,
      cmap='seismic', alpha=0.8, boolPlot=False, cbarPlot=True, addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        alpha (float, optional): opacity. Defaults to 0.8.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar if True.
            Defaults to True.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
    """
    axStyle = self.get_axes()

    # == Plot V ==
    _, _, v = self.get_value(q_func, nx, ny, addBias=addBias)
    vmax = np.ceil(max(np.max(v), np.max(-v)))
    vmin = -vmax

    if boolPlot:
      im = ax.imshow(
          v.T > 0., interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, alpha=alpha
      )
    else:
      im = ax.imshow(
          v.T, interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
            ticks=[vmin, 0, vmax]
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  def plot_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False,
      ax=None, c='k', lw=2, zorder=2
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot crosses
            the boundary if True. Defaults to False.
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        c (str, optional): color of the trajectories. Defaults to 'k'.
        lw (float, optional): linewidth of the trajectories. Defaults to 2.
        zorder (int, optional): graph layers order. Defaults to 2.
    Returns:
        np.ndarray: the binary reach-avoid outcomes.
    """

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    trajectories, results = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
    )

    for traj in trajectories:
      traj_x, traj_y = traj
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

    return results

  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, left=False,
        right=False
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])

  def plot_env(self, scaling=1.0, figureFolder=''):
    nx, ny = 101, 101
    vmin = -1 * scaling
    vmax = 1 * scaling

    v = np.zeros((nx, ny))
    l_x = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)

    it = np.nditer(v, flags=['multi_index'])

    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]

      l_x[idx] = self.target_margin(np.array([x, y]))
      g_x[idx] = self.safety_margin(np.array([x, y]))

      v[idx] = np.maximum(l_x[idx], g_x[idx])
      it.iternext()

    axStyle = self.get_axes()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    ax = axes[0]
    im = ax.imshow(
        l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$\ell(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(
        g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[2]
    im = ax.imshow(
        v.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax
    )
    self.plot_reach_avoid_set(ax)
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
      self.plot_target_failure_set(ax=ax)
      self.plot_formatting(ax=ax)

    fig.tight_layout()
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)