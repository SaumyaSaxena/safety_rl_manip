import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
import os
import wandb
from safety_rl_manip.envs.utils import signed_dist_fn_rectangle, create_grid
import matplotlib.colors as mcolors


class PointMass2DObstacles(gym.Env):
    def __init__(self, device, cfg=None):
        self.env_cfg = cfg
        self.device = device

        self.task = cfg.task
        self.dt = cfg.dt
        self.goal = np.array(self.env_cfg.goal)
        self.epoch = 0

        self.N_O = 3 # Gripper and 2 obstacles
        self.dim = 4
        self.n = 2*self.dim # max number of total objects is 2, dim=4
        self.n_all = self.N_O*self.dim
        self.m = 2
        self.n_modes = 2

        self.N_x = cfg.N_x  # Number of grid points per dimension
        self.env_bound_low = np.array(self.env_cfg.env_bounds.low)
        self.env_bound_high = np.array(self.env_cfg.env_bounds.high)

        self.u_min = np.array(cfg.control_low)
        self.u_max = np.array(cfg.control_high)
        self.action_space = gym.spaces.Box(self.u_min, self.u_max)

        self.observation_space = gym.spaces.Box(
            -np.inf*np.ones(self.n),
            np.inf*np.ones(self.n)
        )
        
        self.reward = self.env_cfg.reward
        self.penalty = self.env_cfg.penalty
        self.doneType = self.env_cfg.doneType
        self.costType = self.env_cfg.costType
        self.return_type = self.env_cfg.return_type
        self.scaling_target = self.env_cfg.scaling_target
        self.scaling_safety = self.env_cfg.scaling_safety

        self.safety_set_low, self.safety_set_high = None, None
        self.target_set_low, self.target_set_high = None, None
        # Visualization Parameters
        self.visual_initial_states = self.sample_initial_state(self.env_cfg.get('num_eval_trajs', 20))

        self.viewer = None

        self.create_env_grid()
        self.target_T = self.target_margin(self.grid_x)
        self.obstacle_T = self.safety_margin(self.grid_x)
        # self.obstacle_T_2 = self.safety_margin(np.concatenate([self.grid_x[...,:4], self.grid_x[...,8:12]], axis=-1))

        self.state = np.zeros((self.n_all))

    def create_env_grid(self):
        gripper_grid = create_grid(
            self.env_cfg.gripper.state_ranges.low[:2], 
            self.env_cfg.gripper.state_ranges.high[:2], 
            self.N_x) # xy of gripper

        obstacle1 = np.concatenate([np.tile(self.env_cfg.obstacle1.initial_pos, (*self.N_x,1)), np.zeros((*self.N_x,2))], axis=-1)
        obstacle2 = np.concatenate([np.tile(self.env_cfg.obstacle2.initial_pos, (*self.N_x,1)), np.zeros((*self.N_x,2))], axis=-1)
        
        self.grid_x = np.concatenate(
            [gripper_grid, # block bottom
            np.zeros((*self.N_x,2)),
            obstacle1,
            obstacle2,
        ], axis=-1)
        self.grid_x_flat = torch.from_numpy(self.grid_x.reshape(-1, self.grid_x.shape[-1])).float().to(self.device)


    def reset(self, start=None):
        #start: shape(self.n)

        self._current_timestep = 0
        if start is None:
            sample_state = self.sample_initial_state(1)[0]
        else:
            sample_state = start.copy()

        self.safety_set_low = None
        self.safety_set_high = None
        self.target_set_low = None
        self.target_set_high = None
        
        self.state = sample_state.copy()
        return sample_state
    
    def sample_initial_state(self, N):
        # sample N initial states
        states = np.zeros((N, self.n_all))
        samples = []
        while(len(samples)<N):
            sample_gripper = np.random.uniform(
                low=self.env_cfg.gripper.state_ranges.low, 
                high=self.env_cfg.gripper.state_ranges.high,
                size=(N,len(self.env_cfg.gripper.state_ranges.low)))
            sample_obs1 = np.random.uniform(
                low=self.env_cfg.gripper.state_ranges.low, 
                high=self.env_cfg.gripper.state_ranges.high,
                size=(N,len(self.env_cfg.gripper.state_ranges.low)))
            sample_obs2 = np.random.uniform(
                low=self.env_cfg.gripper.state_ranges.low, 
                high=self.env_cfg.gripper.state_ranges.high,
                size=(N,len(self.env_cfg.gripper.state_ranges.low)))
            states = np.concatenate([sample_gripper,sample_obs1,sample_obs2],axis=1)
            fail, gx = self.check_failure(states)
            success, lx = self.check_success(states)
            done = self.get_done(states, success, fail)
            samples.extend(states[np.logical_not(np.any(done,axis=-1))])

        samples = np.stack(samples, axis=0)
        return samples[:N]
    
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
    
    def check_within_env(self, state):
        """Checks if the robot is still in the environment.

        Args:
            state (np.ndarray): the state of the agent. shape = (batch, n)

        Returns:
            bool: True if the agent is not in the environment.
        """
        outsideLeft = np.any((state[...,0:2] <= self.env_bound_low[:2]), axis=-1)
        outsideRight = np.any((state[...,0:2] >= self.env_bound_high[:2]), axis=-1)
        outside1 = np.logical_or(outsideLeft, outsideRight)
        outside2 = np.logical_or(outsideLeft, outsideRight)
        return np.stack([outside1, outside2], axis=-1)

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

    def step(self, action):
        ut = np.array(action).flatten()
        xtp1 = self.integrate_forward(self.state, ut)

        fail, gx = self.check_failure(self.state.reshape(1,self.n_all))
        success, lx = self.check_success(self.state.reshape(1,self.n_all))
        done = self.get_done(self.state.reshape(1,self.n_all), success, fail)[0]
        cost = self.get_cost(lx, gx, success, fail)[0]

        self.state = xtp1.copy()
        info = {"g_x": gx, "l_x": lx}
        return np.copy(self.state), cost, done, info
    
    def integrate_forward(self, state, ut):
        """Integrates the dynamics forward by one step.

        Returns:
            np.ndarray: next state.
        """
        xtp1 = state.copy()
        xtp1[0] = state[0] + state[2]*self.dt # x_tp1 = xt + xdot_t *  dt
        xtp1[1] = state[1] + state[3]*self.dt # x_tp1 = xt + xdot_t *  dt

        xtp1[2] = state[2] + ut[0]*self.dt # xdot_tp1 = xdot_t + u *  dt
        xtp1[3] = state[3] + ut[1]*self.dt # xdot_tp1 = xdot_t + u *  dt
        return xtp1

    def safety_margin(self, s, safety_set_low=None, safety_set_high=None):
        # g(x)>0 is obstacle

        # collision_dist1 = np.linalg.norm(self.env_cfg.gripper.size[:2]) + np.linalg.norm(self.env_cfg.obstacle1.size[:2])
        # obstacle1 = (self.env_cfg.thresh + collision_dist1) - np.linalg.norm(s[...,:2]-s[...,4:6])

        obstacle1_high = s[...,4:6] + self.env_cfg.obstacle1.size
        obstacle1_low = s[...,4:6] - self.env_cfg.obstacle1.size
        obstacle1 = signed_dist_fn_rectangle(
            s[...,:2], 
            np.array(obstacle1_low), 
            np.array(obstacle1_high),
            obstacle=True)
        gx1 = self.scaling_safety * obstacle1

        obstacle2_high = s[...,8:10] + self.env_cfg.obstacle2.size
        obstacle2_low = s[...,8:10] - self.env_cfg.obstacle2.size
        obstacle2 = signed_dist_fn_rectangle(
            s[...,:2], 
            np.array(obstacle2_low), 
            np.array(obstacle2_high),
            obstacle=True)
        gx2 = self.scaling_safety * obstacle2
        
        if 'reward' in self.return_type: # g(x)<0 is obstacle
            gx1 = -1.*gx1
            gx2 = -1.*gx2
        return np.stack([gx1, gx2], axis=-1)
    
    def target_margin(self, s, target_set_low=None, target_set_high=None):
        # assumes input to be s = [gripper, relevant obstacle]
        # l(x)<0 is target
        
        goal = np.concatenate([self.goal, np.zeros(3)])
        lx1 = np.linalg.norm(s[...,:2]-self.goal, axis=-1) - self.env_cfg.thresh
        lx2 = np.linalg.norm(s[...,:2]-self.goal, axis=-1) - self.env_cfg.thresh

        lx1 = self.scaling_target * lx1
        lx2 = self.scaling_target * lx2
        if 'reward' in self.return_type: # l(x)>0 is target
            lx1 = -1.*lx1
            lx2 = -1.*lx2
        return np.stack([lx1, lx2], axis=-1)

    def check_failure(self, state):
        g_x = self.safety_margin(state, self.safety_set_low, self.safety_set_high)
        if 'reward' in self.return_type: 
            return g_x<0, g_x
        else:
            return g_x>0, g_x # g(x)>0 is failure
  
    def check_success(self, state):
        l_x = self.target_margin(state, self.target_set_low, self.target_set_high)
        if 'reward' in self.return_type: 
            return l_x>0, l_x
        else:
            return l_x<0, l_x # l(x)<0 is target
        
    def plot_env(self, save_dir=''):
        save_plot_name = os.path.join(save_dir, f'target_and_obstacle_set_PointMass2DObstacles.png')

        fig, axes = plt.subplots(3, figsize=(12, 30))
        level_min = min(-1e-3,self.target_T.min())
        level_max = max(self.target_T.max(), 1e-3)
        norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        ctr_t = axes[0].contourf(self.grid_x[...,0], self.grid_x[...,1], self.target_T[...,0], levels=levels, cmap='seismic', norm=norm)
        targ = axes[0].contour(self.grid_x[...,0], self.grid_x[...,1], self.target_T[...,0], levels=[0], colors='black')
        axes[0].set_title(f'Target set l(x)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].clabel(targ, fontsize=12, inline=1, fmt='target')
        fig.colorbar(ctr_t, ax=axes[0])

        level_min = min(-1e-3,self.obstacle_T[...,0].min())
        level_max = max(self.obstacle_T[...,0].max(), 1e-3)
        norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        ctr_o = axes[1].contourf(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T[...,0], levels=levels, cmap='seismic', norm=norm)
        obst = axes[1].contour(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T[...,0], levels=[0], colors='black')
        axes[1].set_title(f'Obstacle set 1 g(x)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].clabel(obst, fontsize=12, inline=1, fmt='obstacle1')
        fig.colorbar(ctr_o, ax=axes[1])

        # ra = np.maximum(self.obstacle_T_1, self.target_T)
        # level_min = min(-1e-3,ra[:,:,xdot_idx,ydot_idx].min())
        # level_max = max(ra[:,:,xdot_idx,ydot_idx].max(), 1e-3)
        # norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        # levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        # ctr_ra = axes[2].contourf(self.grid_x[:,:,xdot_idx,ydot_idx,0], self.grid_x[:,:,xdot_idx,ydot_idx,1], ra[:,:,xdot_idx,ydot_idx], levels=levels, cmap='seismic', norm=norm)
        # axes[2].contour(self.grid_x[:,:,xdot_idx,ydot_idx,0], self.grid_x[:,:,xdot_idx,ydot_idx,1], ra[:,:,xdot_idx,ydot_idx], levels=[0], colors='black')
        # axes[2].set_title(f'Terminal Value fn')
        # axes[2].set_xlabel('X')
        # axes[2].set_ylabel('Y')
        # fig.colorbar(ctr_ra, ax=axes[2])

        level_min = min(-1e-3,self.obstacle_T[...,1].min())
        level_max = max(self.obstacle_T[...,1].max(), 1e-3)
        norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        ctr_o = axes[2].contourf(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T[...,1], levels=levels, cmap='seismic', norm=norm)
        obst = axes[2].contour(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T[...,1], levels=[0], colors='black')
        axes[2].set_title(f'Obstacle set 2 g(x)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].clabel(obst, fontsize=12, inline=1, fmt='obstacle2')
        fig.colorbar(ctr_o, ax=axes[2])

        # ra = np.maximum(self.obstacle_T_2, self.target_T)
        # level_min = min(-1e-3,ra.min())
        # level_max = max(ra.max(), 1e-3)
        # norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        # levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        # ctr_ra = axes[2].contourf(self.grid_x[...,0], self.grid_x[...,1], ra, levels=levels, cmap='seismic', norm=norm)
        # axes[2].contour(self.grid_x[...,0], self.grid_x[...,1], ra, levels=[0], colors='black')
        # axes[2].set_title(f'Terminal Value fn')
        # axes[2].set_xlabel('X')
        # axes[2].set_ylabel('Y')
        # fig.colorbar(ctr_ra, ax=axes[2])

        plt.savefig(save_plot_name)
        plt.close()

    def plot_value_fn(        
            self, 
            value_fn, 
            grid_x, 
            target_T=None, 
            obstacle_T=None, 
            trajs=[], 
            save_dir='', 
            name='',
            debug=True):
        
        save_plot_name = os.path.join(save_dir, f'Value_fn_{name}.png')
        # Plot contour plot
        plt.figure(figsize=(8, 6))
        level_min = min(-1e-3,value_fn.min())
        level_max = max(value_fn.max(), 1e-3)
        norm = mcolors.TwoSlopeNorm(vmin=level_min, vcenter=0, vmax=level_max)
        levels= np.append(np.linspace(level_min, -1e-6, 15), np.linspace(1e-6, level_max, 15))
        plt.contourf(grid_x[...,0], grid_x[...,1], value_fn, levels=levels, cmap='seismic', norm=norm)
        plt.colorbar(label='Value fn')
        plt.xlabel('X')
        plt.ylabel('X dot')
        plt.contour(grid_x[...,0], grid_x[...,1], value_fn, levels=[0], colors='black', linewidths=2)

        if target_T is not None:
            targ = plt.contour(grid_x[...,0], grid_x[...,1], target_T[...,0], levels=[0], colors='green', linestyles='dashed')
            plt.clabel(targ, fontsize=12, inline=1, fmt='target')
        if obstacle_T is not None:
            obst1 = plt.contour(grid_x[...,0], grid_x[...,1], obstacle_T[...,0], levels=[0], colors='darkred', linestyles='dashed')
            plt.clabel(obst1, fontsize=12, inline=1, fmt='obstacle1')
            obst2 = plt.contour(grid_x[...,0], grid_x[...,1], obstacle_T[...,1], levels=[0], colors='darkred', linestyles='dashed')
            plt.clabel(obst2, fontsize=12, inline=1, fmt='obstacle2')
        
        if trajs is not None:
            for _n, traj in enumerate(trajs):
                plt.plot(traj[:,0],traj[:,1])
                plt.scatter(traj[0,0], traj[0,1], c='white')
                # self.plot_trajectory(traj, t, dt, save_dir)
                # self.create_rollout_video(traj, dt, grid_x, save_dir, name=f'{_n}')
                save_plot_name = f'{save_dir}/Value_fn_wTrajs_PointMass2DObstacles_trajs_{_n}_{name}.png'

        plt.savefig(save_plot_name)
        plt.close()
        if not debug:
            wandb.log({f"Value_fn_{name}": wandb.Image(save_plot_name)})

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/gym_envs.yaml')
    env = PointMass2DObstacles(torch.device("cuda", 0), cfg['point_mass_2D_obstacles_env-v0'])
    env.plot_env(save_dir='/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media')

