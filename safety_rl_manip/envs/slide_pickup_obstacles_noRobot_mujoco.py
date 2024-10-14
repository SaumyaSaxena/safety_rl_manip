import warnings
import os

import numpy as np
import gym
from gym.utils import EzPickle
import matplotlib.pyplot as plt

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import MultiTaskNoWallsArena
import mujoco

from robosuite.models.objects.primitive.box import BoxObject
from safety_rl_manip.envs.utils import create_grid

import torch
import wandb

class SlidePickupObstaclesMujocoEnv(gym.Env, EzPickle):
    def __init__(self, device, cfg):
        EzPickle.__init__(self)
        self.device = device
        self.env_cfg = cfg
        self.frame_skip = self.env_cfg.frame_skip
        self._did_see_sim_exception = False
        self.goal = np.array(self.env_cfg.goal)
        self.N_O = 4 # 2 manipulated blocks and 2 obstacles TODO(saumya): for varied number of obstacles?
        self.n = 3*6 # max number of total objects is 3
        self.n_all = self.N_O*6
        self.m = 3

        self.u_min = np.array(self.env_cfg.control_low)
        self.u_max = np.array(self.env_cfg.control_high)
        self.action_space = gym.spaces.Box(self.u_min, self.u_max)

        self.env_bound_low = np.array(self.env_cfg.env_bounds.low)
        self.env_bound_high = np.array(self.env_cfg.env_bounds.high)

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
        self._setup_procedural_env()
        self.renderer = mujoco.Renderer(self.model, self.env_cfg.img_size[0], self.env_cfg.img_size[1])
        
        # Visualization Parameters
        self.visual_initial_states = self.sample_initial_state(self.env_cfg.get('num_eval_trajs', 20))

        # Will visualize the value function in x-y only for initial positions of bottom block
        self.N_x = self.env_cfg.block_bottom.N_x
        
        self.create_env_grid()
        
        self.target_T = self.target_margin(self.grid_x)
        self.safety_set_top_low = self.grid_x[...,6:9] - self.env_cfg.block_top.safety_set.low
        self.safety_set_top_high = self.grid_x[...,6:9] + self.env_cfg.block_top.safety_set.high
        self.obstacle_T = self.safety_margin(self.grid_x)

    def create_env_grid(self):
        # TODO(saumya): Add variable obstacles here
        self.grid_x = create_grid(
            self.env_cfg.block_bottom.state_ranges.low, 
            self.env_cfg.block_bottom.state_ranges.high, 
            self.N_x)
        
        obsA_state_range_low, obsA_state_range_high, obsB_state_range_low, obsB_state_range_high = self.get_obs_ranges_from_block_states(self.grid_x)

        xy_sample_obsA = (obsA_state_range_low + obsA_state_range_high)/2
        xy_sample_obsB = (obsB_state_range_low + obsB_state_range_high)/2
        
        distA = np.linalg.norm(self.grid_x-xy_sample_obsA, axis=-1)
        distB = np.linalg.norm(self.grid_x-xy_sample_obsB, axis=-1)

        relevantA_idx = distA < distB
        relevant_obs_xy = xy_sample_obsB.copy()
        relevant_obs_xy[relevantA_idx] = xy_sample_obsA[relevantA_idx]
        rel_obs_z = self.block_obsB_z*np.ones((*self.N_x,1))
        rel_obs_z[relevantA_idx] = self.block_obsA_z

        relevant_obs = np.concatenate([relevant_obs_xy, rel_obs_z, np.zeros((*self.N_x,3))], axis=-1)
        
        self.grid_x = np.concatenate(
            [self.grid_x, # block bottom
            self.block_bottom_z*np.ones((*self.N_x,1)),
            np.zeros((*self.N_x,3)),
            self.grid_x, # block top
            self.block_top_z*np.ones((*self.N_x,1)),
            np.zeros((*self.N_x,3)),
            relevant_obs,
            ], axis=-1)
        
        self.grid_x_flat = torch.from_numpy(self.grid_x.reshape(-1, self.grid_x.shape[-1])).float().to(self.device)

    def get_obs_ranges_from_block_states(self, xy_samples):

        obsA_state_range_low = np.maximum(
            xy_samples + np.array(self.env_cfg.block_obsA.state_ranges.low), 
            self.env_bound_low[:2]+np.array(self.env_cfg.block_obsA.size)[:2])
        obsA_state_range_high = np.minimum(
            xy_samples + np.array(self.env_cfg.block_obsA.state_ranges.high), 
            self.env_bound_high[:2]-np.array(self.env_cfg.block_obsA.size)[:2])

        obsB_state_range_low = np.maximum(
            xy_samples + np.array(self.env_cfg.block_obsB.state_ranges.low), 
            self.env_bound_low[:2]+np.array(self.env_cfg.block_obsB.size)[:2])
        obsB_state_range_high = np.minimum(
            xy_samples + np.array(self.env_cfg.block_obsB.state_ranges.high), 
            self.env_bound_high[:2]-np.array(self.env_cfg.block_obsB.size)[:2])
        
        return obsA_state_range_low, obsA_state_range_high, obsB_state_range_low, obsB_state_range_high
    
    def _setup_procedural_env(self):
        world = MujocoWorldBase()
        mujoco_arena = MultiTaskNoWallsArena()
        world.merge(mujoco_arena)
        self.indices_by_name = {}

        # Bottom block
        block_bottom = BoxObject(name='Block_bottom', size=self.env_cfg.block_bottom.size, rgba=self.env_cfg.block_bottom.rgba)
        block_bottom_body = block_bottom.get_obj()
        self.block_bottom_z = -block_bottom.bottom_offset[2]
        block_bottom_body.set('pos', f'{self.env_cfg.block_bottom.initial_pos[0]} {self.env_cfg.block_bottom.initial_pos[1]} {self.block_bottom_z}')
        world.worldbody.append(block_bottom_body)
        world.merge_assets(block_bottom)

        # Top block
        block_top = BoxObject(name='Block_top', size=self.env_cfg.block_top.size, rgba=self.env_cfg.block_top.rgba)
        block_top_body = block_top.get_obj()
        self.block_top_z = (block_bottom.top_offset - block_bottom.bottom_offset - block_top.bottom_offset)[2]
        block_top_body.set('pos', f'{self.env_cfg.block_top.initial_pos[0]} {self.env_cfg.block_top.initial_pos[1]} {self.block_top_z}')
        world.worldbody.append(block_top_body)
        world.merge_assets(block_top)

        # TODO(saumya): Variable number of distractors?

        # Distractor A
        block_obsA = BoxObject(name='Block_obsA', size=self.env_cfg.block_obsA.size, rgba=self.env_cfg.block_obsA.rgba)
        block_obsA_body = block_obsA.get_obj()
        self.block_obsA_z = -block_obsA.bottom_offset[2]
        block_obsA_body.set('pos', f'{self.env_cfg.block_obsA.initial_pos[0]} {self.env_cfg.block_obsA.initial_pos[1]} {self.block_obsA_z}')
        world.worldbody.append(block_obsA_body)
        world.merge_assets(block_obsA)

        # Distractor B
        block_obsB = BoxObject(name='Block_obsB', size=self.env_cfg.block_obsB.size, rgba=self.env_cfg.block_obsB.rgba)
        block_obsB_body = block_obsB.get_obj()
        self.block_obsB_z = -block_obsB.bottom_offset[2]
        block_obsB_body.set('pos', f'{self.env_cfg.block_obsB.initial_pos[0]} {self.env_cfg.block_obsB.initial_pos[1]} {self.block_obsB_z}')
        world.worldbody.append(block_obsB_body)
        world.merge_assets(block_obsB)

        world.root.find('compiler').set('inertiagrouprange', '0 5')
        world.root.find('compiler').set('inertiafromgeom', 'auto')
        self.model = world.get_model(mode="mujoco")
        self.data = mujoco.MjData(self.model)

    def sample_initial_state(self, N):
        # sample N initial states
        states = np.zeros((N, self.n_all))

        xy_sample_blocks = np.random.uniform(
            low=self.env_cfg.block_bottom.state_ranges.low, 
            high=self.env_cfg.block_bottom.state_ranges.high,
            size=(N,len(self.env_cfg.block_bottom.state_ranges.low)))
        
        states[:, 0:2] = xy_sample_blocks
        states[:, 6:8] = xy_sample_blocks
        states[:,2] = self.block_bottom_z
        states[:,8] = self.block_top_z

        obsA_state_range_low, obsA_state_range_high, obsB_state_range_low, obsB_state_range_high = self.get_obs_ranges_from_block_states(xy_sample_blocks)

        xy_sample_obsA = np.random.uniform(
            low=obsA_state_range_low, 
            high=obsA_state_range_high)
        
        xy_sample_obsB = np.random.uniform(
            low=obsB_state_range_low, 
            high=obsB_state_range_high)

        states[:, 12:14] = xy_sample_obsA
        states[:, 18:20] = xy_sample_obsB
        states[:,14] = self.block_obsA_z
        states[:,20] = self.block_obsB_z

        #TODO: do this for variable number of obstacles
        return states
    
    def reset(self, start=None):
        #start: shape(self.n)
        self._did_see_sim_exception = False
        mujoco.mj_resetData(self.model, self.data)

        self._current_timestep = 0
        if start is None:
            sample_state = self.sample_initial_state(1)[0]
        else:
            sample_state = start.copy()

        self.data.qpos[0:3] = sample_state[0:3] # block_bottom
        self.data.qpos[7:10] = sample_state[6:9] # block_top
        self.data.qpos[14:17] = sample_state[12:15] # obsA
        self.data.qpos[21:24] = sample_state[18:21] # obsB

        mujoco.mj_forward(self.model, self.data)
        curr_state = self.get_current_state()
        self.safety_set_top_low = curr_state[6:9] + self.env_cfg.block_top.safety_set.low
        self.safety_set_top_high = curr_state[6:9] + self.env_cfg.block_top.safety_set.high

        self.target_set_low = curr_state[:3] + self.env_cfg.block_bottom.target_set.low
        self.target_set_high = curr_state[:3] + self.env_cfg.block_bottom.target_set.high
        return curr_state

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
        #TODO(saumya): extend to distractor objects
        outsideLeft_bottom = np.any((state[...,0:3] <= self.env_bound_low), axis=-1)
        outsideRight_bottom = np.any((state[...,0:3] >= self.env_bound_high), axis=-1)
        outside_bottom = np.logical_or(outsideLeft_bottom, outsideRight_bottom)

        outsideLeft_top = np.any((state[...,6:9] <= self.env_bound_low), axis=-1)
        outsideRight_top = np.any((state[...,6:9] >= self.env_bound_high), axis=-1)
        outside_top = np.logical_or(outsideLeft_top, outsideRight_top)

        return np.logical_or(outside_bottom, outside_top)

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

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, action, n_frames=None):
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.data.xfrc_applied[self.model.body('Block_bottom_main').id] = action # [0.0, 0.0, 50.0, 0.0, 0.0, 0.0]

        for _ in range(n_frames):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def get_current_state(self):
        # Current state contains the obstacle that is closer to block_bottom
        distA = np.linalg.norm(self.data.qpos[0:2]-self.data.qpos[14:16])
        distB = np.linalg.norm(self.data.qpos[0:2]-self.data.qpos[21:23])
        if distA < distB:
            relevant_obs = np.concatenate([self.data.qpos[14:17], self.data.qvel[12:15]])
            self.relevant_obj = 'block_obsA'
        else:
            relevant_obs = np.concatenate([self.data.qpos[21:24], self.data.qvel[18:21]])
            self.relevant_obj = 'block_obsB'
        
        return np.concatenate([self.data.qpos[0:3], self.data.qvel[0:3], self.data.qpos[7:10], self.data.qvel[6:9], relevant_obs])
    
    # == Dynamics ==
    def step(self, action):
        xt = self.get_current_state()
        self.do_simulation(np.concatenate([action.flatten(), np.zeros(3)]))
        xtp1 = self.get_current_state()

        fail, g_x = self.check_failure(xt.reshape(1,self.n))
        success, l_x = self.check_success(xt.reshape(1,self.n))
        done = self.get_done(xt.reshape(1,self.n), success, fail)[0]
        cost = self.get_cost(l_x, g_x, success, fail)[0]

        info = {"g_x": g_x[0], "l_x": l_x[0]}
        return xtp1, cost, done, info

    def render(self):
        pass

      # == Getting Margin ==
    def safety_margin(self, s, safety_set_top_low=None, safety_set_top_high=None):
        # Input:s top block position, shape (batch, 3)
        # g(x)>0 is obstacle

        top_block_pos = s[...,6:9]
        if safety_set_top_low is None:
            safety_set_top_low = top_block_pos + self.env_cfg.block_top.safety_set.low

        if safety_set_top_high is None:
            safety_set_top_high = top_block_pos + self.env_cfg.block_top.safety_set.high

        lower_boundary = np.max(self.safety_set_top_low - top_block_pos, axis=-1)
        upper_boundary = np.max(top_block_pos - self.safety_set_top_high, axis=-1)
        
        top_block_bounds = np.maximum(lower_boundary, upper_boundary)

        collision_dist = np.linalg.norm(self.env_cfg.block_bottom.size[:2]) + np.linalg.norm(self.env_cfg.block_obsA.size[:2])
        obstacle = (self.env_cfg.thresh + collision_dist) - np.linalg.norm(s[...,:3]-s[...,12:15])
        
        gx = self.scaling_safety * np.maximum(top_block_bounds, obstacle)

        if 'reward' in self.return_type: # g(x)<0 is obstacle
            gx = -1.*gx
        return gx

    def target_margin(self, s, target_set_low=None, target_set_high=None):
        """Computes the margin (e.g. distance) between the state and the target set.

        Args:
            s (np.ndarray): the state of the agent. shape (batch, n)

        Returns:
            float: negative numbers indicate reaching the target. If the target set
                is not specified, return None.
        """
        # l(x)<0 is target
        # bottom block reached goal region
        goal = np.concatenate([self.goal, np.zeros(3)])

        bottom_block_pos = s[...,:3]
        if target_set_low is None:
            target_set_low = bottom_block_pos + self.env_cfg.block_bottom.target_set.low

        if target_set_high is None:
            target_set_high = bottom_block_pos + self.env_cfg.block_bottom.target_set.high

        # lx = np.linalg.norm(s[...,:6]-goal, axis=-1) - self.env_cfg.thresh

        lower_boundary = np.max(target_set_low - s[...,:3], axis=-1)
        upper_boundary = np.max(s[...,:3] - target_set_high, axis=-1)
        
        lx = np.maximum(lower_boundary, upper_boundary)

        lx = self.scaling_target * lx

        if 'reward' in self.return_type: # l(x)>0 is target
            lx = -1.*lx

        return lx

    def check_failure(self, state):
        g_x = self.safety_margin(state, self.safety_set_top_low, self.safety_set_top_high)
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
    
    def plot_trajectory(self, state, action, save_filename):
        # state shape = (T+1, self.n)
        # action shape = (T, self.m)

        fig, axes = plt.subplots(3, 3, figsize=(16, 16))

        # plot position
        axes[0,0].plot(state[:,0], label='bottom block x')
        axes[0,0].plot(state[:,6], label='top block x')
        axes[0,0].plot(state[:,12], label='relevant obs x')
        axes[0,0].set_xlabel(f't')
        axes[0,0].set_ylabel(f'x')
        axes[0,0].set_title(f'x pos')
        axes[0,0].legend()

        axes[0,1].plot(state[:,1], label='bottom block y')
        axes[0,1].plot(state[:,7], label='top block y')
        axes[0,1].plot(state[:,13], label='relevant obs y')
        axes[0,1].set_xlabel(f't')
        axes[0,1].set_ylabel(f'y')
        axes[0,1].set_title(f'y pos')
        axes[0,1].legend()

        axes[0,2].plot(state[:,2], label='bottom block z')
        axes[0,2].plot(state[:,8], label='top block z')
        axes[0,2].plot(state[:,14], label='relevant obs z')
        axes[0,2].set_xlabel(f't')
        axes[0,2].set_ylabel(f'z')
        axes[0,2].set_title(f'z pos')
        axes[0,2].legend()

        # plot velocity
        axes[1,0].plot(state[:,3], label='bottom block x')
        axes[1,0].plot(state[:,9], label='top block x')
        axes[1,0].plot(state[:,15], label='relevant obs x')
        axes[1,0].set_xlabel(f't')
        axes[1,0].set_ylabel(f'xdot')
        axes[1,0].set_title(f'x vel')
        axes[1,0].legend()

        axes[1,1].plot(state[:,4], label='bottom block y')
        axes[1,1].plot(state[:,10], label='top block y')
        axes[1,1].plot(state[:,16], label='relevant obs y')
        axes[1,1].set_xlabel(f't')
        axes[1,1].set_ylabel(f'ydot')
        axes[1,1].set_title(f'y vel')
        axes[1,1].legend()

        axes[1,2].plot(state[:,5], label='bottom block z')
        axes[1,2].plot(state[:,11], label='top block z')
        axes[1,2].plot(state[:,17], label='relevant obs z')
        axes[1,2].set_xlabel(f't')
        axes[1,2].set_ylabel(f'zdot')
        axes[1,2].set_title(f'z vel')
        axes[1,2].legend()

        # control on bottom block
        axes[2,0].plot(action[:,0], label='u_x')
        axes[2,0].set_xlabel(f't')
        axes[2,0].set_ylabel(f'u_x')
        axes[2,0].set_title(f'Control x')
        axes[2,0].legend()

        axes[2,1].plot(action[:,1], label='u_y')
        axes[2,1].set_xlabel(f't')
        axes[2,1].set_ylabel(f'u_y')
        axes[2,1].set_title(f'Control y')
        axes[2,1].legend()

        axes[2,2].plot(action[:,2], label='u_z')
        axes[2,2].set_xlabel(f't')
        axes[2,2].set_ylabel(f'u_z')
        axes[2,2].set_title(f'Control z')
        axes[2,2].legend()

        plt.savefig(save_filename)
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
            # plt.clabel(targ, fontsize=12, inline=1, fmt='target')
        if obstacle_T is not None:
            obst = plt.contour(grid_x[...,0], grid_x[...,1], obstacle_T, levels=[0], colors='darkred', linestyles='dashed')
            # plt.clabel(obst, fontsize=12, inline=1, fmt='obstacle')

        for traj in trajs:
            plt.plot(traj[:,0],traj[:,1])

        plt.savefig(save_plot_name)
        plt.close()
        if not debug:
            wandb.log({f"Value_fn_{name}": wandb.Image(save_plot_name)})

    def plot_env(self, save_dir=''):
      
        save_plot_name = os.path.join(save_dir, f'target_and_obstacle_set_SlidePickupMujocoEnv.png')

        fig, axes = plt.subplots(3, figsize=(12, 12))

        max_V = np.max(np.abs(self.target_T))
        levels=np.arange(-max_V, max_V, 0.01)
        levels=np.linspace(-max_V, max_V, 11) if len(levels) < 11 else levels
            
        ctr_t = axes[0].contourf(self.grid_x[...,0], self.grid_x[...,1], self.target_T, levels=levels, cmap='seismic')
        targ = axes[0].contour(self.grid_x[...,0], self.grid_x[...,1], self.target_T, levels=[0], colors='black')
        axes[0].set_title(f'Target set l(x)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        # axes[0].clabel(targ, fontsize=12, inline=1, fmt='target')
        fig.colorbar(ctr_t, ax=axes[0])


        max_V = np.max(np.abs(self.obstacle_T))
        levels=np.arange(-max_V, max_V, 0.01)
        levels=np.linspace(-max_V, max_V, 11) if len(levels) < 11 else levels
        ctr_o = axes[1].contourf(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T, levels=levels, cmap='seismic')
        obst = axes[1].contour(self.grid_x[...,0], self.grid_x[...,1], self.obstacle_T, levels=[0], colors='black')
        axes[1].set_title(f'Obstacle set g(x)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        # axes[1].clabel(obst, fontsize=12, inline=1, fmt='obstacle')
        fig.colorbar(ctr_o, ax=axes[1])

        ra = np.maximum(self.obstacle_T, self.target_T)

        ctr_ra = axes[2].contourf(self.grid_x[...,0], self.grid_x[...,1], ra, levels=np.arange(-2, 2, 0.1), cmap='seismic')
        axes[2].contour(self.grid_x[...,0], self.grid_x[...,1], ra, levels=[0], colors='black')
        axes[2].set_title(f'Terminal Value fn')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        fig.colorbar(ctr_ra, ax=axes[2])

        plt.savefig(save_plot_name)
        plt.close()


if __name__ == "__main__":
    import safety_rl_manip

    from omegaconf import OmegaConf
    from moviepy.editor import ImageSequenceClip

    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')

    env_name = "slide_pickup_obstacles_mujoco_env-v0"
    env = gym.make(env_name, device=0, cfg=env_cfg[env_name])

    env.plot_env('/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media')

    save_gif = True
    num_episodes = 10
    max_ep_len = 300

    for i in range(num_episodes):
        xt_all, at_all, imgs = [], [], []
        xt = env.reset()
        xt_all.append(xt)
        
        done = False
        ep_len = 0
        while not (done or ep_len == max_ep_len):
            at = env.action_space.sample()
            at_all.append(at)
            xt, _, done, _ = env.step(at)
            xt_all.append(xt)
            env.renderer.update_scene(env.data, camera='left_cap3') 
            img = env.renderer.render()
            imgs.append(img)
            ep_len += 1
        if save_gif:
            filename = f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_slideup_obstacles_{i}.gif'
            cl = ImageSequenceClip(imgs[::4], fps=500)
            cl.write_gif(filename, fps=500)
            env.plot_trajectory(np.stack(xt_all,axis=0), np.stack(at_all,axis=0), f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_slideup_obstacles_{i}.png')

