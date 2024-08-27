import warnings

import numpy as np
import gym
from gym.utils import EzPickle
from gym import spaces

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import TableArena, MultiTaskNoWallsArena
import mujoco
from mujoco import viewer
from moviepy.editor import ImageSequenceClip
from robosuite.models.objects.primitive.box import BoxObject

class SlidePickupMujocoEnv(gym.Env, EzPickle):
    def __init__(self, device, cfg):
        EzPickle.__init__(self)
        
        self.device = device
        self.env_cfg = cfg
        self.frame_skip = self.env_cfg.frame_skip
        self._did_see_sim_exception = False
        self.goal = np.array(self.env_cfg.goal)

        self.N_O = 2 # TODO(saumya): add obstacles
        self.n = self.N_O*6
        self.m = 3

        self.u_min = np.array(self.env_cfg.control_low)
        self.u_max = np.array(self.env_cfg.control_high)
        self.action_space = gym.spaces.Box(self.u_min, self.u_max)
        # self.observation_space = gym.spaces.Box(
        #     np.float32(),
        #     np.float32(self.midpoint + self.interval / 2)
        # )
        self.doneType = self.env_cfg.doneType
        self.costType = self.env_cfg.costType

        self._setup_procedural_env()
        self.renderer = mujoco.Renderer(self.model, self.env_cfg.img_size[0], self.env_cfg.img_size[1])

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

        # TODO(saumya): Add distractors

        world.root.find('compiler').set('inertiagrouprange', '0 5')
        world.root.find('compiler').set('inertiafromgeom', 'auto')
        self.model = world.get_model(mode="mujoco")
        self.data = mujoco.MjData(self.model)

    
    def reset(self, start=None):
        self._did_see_sim_exception = False
        mujoco.mj_resetData(self.model, self.data)

        self._current_timestep = 0
        if start is None:
            xy_sample = np.random.uniform(low=self.env_cfg.block_bottom.state_ranges.low, high=self.env_cfg.block_bottom.state_ranges.high)
            self.data.qpos[0:2] = xy_sample
            self.data.qpos[7:9] = xy_sample
        else:
            self.data.qpos[0:2] = start
            self.data.qpos[7:9] = start

        self.data.qpos[2] = self.block_bottom_z
        self.data.qpos[9] = self.block_top_z
        mujoco.mj_forward(self.model, self.data)
        return self.get_current_state()

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
        outsideLeft = (state[...,0] <= self.bounds[0, 0])
        outsideRight = (state[...,0] >= self.bounds[0, 1])
        return np.logical_or(outsideLeft, outsideRight)

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
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
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
        return np.concatenate([self.data.qpos[0:3], self.data.qvel[0:3], self.data.qpos[7:10], self.data.qvel[6:9]])
    
    # == Dynamics ==
    def step(self, action):
        self.do_simulation(np.concatenate(action.flatten(), np.zeros(3)))
        xtp1 = self.get_current_state()

        fail, g_x = self.check_failure(self.state.reshape(1,self.n))
        success, l_x = self.check_success(self.state.reshape(1,self.n))
        done = self.get_done(self.state.reshape(1,self.n), success, fail)[0]
        cost = self.get_cost(l_x, g_x, success, fail)[0]

        self.state = xtp1
        info = {"g_x": g_x[0], "l_x": l_x[0]}
        return xtp1, cost, done, info

    def render(self):
        pass

      # == Getting Margin ==
    def safety_margin(self, s):
        # g(x)>0 is obstacle
        left_boundary = -0.4-s[:,6]
        right_boundary = s[:,6]-0.4
        obstacle = np.maximum(left_boundary, right_boundary)
        
        gx = self.scaling * obstacle

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
        # bottom block reached goal region
        lx = np.linalg.norm(s[:,:3]-self.goal, axis=1) - self.env_cfg.thresh

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


if __name__ == "__main__":
    from 

    from gym_reachability import gym_reachability  # register Custom Gym envs.
    from omegaconf import OmegaConf
    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')

    env = gym.make("slide_pickup_mujoco_env-v0", device=0, cfg=env_cfg)

    env.reset()
    save_gif = True
    imgs = []
    for i in range(1000):
        env.step(env.action_space.sample())
        env.renderer.update_scene(env.data, camera='left_cap3') 

        img = env.renderer.render()
        imgs.append(img)

    if save_gif:
        filename = f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_slide_up.gif'
        cl = ImageSequenceClip(imgs[::4], fps=500)
        cl.write_gif(filename, fps=500)