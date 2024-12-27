import warnings
import os

import numpy as np
import gym
from gym.utils import EzPickle
import matplotlib.pyplot as plt

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import MultiTaskNoWallsArena
import mujoco
from mujoco import viewer

from robosuite.models.objects.primitive.box import BoxObject
from robosuite.models.objects.utils import get_obj_from_name

from safety_rl_manip.envs.utils import create_grid
from scipy.spatial.transform import Rotation
import torch
import wandb
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.utils.mjcf_utils import recolor_collision_geoms, new_site, string_to_array
from robosuite.utils.binding_utils import MjSim
import robosuite.utils.sim_utils as SU

from safety_rl_manip.envs.utils import create_grid, signed_dist_fn_rectangle


class SlidePickupClutterMujocoEnv(gym.Env, EzPickle):
    def __init__(self, device, cfg):
        EzPickle.__init__(self)
        self.device = device
        self.env_cfg = cfg
        self.frame_skip = self.env_cfg.frame_skip
        self._did_see_sim_exception = False
        self.goal = np.array(self.env_cfg.goal)

        self.all_object_names = [self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name] + self.env_cfg.objects.names

        self.N_O = len(self.all_object_names)
        self.N_objs = len(self.env_cfg.objects.names)

        self.n = self.env_cfg.n_rel_objs*6 # max number of total objects is 3
        self.n_all = self.N_O*6
        self.m = 3

        assert self.N_objs <= len(self.env_cfg.objects.initial_poses), f"Number of initial poses {len(self.env_cfg.objects.initial_poses)} less than number of object names {self.N_objs}"
        
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

        self.model.vis.global_.offheight = self.env_cfg.img_size[0]
        self.model.vis.global_.offwidth = self.env_cfg.img_size[1]

        self.renderer = mujoco.Renderer(self.model, height=self.env_cfg.img_size[0], width=self.env_cfg.img_size[1])
        
        self.initialize_procedural_sampler()
        self.reset_mocap_welds()

        self.all_body_ids = list(range(self.model.nbody))
        self.all_body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in self.all_body_ids]
        
        self.all_geom_ids = list(range(self.model.ngeom))
        self.all_geom_names=[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in self.all_geom_ids]
        # self.all_geom_names = ['unknown' if geom_name is None else geom_name for geom_name in self.all_geom_names]
        
        self.reset()

        self.side_cam_name = "side_cap"
        self.front_cam_name = "front_cap3"
        self.eye_in_hand_cam_name = "robot0_eye_in_hand"

        self.side_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.side_cam_name)
        self.front_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.front_cam_name)
        self.eye_in_hand_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.eye_in_hand_cam_name)
        
        # self.side_cam_pos, self.side_cam_quat_wxyz = self.get_static_cam_pose(self.side_camera_id)
        # self.front_cam_pos, self.front_cam_quat_wxyz = self.get_static_cam_pose(self.front_camera_id)
        # self.eye_in_hand_cam_pos, self.eye_in_hand_cam_quat_wxyz = self.get_static_cam_pose(self.eye_in_hand_camera_id)

        # Will visualize the value function in x-y only for initial positions of bottom block
        self.N_x = self.env_cfg.block_bottom.N_x
        
        # self.create_env_grid()
        
        # self.target_T = self.target_margin(self.grid_x)
        # self.safety_set_top_low = self.grid_x[...,6:9] - self.env_cfg.block_top.safety_set.low
        # self.safety_set_top_high = self.grid_x[...,6:9] + self.env_cfg.block_top.safety_set.high
        # self.obstacle_T = self.safety_margin(self.grid_x)

        # Visualization Parameters
        # self.visual_initial_states = self.sample_initial_state(self.env_cfg.get('num_eval_trajs', 20)) # samples n_all

    
    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self._get_site_pos('robot0_rightfinger')
        left_finger_pos = self._get_site_pos('robot0_leftfinger')
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center
    
    @property
    def ee_pos(self):
        # body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper0_mocap')
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper0_eef')
        return self.data.xpos[body_id]
    
    @property
    def top_block_pos(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{self.env_cfg.block_top.block_name}_main')
        # self.data.subtree_com[body_id]
        return self.data.xpos[body_id]

    @property
    def bottom_block_pos(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{self.env_cfg.block_bottom.block_name}_main')
        return self.data.xpos[body_id]
    
    def _get_site_pos(self, siteName):
        # _id = self.model.site_names.index(siteName)
        # return self.data.site_xpos[_id].copy()
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, siteName)
        return self.data.site_xpos[sid]
    
    def get_static_cam_pose(self, cam_id):
        cam_pos = self.data.cam_xpos[cam_id]
        cam_rot = self.data.cam_xmat[cam_id].reshape(3,3)
        rotation_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        rotation_x = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        rotation_y = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        new_rotation_matrix =  cam_rot @ rotation_z @ rotation_y
        q_wxyz = Rotation.from_matrix((new_rotation_matrix)).as_quat(scalar_first=True)
        return cam_pos, q_wxyz


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
        
        if self.block_obsA_active and self.block_obsB_active:
            relevantA_idx = distA < distB
            relevant_obs_xy = xy_sample_obsB.copy()
            relevant_obs_xy[relevantA_idx] = xy_sample_obsA[relevantA_idx]
            rel_obs_z = self.block_obsB_z*np.ones((*self.N_x,1))
            rel_obs_z[relevantA_idx] = self.block_obsA_z
            relevant_obs = np.concatenate([relevant_obs_xy, rel_obs_z, np.zeros((*self.N_x,3))], axis=-1)
        elif self.block_obsA_active:
            rel_obs_z = self.block_obsA_z*np.ones((*self.N_x,1))
            relevant_obs = np.concatenate([xy_sample_obsA, rel_obs_z, np.zeros((*self.N_x,3))], axis=-1)
        elif self.block_obsB_active:
            rel_obs_z = self.block_obsB_z*np.ones((*self.N_x,1))
            relevant_obs = np.concatenate([xy_sample_obsB, rel_obs_z, np.zeros((*self.N_x,3))], axis=-1)
        else:
            raise NotImplementedError("Atleast one of the obstacles should be active!")
        
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

        self.mujoco_robot = Panda()
        self.gripper = gripper_factory('PandaGripper')
        self.mujoco_robot.set_base_xpos(self.env_cfg.robot_base_pos)
        self.mujoco_robot.add_gripper(self.gripper)
        recolor_collision_geoms(root=self.mujoco_robot.worldbody, rgba=[0, 0, 0., 0.])
        world.merge(self.mujoco_robot)

        self.all_mujoco_objects = {}
        # Bottom block
        if 'block' in self.env_cfg.block_bottom.block_name:
            block_bottom = BoxObject(name='block_bottom', size=self.env_cfg.block_bottom.size, rgba=self.env_cfg.block_bottom.rgba)
        else:
            block_bottom = get_obj_from_name(self.env_cfg.block_bottom.block_name)
        block_bottom_body = block_bottom.get_obj()
        self.block_bottom_z = -block_bottom.bottom_offset[2]
        block_bottom_body.set('pos', f'{self.env_cfg.block_bottom.initial_pos[0]} {self.env_cfg.block_bottom.initial_pos[1]} {self.block_bottom_z}')
        world.worldbody.append(block_bottom_body)
        world.merge_assets(block_bottom)
        self.all_mujoco_objects['block_bottom'] = block_bottom
        hor_site = block_bottom.worldbody.find(f"./body/site[@name='{self.env_cfg.block_bottom.block_name}_horizontal_radius_site']")
        self.bottom_hor_rad = string_to_array(hor_site.get("pos"))

        rot_z = Rotation.from_euler('z', -90, degrees=True)  # 90 degrees about Z-axis
        rot_x = Rotation.from_euler('x', -90, degrees=True)  # 90 degrees about X-axis
        relquat = (rot_x*rot_z).as_quat(scalar_first=True)
        
        suction_site = new_site(
            name=f"suction_site", 
            pos=(-self.bottom_hor_rad[0]+0.01, -self.block_bottom_z, 0), 
            quat=relquat,
            size=(0.01, 0.01, 0.01), 
            rgba=(1, 0, 0, 1)
        )
        block_bottom_body.append(suction_site)
        
        # Top block
        if 'block' in self.env_cfg.block_top.block_name:
            block_top = BoxObject(name='block_top', size=self.env_cfg.block_top.size, rgba=self.env_cfg.block_top.rgba)
        else:
            block_top = get_obj_from_name(self.env_cfg.block_top.block_name)
        block_top_body = block_top.get_obj()
        self.block_top_z = (block_bottom.top_offset - block_bottom.bottom_offset - block_top.bottom_offset)[2]
        block_top_body.set('pos', f'{self.env_cfg.block_top.initial_pos[0]} {self.env_cfg.block_top.initial_pos[1]} {self.block_top_z}')
        world.worldbody.append(block_top_body)
        world.merge_assets(block_top)
        self.all_mujoco_objects['block_top'] = block_top

        self.object_z = []
        for obj_name, obj_pos in zip(self.env_cfg.objects.names, self.env_cfg.objects.initial_poses):
            obj = get_obj_from_name(obj_name)
            obj_body = obj.get_obj()
            self.object_z.append(-obj.bottom_offset[2])
            obj_body.set('pos', f'{obj_pos[0]} {obj_pos[1]} {-obj.bottom_offset[2]}')
            world.worldbody.append(obj_body)
            world.merge_assets(obj)
            self.all_mujoco_objects[obj_name] = obj
        self.object_z = np.array(self.object_z)

        # world.root.find('compiler').set('inertiagrouprange', '0 5')
        # world.root.find('compiler').set('inertiafromgeom', 'auto')
        self.model = world.get_model(mode="mujoco")
        self.data = mujoco.MjData(self.model)

        self.sim = MjSim(self.model)
        self.arm_qpos_index = [self.sim.model.get_joint_qpos_addr(x) for x in self.mujoco_robot.joints if 'wrist' not in x]
        self.qvel_index = [self.sim.model.get_joint_qvel_addr(x) for x in self.mujoco_robot.joints]

        self.wrist_qpos_index = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper.joints if 'wrist' in x]
        # self.wrist_qpos_index = [self.sim.model.get_joint_qpos_addr(x) for x in self.mujoco_robot.joints if 'wrist' in x]

        self.gripper_qpos_index = [self.sim.model.get_joint_qpos_addr(x) for x in self.gripper.joints if 'finger' in x]

        self._ref_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.mujoco_robot.actuators
        ]
        self._ref_gripper_fingers_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.gripper.actuators if 'finger' in actuator
        ]

        self.robot_dof = len(self.mujoco_robot.actuators) + len(self.gripper.actuators)
          
    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        # sim = self.sim
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i, :] = np.array(
                        [0.,  0.,  0.,
                        0.0,  0.,  0.,
                        0., 1.,  0.,
                        0.,  1.])
        
        mujoco.mj_forward(self.model, self.data)

    def initialize_procedural_sampler(self):

        object_samplers = []
        block_bottom_sampler = UniformRandomSampler(
            name=f"{self.env_cfg.block_bottom.block_name}_sampler",
            x_range=[self.env_cfg.block_bottom.state_ranges.low[0], self.env_cfg.block_bottom.state_ranges.high[0]],
            y_range=[self.env_cfg.block_bottom.state_ranges.low[1], self.env_cfg.block_bottom.state_ranges.high[1]],
            rotation=0,  # do not Randomize orientation
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
        )
        object_samplers.append(block_bottom_sampler)
        block_top_sampler = UniformRandomSampler(
            name=f"{self.env_cfg.block_top.block_name}_sampler",
            x_range=[self.env_cfg.block_top.state_ranges.low[0], self.env_cfg.block_top.state_ranges.high[0]],
            y_range=[self.env_cfg.block_top.state_ranges.low[1], self.env_cfg.block_top.state_ranges.high[1]],
            rotation=0,  # do not Randomize orientation
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            z_offset=0.02,
        )
        object_samplers.append(block_top_sampler)

        for obj_name in self.env_cfg.objects.names:
            object_samplers.append(UniformRandomSampler(
                name=f"{obj_name}_sampler",
                x_range=[self.env_cfg.objects.state_ranges.low[0], self.env_cfg.objects.state_ranges.high[0]],
                y_range=[self.env_cfg.objects.state_ranges.low[1], self.env_cfg.objects.state_ranges.high[1]],
                rotation=0,  # do not Randomize orientation
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
            ))

        self.composite_sampler = SequentialCompositeSampler(name="table_sampler")
        self.composite_sampler.append_sampler(object_samplers[0])
        self.composite_sampler.append_sampler(object_samplers[1], sample_args={'reference': self.env_cfg.block_bottom.block_name, 'on_top': True})
        for i in range(self.N_objs):
            self.composite_sampler.append_sampler(object_samplers[2+i])

        for obj_name, mj_obj in self.all_mujoco_objects.items():
            self.composite_sampler.add_objects_to_sampler(sampler_name=f"{obj_name}_sampler", mujoco_objects=mj_obj)

    def sample_initial_state(self, N):
        states = np.zeros((N, self.n_all))

        if self.env_cfg.randomize_locations:
            sample = self.composite_sampler.sample()
            for i, obj_name in enumerate(self.all_object_names):
                states[:, i*6:i*6+3] = np.array(sample[obj_name][0])
        else:
            states[:, 0:2] = self.env_cfg.block_bottom.initial_pos
            states[:, 6:8] = self.env_cfg.block_top.initial_pos
            states[:,2] = self.block_bottom_z
            states[:,8] = self.block_top_z
            for i in range(self.N_objs):
                states[:, 12+i*6:12+i*6+2] = self.env_cfg.objects.initial_poses[i]
                states[:, 12+i*6+2] = self.object_z[i]

        return states
    
    def _reset_hand(self, steps=50):
        self.data.qpos[self.arm_qpos_index] = np.array(self.env_cfg.init_arm_qpos) # robot joints
        self.data.qpos[self.wrist_qpos_index] = np.array(self.env_cfg.init_wrist_qpos) # gripper wrist
        self.data.qpos[self.gripper_qpos_index] = np.array(self.env_cfg.init_fingers_qpos) # gripper fingers

        neq = self.model.eq_data.shape[0]
        self.model.eq_active0[neq-1] = 0
        self.data.eq_active[neq-1] = 0
        self.suction_gripper_active = False
        
        for _ in range(steps):
            # self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

            # self.data.qpos[self.arm_qpos_index] = np.array(self.env_cfg.init_arm_qpos) # robot joints
            # self.data.qpos[self.wrist_qpos_index] = np.array(self.env_cfg.init_wrist_qpos) # gripper wrist
            # self.data.qpos[self.gripper_qpos_index] = np.array(self.env_cfg.init_fingers_qpos) # gripper fingers

            self.data.mocap_pos = np.array(self.env_cfg.hand_init_pos)
            self.data.mocap_quat = np.array(self.env_cfg.mocap_quat)

            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center
    
    def _gravity_compensation(self):
        torques = self.data.qfrc_bias[self.qvel_index]
        self.data.ctrl[self._ref_joint_actuator_indexes] = torques
        # low = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 0]
        # high = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 1]

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.env_cfg.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            np.array(self.env_cfg.mocap_low),
            np.array(self.env_cfg.mocap_high),
        )
        # self.data.set_mocap_pos('mocap', new_mocap_pos)
        # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

        self.data.mocap_pos = new_mocap_pos.copy()
        self.data.mocap_quat = np.array(self.env_cfg.mocap_quat)

    def reset(self, start=None):
        self._did_see_sim_exception = False
        mujoco.mj_resetData(self.model, self.data)
        self._reset_hand()

        self._current_timestep = 0

        if start is None:
            sample_state = self.sample_initial_state(1)[0]
        else:
            sample_state = start.copy()

        for i in range(self.N_O):
            self.data.qpos[self.robot_dof+i*7:self.robot_dof+i*7+3] = sample_state[i*6:i*6+3]
        
        for _ in range(20):
            self.do_simulation([-1, 1], self.frame_skip)
        
        # mujoco.mj_forward(self.model, self.data)

        self.ee_pos_tm1 = self.ee_pos.copy()
        self.ee_pos_t = self.ee_pos.copy()
        self.ee_vel_t = (self.ee_pos_t-self.ee_pos_tm1)/self.dt

        curr_state = self.get_current_state()

        self.safety_set_top_low = curr_state[12:15] + self.env_cfg.block_top.safety_set.low
        self.safety_set_top_high = curr_state[12:15] + self.env_cfg.block_top.safety_set.high

        self.target_set_low = curr_state[6:9] + self.env_cfg.block_bottom.target_set.low
        self.target_set_high = curr_state[6:9] + self.env_cfg.block_bottom.target_set.high


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
        # EE within table
        outsideLeft_ee = np.any((state[...,0:3] <= self.env_bound_low), axis=-1)
        outsideRight_ee = np.any((state[...,0:3] >= self.env_bound_high), axis=-1)
        outside_ee = np.logical_or(outsideLeft_ee, outsideRight_ee)

        # Bottom bottom within table
        outsideLeft_bottom = np.any((state[...,6:9] <= self.env_bound_low), axis=-1)
        outsideRight_bottom = np.any((state[...,6:9] >= self.env_bound_high), axis=-1)
        outside_bottom = np.logical_or(outsideLeft_bottom, outsideRight_bottom)

        # Bottom top within table
        outsideLeft_top = np.any((state[...,12:15] <= self.env_bound_low), axis=-1)
        outsideRight_top = np.any((state[...,12:15] >= self.env_bound_high), axis=-1)
        outside_top = np.logical_or(outsideLeft_top, outsideRight_top)

        return np.logical_or(np.logical_or(outside_bottom, outside_top), outside_ee)

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

    def do_simulation(self, ctrl, n_frames=None):
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        # self.data.xfrc_applied[self.model.body('block_bottom_main').id] = action # [0.0, 0.0, 50.0, 0.0, 0.0, 0.0]

        if self.env_cfg.gravity_compensation:
            self._gravity_compensation()
        
        self.data.ctrl[self._ref_gripper_fingers_actuator_indexes] = ctrl

        for _ in range(n_frames):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def get_current_state(self):
        xt = np.append(self.ee_pos_t, self.ee_vel_t)
        for i, body_name in enumerate(self.all_object_names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{body_name}_main')
            body_state = np.append(self.data.xpos[body_id], self.data.qvel[self.robot_dof+i*6: self.robot_dof+i*6+3])
            xt = np.append(xt, body_state)
        return xt
    
    def check_contact_slide(self):
        for contact in self.data.contact[: self.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            # g1, g2 = self.model.geom_id2name(contact.geom1), self.model.geom_id2name(contact.geom2)

            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if (g1 in self.gripper.contact_geoms) and (self.env_cfg.block_bottom.block_name in g2):
                return True

            if (g2 in self.gripper.contact_geoms) and (self.env_cfg.block_bottom.block_name in g1):
                return True
            
        return False


    def check_suction_grasp(self):
        grasp_object_idx = 0
        if self.check_contact_slide() and not self.suction_gripper_active:
            
            # body1_name = "gripper0_eef"
            # body2_name = f"{self.all_object_names[grasp_object_idx]}_main"
            # body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
            # body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
            # relpos = self.bottom_block_pos - self.ee_pos
            # # Compute the relative orientation
            # quat_body1 = self.data.xquat[body1_id]
            # quat_body2 = self.data.xquat[body2_id]
            # neg_quat_body1 = np.zeros(4)
            # mujoco.mju_negQuat(neg_quat_body1, quat_body1)
            # relquat = np.zeros(4)
            # mujoco.mju_mulQuat(relquat, quat_body2, neg_quat_body1)
            # rot_z = Rotation.from_euler('y', 90, degrees=True)  # 90 degrees about Z-axis
            # rot_x = Rotation.from_euler('y', 0, degrees=True)  # 90 degrees about X-axis
            # relquat = (rot_x*rot_z).as_quat(scalar_first=True)
            # neq = self.model.eq_data.shape[0]
            # self.model.eq_data[neq-1, :3] = relpos
            # # self.model.eq_data[neq-1, 7:10] = self.ee_pos - self.bottom_block_pos
            # self.model.eq_data[neq-1, 3:3+4] = relquat
            # self.model.eq_type[neq-1] = 1  # 1 corresponds to a 'weld' constraint
            # self.model.eq_obj1id[neq-1] = body1_id
            # self.model.eq_obj2id[neq-1] = body2_id
            # self.model.eq_active0[neq-1] = 1
            # self.data.eq_active[neq-1] = 1
            # self.model.eq_objtype[neq-1] = 1 # body
            # self.suction_gripper_active = True

            print([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)])

            site1_name = "gripper0_grip_site"
            site2_name = "suction_site"
            site1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site1_name)
            site2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site2_name)
            neq = self.model.eq_data.shape[0]
            self.model.eq_type[neq-1] = 1  # 1 corresponds to a 'weld' constraint
            self.model.eq_obj1id[neq-1] = site1_id
            self.model.eq_obj2id[neq-1] = site2_id
            self.model.eq_active0[neq-1] = 1
            self.data.eq_active[neq-1] = 1
            self.model.eq_objtype[neq-1] = 6 # site
            self.suction_gripper_active = True

            self.sim.forward()
            # viewer.launch(self.model, self.data)

    # == Dynamics ==
    def step(self, action):
        # self.do_simulation(np.concatenate([action.flatten(), np.zeros(3)])) # move blocks directly
        self.ee_pos_tm1 = self.ee_pos.copy()

        xt = self.get_current_state()
        fail, g_x = self.check_failure(xt.reshape(1,-1))
        success, l_x = self.check_success(xt.reshape(1,-1))
        done = self.get_done(xt.reshape(1,-1), success, fail)[0]
        cost = self.get_cost(l_x, g_x, success, fail)[0]
        info = {"g_x": g_x[0], "l_x": l_x[0]}

        self.set_xyz_action(action[:3])
        self.do_simulation([-action[-1], action[-1]])
        self.check_suction_grasp()

        self.ee_pos_t = self.ee_pos.copy()
        self.ee_vel_t = (self.ee_pos_t-self.ee_pos_tm1)/self.dt
        
        xtp1 = self.get_current_state()
        return xtp1, cost, done, info

    def render(self):
        pass
    
    def get_suction_target(self):

        # target_pos = self.bottom_block_pos.copy()
        # target_pos[0] = target_pos[0] - self.bottom_hor_rad[0] + 0.03

        target_pos = self._get_site_pos('suction_site')
                
        return target_pos

    def get_action(self):
        action = np.zeros(4)
        action[3] = 1.0
        if self.suction_gripper_active:
            action[:3] = [-0.2, 0, 0.1]
            # action[0] = 0
            print("suction active")
        else:
            target_pos = self.get_suction_target()
            action[:3] = target_pos - self.ee_pos

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'placeSiteB')
            self.model.site_pos[site_id] = target_pos
            # viewer.launch(self.model, self.data)

        return action

    def collision_distance(self, obj1_name, obj1_state, obj2_name, obj2_state):
        # g(x)>0 is obstacle
        hor_site = self.all_mujoco_objects[obj1_name].worldbody.find(f"./body/site[@name='{obj1_name}_horizontal_radius_site']")
        obj1_hor_rad = string_to_array(hor_site.get("pos"))

        hor_site = self.all_mujoco_objects[obj2_name].worldbody.find(f"./body/site[@name='{obj2_name}_horizontal_radius_site']")
        obj2_hor_rad = string_to_array(hor_site.get("pos"))

        obstacle_high = obj1_state[...,:3] + obj1_hor_rad + obj2_hor_rad
        obstacle_low = obj1_state[...,:3] - obj1_hor_rad - obj2_hor_rad
        obstacle = signed_dist_fn_rectangle(
            obj2_state[...,:3], 
            obstacle_low, 
            obstacle_high,
            obstacle=True)

        # collision_dist = self.all_mujoco_objects[0].horizontal_radius + self.all_mujoco_objects[2+i].horizontal_radius
        # obstacle = (self.env_cfg.thresh + collision_dist) - np.linalg.norm(s[...,6:9]-s[...,18+i*6:18+i*6+3])
        return obstacle

    def safety_margin(self, s, safety_set_top_low=None, safety_set_top_high=None):
        # Input:s top block position, shape (batch, 3)
        # g(x)>0 is obstacle

        top_block_pos = s[...,12:15]
        if safety_set_top_low is None:
            safety_set_top_low = top_block_pos + self.env_cfg.block_top.safety_set.low

        if safety_set_top_high is None:
            safety_set_top_high = top_block_pos + self.env_cfg.block_top.safety_set.high

        lower_boundary = np.max(safety_set_top_low - top_block_pos, axis=-1)
        upper_boundary = np.max(top_block_pos - safety_set_top_high, axis=-1)
        gx = np.maximum(lower_boundary, upper_boundary) # top block safety

        # # gripper should not hit top block
        # obstacle_high = s[...,12:15] + self.env_cfg.block_top.size
        # obstacle_low = s[...,12:15] - self.env_cfg.block_top.size
        # obstacle = signed_dist_fn_rectangle(
        #     s[...,:3], 
        #     obstacle_low, 
        #     obstacle_high,
        #     obstacle=True)
        # gx = np.maximum(gx, obstacle)

        # obstacle avoidance between bottom block and distractors
        for i, obj_name in enumerate(self.env_cfg.objects.names):
            obstacle = self.collision_distance('block_bottom', s[...,6:9], obj_name, s[...,18+i*6:18+i*6+3])
            gx = np.maximum(gx, obstacle)
        
        gx = self.scaling_safety * gx

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
        # goal = np.concatenate([self.goal, np.zeros(3)])

        bottom_block_pos = s[...,6:9]
        if target_set_low is None:
            target_set_low = bottom_block_pos + self.env_cfg.block_bottom.target_set.low

        if target_set_high is None:
            target_set_high = bottom_block_pos + self.env_cfg.block_bottom.target_set.high
        
        # site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'placeSiteB')
        # self.model.site_pos[site_id] = (target_set_high+target_set_low)/2

        # lx = np.linalg.norm(s[...,:6]-goal, axis=-1) - self.env_cfg.thresh

        if self.suction_gripper_active:
            lower_boundary = np.max(target_set_low - bottom_block_pos, axis=-1)
            upper_boundary = np.max(bottom_block_pos - target_set_high, axis=-1)
            lx = np.maximum(lower_boundary, upper_boundary)
        else:
            target_pos = self.get_suction_target()
            lx = np.linalg.norm(s[...,:3]-target_pos, axis=-1) - self.env_cfg.thresh

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
    # from moviepy.editor import ImageSequenceClip
    import imageio
    from PIL import Image

    out_folder = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/slide_pickup_clutter/'
    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')

    env_name = "slide_pickup_clutter_mujoco_env-v0"
    env = gym.make(env_name, device=0, cfg=env_cfg[env_name])

    save_gif = True
    num_episodes = 1
    max_ep_len = 200
    down_action = np.array([0.0,0,-0.3,0])
    up_action = np.array([0.3,0,0.3,0])

    for i in range(num_episodes):
        xt_all, at_all, imgs = [], [], []
        xt = env.reset()
        xt_all.append(xt)
        
        done = False
        ep_len = 0
        action = down_action.copy()
        # while not (done or ep_len == max_ep_len):
        while not (ep_len == max_ep_len):
            at = env.action_space.sample()
            at_all.append(at)
            xt, _, done, _ = env.step(env.get_action())
            # xt, _, done, _ = env.step(np.zeros(4))
            

            # body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper0_mocap')
            # print(f"mocap pos: {env.data.xpos[body_id]}")
            # body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'gripper0_eef')
            # print(f"eef pos: {env.data.xpos[body_id]}")
            
            # if ep_len>150:
            #     print(env.all_geom_names)
            #     g_geoms = [env.gripper.important_geoms["left_finger"], env.gripper.important_geoms["right_finger"]]
            #     g_pad_geoms = [env.gripper.important_geoms["left_fingerpad"], env.gripper.important_geoms["right_fingerpad"]]
            #     o_geoms = env.block_top.contact_geoms

            #     # print(SU.check_contact(sim=env.sim, geoms_1='gripper0_finger1_collision', geoms_2='block_top_g0_vis'))
                
            #     for g_group in g_geoms:
            #         # if not self.check_contact(g_group, o_geoms):
            #         print(f'At t:{ep_len}, contact between {g_group} and {o_geoms}: {SU.check_contact(sim=env.sim, geoms_1=g_group, geoms_2=o_geoms)}')
            #     print(SU.check_contact(sim=env.sim, geoms_1=env.gripper.contact_geoms, geoms_2=env.block_top.contact_geoms))

            # print(get_contacts(env, env.block_top))
            # print(check_contact(env, geoms_1=env.gripper.contact_geoms, geoms_2=env.block_top.contact_geoms))


            xt_all.append(xt)

            env.renderer.update_scene(env.data, camera=env.front_cam_name)
            img = env.renderer.render()
            Image.fromarray(img).save(out_folder + f"rgb_slide_pickup_clutter_t_{ep_len}_{i}_front.png")
            imgs.append(img)

            env.renderer.update_scene(env.data, camera=env.side_cam_name) 
            img = env.renderer.render()
            # Image.fromarray(img).save(out_folder + f"rgb_slide_pickup_clutter_t_{ep_len}_{i}_side.png")
            # Image.fromarray(depth.astype(np.uint8)).save(out_folder + f"clutter_depth_{i}_front.png")
            
            ep_len += 1
        print(f'ep_len: {ep_len}')
        if save_gif:
            filename = out_folder + f'test_slide_pickup_clutter_box_{i}.gif'
            imageio.mimsave(filename, imgs[::4], duration=100)
