import warnings
import os

import numpy as np
import random
import gym
import copy
from gym.utils import EzPickle
import matplotlib.pyplot as plt

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import MultiTaskNoWallsArena
import mujoco
from mujoco import viewer

from robosuite.models.objects.primitive.box import BoxObject
from robosuite.models.objects.utils import get_obj_from_name

from safety_rl_manip.envs.utils import create_grid, draw_bounding_boxes_cv2, get_bounding_boxes, signed_dist_fn_rectangle, plot_colored_segmentation
from scipy.spatial.transform import Rotation
import torch
import wandb
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.utils.mjcf_utils import recolor_collision_geoms, new_site, string_to_array
from robosuite.utils.binding_utils import MjSim
import robosuite.utils.sim_utils as SU
from PIL import Image
from scipy.ndimage import median_filter

class SlidePickupClutterMujocoMultimodalEnv(gym.Env, EzPickle):
    def __init__(self, device, cfg):
        EzPickle.__init__(self)
        self.device = device
        self.env_cfg = cfg
        self.n_rel_objs = self.env_cfg.n_rel_objs
        self.frame_skip = self.env_cfg.frame_skip
        self._did_see_sim_exception = False
        self.goal = np.array(self.env_cfg.goal)
        self.observations = self.env_cfg.observations
        self.constraint_type_repr = self.env_cfg.get('constraint_type_repr', 'int')

        self.all_blocks_object_names = [self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name] + self.env_cfg.objects.names

        self.N_all_objects = len(self.all_blocks_object_names)
        self.N_dist_objects = len(self.env_cfg.objects.names)

        self.use_constraint_types = self.env_cfg.use_constraint_types

        self.object_type_to_int_mapping = {
            f'ee': 0,
            f'{self.env_cfg.block_bottom.block_name}': 1,
            f'{self.env_cfg.block_top.block_name}': 2
        }
        self.constraint_to_int_mapping = {
            'any_contact': 3,
            'soft_contact': 4,
            'no_contact': 5
        }
        self.max_constraint_types = 6
    
        if self.use_constraint_types:
            if self.constraint_type_repr == 'int':
                self.low_dim_sizes = {'robot0': 7, 'objects': 7}
            elif self.constraint_type_repr == 'one_hot':
                self.low_dim_sizes = {'robot0': 6+self.max_constraint_types, 'objects': 6+self.max_constraint_types}
        else:
            self.low_dim_sizes = {'robot0': 6, 'objects': 6}

        self.low_dim_sizes['full_size'] = self.low_dim_sizes['robot0'] + (2+self.n_rel_objs)*self.low_dim_sizes['objects']
        self.set_observation_shapes()

        self.obj_to_constraint_map = self.env_cfg.obj_to_constraint_map
        self.obj_to_constraint_map_gt = copy.deepcopy(self.obj_to_constraint_map)

        # self.m = 3

        assert self.N_dist_objects <= len(self.env_cfg.objects.initial_poses), f"Number of initial poses {len(self.env_cfg.objects.initial_poses)} less than number of object names {self.N_dist_objects}"
        assert self.env_cfg.n_rel_objs <= len(self.env_cfg.objects.names)
        
        self.u_min = np.array(self.env_cfg.control_low)
        self.u_max = np.array(self.env_cfg.control_high)
        self.action_space = gym.spaces.Box(self.u_min, self.u_max)

        self.env_bound_low = np.array(self.env_cfg.env_bounds.low)
        self.env_bound_high = np.array(self.env_cfg.env_bounds.high)

        # self.observation_space = gym.spaces.Box(
        #     self.env_bound_low,
        #     self.env_bound_high
        # )

        self.observation_space = gym.spaces.Box(
            -np.inf*np.ones(self.low_dim_sizes['full_size']),
            np.inf*np.ones(self.low_dim_sizes['full_size'])
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

        self.side_cam_name = "side_cap"
        self.front_cam_name = "front_cap3"
        self.eye_in_hand_cam_name = "robot0_eye_in_hand"
        
        self.reset()

        self.side_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.side_cam_name)
        self.front_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.front_cam_name)
        self.eye_in_hand_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.eye_in_hand_cam_name)
        
        self.side_cam_pos, self.side_cam_quat_wxyz = self.get_static_cam_pose(self.side_camera_id)
        self.front_cam_pos, self.front_cam_quat_wxyz = self.get_static_cam_pose(self.front_camera_id)
        self.eye_in_hand_cam_pos, self.eye_in_hand_cam_quat_wxyz = self.get_static_cam_pose(self.eye_in_hand_camera_id)

        # Camera info
        height, width = self.renderer._height, self.renderer._width
        fx = width / (2.0 * np.tan(float(self.model.cam_fovy[self.front_camera_id]) * np.pi / 360.0))
        fy = height / (2.0 * np.tan(float(self.model.cam_fovy[self.front_camera_id]) * np.pi / 360.0))
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
        self.camera_info = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
            "K": K,
            "front_cam_pos": self.front_cam_pos,
            "front_cam_quat_wxyz": self.front_cam_quat_wxyz,
        }
        # Will visualize the value function in x-y only for initial positions of bottom block
        self.N_x = self.env_cfg.block_bottom.N_x
        
        # self.create_env_grid()
        
        # self.target_T = self.target_margin(self.grid_x)
        # self.safety_set_top_low = self.grid_x[...,6:9] - self.env_cfg.block_top.safety_set.low
        # self.safety_set_top_high = self.grid_x[...,6:9] + self.env_cfg.block_top.safety_set.high
        # self.obstacle_T = self.safety_margin(self.grid_x)
        
        # Visualization Parameters
        # self.visual_initial_states = self.sample_initial_state(self.env_cfg.get('num_eval_trajs', 20)) # samples n_all
        self.visual_initial_states = None
        self.all_failure_modes = ['top_block_oob', 'collision_no_contact', 'collision_soft_contact', 'out_of_env', 'success']
    
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
    def env_observation_shapes(self):
        return self.observation_shapes

    @property
    def ee_pos(self):
        # body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper0_mocap')
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper0_eef')
        return self.data.xpos[body_id]
    
    @property
    def top_block_pos(self):
        # self.data.subtree_com[body_id]
        return self._get_body_pos(self.env_cfg.block_top.block_name)

    @property
    def bottom_block_pos(self):
        return self._get_body_pos(self.env_cfg.block_bottom.block_name)

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def set_observation_shapes(self):
        self.observation_shapes = {}
        self.observation_shapes['robot_state'] = (self.low_dim_sizes['robot0'],)
        if 'objects_state' in self.observations['low_dim']:
            self.observation_shapes['objects_state'] = (self.N_all_objects, self.low_dim_sizes['objects'])
        for k in self.observations['rgb']:
            self.observation_shapes[k] = (self.env_cfg.img_size[0], self.env_cfg.img_size[1], 3)

    def update_relevant_objs(self, objs=None):
        if objs is None: # Training (Run on reset): randomly sample n_rel_objs from all objects
            self.rel_obj_names = random.sample(self.env_cfg.objects.names, len(self.env_cfg.objects.names))[:self.env_cfg.n_rel_objs]
        else: # Eval
            if not (set(self.rel_obj_names) == set(objs)): # do not update if the unique elements are the same
                self.rel_obj_names = [o for o in objs]
                    
    def update_constraint_types(self, const_types=None):
        if const_types is None: # Training: pred constraint types and GT constraint types are the same
            self.pred_constraint_types = {obj_name: random.choice(self.env_cfg.constraint_types) for obj_name in self.env_cfg.objects.names}
            self.obj_to_constraint_map = copy.deepcopy(self.pred_constraint_types) # used for safety analysis
        else: # Eval: Use GT constraint types specified by user
            self.pred_constraint_types = const_types
            self.obj_to_constraint_map = copy.deepcopy(self.obj_to_constraint_map_gt) # used for safety analysis
        
    def _get_site_pos(self, siteName):
        # _id = self.model.site_names.index(siteName)
        # return self.data.site_xpos[_id].copy()
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, siteName)
        return self.data.site_xpos[sid]
    
    def _get_body_pos(self, bodyName):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{bodyName}_main')
        return self.data.xpos[body_id]
    
    def _get_body_vel(self, bodyName):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{bodyName}_main')
        dof_addr = self.model.body_dofadr[body_id]  # Degree of freedom address
        velocity = self.data.qvel[dof_addr:dof_addr+self.model.body_dofnum[body_id]]
        return velocity[:3] # linear velocity
    
    def get_static_cam_pose(self, cam_id):
        cam_pos = self.data.cam_xpos[cam_id]
        cam_rot = self.data.cam_xmat[cam_id].reshape(3,3) # camera points along -z-axis
        rotation_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        rotation_y = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        new_rotation_matrix =  cam_rot @ rotation_z @ rotation_y # converts camera to point in +z axis

        q_wxyz = Rotation.from_matrix(new_rotation_matrix).as_quat(scalar_first=True)
        return cam_pos, q_wxyz
    
    def get_current_image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render()

    def get_current_segmentation_mask(self, cam_name):
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=cam_name)
        seg_img = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        seg_geoms = seg_img[:,:,0]
        seg_geoms[seg_geoms==-1] = 0
        smoothed_seg_geoms = median_filter(seg_geoms, size=3)
        seg_body_id = self.model.geom_bodyid[smoothed_seg_geoms]
        
        # Image.fromarray(seg_body_id.astype(np.uint8)).save("seg_img.png")
        # plot_colored_segmentation(seg_body_id)
        return seg_body_id
    
    def get_img_with_bb(self, cam_name):
        rgb_img = self.get_current_image(cam_name)
        seg_img = self.get_current_segmentation_mask(cam_name)
        bboxes = get_bounding_boxes(seg_img)

        sorted_obj_names = self.get_sorted_objects()
        xml_obj_names = [f'{obj}_main' for obj in sorted_obj_names]
        xml_obj_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in xml_obj_names]
        
        bboxes_to_plot = {}
        bboxes_with_names = {}
        for i in range(self.N_dist_objects):
            if xml_obj_ids[i] in bboxes.keys():
                bboxes_to_plot[i] = bboxes[xml_obj_ids[i]]
                bboxes_with_names[sorted_obj_names[i]] = bboxes[xml_obj_ids[i]]

        rgb_bb = draw_bounding_boxes_cv2(rgb_img, bboxes_to_plot, sorted_obj_names)
        return rgb_bb, bboxes_with_names
    
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
        self.all_mujoco_objects[self.env_cfg.block_bottom.block_name] = block_bottom
        hor_site = block_bottom.worldbody.find(f"./body/site[@name='{self.env_cfg.block_bottom.block_name}_horizontal_radius_site']")
        self.bottom_hor_rad = string_to_array(hor_site.get("pos"))

        if 'lego' in self.env_cfg.block_bottom.block_name:
            rot_z = Rotation.from_euler('z', -90, degrees=True)  # 90 degrees about Z-axis
            rot_x = Rotation.from_euler('x', -90, degrees=True)  # 90 degrees about X-axis
            relquat = (rot_x*rot_z).as_quat(scalar_first=True)
            suction_site = new_site(
                name=f"suction_site", 
                pos=(-self.bottom_hor_rad[0]+0.025, -self.block_bottom_z+0.01, 0), 
                quat=relquat,
                size=(0.01, 0.01, 0.01), 
                rgba=(1, 0, 0, 1)
            )
        else:
            rot_z = Rotation.from_euler('z', 0, degrees=True)  # 90 degrees about Z-axis
            rot_x = Rotation.from_euler('x', -90, degrees=True)  # 90 degrees about X-axis
            relquat = (rot_x*rot_z).as_quat(scalar_first=True)
            suction_site = new_site(
                name=f"suction_site", 
                pos=(0, -self.block_bottom_z+0.01, self.bottom_hor_rad[0]-0.025), 
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
        self.all_mujoco_objects[self.env_cfg.block_top.block_name] = block_top

        self.object_bottom_z = []
        for obj_name, obj_pos in zip(self.env_cfg.objects.names, self.env_cfg.objects.initial_poses):
            obj = get_obj_from_name(obj_name)
            obj_body = obj.get_obj()
            self.object_bottom_z.append(-obj.bottom_offset[2])
            obj_body.set('pos', f'{obj_pos[0]} {obj_pos[1]} {-obj.bottom_offset[2]}')
            world.worldbody.append(obj_body)
            world.merge_assets(obj)
            self.all_mujoco_objects[obj_name] = obj
        self.object_bottom_z = np.array(self.object_bottom_z)

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
        for i in range(self.N_dist_objects):
            self.composite_sampler.append_sampler(object_samplers[2+i])

        for obj_name, mj_obj in self.all_mujoco_objects.items():
            self.composite_sampler.add_objects_to_sampler(sampler_name=f"{obj_name}_sampler", mujoco_objects=mj_obj)

    def check_bottom_block_crowding(self, state):
        block_bottom_pos = state[:2]
        block_bottom_low, block_bottom_high = self.get_object_bounds(self.env_cfg.block_bottom.block_name)

        block_bottom_low[0] += 2*block_bottom_low[0]
        block_bottom_low[1] += block_bottom_low[1]
        block_bottom_high[1] += block_bottom_high[1]
        
        gx = -1e6
        for i, obj_name in enumerate(self.env_cfg.objects.names):
            obj_pos = state[12+i*6:12+i*6+2]
            obj_low, obj_high = self.get_object_bounds(obj_name)

            obstacle_high = block_bottom_pos[:2] + block_bottom_high[:2] - obj_low[:2] + self.env_cfg.thresh
            obstacle_low = block_bottom_pos[:2] + block_bottom_low[:2] - obj_high[:2] - self.env_cfg.thresh
            obstacle = signed_dist_fn_rectangle(
                obj_pos[:2], 
                obstacle_low, 
                obstacle_high,
                obstacle=True)
            
            gx = np.maximum(gx, obstacle)
        return gx>0
            
    def sample_initial_state(self, N):
        # samples only the blocks and object locations
        states = np.zeros((N, self.N_all_objects*self.low_dim_sizes['objects']))

        if self.env_cfg.randomize_locations:
            for idx in range(N):
                succ = False
                while not succ:
                    sample = self.composite_sampler.sample()
                    for i, obj_name in enumerate(self.all_blocks_object_names):
                        states[idx, i*6:i*6+3] = np.array(sample[obj_name][0])
                    full_state = np.concatenate([self.ee_pos.copy(), np.zeros(3), states[idx]]).reshape(1,-1) # append ee state
                    fail, _ = self.check_failure(full_state)
                    if self.env_cfg.get('reset_uncrowded', True):
                        crowded = self.check_bottom_block_crowding(states[idx])
                        succ = False if (fail[0] or crowded) else True
                    else:
                        succ = False if fail[0] else True
        else:
            states[:, 0:2] = self.env_cfg.block_bottom.initial_pos
            states[:, 6:8] = self.env_cfg.block_top.initial_pos
            states[:,2] = self.block_bottom_z
            states[:,8] = self.block_top_z
            for i in range(self.N_dist_objects):
                states[:, 12+i*6:12+i*6+2] = self.env_cfg.objects.initial_poses[i]
                states[:, 12+i*6+2] = self.object_bottom_z[i]

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
        self.failure_mode = None
        self.colliding_obj_name = None

        self.update_relevant_objs()
        if self.use_constraint_types:
            self.update_constraint_types()

        self.safety_set_top_low, self.safety_set_top_high = None, None
        if start is None:
            sample_state = self.sample_initial_state(1)[0]
        else:
            sample_state = start.copy()

        self.failure_mode = None
        self.colliding_obj_name = None
        self._init_knn_objects = None

        for i in range(self.N_all_objects):
            self.data.qpos[self.robot_dof+i*7:self.robot_dof+i*7+3] = sample_state[i*6:i*6+3]
        
        for _ in range(20):
            self.do_simulation([-1, 1], self.frame_skip)
        
        self.ee_pos_tm1 = self.ee_pos.copy()
        curr_state = self.get_current_relevant_state()

        self.safety_set_top_low = curr_state['objects_state'][1, :3] + self.env_cfg.block_top.safety_set.low
        self.safety_set_top_high = curr_state['objects_state'][1, :3] + self.env_cfg.block_top.safety_set.high

        if self.env_cfg.block_bottom.target_set_type == 'relative':
            self.target_set_low = curr_state['objects_state'][0, :3] + self.env_cfg.block_bottom.target_set.low
            self.target_set_high = curr_state['objects_state'][0, :3] + self.env_cfg.block_bottom.target_set.high
        elif self.env_cfg.block_bottom.target_set_type == 'absolute':
            self.target_set_low = self.env_cfg.block_bottom.target_set.low
            self.target_set_high = self.env_cfg.block_bottom.target_set.high
        else:
            raise NotImplementedError('Target set type not implemented.')

        if self.env_cfg.reset_grasped:
            while not self.suction_gripper_active:
                target_pos = self.get_suction_target()
                action = target_pos - self.ee_pos
                self.step(action)
            self._current_timestep = 0

        self.init_obj_pos = {}
        for obj_name in self.env_cfg.objects.names:
            self.init_obj_pos[obj_name] = self._get_body_pos(obj_name).copy()
        
        return self.get_current_relevant_state()

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
        
        # if 'reward' in self.return_type:
        #     cost[success] = -1.*self.reward
        #     cost[fail] = -1.*self.penalty
        # else:
        #     cost[success] = self.reward
        #     cost[fail] = self.penalty

        if self.env_cfg.get('shape_reward', False):
            if 'reward' in self.return_type:
                cost[success] = -1.*self.reward
                cost[fail] = -1.*self.penalty

                l_x[success] = -1.*self.reward
                g_x[fail] = -1.*self.penalty
            else:
                cost[success] = self.reward
                cost[fail] = self.penalty

                l_x[success] = self.reward
                g_x[fail] = self.penalty

        return cost, l_x, g_x
      
    def check_within_env(self, state=None):
        """Checks if the robot is still in the environment.

        Args:
            state (np.ndarray): the state of the agent. shape = (batch, n)

        Returns:
            bool: True if the agent is not in the environment.
        """
        if state is None:
            _ee_pos = self.ee_pos.reshape(1,-1) # shape = (batch, 3)
            _bottom_block_pos = self.bottom_block_pos.reshape(1,-1) # shape (batch, 3)
            _top_block_pos = self.top_block_pos.reshape(1,-1) # shape (batch, 3)
        else:
            _ee_pos = state[...,:3].copy()
            _bottom_block_pos = state[...,6:9].copy()
            _top_block_pos = state[...,12:15].copy()

        # EE within table
        outsideLeft_ee = np.any((_ee_pos <= self.env_bound_low), axis=-1)
        outsideRight_ee = np.any((_ee_pos >= self.env_bound_high), axis=-1)
        outside_ee = np.logical_or(outsideLeft_ee, outsideRight_ee)

        # Bottom bottom within table
        outsideLeft_bottom = np.any((_bottom_block_pos <= self.env_bound_low), axis=-1)
        outsideRight_bottom = np.any((_bottom_block_pos >= self.env_bound_high), axis=-1)
        outside_bottom = np.logical_or(outsideLeft_bottom, outsideRight_bottom)

        # Bottom top within table
        outsideLeft_top = np.any((_top_block_pos <= self.env_bound_low), axis=-1)
        outsideRight_top = np.any((_top_block_pos >= self.env_bound_high), axis=-1)
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
            if self.failure_mode is None and self.check_within_env(state):
                self.failure_mode = 'out_of_env'
            elif success:
                self.failure_mode = 'success'
        elif self.doneType == 'real':
            real_fail = self.check_real_failure()
            done = np.logical_or(np.logical_or(real_fail, success), self.check_within_env(state))
            if self.failure_mode is None and self.check_within_env(state):
                self.failure_mode = 'out_of_env'
            elif success:
                self.failure_mode = 'success'
        else:
            raise ValueError("invalid done type!")
        return done

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

    def get_current_relevant_state(self):
        """ 
            Returns states of self.observations for the environment:
        """
        obs = {} # observation dict
        ee_vel_t = (self.ee_pos-self.ee_pos_tm1)/self.dt

        if self.constraint_type_repr == 'int':
            const_type_rep = [self.object_type_to_int_mapping['ee']]
        elif self.constraint_type_repr == 'one_hot':
            const_type_rep = np.zeros(self.max_constraint_types)
            const_type_rep[self.object_type_to_int_mapping['ee']] = 1

        ee_state = np.concatenate([self.ee_pos, ee_vel_t, const_type_rep])
        obs['robot_state'] = ee_state

        if 'objects_state' in self.observations['low_dim']:
            xt = []
            for i, body_name in enumerate([self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name]):
                if self.constraint_type_repr == 'int':
                    const_type_rep = [self.object_type_to_int_mapping[body_name]]
                elif self.constraint_type_repr == 'one_hot':
                    const_type_rep = np.zeros(self.max_constraint_types)
                    const_type_rep[self.object_type_to_int_mapping[body_name]] = 1

                body_state = np.concatenate([
                    self._get_body_pos(body_name), 
                    self._get_body_vel(body_name),
                    const_type_rep]
                )
                xt.append(body_state)

            for i, body_name in enumerate(self.rel_obj_names):
                if self.use_constraint_types:
                    if self.constraint_type_repr == 'int':
                        const_type_rep = [self.constraint_to_int_mapping[self.pred_constraint_types[body_name]]]
                    elif self.constraint_type_repr == 'one_hot':
                        const_type_rep = np.zeros(self.max_constraint_types)
                        const_type_rep[self.constraint_to_int_mapping[self.pred_constraint_types[body_name]]] = 1

                    body_state = np.concatenate([
                        self._get_body_pos(body_name), 
                        self._get_body_vel(body_name), 
                        const_type_rep])
                else:
                    body_state = np.append(self._get_body_pos(body_name), self._get_body_vel(body_name))
                xt.append(body_state)
            obs['objects_state'] = np.stack(xt, axis=0)

        if 'rgb_front_cam' in self.observations['rgb']:
            obs['rgb_front_cam'] = self.get_current_image(self.front_cam_name)
        if 'rgb_eye_in_hand_cam' in self.observations['rgb']:
            obs['rgb_eye_in_hand_cam'] = self.get_current_image(self.eye_in_hand_cam_name)
        return obs
    
    def get_current_full_state(self):
        ee_vel_t = (self.ee_pos-self.ee_pos_tm1)/self.dt
        xt = np.append(self.ee_pos, ee_vel_t)
        for i, body_name in enumerate([self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name]):
            body_state = np.append(self._get_body_pos(body_name), self._get_body_vel(body_name))
            xt = np.append(xt, body_state)

        for i, body_name in enumerate(self.env_cfg.objects.names):
            if self.use_constraint_types:
                body_state = np.concatenate([
                    self._get_body_pos(body_name), 
                    self._get_body_vel(body_name), 
                    [self.constraint_to_int_mapping[self.pred_constraint_types[body_name]]]])
            else:
                body_state = np.append(self._get_body_pos(body_name), self._get_body_vel(body_name))
            xt = np.append(xt, body_state)
        return xt
    
    def check_contact_slide(self):
        for contact in self.data.contact[: self.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            # g1, g2 = self.model.geom_id2name(contact.geom1), self.model.geom_id2name(contact.geom2)

            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if (g1 is None) or (g2 is None):
                continue

            if (g1 in self.gripper.contact_geoms) and (self.env_cfg.block_bottom.block_name in g2):
                return True

            if (g2 in self.gripper.contact_geoms) and (self.env_cfg.block_bottom.block_name in g1):
                return True
            
        return False

    def check_suction_grasp(self):
        grasp_object_idx = 0
        if self.check_contact_slide() and not self.suction_gripper_active:
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
        xt = self.get_current_relevant_state()
        self.ee_pos_tm1 = self.ee_pos.copy()

        fail, g_x = self.check_failure()
        success, l_x = self.check_success()

        done = self.get_done(None, success, fail)[0]
        cost, l_x, g_x = self.get_cost(l_x, g_x, success, fail)
        info = {"g_x": g_x[0], "l_x": l_x[0]}
        info['failure_mode'] = self.failure_mode
        
        self.set_xyz_action(action[:3])
        self.do_simulation([-1, 1]) # Gripper always closed
        self.check_suction_grasp()
        
        xtp1 = self.get_current_relevant_state()
        self._current_timestep += 1
        return xtp1, cost[0], done, info

    def render(self):
        pass
    
    def get_suction_target(self):

        # target_pos = self.bottom_block_pos.copy()
        # target_pos[0] = target_pos[0] - self.bottom_hor_rad[0] + 0.03

        target_pos = self._get_site_pos('suction_site')

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'placeSiteB')
        self.model.site_pos[site_id] = target_pos             
        return target_pos

    def get_action(self):
        action = np.zeros(3)
        if self.suction_gripper_active:
            action[:3] = [-0.4, 0, 0.1]
            # target_pos = [0., 0.3, 0.1]
            # action[:3] = target_pos - self.ee_pos
        else:
            target_pos = self.get_suction_target()
            action[:3] = target_pos - self.ee_pos
            # viewer.launch(self.model, self.data)

        return action

    def get_object_bounds(self, obj_name):
        hor_site = self.all_mujoco_objects[obj_name].worldbody.find(f"./body/site[@name='{obj_name}_horizontal_radius_site']")
        obj_hor_rad = string_to_array(hor_site.get("pos"))
        obj_hor_rad[2] = 0

        obj_low = -1*obj_hor_rad
        obj_high = 1*obj_hor_rad

        bottom_site = self.all_mujoco_objects[obj_name].worldbody.find(f"./body/site[@name='{obj_name}_bottom_site']")
        obj_low[2] = string_to_array(bottom_site.get("pos"))[2]

        top_site = self.all_mujoco_objects[obj_name].worldbody.find(f"./body/site[@name='{obj_name}_top_site']")
        obj_high[2] = string_to_array(top_site.get("pos"))[2]

        return obj_low, obj_high
        
    def collision_distance(self, obj1_name, obj1_state, obj2_name, obj2_state):
        # g(x)>0 is obstacle

        obj1_low, obj1_high = self.get_object_bounds(obj1_name)
        obj2_low, obj2_high = self.get_object_bounds(obj2_name)

        obstacle_high = obj1_state[...,:3] + obj1_high - obj2_low + self.env_cfg.thresh
        obstacle_low = obj1_state[...,:3] + obj1_low - obj2_high - self.env_cfg.thresh
        obstacle = signed_dist_fn_rectangle(
            obj2_state[...,:3], 
            obstacle_low, 
            obstacle_high,
            obstacle=True)
        # print(f'{obstacle=}')
        return obstacle

    def constraint_distance(
        self, 
        obj1_name, 
        obj1_pos,
        obj1_vel, 
        obj2_name, 
        obj2_pos,
        obj2_vel,
    ):
        # g(x)>0 is obstacle
                
        obstacle = self.collision_distance(obj1_name, obj1_pos, obj2_name, obj2_pos)

        if 'no_contact' in self.obj_to_constraint_map[obj2_name]:
            if obstacle[0]>0 and self.failure_mode is None:
                self.failure_mode = 'collision_no_contact'
                self.colliding_obj_name = obj2_name
        elif 'any_contact' in self.obj_to_constraint_map[obj2_name]:
            obstacle = -1e6*np.ones(obstacle.shape)
        elif 'soft_contact' in self.obj_to_constraint_map[obj2_name]:
            velocity_const =  np.linalg.norm(obj1_vel-obj2_vel, axis=-1) - self.env_cfg.vel_thresh
            obstacle = velocity_const*(obstacle>0) + -1e6*np.ones(obstacle.shape)*(obstacle<0)
            if obstacle[0]>0 and self.failure_mode is None:
                self.failure_mode = 'collision_soft_contact'
                self.colliding_obj_name = obj2_name
        else:
            raise NotImplementedError('Contraint type not implemented!')

        return obstacle
        

    def safety_margin(self, s=None, safety_set_top_low=None, safety_set_top_high=None):
        # Input:s top block position, shape (batch, 3)
        # g(x)>0 is obstacle

        if s is None:
            _ee_pos = self.ee_pos.reshape(1,-1)
            _bottom_block_pos = self.bottom_block_pos.reshape(1,-1) # shape (batch, 3)
            _bottom_block_vel = self._get_body_vel(self.env_cfg.block_bottom.block_name).reshape(1,-1) # shape (batch, 3)
            _top_block_pos = self.top_block_pos.reshape(1,-1) # shape (batch, 3)
        else:
            _ee_pos = s[...,:3].copy()
            _bottom_block_pos = s[...,6:9].copy()
            _bottom_block_vel = s[...,9:12].copy()
            _top_block_pos = s[...,12:15].copy()
            
        if safety_set_top_low is None:
            safety_set_top_low = _top_block_pos + self.env_cfg.block_top.safety_set.low

        if safety_set_top_high is None:
            safety_set_top_high = _top_block_pos + self.env_cfg.block_top.safety_set.high

        # top block safety: stay within safety bounds
        gx = signed_dist_fn_rectangle(
            _top_block_pos, 
            safety_set_top_low, 
            safety_set_top_high)

        if gx[0]>0:
            self.failure_mode = 'top_block_oob'

        if not self.env_cfg.reset_grasped:
            # gripper should not hit top block
            obj_low, obj_high = self.get_object_bounds(self.env_cfg.block_top.block_name)
            obstacle_high = _top_block_pos + obj_high + self.env_cfg.thresh
            obstacle_low = _top_block_pos + obj_low - self.env_cfg.thresh
            obstacle = signed_dist_fn_rectangle(
                _ee_pos, 
                obstacle_low, 
                obstacle_high,
                obstacle=True)
            gx = np.maximum(gx, obstacle)

        # obstacle avoidance between bottom block and distractors
        for i, obj_name in enumerate(self.env_cfg.objects.names):
            _obj_pos = self._get_body_pos(obj_name) if s is None else s[...,18+i*6:18+i*6+3]
            _obj_vel = self._get_body_vel(obj_name) if s is None else s[...,18+i*6+3:18+i*6+6]
            obstacle = self.constraint_distance(
                self.env_cfg.block_bottom.block_name, 
                _bottom_block_pos, 
                _bottom_block_vel,
                obj_name, 
                _obj_pos, 
                _obj_vel,)
            
            gx = np.maximum(gx, obstacle)
        
        gx = self.scaling_safety * gx

        if 'reward' in self.return_type: # g(x)<0 is obstacle
            gx = -1.*gx
        return gx

    def target_margin(self, s=None, target_set_low=None, target_set_high=None):
        """Computes the margin (e.g. distance) between the state and the target set.

        Args:
            s (np.ndarray): the state of the agent. shape (batch, n)

        Returns:
            float: negative numbers indicate reaching the target. If the target set
                is not specified, return None.
        """
        # l(x)<0 is target

        if s is None:
            _ee_pos = self.ee_pos.reshape(1,-1)
            _bottom_block_pos = self.bottom_block_pos.reshape(1,-1) # shape (batch, 3)
            _top_block_pos = self.top_block_pos.reshape(1,-1) # shape (batch, 3)
        else:
            _ee_pos = s[...,:3].copy()
            _bottom_block_pos = s[...,6:9].copy()
            _top_block_pos = s[...,12:15].copy()

        if target_set_low is None:
            if self.env_cfg.block_bottom.target_set_type == 'relative':
                target_set_low = _bottom_block_pos + self.env_cfg.block_bottom.target_set.low
            elif self.env_cfg.block_bottom.target_set_type == 'absolute':
                target_set_low = self.env_cfg.block_bottom.target_set.low
            else:
                raise NotImplementedError('Target set type not implemented.')
        if target_set_high is None:
            if self.env_cfg.block_bottom.target_set_type == 'relative':
                target_set_high = _bottom_block_pos + self.env_cfg.block_bottom.target_set.high
            elif self.env_cfg.block_bottom.target_set_type == 'absolute':
                target_set_high = self.env_cfg.block_bottom.target_set.high
            else:
                raise NotImplementedError('Target set type not implemented.')

        # block bottom goal
        lx = signed_dist_fn_rectangle(
            _bottom_block_pos, 
            target_set_low, 
            target_set_high)

        # Goto suction target
        if not self.env_cfg.reset_grasped:
            _target_pos = self.get_suction_target()
            suction_lx = np.linalg.norm(_ee_pos-_target_pos, axis=-1) - self.env_cfg.thresh
            lx = np.maximum(suction_lx, lx)

        lx = self.scaling_target * lx
        if 'reward' in self.return_type: # l(x)>0 is target
            lx = -1.*lx

        return lx

    def check_failure(self, state=None):
        g_x = self.safety_margin(state, self.safety_set_top_low, self.safety_set_top_high)
        if 'reward' in self.return_type: 
            return g_x<0, g_x
        else:
            return g_x>0, g_x # g(x)>0 is failure
    
    def check_contact(self, obj1_name, obj2_name):
        for contact in self.data.contact[: self.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            # g1, g2 = self.model.geom_id2name(contact.geom1), self.model.geom_id2name(contact.geom2)

            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if (g1 is None) or (g2 is None):
                continue

            if (obj2_name in g1) and (obj1_name in g2):
                return True

            if (obj1_name in g1) and (obj2_name in g2):
                return True
            
        return False
    
    def is_toppled(self, obj_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{obj_name}_main')
        quat_body = self.data.xquat[body_id]

        xyz = Rotation.from_quat(quat_body, scalar_first=True).as_euler('xyz', degrees=True)
        del_z = np.abs(xyz[2])
        shortest_angle = np.minimum(del_z, 180-del_z)
        print(shortest_angle)

        # obj_pos = self._get_body_pos(obj_name)
        # toppled = np.abs(self.init_obj_pos[obj_name][2]-obj_pos[2]) > self.env_cfg.toppl_thresh

        toppled = shortest_angle > self.env_cfg.get('toppl_thresh', 15)
        return toppled

    def check_real_failure(self):

        # top block safety: stay within safety bounds
        gx = signed_dist_fn_rectangle(
            self.top_block_pos.reshape(1,-1), 
            self.safety_set_top_low, 
            self.safety_set_top_high)

        if gx[0]>0:
            self.failure_mode = 'top_block_oob'
            return True
        else:
            # Check for collision
            for i, obj_name in enumerate(self.env_cfg.objects.names):
                in_collision = self.check_contact(self.env_cfg.block_bottom.block_name, obj_name)

                if 'no_contact' in self.obj_to_constraint_map[obj_name]:
                    if in_collision:
                        self.failure_mode = 'collision_no_contact'
                        self.colliding_obj_name = obj_name
                        return True
                elif 'any_contact' in self.obj_to_constraint_map[obj_name]:
                    continue
                elif 'soft_contact' in self.obj_to_constraint_map[obj_name]:
                    if in_collision and self.is_toppled(obj_name):
                        self.failure_mode = 'collision_soft_contact'
                        self.colliding_obj_name = obj_name
                        return True
                else:
                    raise NotImplementedError('Contraint type not implemented!')
        return False

    def check_success(self, state=None):
        l_x = self.target_margin(state, self.target_set_low, self.target_set_high)
        if 'reward' in self.return_type: 
            return l_x>0, l_x
        else:
            return l_x<0, l_x # l(x)<0 is target
    
    def get_sorted_objects(self):
        dist = np.zeros(self.N_dist_objects)
        for i, body_name in enumerate(self.env_cfg.objects.names):
            dist[i] = np.linalg.norm(self._get_body_pos(body_name) - self.bottom_block_pos)
        sorted_indices = np.argsort(dist)
        rel_objs = [self.env_cfg.objects.names[int(sorted_indices[j])] for j in range(len(sorted_indices))]
        return rel_objs
    
    def get_rel_objects(self, mode):
        if 'dynamic_knn' in mode.lower():
            rel_objs = self.get_sorted_objects()[:self.env_cfg.n_rel_objs]
        elif 'static_knn' in mode.lower():
            if self._init_knn_objects is None:
                rel_objs = self.get_sorted_objects()[:self.env_cfg.n_rel_objs]
                self._init_knn_objects = rel_objs.copy()
            else:
                rel_objs = self._init_knn_objects.copy()
        elif 'none' in mode.lower():
            return [self.env_cfg.objects.names[i] for i in range(self.env_cfg.n_rel_objs)]
        elif 'gt' in mode.lower():
            # Choose the closest fragile objects
            rel_objs = [obj for obj in self.get_sorted_objects() if 'mug' in obj][:self.env_cfg.n_rel_objs]
        else:
            raise NotImplementedError(f"Relevant object detector: {mode} not implemented.")
        
        return rel_objs

    def plot_trajectory(self, state, action, save_filename):
        """ 
            state shape = (T+1, self.n)
            action shape = (T, self.m)
        """

        _rows = 4 if self.use_constraint_types else 3
        fig, axes = plt.subplots(_rows, 3, figsize=(16, 16))

        # plot position
        axes[0,0].plot(state[:,0], label='EE x')
        axes[0,0].plot(state[:,6], label='Bottom block x')
        axes[0,0].plot(state[:,12], label='Top block x')

        axes[0,1].plot(state[:,1], label='EE y')
        axes[0,1].plot(state[:,7], label='Bottom block y')
        axes[0,1].plot(state[:,13], label='Top block y')

        axes[0,2].plot(state[:,2], label='EE z')
        axes[0,2].plot(state[:,8], label='Bottom block z')
        axes[0,2].plot(state[:,14], label='Top block z')

        # plot velocity
        axes[1,0].plot(state[:,3], label='EE x vel')
        axes[1,0].plot(state[:,9], label='Bottom block x vel')
        axes[1,0].plot(state[:,15], label='Top block x vel')

        axes[1,1].plot(state[:,4], label='EE y vel')
        axes[1,1].plot(state[:,10], label='Bottom block y vel')
        axes[1,1].plot(state[:,16], label='Top block y vel')

        axes[1,2].plot(state[:,5], label='EE z vel')
        axes[1,2].plot(state[:,11], label='Bottom block z vel')
        axes[1,2].plot(state[:,17], label='Top block z vel')

        for i in range(self.env_cfg.n_rel_objs):
            axes[0,0].plot(state[:,18+self.low_dim_sizes['objects']*i], label=f'{self.rel_obj_names[i]} x')
            axes[0,1].plot(state[:,18+self.low_dim_sizes['objects']*i+1], label=f'{self.rel_obj_names[i]} y')
            axes[0,2].plot(state[:,18+self.low_dim_sizes['objects']*i+2], label=f'{self.rel_obj_names[i]} z')

            axes[1,0].plot(state[:,18+self.low_dim_sizes['objects']*i+3], label=f'{self.rel_obj_names[i]} x vel')
            axes[1,1].plot(state[:,18+self.low_dim_sizes['objects']*i+4], label=f'{self.rel_obj_names[i]} y vel')
            axes[1,2].plot(state[:,18+self.low_dim_sizes['objects']*i+5], label=f'{self.rel_obj_names[i]} z vel')

            if self.use_constraint_types:
                axes[3,0].plot(state[:,18+self.low_dim_sizes['objects']*i+6], label=f'{self.rel_obj_names[i]} x')

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

        axes[0,0].set_xlabel(f't')
        axes[0,0].set_ylabel(f'x')
        axes[0,0].set_title(f'x pos')
        axes[0,0].legend()
        
        axes[0,1].set_xlabel(f't')
        axes[0,1].set_ylabel(f'y')
        axes[0,1].set_title(f'y pos')
        axes[0,1].legend()

        axes[0,2].set_xlabel(f't')
        axes[0,2].set_ylabel(f'z')
        axes[0,2].set_title(f'z pos')
        axes[0,2].legend()

        axes[1,0].set_xlabel(f't')
        axes[1,0].set_ylabel(f'xdot')
        axes[1,0].set_title(f'x vel')
        axes[1,0].legend()

        axes[1,1].set_xlabel(f't')
        axes[1,1].set_ylabel(f'ydot')
        axes[1,1].set_title(f'y vel')
        axes[1,1].legend()

        axes[1,2].set_xlabel(f't')
        axes[1,2].set_ylabel(f'zdot')
        axes[1,2].set_title(f'z vel')
        axes[1,2].legend()
        if self.use_constraint_types:
            axes[3,0].set_xlabel(f't')
            axes[3,0].set_ylabel(f'constraint type')
            axes[3,0].set_title(f'constraint type')
            axes[3,0].legend()

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
    import imageio
    from PIL import Image

    out_folder = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/slide_pickup_clutter/'
    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')

    env_name = "slide_pickup_clutter_mujoco_multimodal_env-v0"
    env = gym.make(env_name, device=0, cfg=env_cfg[env_name])

    save_gif = True
    num_episodes = 10
    max_ep_len = 200
    down_action = np.array([0.0,0,-0.3,0])
    up_action = np.array([0.3,0,0.3,0])

    for i in range(num_episodes):
        print(f'episode:{i}')
        xt_all, at_all, imgs = [], [], []
        xt = env.reset()
        import ipdb; ipdb.set_trace()
        xt_all.append(xt)
        
        done = False
        ep_len = 0
        action = down_action.copy()
        while not (done or ep_len == max_ep_len):
        # while not (ep_len == max_ep_len):
            at = env.action_space.sample()
            at_all.append(at)
            xt, _, done, _ = env.step(env.get_action())

            xt_all.append(xt)

            env.renderer.update_scene(env.data, camera=env.front_cam_name)
            img = env.renderer.render()
            # Image.fromarray(img).save(out_folder + f"cereal_t_{ep_len}_{i}_front.png")
            imgs.append(img)

            env.renderer.update_scene(env.data, camera=env.side_cam_name) 
            img = env.renderer.render()
            
            ep_len += 1
        print(f'ep_len: {ep_len}')
        if save_gif:
            file_name = f'test_slide_pickup_clutter_all_up_{i}_{env.failure_mode}_{env.colliding_obj_name}'
            imageio.mimsave(out_folder+f'{file_name}.gif', imgs[::4], duration=100)
            env.plot_trajectory(np.stack(xt_all,axis=0), np.stack(at_all,axis=0), os.path.join(out_folder, f'{file_name}.png'))