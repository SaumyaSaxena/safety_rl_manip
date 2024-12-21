import safety_rl_manip
import gym
import numpy as np
import time
from omegaconf import OmegaConf
# from moviepy.editor import ImageSequenceClip
import mujoco
from PIL import Image
import hydra_python as hydra
from hydra_python.utils import initialize_hydra_pipeline_mujoco
from hydra_python.stretch_ai_utils.utils import write_config_yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
from hydra_python import RRLogger
from hydra_python.utils import hydra_get_mesh
import rerun as rr
import torch

import numpy as np
from scipy.spatial.transform import Rotation as R

def render_camera_image(camera_id, cam_pos, cam_quat_wxyz, env, rr_logger, sg_sim, idx, output_path):
    rr_logger.log_clear("world/hydra_graph")
    rr_logger.log_clear("/world/annotations/bb")
    
    env.renderer.update_scene(env.data, camera=camera_id) 
    env.renderer.disable_depth_rendering()
    env.renderer.disable_segmentation_rendering()
    img_side = env.renderer.render()

    env.renderer.enable_depth_rendering()
    depth_side = env.renderer.render().astype(np.float32)
    depth_norm = (depth_side - depth_side.min()) / (depth_side.max() - depth_side.min()) * 255
    env.renderer.enable_segmentation_rendering()
    seg_side = env.renderer.render()
    seg_body_id_side = env.model.geom_bodyid[seg_side[:,:,0]]

    image = Image.fromarray(img_side)
    image.save(output_path / f"clutter_rgb_{idx}_{camera_id}.png")
    Image.fromarray(depth_norm.astype(np.uint8)).save(output_path / f"clutter_depth_{idx}_{camera_id}.png")
    Image.fromarray(seg_side[:,:,0].astype(np.uint8)).save(output_path / f"clutter_segid_{idx}_{camera_id}.png")
    Image.fromarray(seg_side[:,:,1].astype(np.uint8)).save(output_path / f"clutter_segtype_{idx}_{camera_id}.png")
    
    for _ in range(2):
        pipeline.step(idx, cam_pos, cam_quat_wxyz, depth_side, seg_body_id_side.astype(np.int32), img_side.astype(np.uint8))
    mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)
    rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
    rr_logger.log_camera_tf(cam_pos, cam_quat_wxyz, cam_entity=camera_id)
    sg_sim.update()
    rr_logger.step()
    return img_side, seg_body_id_side, depth_side

if __name__ == "__main__":

    hydra_cfg = OmegaConf.load('/home/saumyas/catkin_ws_semnav/src/hydra/python/src/hydra_python/commands/cfg/vlm_eqa_strange_mujoco.yaml')
    OmegaConf.resolve(hydra_cfg)

    output_path = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/test3/'
    output_path = hydra.resolve_output_path(output_path)

    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')
    env_name = "slide_pickup_clutter_mujoco_env-v0"
    env = gym.make(env_name, device=0, cfg=env_cfg[env_name])

    # env.plot_env('/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media')

    sensor_categories_mapping = {k: v for k, v in zip(env.all_body_ids, env.all_body_names)}
    write_config_yaml(sensor_categories_mapping, out_path=Path('/home/saumyas/catkin_ws_semnav/src/hydra/config/label_spaces/mujoco_label_space.yaml'))
    
    save_gif = True
    num_episodes = 10
    max_ep_len = 300

    height, width = env.renderer._height, env.renderer._width
    focal_length = width / (2.0 * np.tan(float(env.model.cam_fovy[env.side_camera_id]) * np.pi / 360.0))
    camera_info = {
        "fx": float(focal_length),
        "fy": float(focal_length),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "width": int(width),
        "height": int(height),
    }
        
    pipeline = initialize_hydra_pipeline_mujoco(hydra_cfg.hydra, camera_info, env.all_body_names, output_path)
    rr_logger = RRLogger(output_path)

    device = f"cuda:{hydra_cfg.gpu}" if torch.cuda.is_available() else "cpu"
    sg_sim = hydra.SceneGraphSim(
        hydra_cfg, 
        output_path, 
        pipeline, 
        rr_logger, 
        device=device)

    for i in range(num_episodes):
        xt_all, at_all, imgs = [], [], []
        xt = env.reset()
        xt_all.append(xt)
        
        done = False
        ep_len = 0
        while not (done or ep_len == max_ep_len):
            start = time.time()
            at = env.action_space.sample()
            at_all.append(at)
            xt, _, done, _ = env.step(at)
            xt_all.append(xt)

            img_side, seg_side, depth_side = render_camera_image(env.side_cam_name, env.side_cam_pos, env.side_cam_quat_wxyz, env, rr_logger, sg_sim, ep_len, output_path)
            ep_len += 1
            img_front, seg_front, depth_front = render_camera_image(env.front_cam_name, env.front_cam_pos, env.front_cam_quat_wxyz, env, rr_logger, sg_sim, ep_len, output_path)
            ep_len += 1
            img_eye, seg_eye, depth_eye = render_camera_image(env.eye_in_hand_cam_name, env.eye_in_hand_cam_pos, env.eye_in_hand_cam_quat_wxyz, env, rr_logger, sg_sim, ep_len, output_path)

            rr_logger.log_img_data(
                np.concatenate([img_front, img_side, img_eye], axis=1), 
                np.concatenate([seg_front, seg_side, seg_eye], axis=1))
            rr.log(f"{rr_logger.primary_camera_entity}/depth", rr.DepthImage(np.concatenate([depth_front, depth_side, depth_eye], axis=1), meter=1.0))
            

            print(f"Time elapsed: {time.time()-start}")
            import ipdb; ipdb.set_trace()
            
            
        if save_gif:
            filename = f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_slideup_obstacles_{i}.gif'
            cl = ImageSequenceClip(imgs[::4], fps=500)
            cl.write_gif(filename, fps=500)
            env.plot_trajectory(np.stack(xt_all,axis=0), np.stack(at_all,axis=0), f'/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/env_test_slideup_obstacles_{i}.png')

