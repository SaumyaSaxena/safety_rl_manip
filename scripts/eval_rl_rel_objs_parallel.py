import os
import numpy as np
import torch
from pathlib import Path
import logging
import wandb
import time, json
from omegaconf import OmegaConf
import imageio, cv2
from PIL import Image
from tqdm import trange
import copy

from safety_rl_manip.models.RARL.utils import set_seed

from safety_rl_manip.models.RARL.DDPG import DDPG
from safety_rl_manip.models.RARL.SAC import SAC
from safety_rl_manip.models.vlm_semantic_safety import SafePlanner
from safety_rl_manip.models.vlm_semantic_safety_constraint_type import VLMSafetyCriteria
from safety_rl_manip.utils.logging import log_experiment_status
from scipy.spatial.transform import Rotation
from safety_rl_manip.envs.utils import draw_bounding_boxes_cv2

logger = logging.getLogger(__name__)

def world_to_image(world_coords, K, R, T):
    """
    Convert world coordinates (XYZ) to image coordinates (u, v) given camera parameters.
    Camera points along +z-axis
    
    Parameters:
    - world_coords: numpy array of shape (N, 3), world coordinates (X, Y, Z).
    - K: Camera intrinsic matrix (3x3).
    - R: Camera rotation matrix (3x3).
    - T: Camera translation vector (3x1).
    
    Returns:
    - image_coords: numpy array of shape (N, 2), image coordinates (u, v).
    """
    # Convert world coordinates to camera coordinates
    camera_coords = R.T @ (world_coords.T - T.reshape(-1, 1))
    
    # Project onto the image plane
    image_coords_homogeneous = np.dot(K, camera_coords)  # shape (3, N)
    
    # Normalize to get (u, v)
    image_coords = image_coords_homogeneous[:2, :] / image_coords_homogeneous[2, :]
    return image_coords.T  # Shape (N, 2)

def plot_traj_on_image(image, trajectory, camera_info):
    """
    Plot the trajectory (in world coordinates) on the image, including an arrow at the end.
    
    Parameters:
    - image: The 2D image (numpy array).
    - trajectory: A numpy array of shape (N, 3), world coordinates (X, Y, Z).
    - K: Camera intrinsic matrix (3x3).
    - R: Camera rotation matrix (3x3).
    - T: Camera translation vector (3x1).
    """
    cam_rot = Rotation.from_quat(camera_info['front_cam_quat_wxyz'], scalar_first=True).as_matrix()
    # Convert world trajectory to image coordinates
    image_coords = world_to_image(trajectory, camera_info['K'], cam_rot, camera_info['front_cam_pos'])
    
    # Convert image coordinates to integer values for plotting
    image_coords_int = np.round(image_coords).astype(int)
    
    # Plot the trajectory by connecting the points
    for i in range(1, len(image_coords_int)):
        cv2.line(image, tuple(image_coords_int[i-1]), tuple(image_coords_int[i]), (0, 0, 0), 3)
    
    if len(image_coords_int)>1:
        # Add an arrow at the last point
        start_point = tuple(image_coords_int[-2])
        end_point = tuple(image_coords_int[-1])
        cv2.arrowedLine(image, start_point, end_point, (0, 0, 0), 3, tipLength=1)

    return image

def label_and_append_images(imgs, ee_traj, camera_info):
    imgs_labelled = []
    for i, im in enumerate(imgs):
        color_img = im.copy()
        cv2.putText(color_img, str(f"Image {i+1}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        color_img = plot_traj_on_image(color_img, ee_traj[i], camera_info)
        imgs_labelled.append(color_img)
    return np.concatenate(imgs_labelled, axis=1)


def main():
    cfg = OmegaConf.load('cfg/eval/eval_rl_rel_objs_parallel.yaml')
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    eval_path = Path('outputs/evals')
    model_path = os.path.join(eval_path, 'ckpts')

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")

    eval_cfg = cfg.eval_cfg
    vlm_cfg = cfg.vlm_cfg
    device = (
        torch.device("cuda", eval_cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    # Load checkpoint
    ckpt_file = wandb.restore(
        str(eval_cfg.wandb_load.file), 
        run_path=eval_cfg.wandb_load.run_path,
        root=str(model_path), replace=True,
    )
    ckpt = torch.load(ckpt_file.name, map_location=device)
    env_name = ckpt['env_name']
    env_cfg = ckpt['env_cfg']
    eval_env_cfg = cfg.env_cfg

    OmegaConf.set_struct(env_cfg, False) # to allow merging new fields into the config
    updated_env_cfg = OmegaConf.merge(env_cfg, eval_env_cfg)
    OmegaConf.set_struct(env_cfg, True)

    mode = ckpt['train_cfg']['mode']
    algo_name = ckpt['train_cfg']['algo_name']

    if updated_env_cfg['use_constraint_types']:
        _updaters_name = '_'.join(f"{a}_{b}" for a, b in zip(eval_cfg['relevant_obj_updaters'], eval_cfg['constraint_type_updaters']))
        time_str = time_str + '_' + ckpt['train_cfg']['run_variant'] + '_' + _updaters_name + '_' + f'{cfg.suffix}'
    else:
        time_str = time_str + '_' + ckpt['train_cfg']['run_variant'] + '_' + str(f"{'_'.join(eval_cfg['relevant_obj_updaters'])}") + '_' + f'{cfg.suffix}'
    
    eval_path = os.path.join(eval_path, f'{env_name}_{algo_name}_{mode}', time_str)
    agent = eval(algo_name)(
        env_name, device, train_cfg=ckpt['train_cfg'], eval_cfg=eval_cfg,
        env_cfg=updated_env_cfg,
        outFolder=eval_path, debug=cfg.debug
    )
    agent.load_state_dict(ckpt["state_dict"])

    if eval_cfg.check_feasibility:
        # Load checkpoint for checking feasibility
        ckpt_file_feas = wandb.restore(
            str(eval_cfg.wandb_load.feasibility_file), 
            run_path=eval_cfg.wandb_load.feasibility_run_path,
            root=str(model_path), replace=True,
        )
        ckpt_feas = torch.load(ckpt_file_feas.name, map_location=device)
        agent_feas = eval(algo_name)(
            env_name, device, train_cfg=ckpt_feas['train_cfg'], eval_cfg=eval_cfg,
            env_cfg=ckpt_feas['env_cfg'],
            outFolder=eval_path, debug=cfg.debug
        )
        agent_feas.load_state_dict(ckpt_feas["state_dict"])

    set_seed(eval_cfg['seed'])

    if 'vlm' in eval_cfg.relevant_obj_updaters or 'vlm' in eval_cfg.constraint_type_updaters:
        vlm_safety = SafePlanner(vlm_cfg.vlm_type, updated_env_cfg['n_rel_objs'])

    results_filenames = [os.path.join(
        agent.outFolder, 
        f'results_rollouts_{a}_{b}_f{c}.json') 
        for a, b, c in zip(
            eval_cfg['relevant_obj_updaters'], 
            eval_cfg['constraint_type_updaters'], 
            vlm_cfg.rel_obj_update_freq)]

    n_evals = len(eval_cfg['relevant_obj_updaters'])
    failure_dict = {k:0 for k in agent.test_env.all_failure_modes+['hit_irrelevant_obj', 'timeout']}
    failure_analysis_dicts = [copy.deepcopy(failure_dict) for _ in range(n_evals)]
    FP, FN, TP, TN, num_pred_success, num_gt_success, total_episode_len = [0]*n_evals, [0]*n_evals, [0]*n_evals, [0]*n_evals, [0]*n_evals, [0]*n_evals, [0]*n_evals
    for i in trange(eval_cfg.num_visualization_rollouts, desc="Relevant obj testing"):
        
        result_folder = os.path.join(agent.figureFolder, f"{i}")
        os.makedirs(result_folder, exist_ok=True)

        sample_state = agent.test_env.sample_initial_state(1)[0]
        for j in range(n_evals):
            _rel_obj_phrase = f'{eval_cfg.relevant_obj_updaters[j]}_{eval_cfg.constraint_type_updaters[j]}_f{vlm_cfg.rel_obj_update_freq[j]}'
            imgs, imgs_obj_label, imgs_bb, bb_names, ee_pos = [], [], [], [], []
            o, d = agent.test_env.reset(start=sample_state), False

            if eval_cfg.check_feasibility:
                # Check if full state is feasible        
                o_full = agent.test_env.get_current_full_state()
                pred_v = agent_feas.ac_targ.q(
                    torch.from_numpy(o_full).float().to(agent.device), 
                    agent_feas.ac_targ.pi(torch.from_numpy(o_full).float().to(agent.device))).detach().cpu().numpy()
                pred_success = pred_v > 0.
                num_pred_success[j] += pred_success

            if (
                eval_cfg.save_media or
                ('vlm' in eval_cfg.relevant_obj_updaters[j]) or
                ('vlm' in eval_cfg.constraint_type_updaters[j] and eval_cfg.use_image_for_vlm_contraint_type)
            ):
                # imgs.append(agent.test_env.get_current_image(agent.test_env.front_cam_name))
                _rgb_bb, bboxes_with_names = agent.test_env.get_img_with_bb(agent.test_env.front_cam_name)
                imgs_bb.append(_rgb_bb)
                bb_names.append(bboxes_with_names)
            
            ee_pos.append(agent.test_env.ee_pos.copy())
            metrics = {}
            # Update constraint type
            if updated_env_cfg['use_constraint_types']:
                if 'conser' in eval_cfg.constraint_type_updaters[j]:
                    obj_const_types = {obj_name: 'no_contact' for obj_name in updated_env_cfg['objects']['names']}
                elif 'vlm' in eval_cfg.constraint_type_updaters[j]:
                    img_path = os.path.join(result_folder, f"current_init_img_{agent.test_env.current_timestep}.png")
                    if eval_cfg.use_image_for_vlm_contraint_type:
                        Image.fromarray(imgs_bb[-1]).save(img_path)
                    output = vlm_safety.get_constraint_types(
                        img_path, 
                        updated_env_cfg['objects']['names'], 
                        updated_env_cfg['constraint_types'],
                        use_image=eval_cfg.use_image_for_vlm_contraint_type)
                    metrics['explanation_constraint_types'] = output[0]
                    if eval_cfg.use_image_for_vlm_contraint_type:
                        metrics['image_description_const'] = output[2]
                    obj_const_types = output[1]
                elif 'gt' in eval_cfg.constraint_type_updaters[j]:
                    obj_const_types = agent.test_env.obj_to_constraint_map_gt.copy()
                agent.test_env.update_constraint_types(obj_const_types)
                metrics['constraint_types'] = obj_const_types

            while not(d or (agent.test_env.current_timestep == agent.max_ep_len)):
                if (
                    agent.test_env.current_timestep == 0 or (
                        # agent.test_env.current_timestep > vlm_cfg.subsample_freq*vlm_cfg.num_frames and
                        agent.test_env.current_timestep % vlm_cfg.rel_obj_update_freq[j] == 0 and 
                        agent.test_env.suction_gripper_active
                    )  
                ):
                    if 'vlm' in eval_cfg.relevant_obj_updaters[j]:
                        rel_imgs = imgs_bb[::-vlm_cfg.subsample_freq][::-1][-vlm_cfg.num_frames:]
                        # _t_start = -vlm_cfg.subsample_freq*(vlm_cfg.num_frames-1) # plot traj from t=-n*delta to current time step
                        _t_start = 0 # plot traj from t=0 to current time step
                        rel_ee_pos = [
                            np.array(ee_pos[_t_start:len(ee_pos)-vlm_cfg.subsample_freq*idz]) 
                            for idz in range(vlm_cfg.num_frames-1,-1,-1)
                        ]
                        rel_imgs_labelled = label_and_append_images(rel_imgs, rel_ee_pos, agent.test_env.camera_info)
                        img_path = os.path.join(result_folder, f"{_rel_obj_phrase}_{agent.test_env.current_timestep}.png")
                        Image.fromarray(rel_imgs_labelled).save(img_path)
                        
                        sorted_objs = agent.test_env.get_sorted_objects()
                        
                        objs, obj_text, img_desc = vlm_safety.get_rel_objects(img_path, sorted_objs)
                        metrics[f'timestep_{agent.test_env.current_timestep}'] = {
                            'relevant_objects': objs,
                            'Relevent_object_text': obj_text,
                            'Image_description': img_desc,
                        }
                    else:
                        objs = agent.test_env.get_rel_objects(eval_cfg.relevant_obj_updaters[j])
                        metrics[f'timestep_{agent.test_env.current_timestep}'] = {'relevant_objects': objs}

                    agent.test_env.update_relevant_objs(objs)
                    o = agent.test_env.get_current_relevant_state()
                    
                if eval_cfg.save_media:
                    img_obj = imgs_bb[-1].copy()
                    cv2.putText(img_obj, str(f"{' '.join(objs)}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # Put green BB around selected objects
                    _bbs = {idx: bb_names[-1][_o] for idx, _o in enumerate(objs) if _o in bb_names[-1].keys()}
                    img_obj = draw_bounding_boxes_cv2(img_obj, _bbs, color=(0,255,0))
                    imgs_obj_label.append(img_obj)

                o, _, d, _ = agent.test_env.step(agent.get_action(o, 0))
                if eval_cfg.save_media or ('vlm' in eval_cfg.relevant_obj_updaters[j]):
                    # imgs.append(agent.test_env.get_current_image(agent.test_env.front_cam_name))
                    _rgb_bb, bboxes_with_names = agent.test_env.get_img_with_bb(agent.test_env.front_cam_name)
                    imgs_bb.append(_rgb_bb)
                    bb_names.append(bboxes_with_names)

                ee_pos.append(agent.test_env.ee_pos.copy())
            
            gt_success = agent.test_env.failure_mode=='success'
            num_gt_success[j] += gt_success
            total_episode_len[j] += agent.test_env.current_timestep 
            if d:
                failure_analysis_dicts[j][agent.test_env.failure_mode] += 1
                metrics['failure_mode'] = agent.test_env.failure_mode
                metrics['episode_len'] = agent.test_env.current_timestep
                if 'collision' in agent.test_env.failure_mode:
                    metrics['colliding_object_name'] = agent.test_env.colliding_obj_name
                    if agent.test_env.colliding_obj_name not in objs:
                        failure_analysis_dicts[j]['hit_irrelevant_obj'] += 1
                        metrics['hit_irrelevant_obj'] = True
                    else:
                        metrics['hit_irrelevant_obj'] = False
            else:
                failure_analysis_dicts[j]['timeout'] += 1
        
            if eval_cfg.check_feasibility:
                metrics['pred_success'] = bool(pred_success)
                FP[j] += np.sum(np.logical_and((gt_success == False), (pred_success == True)))
                FN[j] += np.sum(np.logical_and((gt_success == True), (pred_success == False)))
                TP[j] += np.sum(np.logical_and((gt_success == True), (pred_success == True)))
                TN[j] += np.sum(np.logical_and((gt_success == False), (pred_success == False)))

            if eval_cfg.save_media:
                filename = os.path.join(result_folder, f'{_rel_obj_phrase}_{agent.test_env.failure_mode}_{agent.test_env.colliding_obj_name}.gif')
                imageio.mimsave(filename, imgs_obj_label[::4], duration=200)
            
            log_experiment_status(experiment_id=i, success=gt_success, metrics=metrics, filename=results_filenames[j])

            # LOG SUMMARY
            success_rate = num_gt_success[j]/(i+1)
            info = {
                'Total_num_episodes': int((i+1)),
                'num_gt_success': int(num_gt_success[j]),
                'success_rate': float(success_rate),
                'failure_analysis': failure_analysis_dicts[j],
                'avg_episode_len': total_episode_len[j]/(i+1),
            }
            if eval_cfg.check_feasibility:
                false_pos_rate = FP[j]/(FP[j]+TN[j])
                false_neg_rate = FN[j]/(FN[j]+TP[j])
                info.update({
                    'False_positive_rate': float(false_pos_rate),
                    'False_negative_rate': float(false_neg_rate),
                    'FP': float(FP[j]),
                    'FN': float(FN[j]),
                    'TP': float(TP[j]),
                    'TN': float(TN[j]),
                    'num_pred_success': int(num_pred_success[j]),
                })
            log_experiment_status(experiment_id='summary', metrics=info, filename=results_filenames[j])
    
if __name__ == "__main__":
    main()