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

from safety_rl_manip.models.RARL.utils import set_seed

from safety_rl_manip.models.RARL.DDPG import DDPG
from safety_rl_manip.models.RARL.SAC import SAC
from safety_rl_manip.models.vlm_semantic_safety import SafePlanner

logger = logging.getLogger(__name__)

def label_and_append_images(imgs):
    imgs_labelled = []
    for i, im in enumerate(imgs):
        color_img = im.copy()
        cv2.putText(color_img, str(f"Image {i+1}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        imgs_labelled.append(color_img)
    return np.concatenate(imgs_labelled, axis=1)


def main():
    cfg = OmegaConf.load('cfg/eval/eval_rl_vlm.yaml')

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")

    eval_cfg = cfg.eval_cfg
    vlm_cfg = cfg.vlm_cfg
    
    device = (
        torch.device("cuda", eval_cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    eval_path = Path('outputs/evals')
    model_path = os.path.join(eval_path, 'ckpts')

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
    updated_env_cfg = OmegaConf.merge(env_cfg, eval_env_cfg)

    mode = ckpt['train_cfg']['mode']
    algo_name = ckpt['train_cfg']['algo_name']

    if 'run_variant' in ckpt['train_cfg']:
        time_str = time_str + '_' + ckpt['train_cfg']['run_variant'] + '_' + eval_cfg['relevant_obj_updater']

    eval_path = os.path.join(eval_path, f'{env_name}_{algo_name}_{mode}', time_str)

    agent = eval(algo_name)(
        env_name, device, train_cfg=ckpt['train_cfg'], eval_cfg=eval_cfg,
        env_cfg=updated_env_cfg,
        outFolder=eval_path, debug=cfg.debug
    )
    agent.load_state_dict(ckpt["state_dict"])

    set_seed(eval_cfg['seed'])

    if 'vlm' in eval_cfg.relevant_obj_updater:
        vlm_safety = SafePlanner(vlm_cfg.vlm_type, updated_env_cfg['objects']['names'])

    failure_analysis = {k:0 for k in agent.test_env.all_failure_modes}
    failure_analysis['hit_irrelevant_obj'] = 0
    FP, FN, TP, TN, num_pred_success, num_gt_success = 0, 0, 0, 0, 0, 0
    for i in trange(eval_cfg.num_visualization_rollouts, desc="Relevant obj testing"):
        
        result_folder = os.path.join(agent.figureFolder, f"{i}")
        os.makedirs(result_folder, exist_ok=True)
        
        imgs, imgs_obj_label = [], []
        o, d = agent.test_env.reset(), False

        # Check if state is feasible
        # pred_success = False
        # for obj_name in agent.test_env.env_cfg.objects.names:
        #     agent.test_env.update_relevant_objs([obj_name])
        #     o = agent.test_env.get_current_state()
        #     pred_v = agent.ac_targ.q(
        #         torch.from_numpy(o).float().to(agent.device), 
        #         agent.ac_targ.pi(torch.from_numpy(o).float().to(agent.device))).detach().cpu().numpy()
        #     pred_success = pred_v > 0. or pred_success
        # num_pred_success += pred_success

        imgs.append(agent.test_env.get_current_image(agent.test_env.front_cam_name))

        while not(d or (agent.test_env.current_timestep == agent.max_ep_len)):
            if (
                agent.test_env.current_timestep == 0 or (
                    agent.test_env.current_timestep > vlm_cfg.subsample_freq*vlm_cfg.num_frames and
                    agent.test_env.current_timestep % vlm_cfg.subsample_freq == 0 and 
                    agent.test_env.suction_gripper_active
                )  
            ):
                if 'vlm' in eval_cfg.relevant_obj_updater:
                    rel_imgs = imgs[::-vlm_cfg.subsample_freq][::-1][-vlm_cfg.num_frames:]
                    rel_imgs_labelled = label_and_append_images(rel_imgs)
                    img_path = os.path.join(result_folder, f"current_img_mug_teapot_{agent.test_env.current_timestep}.png")
                    Image.fromarray(rel_imgs_labelled).save(img_path)
                    objs = vlm_safety.get_rel_objects(img_path)
                else:
                    objs = agent.test_env.get_rel_objects(eval_cfg.relevant_obj_updater)

                agent.test_env.update_relevant_objs(objs)
                o = agent.test_env.get_current_state()

            img_obj = imgs[-1].copy()
            cv2.putText(img_obj, str(f"{' '.join(objs)}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            imgs_obj_label.append(img_obj)

            o, _, d, _ = agent.test_env.step(agent.get_action(o, 0))
            imgs.append(agent.test_env.get_current_image(agent.test_env.front_cam_name))

        gt_success = agent.test_env.failure_mode=='success'
        num_gt_success += gt_success
        if d:
            failure_analysis[agent.test_env.failure_mode] += 1
            if agent.test_env.failure_mode == 'hit_obstacle' and agent.test_env.colliding_obj_name not in objs:
                failure_analysis['hit_irrelevant_obj'] += 1
        
        FP += np.sum(np.logical_and((gt_success == False), (pred_success == True)))
        FN += np.sum(np.logical_and((gt_success == True), (pred_success == False)))
        TP += np.sum(np.logical_and((gt_success == True), (pred_success == True)))
        TN += np.sum(np.logical_and((gt_success == False), (pred_success == False)))

        filename = os.path.join(result_folder, f'test_slide_pickup_reset_grasped_{agent.test_env.failure_mode}_{agent.test_env.colliding_obj_name}.gif')
        imageio.mimsave(filename, imgs_obj_label[::4], duration=200)

    false_pos_rate = FP/(FP+TN)
    false_neg_rate = FN/(FN+TP)
    success_rate = num_gt_success/eval_cfg.num_visualization_rollouts
    info = {
        'Total_num_episodes': eval_cfg.num_visualization_rollouts,
        'False_positive_rate': false_pos_rate,
        'False_negative_rate': false_neg_rate,
        'FP': float(FP),
        'FN': float(FN),
        'TP': float(TP),
        'TN': float(TN),
        'num_pred_success': num_pred_success,
        'num_gt_success': num_gt_success,
        'success_rate': float(success_rate),
        'failure_analysis': failure_analysis,
    }
    fname = os.path.join(agent.outFolder, 'results_rollouts.json')
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":
    main()