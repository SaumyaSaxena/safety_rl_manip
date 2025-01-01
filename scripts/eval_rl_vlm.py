import os
import numpy as np
import torch
from pathlib import Path
import logging
import wandb
import time
from omegaconf import OmegaConf
import imageio, cv2
from PIL import Image
from safety_rl_manip.models.RARL.utils import set_seed

from safety_rl_manip.models.RARL.DDPG import DDPG
from safety_rl_manip.models.RARL.SAC import SAC
from safety_rl_manip.models.vlm_semantic_safety import SafePlanner
from safety_rl_manip.envs.utils import add_text_to_img

logger = logging.getLogger(__name__)

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

    env_cfg['objects']['names'] = ['porcelain_teapot', 'porcelain_mug']
    env_cfg['randomize_locations'] = False
    env_cfg['objects']['initial_poses'] = [[-0.1, 0.39], [-0.5, 0.6]]
    # env_cfg['action_scale'] = [0.01, 0.01, 0.01]

    mode = ckpt['train_cfg']['mode']
    algo_name = ckpt['train_cfg']['algo_name']

    if 'run_variant' in ckpt['train_cfg']:
        time_str = time_str + '_' + ckpt['train_cfg']['run_variant']

    eval_path = os.path.join(eval_path, f'{env_name}_{algo_name}_{mode}', time_str)

    agent = eval(algo_name)(
        env_name, device, train_cfg=ckpt['train_cfg'], eval_cfg=eval_cfg,
        env_cfg=env_cfg,
        outFolder=eval_path, debug=cfg.debug
    )
    agent.load_state_dict(ckpt["state_dict"])

    set_seed(eval_cfg['seed'])

    vlm_safety = SafePlanner(vlm_cfg.vlm_type, env_cfg['objects']['names'])

    o, d, ep_ret, ep_len = agent.test_env.reset(), False, 0, 0
    imgs = []
    imgs_obj_label = []
    while not(d or (ep_len == agent.max_ep_len)):
        o, r, d, _ = agent.test_env.step(agent.get_action(o, 0))

        imgs.append(agent.test_env.get_current_image(agent.test_env.front_cam_name))
        objs=[agent.test_env.rel_obj_names[0]]
        objs = ['porcelain_mug']
        agent.test_env.rel_obj_names[0] = objs[0]
        # if agent.test_env.current_timestep > vlm_cfg.subsample_freq*vlm_cfg.num_frames and (agent.test_env.current_timestep % vlm_cfg.subsample_freq == 0) and agent.test_env.suction_gripper_active:
        if (agent.test_env.current_timestep % vlm_cfg.subsample_freq == 0) and agent.test_env.suction_gripper_active:
            rel_imgs = imgs[::-vlm_cfg.subsample_freq][::-1][-vlm_cfg.num_frames:]
            rel_imgs_labelled = []
            for i, im in enumerate(rel_imgs):
                color_img = im.copy()
                cv2.putText(color_img, str(f"Image {i+1}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                rel_imgs_labelled.append(color_img)
            
            rgb_img = np.concatenate(rel_imgs_labelled, axis=1)
            Image.fromarray(rgb_img).save(os.path.join(agent.figureFolder, f"current_img_mug_teapot_{agent.test_env.current_timestep}.png"))

            objs = ['porcelain_mug']
            # vlm_output = vlm_safety.get_gpt_output(os.path.join(agent.figureFolder, f"current_img_mug_teapot_{agent.test_env.current_timestep}.png"))
            # text, objs = vlm_safety.get_text_from_parsed_output(vlm_output)
            # agent.test_env.rel_obj_names[0] = objs[0]
            # o = agent.test_env.get_current_state()
            # img_vlm_out = add_text_to_img(rgb_img, text)
            # Image.fromarray(img_vlm_out).save(os.path.join(agent.figureFolder, f"current_img_mug_teapot_{agent.test_env.current_timestep}_vlm_output.png"))

        img_obj = imgs[-1].copy()
        cv2.putText(img_obj, str(f"{objs[0]}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        imgs_obj_label.append(img_obj)

    filename = os.path.join(agent.figureFolder, f'test_slide_pickup_reset_grasped.gif')
    imageio.mimsave(filename, imgs_obj_label[::4], duration=200)

if __name__ == "__main__":
    main()