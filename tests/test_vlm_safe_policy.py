import mujoco
import safety_rl_manip
from omegaconf import OmegaConf
import imageio, cv2
from PIL import Image
import gym
import numpy as np
from safety_rl_manip.models.vlm_semantic_safety import SafePlanner
from safety_rl_manip.envs.utils import add_text_to_img

if __name__ == "__main__":

    output_path = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/gpt_safety/panda_slide_pickup/'
    env_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/envs/mujoco_envs.yaml')
    eval_cfg = OmegaConf.load('/home/saumyas/Projects/safe_control/safety_rl_manip/cfg/eval/eval_rl_vlm.yaml')
    
    vlm_cfg = eval_cfg.vlm_cfg
    env_name = "slide_pickup_clutter_mujoco_env-v0"
    env_cfg = env_cfg[env_name]
    env = gym.make(env_name, device=0, cfg=env_cfg)

    vlm_safety = SafePlanner(vlm_cfg.vlm_type, env_cfg.objects.names)

    max_ep_len = 200
    xt_all, at_all, imgs = [], [], []
    xt = env.reset()
    xt_all.append(xt)    
    done = False
    while not (done or env.current_timestep == max_ep_len):
        xt, _, done, _ = env.step(env.get_action())
        xt_all.append(xt)

        imgs.append(env.get_current_image(env.front_cam_name))

        if env.current_timestep > vlm_cfg.subsample_freq*vlm_cfg.num_frames and (env.current_timestep % vlm_cfg.subsample_freq == 0) and env.suction_gripper_active:
            rel_imgs = imgs[::-vlm_cfg.subsample_freq][::-1][-vlm_cfg.num_frames:]
            rel_imgs_labelled = []
            for i, im in enumerate(rel_imgs):
                color_img = im.copy()
                cv2.putText(color_img, str(f"Image {i+1}"), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                rel_imgs_labelled.append(color_img)
            
            rgb_img = np.concatenate(rel_imgs_labelled, axis=1)
            # Image.fromarray(rgb_img).save(output_path + f"current_img_mug_teapot_{env.current_timestep}.png")

            # vlm_output = vlm_safety.get_gpt_output(output_path + f"current_img_mug_teapot_{env.current_timestep}.png")
            # text, objs = vlm_safety.get_text_from_parsed_output(vlm_output)
            # img_vlm_out = add_text_to_img(rgb_img, text)
            # Image.fromarray(img_vlm_out).save(output_path + f"current_img_mug_teapot_{env.current_timestep}_vlm_output.png")

            # import ipdb; ipdb.set_trace()
    filename = output_path + f'test_slide_pickup_reset_grasped.gif'
    imageio.mimsave(filename, imgs[::4], duration=100)