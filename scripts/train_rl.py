import os
from warnings import simplefilter
import numpy as np
import torch
import hydra
from pathlib import Path
import logging
import wandb
from omegaconf import OmegaConf


from safety_rl_manip.models.RARL.DDPG import DDPG
from safety_rl_manip.models.RARL.DDPG_multimodal import DDPGMultimodal
from safety_rl_manip.models.RARL.SAC_multimodal import SACMultimodal
from safety_rl_manip.models.RARL.DDPG_switching import DDPGSwitching
from safety_rl_manip.models.RARL.SAC import SAC


logger = logging.getLogger(__name__)

# @hydra.main(config_path='../cfg/', config_name='train_pickup_slide_mujoco_ddpg.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_pickup_slide_obstacles_mujoco_ddpg.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_pickup_slide_panda_mujoco_ddpg.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_pickup_slide_panda_mujoco_ddpg_mulitmodal.yaml')
@hydra.main(config_path='../cfg/', config_name='train_pickup_slide_panda_mujoco_sac_mulitmodal_low_dim.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_point_mass_2D_obstacles_switching.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_point_mass_cont_ddpg.yaml')
# @hydra.main(config_path='../cfg/', config_name='train_point_mass_cont_sac.yaml')
def main(cfg):
    
    hydra_dir = Path(os.getcwd())

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")
    if not cfg.debug:
        wandb_run = wandb.init(project=cfg.wandb.project, 
            entity=cfg.wandb.entity, 
            group=cfg.wandb.group,
            name=f'{cfg.wandb.name}_{cfg.tag}',
            dir=hydra_dir)
        wandb.config.update(dict(cfg))
    
    device = (
        torch.device("cuda", cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    env_name = cfg.env_name
    env_cfg = cfg.envs[env_name]

    train_cfg = cfg.train_cfg
    eval_cfg = cfg.eval_cfg
    agent = eval(train_cfg.algo_name)(
        env_name, device, train_cfg=train_cfg, eval_cfg=eval_cfg, env_cfg=env_cfg,
        outFolder=hydra_dir, debug=cfg.debug
    )

    # Load checkpoint
    ckpt = None
    if train_cfg.resume_from_ckpt:
        ckpt_file = wandb.restore(
            str(train_cfg.wandb_load.file), 
            run_path=train_cfg.wandb_load.run_path,
            root=str(hydra_dir), replace=True,
        )
        ckpt = torch.load(ckpt_file.name, map_location=device)

    step=0
    if train_cfg.warmup:
        step = agent.initQ()
    
    agent.learn(step, ckpt)

if __name__ == "__main__":
    main()