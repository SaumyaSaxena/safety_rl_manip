import os
from warnings import simplefilter
import numpy as np
import torch
import hydra
from pathlib import Path
import logging
import wandb
from omegaconf import OmegaConf

from RARL.DDPG import DDPG

from gym_reachability import gym_reachability  # Custom Gym env.

logger = logging.getLogger(__name__)

@hydra.main(config_path='cfg/', config_name='train/train_point_mass_cont_ddpg.yaml')
def main(cfg):
    hydra_dir = Path(os.getcwd())
    cfg = cfg.train

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

    # Making env
    env_name = cfg.env_name
    env_cfg = cfg.envs[env_name]

    if env_cfg.doneType == 'toEnd':
        env_cfg.sample_inside_obs = True
    elif env_cfg.doneType == 'TF' or env_cfg.doneType == 'fail':
        env_cfg.sample_inside_obs = False
    
    agent = DDPG(
        env_name, device, env_cfg=env_cfg,
        mode=cfg.mode, outFolder=hydra_dir, debug=cfg.debug
    )

    agent.learn()

if __name__ == "__main__":
    main()