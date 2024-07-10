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

from gym_reachability import gym_reachability  # register Custom Gym envs.

logger = logging.getLogger(__name__)

@hydra.main(config_path='cfg/', config_name='train/train_pickup1D_ddpg.yaml')
# @hydra.main(config_path='cfg/', config_name='train/train_point_mass_cont_ddpg.yaml')
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

    env_name = cfg.env_name
    env_cfg = cfg.envs[env_name]
    train_cfg = cfg.train_cfg
    
    agent = DDPG(
        env_name, device, train_cfg=train_cfg, env_cfg=env_cfg,
        outFolder=hydra_dir, debug=cfg.debug
    )

    agent.learn()

if __name__ == "__main__":
    main()