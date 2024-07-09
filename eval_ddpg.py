import os
import numpy as np
import torch
from pathlib import Path
import logging
import wandb
import time
from omegaconf import OmegaConf

from RARL.DDPG import DDPG

logger = logging.getLogger(__name__)

def main():
    cfg = OmegaConf.load('cfg/eval/eval_ddpg.yaml')

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")

    eval_cfg = cfg.eval_cfg
    
    device = (
        torch.device("cuda", cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    eval_path = os.path('outputs/evals')
    model_path = os.path.join(eval_path, 'ckpts')

    # Load checkpoint
    ckpt_file = wandb.restore(
        str(eval_cfg.wandb_load.file), 
        run_path=eval_cfg.wandb_load.run_path,
        root=str(model_path), replace=True,
    )
    ckpt = torch.load(ckpt_file.name, map_location=device)
    
    env_name = ckpt['env_name']
    mode = ckpt['train_cfg']['mode']
    eval_path = os.path.join(eval_path, f'{env_name}_DDPG_{mode}', time_str)

    agent = DDPG(
        env_name, device, train_cfg=ckpt['train_cfg'], env_cfg=ckpt['env_cfg'],
        outFolder=eval_path, debug=cfg.debug
    )
    agent.eval(ckpt)

if __name__ == "__main__":
    main()