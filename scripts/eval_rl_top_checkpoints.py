import os
import numpy as np
import torch
from pathlib import Path
import logging
import wandb
import time
from omegaconf import OmegaConf

from safety_rl_manip.models.RARL.DDPG import DDPG
from safety_rl_manip.models.RARL.DDPG_switching import DDPGSwitching
from safety_rl_manip.models.RARL.DDPG_multimodal import DDPGMultimodal
from safety_rl_manip.models.RARL.DDPG_multimodal_indep import DDPGMultimodalIndep
from safety_rl_manip.models.RARL.SAC import SAC

logger = logging.getLogger(__name__)

def main():
    cfg = OmegaConf.load('cfg/eval/eval_rl.yaml')

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")

    eval_cfg = cfg.eval_cfg
    
    device = (
        torch.device("cuda", eval_cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    eval_dir = Path('outputs/evals')
    model_path = os.path.join(eval_dir, 'ckpts')

    api = wandb.Api()
    run = api.run(eval_cfg.wandb_load.run_path)

    filenames, succ_rates = [], []
    for file in run.files():
        if file.name.startswith('model') and file.name.endswith('.pth'):
            filenames.append(file.name)
            succ_rates.append(float(file.name[-8:-4]))

    succ_rates = np.array(succ_rates)
    topk_idx = np.argsort(succ_rates)[-eval_cfg.top_k:][::-1]
    topk_files = [filenames[i] for i in topk_idx]
    topk_succ_rates = [succ_rates[i] for i in topk_idx]

    for i, ckpt_filename in enumerate(topk_files):
        print("Evaluating checkpoint: ", ckpt_filename)

        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        # Load checkpoint
        ckpt_file = wandb.restore(
            str(ckpt_filename), 
            run_path=eval_cfg.wandb_load.run_path,
            root=str(model_path), replace=True,
        )
        ckpt = torch.load(ckpt_file.name, map_location=device)
    
        env_name = ckpt['env_name']
        mode = ckpt['train_cfg']['mode']
        algo_name = ckpt['train_cfg']['algo_name']

        env_cfg = ckpt['env_cfg']
        if 'env_cfg' in cfg:
            eval_env_cfg = cfg.env_cfg
            OmegaConf.set_struct(env_cfg, False) # to allow merging new fields into the config
            env_cfg = OmegaConf.merge(env_cfg, eval_env_cfg)
            OmegaConf.set_struct(env_cfg, True)

        if 'run_variant' in ckpt['train_cfg']:
            time_str = time_str + '_' + ckpt['train_cfg']['run_variant'] + '_' + f'ckpt_succ_rate_{topk_succ_rates[i]*100:.0f}'

        eval_path = os.path.join(eval_dir, f'{env_name}_{algo_name}_{mode}', time_str)

        agent = eval(algo_name)(
            env_name, device, train_cfg=ckpt['train_cfg'], eval_cfg=eval_cfg,
            env_cfg=env_cfg,
            outFolder=eval_path, debug=cfg.debug
        )
        agent.eval(ckpt, eval_cfg=eval_cfg)

if __name__ == "__main__":
    main()