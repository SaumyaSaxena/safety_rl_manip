import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import hydra
from pathlib import Path
import logging
import wandb

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # Custom Gym env.

logger = logging.getLogger(__name__)

@hydra.main(config_path='cfg/', config_name='train/train_sim_naive.yaml')
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
    
    device = (
        torch.device("cuda", cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    updatePeriod = int(cfg.maxUpdates / cfg.updateTimes)
    vmin = -1 * cfg.scaling
    vmax = 1 * cfg.scaling

    if cfg.mode == 'lagrange':
        envMode = 'normal'
        agentMode = 'normal'
        GAMMA_END = cfg.gamma
        EPS_PERIOD = updatePeriod
        EPS_RESET_PERIOD = cfg.maxUpdates
    elif cfg.mode == 'RA':
        envMode = 'RA'
        agentMode = 'RA'
        if cfg.annealing:
            GAMMA_END = 0.999999
            EPS_PERIOD = int(updatePeriod / 10)
            EPS_RESET_PERIOD = updatePeriod
        else:
            GAMMA_END = cfg.gamma
            EPS_PERIOD = updatePeriod
            EPS_RESET_PERIOD = cfg.maxUpdates

    # Making env
    env_name = cfg.env_name
    env_cfg = cfg.envs[env_name]
    env_cfg.mode = envMode
    if env_cfg.doneType == 'toEnd':
        env_cfg.sample_inside_obs = True
    elif env_cfg.doneType == 'TF' or env_cfg.doneType == 'fail':
        env_cfg.sample_inside_obs = False
    
    env = gym.make(env_name, device=device, **env_cfg)
    env.set_costParam(cfg.penalty, cfg.reward, cfg.costType, cfg.scaling)
    env.set_seed(cfg.seed)
    figureFolder = os.path.join(hydra_dir, 'figure')
    os.makedirs(figureFolder, exist_ok=True)
    env.plot_env(scaling=1.0, figureFolder=figureFolder)

    stateDim = env.state.shape[0]
    actionNum = env.action_space.n
    action_list = np.arange(actionNum)

    # Make agent
    CONFIG = dqnConfig(
        DEVICE=device, ENV_NAME=env_name, SEED=cfg.seed,
        MAX_UPDATES=cfg.maxUpdates, MAX_EP_STEPS=cfg.maxSteps, BATCH_SIZE=cfg.batch_size,
        MEMORY_CAPACITY=cfg.memoryCapacity, ARCHITECTURE=cfg.architecture,
        ACTIVATION=cfg.actType, GAMMA=cfg.gamma, GAMMA_PERIOD=updatePeriod,
        GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
        EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=cfg.learningRate,
        LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100
    )

    dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
    agent = DDQNSingle(
        CONFIG, actionNum, action_list, dimList=dimList, mode=agentMode,
        terminalType=cfg.terminalType, debug=cfg.debug,
    )

    if cfg.warmupQ:
        print("\n== Warmup Q ==")
        lossList = agent.initQ(
            env, cfg.warmupIter, hydra_dir, num_warmup_samples=cfg.num_warmup_samples, vmin=vmin,
            vmax=vmax, plotFigure=cfg.plotFigure, storeFigure=cfg.storeFigure
        )
    
    trainRecords, trainProgress = agent.learn(
        env, MAX_UPDATES=cfg.maxUpdates, MAX_EP_STEPS=cfg.maxSteps, warmupQ=False,
        doneTerminate=True, curUpdates=cfg.warmupIter, vmin=vmin, vmax=vmax, numRndTraj=cfg.numRndTraj,
        checkPeriod=cfg.checkPeriod, outFolder=hydra_dir, plotFigure=cfg.plotFigure,
        storeFigure=cfg.storeFigure
    )

if __name__ == "__main__":
    main()