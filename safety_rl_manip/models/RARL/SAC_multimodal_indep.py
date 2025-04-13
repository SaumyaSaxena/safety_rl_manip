from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import AdamW
import gym
import time
import json
from .SAC_core import SACTransformerActorCritic, SACTransformerIndepActorCritic, SACTransformerIndepActorCriticSS
from .TD3_core import TD3TransformerIndepActorCriticSS
from .DDPG_multimodal_indep import DDPGMultimodalIndep

import wandb
from tqdm import trange
from .utils import calc_false_pos_neg_rate, TopKLogger, ReplayBufferMultimodal, set_seed

from .datasets import *
import imageio
from safety_rl_manip.models.RARL.utils import print_parameters

class SACMultimodalIndep(DDPGMultimodalIndep):

    def __init__(
        self, env_name, device, train_cfg=None, eval_cfg=None,
        env_cfg=None, outFolder='', debug=False,
    ):
        super().__init__(env_name, device, train_cfg=train_cfg, eval_cfg=eval_cfg,
            env_cfg=env_cfg, outFolder=outFolder, debug=debug,)

        self.alpha = train_cfg.alpha


    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        lx, gx = data['lx'], data['gx']

        q1, q2 = self.ac.action_value(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            q1_pi_targ, q2_pi_targ = self.ac_targ.action_value(o2, a2)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            if self.mode == 'RA':
                backup = torch.zeros(q_pi_targ.shape).float().to(self.device)

                q_pi_targ_ent = q_pi_targ - self.alpha * logp_a2
                non_terminal = torch.min(
                    gx,
                    torch.max(lx, q_pi_targ_ent),
                )
                terminal = torch.min(lx, gx)
                backup = non_terminal * self.gamma + terminal * (1 - self.gamma)
            else:
                backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = self.train_cfg.scale_q_loss * ((q1 - backup)**2).mean()
        loss_q2 = self.train_cfg.scale_q_loss * ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.mean().detach().cpu().numpy(),
                        Q2Vals=q2.mean().detach().cpu().numpy(),
                        QTarg=backup.mean().detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        outputs = self.ac.value(o)
        q_pi = torch.min(outputs['q1_policy'], outputs['q2_policy'])

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * outputs['logp_action'] - q_pi).mean()

        return loss_pi