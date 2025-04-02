from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import AdamW
import gym
import time
import json
from .TD3_core import TD3TransformerIndepActorCriticSS
from .SAC_multimodal_indep import SACMultimodalIndep
import wandb
from tqdm import trange
from .utils import calc_false_pos_neg_rate, TopKLogger, ReplayBufferMultimodal, set_seed

from .datasets import *
import imageio
from safety_rl_manip.models.RARL.utils import print_parameters

class TD3MultimodalIndep(SACMultimodalIndep):

    def __init__(
        self, env_name, device, train_cfg=None, eval_cfg=None,
        env_cfg=None, outFolder='', debug=False,
    ):
        super().__init__(env_name, device, train_cfg=train_cfg, eval_cfg=eval_cfg,
            env_cfg=env_cfg, outFolder=outFolder, debug=debug,)
        
        self.target_noise = train_cfg.target_noise
        self.noise_clip = train_cfg.noise_clip
        self.policy_delay = train_cfg.policy_delay
        self.action_min = torch.tensor(self.env.action_space.low, device=device).float()
        self.action_max = torch.tensor(self.env.action_space.high, device=device).float()
    

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        lx, gx = data['lx'], data['gx']

        q1, q2 = self.ac.action_value(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, self.action_min, self.action_max)

            q1_pi_targ, q2_pi_targ = self.ac_targ.action_value(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if self.mode == 'RA':
                backup = torch.zeros(q_pi_targ.shape).float().to(self.device)

                non_terminal = torch.min(
                    gx,
                    torch.max(lx, q_pi_targ),
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
        q1_pi = self.ac.value(o)['q1_policy']
        return -q1_pi.mean()
    
    def update(self, data, epoch, timer):
        # First run one gradient descent step for Q.

        self.ac.q_optimizer.zero_grad()        
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)
        self.ac.q_optimizer.step()
        self.ac.q_scheduler.step(epoch)

        loss_pi = None
        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:
            self.ac.freeze_q_params()

            self.ac.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)
            self.ac.pi_optimizer.step()
            self.ac.pi_scheduler.step(epoch)
            
            self.ac.unfreeze_q_params()

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        _loss_pi = loss_pi.item() if loss_pi is not None else None
        return loss_q.item(), _loss_pi, loss_info