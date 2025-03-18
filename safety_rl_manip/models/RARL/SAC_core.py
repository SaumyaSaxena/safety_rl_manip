import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from timm.scheduler.scheduler_factory import create_scheduler
from torch.optim import AdamW

from .utils import mlp

from safety_rl_manip.models.encoders.octo_transformer import OctoTransformer
from safety_rl_manip.models.encoders.tokenizers import *
from safety_rl_manip.models.encoders.action_heads import *

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_min, act_max):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_min = act_min
        self.act_max = act_max

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        pi_action = (pi_action + 1.)/2.*(self.act_max-self.act_min) + self.act_min

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(
            self, 
            observation_space, 
            action_space, 
            device,
            hidden_sizes=(256,256),
            activation=nn.ReLU
        ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        act_min = torch.from_numpy(action_space.low).to(device)
        act_max = torch.from_numpy(action_space.high).to(device)

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_min, act_max)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
        
class SACTransformerActorCritic(nn.Module):

    def __init__(
            self, 
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = dict(),
        ):
        super().__init__()

        self.window_size = ac_kwargs.transformer_kwargs.window_size
        action_dim = action_space.shape[0]
        min_action = torch.from_numpy(action_space.low).to(device)
        max_action = torch.from_numpy(action_space.high).to(device)
        self.device = device

        # create observation tokenizers
        self.observation_tokenizers, obs_tokenizer_names = nn.ModuleList(), []
        for obs_tokenizer, tokenizer_kwargs in ac_kwargs.observation_tokenizers.items():
            if len(tokenizer_kwargs.kwargs.obs_stack_keys) > 0:
                self.observation_tokenizers.append(eval(tokenizer_kwargs.name)(
                    **tokenizer_kwargs.kwargs, 
                    env_observation_shapes=env_observation_shapes, 
                    device=device))
                obs_tokenizer_names.append(obs_tokenizer)
        
        self.octo_transformer = OctoTransformer(
            observation_tokenizers_names = obs_tokenizer_names,
            observation_tokenizers=self.observation_tokenizers,
            task_tokenizers_names=[],
            task_tokenizers=[],
            readouts=ac_kwargs.readouts,
            token_embedding_size=ac_kwargs.token_embedding_size,
            max_tokens=ac_kwargs.max_tokens,
            transformer_kwargs=ac_kwargs.transformer_kwargs,
            device=device,
        )

        self.head_names = list(ac_kwargs.heads.keys())
        # self.heads = nn.ModuleList([eval(spec.name)(**spec.kwargs, device=device)
        #     for k, spec in ac_kwargs.heads.items()])

        # Action head
        self.action_head = eval(ac_kwargs.heads.action.name)(
            **ac_kwargs.heads.action.kwargs,
            action_dim=action_dim,
            min_action=min_action,
            max_action=max_action,
            device=device)
        
        # Value head 1 
        self.value_head1 = eval(ac_kwargs.heads.value1.name)(
            **ac_kwargs.heads.value1.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + action_dim,
            device=device)

        # Value head 2
        self.value_head2 = eval(ac_kwargs.heads.value2.name)(
            **ac_kwargs.heads.value2.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + action_dim,
            device=device)
        
    def freeze_q_params(self):
        for p in self.value_head1.parameters():
            p.requires_grad = False
        for p in self.value_head2.parameters():
            p.requires_grad = False
    
    def unfreeze_q_params(self):
        for p in self.value_head1.parameters():
            p.requires_grad = True
        for p in self.value_head2.parameters():
            p.requires_grad = True
    
    def encode_obs(self, obs, train=True):
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)

        transformer_outputs = self.octo_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        return transformer_outputs
    
    def forward_action(self, transformer_outputs, deterministic=False, with_logprob=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        return self.action_head(transformer_outputs, deterministic=deterministic, with_logprob=with_logprob)
    
    def forward_value(self, transformer_outputs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        q1 = self.value_head1(transformer_outputs, action, train=train)
        q2 = self.value_head2(transformer_outputs, action, train=train)
        return q1, q2
    
    def forward(self, obs, action, train=True, verbose=False):
        transformer_outputs = self.encode_obs(obs, train=train)

        q1_sample, q2_sample = self.forward_value(transformer_outputs, action, train=train)

        action_pred, logp_action = self.forward_action(transformer_outputs, train=train)

        q_policy = self.forward_value(transformer_outputs, action_pred[:,:,0,:], train=train) # TODO: considering the action at current time step
        outputs = {
            'action': action_pred[:,-1,0,:],
            'logp_action': logp_action[:,-1,0],
            'q1_sample': q1_sample[:,-1,0,0],
            'q2_sample': q2_sample[:,-1,0,0],
            'q_policy': q_policy[:,-1,0,0],
        }
        return outputs
    
    def create_optimizers(self, optimizer_cfg):
        # Set up optimizers and schedulers for policy and q-function
        self.optimizer = AdamW(self.parameters(), lr=optimizer_cfg.lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)
        self.scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.optimizer)

    def pi(self, obs):
        transformer_outputs = self.encode_obs(obs, train=True)
        action_pred, logp_action = self.forward_action(transformer_outputs)
        return action_pred[:,-1,0,:], logp_action[:,-1,0]

    def act(self, obs):
        with torch.no_grad():
            transformer_outputs = self.encode_obs(obs, train=False)
            action_pred, logp_action = self.forward_action(transformer_outputs, deterministic=True, with_logprob=False)
            return action_pred[:,-1,0,:].cpu().numpy()
    
    def action_value(self, obs, action):
        transformer_outputs = self.encode_obs(obs)

        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q1_sample, q2_sample = self.forward_value(transformer_outputs, action)
        outputs = {
            'q1_sample': q1_sample[:,-1,0,0],
            'q2_sample': q2_sample[:,-1,0,0],
        }
        return outputs
    
    def value(self, obs):
        with torch.no_grad():
            transformer_outputs = self.encode_obs(obs, train=False)
            action_pred, logp_action = self.forward_action(transformer_outputs)
            q1_policy, q2_policy = self.forward_value(transformer_outputs, action_pred[:,:,0,:], train=False)
            outputs = {
                'action': action_pred[:,-1,0,:],
                'logp_action': logp_action[:,-1,0],
                'q1_policy': q1_policy[:,-1,0,0],
                'q2_policy': q2_policy[:,-1,0,0],
            }
        return outputs