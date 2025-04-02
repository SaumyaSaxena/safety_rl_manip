import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import itertools

from timm.scheduler.scheduler_factory import create_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW

from .utils import mlp

from safety_rl_manip.models.encoders.octo_transformer import OctoTransformer
from safety_rl_manip.models.encoders.tokenizers import *
from safety_rl_manip.models.encoders.action_heads import *
from safety_rl_manip.models.encoders.transformer import Transformer, SinusoidalPositionalEncoding
import einops

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

class SACTransformerIndepActorCritic(nn.Module):

    def __init__(
            self, 
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = dict(),
        ):
        super().__init__()

        self.window_size = ac_kwargs.window_size
        self.action_dim = action_space.shape[0]
        min_action = torch.from_numpy(action_space.low).to(device)
        max_action = torch.from_numpy(action_space.high).to(device)
        self.device = device

        env_observation_shapes['action'] = (self.action_dim,)

        # create observation tokenizers
        self.observation_tokenizers, obs_tokenizer_names = nn.ModuleList(), []
        for obs_tokenizer, tokenizer_kwargs in ac_kwargs.observation_tokenizers.items():
            if len(tokenizer_kwargs.kwargs.obs_stack_keys) > 0:
                self.observation_tokenizers.append(eval(tokenizer_kwargs.name)(
                    **tokenizer_kwargs.kwargs, 
                    env_observation_shapes=env_observation_shapes, 
                    device=device))
                obs_tokenizer_names.append(obs_tokenizer)
        
        self.Q1_transformer = OctoTransformer(
            observation_tokenizers_names = obs_tokenizer_names,
            observation_tokenizers=self.observation_tokenizers,
            task_tokenizers_names=[],
            task_tokenizers=[],
            readouts=ac_kwargs.readouts_critic,
            token_embedding_size=ac_kwargs.token_embedding_size,
            max_tokens=ac_kwargs.max_tokens,
            transformer_kwargs=ac_kwargs.Q_transformer_kwargs,
            device=device,
        )

        self.Q2_transformer = OctoTransformer(
            observation_tokenizers_names = obs_tokenizer_names,
            observation_tokenizers=self.observation_tokenizers,
            task_tokenizers_names=[],
            task_tokenizers=[],
            readouts=ac_kwargs.readouts_critic,
            token_embedding_size=ac_kwargs.token_embedding_size,
            max_tokens=ac_kwargs.max_tokens,
            transformer_kwargs=ac_kwargs.Q_transformer_kwargs,
            device=device,
        )

        self.pi_transformer = OctoTransformer(
            observation_tokenizers_names = obs_tokenizer_names,
            observation_tokenizers=self.observation_tokenizers,
            task_tokenizers_names=[],
            task_tokenizers=[],
            readouts=ac_kwargs.readouts_actor,
            token_embedding_size=ac_kwargs.token_embedding_size,
            max_tokens=ac_kwargs.max_tokens,
            transformer_kwargs=ac_kwargs.pi_transformer_kwargs,
            device=device,
        )

        # Action head
        self.action_head = eval(ac_kwargs.heads.action.name)(
            **ac_kwargs.heads.action.kwargs,
            action_dim=self.action_dim,
            min_action=min_action,
            max_action=max_action,
            device=device)
        
        # Value head 1
        self.value_head1 = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + self.action_dim,
            device=device)
        
        # Value head 2
        self.value_head2 = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + self.action_dim,
            device=device)
        
    def get_q1_parameters(self):
        return itertools.chain(self.value_head1.parameters(), self.Q1_transformer.parameters())

    def get_q2_parameters(self):
        return itertools.chain(self.value_head2.parameters(), self.Q2_transformer.parameters())
    
    def get_q_parameters(self):
        return itertools.chain(self.get_q1_parameters(), self.get_q2_parameters())

    def get_pi_parameters(self):
        return itertools.chain(self.action_head.parameters(), self.pi_transformer.parameters())
    
    def freeze_q_params(self):
        for p in self.get_q_parameters():
            p.requires_grad = False
    
    def unfreeze_q_params(self):
        for p in self.get_q_parameters():
            p.requires_grad = True

    def forward_action(self, obs, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)
        # obs['action'] = torch.zeros((batch_size, self.window_size, self.action_dim), device=self.device)
        transformer_outputs = self.pi_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )

        return self.action_head(transformer_outputs)
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)
        obs['action'] = action
        transformer_outputs1 = self.Q1_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        q1 = self.value_head1(transformer_outputs1, action, train=train)


        transformer_outputs2 = self.Q2_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        q2 = self.value_head2(transformer_outputs2, action, train=train)

        return q1, q2
    
    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[:,-1,0,:].cpu().numpy()
        
    def value(self, obs):
        action_pred, logp_action = self.forward_action(obs)
        q1_policy, q2_policy = self.forward_value(obs, action_pred[:,:,0,:])
        outputs = {
            'action': action_pred[:,-1,0,:],
            'logp_action': logp_action[:,-1,0],
            'q1_policy': q1_policy[:,-1,0,0],
            'q2_policy': q2_policy[:,-1,0,0],
        }
        return outputs

    def action_value(self, obs, action):

        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q1_sample, q2_sample = self.forward_value(obs, action)
        return q1_sample[:,-1,0,0], q2_sample[:,-1,0,0]
    
    def pi(self, obs):
        action_pred, logp_action = self.forward_action(obs)
        return action_pred[:,-1,0,:], logp_action[:,-1,0]
    
    def create_optimizers(self, optimizer_cfg):
        # Set up optimizers and schedulers for policy and q-function
        self.q_optimizer = AdamW(self.get_q_parameters(), lr=optimizer_cfg.q_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)
        self.q_scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.q_optimizer)

        self.pi_optimizer = AdamW(self.get_pi_parameters(), lr=optimizer_cfg.pi_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)
        self.pi_scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.pi_optimizer)

class SACTransformerIndepActorCriticSS(nn.Module):

    def __init__(
            self, 
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = dict(),
        ):
        super().__init__()

        self.ac_kwargs = ac_kwargs
        self.env_observation_shapes = env_observation_shapes
        self.window_size = ac_kwargs.window_size
        self.pred_horizon = ac_kwargs.pred_horizon
        self.token_embedding_size = ac_kwargs.token_embedding_size
        self.position_embedding_type = ac_kwargs.position_embedding_type
        self.action_dim = action_space.shape[0]
        self.min_action = torch.from_numpy(action_space.low).to(device)
        self.max_action = torch.from_numpy(action_space.high).to(device)
        self.device = device

        self.early_q_action_condn = 'early' in ac_kwargs.q_action_condn_type
        self.late_q_action_condn = 'late' in ac_kwargs.q_action_condn_type
        self.use_q_readout_tokens = ac_kwargs.Q_transformer_kwargs.transformer_output_type == 'cls'
        self.use_pi_readout_tokens = ac_kwargs.pi_transformer_kwargs.transformer_output_type == 'cls'

        self.num_obs_tokens = self.find_num_obs_tokens(ac_kwargs.observation_tokenizers)
        
        # create observation dense layers (tokenizers) for Q network (NOTE: no tokenizers, assuming all low-dim inputs)
        self.q1_observation_tokenizers, self.q1_obs_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.observation_tokenizers)
        self.q1_position_embs = self.create_pos_embeddings()

        self.q2_observation_tokenizers, self.q2_obs_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.observation_tokenizers)
        self.q2_position_embs = self.create_pos_embeddings()
        if self.early_q_action_condn: # # create action tokenizers for Q network for 'SA' or 'CA' with the action tokens
            self.env_observation_shapes['action'] = (self.action_dim,)
            self.q1_action_tokenizers, self.q1_action_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.action_tokenizers)
            self.q2_action_tokenizers, self.q2_action_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.action_tokenizers)
        if self.use_q_readout_tokens: # Add readout tokens for Q network
            self.readout_pos_emb_Q1, self.num_readout_tokens_Q1 = self.create_readout_embeddings(ac_kwargs.readouts_critic)
            self.readout_pos_emb_Q2, self.num_readout_tokens_Q2 = self.create_readout_embeddings(ac_kwargs.readouts_critic)
        self.Q1_transformer = Transformer(**ac_kwargs.Q_transformer_kwargs, device=device).to(device)
        self.Q2_transformer = Transformer(**ac_kwargs.Q_transformer_kwargs, device=device).to(device)

        # create observation tokenizers for pi network
        self.pi_observation_tokenizers, self.pi_obs_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.observation_tokenizers)
        self.pi_position_embs = self.create_pos_embeddings()

        if self.use_pi_readout_tokens: # Add readout tokens for pi network
            self.readout_pos_emb_pi, self.num_readout_tokens_pi = self.create_readout_embeddings(ac_kwargs.readouts_actor)
        self.pi_transformer = Transformer(**ac_kwargs.pi_transformer_kwargs, device=device).to(device)

        # Value head
        embedding_size = self.token_embedding_size + self.action_dim if self.late_q_action_condn else self.token_embedding_size
        self.value_head1 = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=embedding_size,
            device=device).to(device)
        self.value_head2 = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=embedding_size,
            device=device).to(device)

        # Action head
        self.action_head = eval(ac_kwargs.heads.action.name)(
            **ac_kwargs.heads.action.kwargs,
            action_dim=self.action_dim,
            min_action=self.min_action,
            max_action=self.max_action,
            device=device).to(device)
        
        self.reset_parameters()

    def find_num_obs_tokens(self, tokenizer_kwargs):
        n_tokens = 0
        for name, kwargs in tokenizer_kwargs.items():
            if 'low_dim' in name:
                for stack_key in kwargs.kwargs.obs_stack_keys:
                    if len(self.env_observation_shapes[stack_key]) == 1:
                        n_tokens += 1
                    else:
                        n_tokens += self.env_observation_shapes[stack_key][0]
            else:
                for stack_key in kwargs.kwargs.obs_stack_keys:
                    n_tokens += kwargs.kwargs.num_tokens
        return n_tokens
    
    def create_tokenizers(self, tokenizer_kwargs):
        tokenizers, kwargs_list = nn.ModuleList(), []
        for _, kwargs in tokenizer_kwargs.items():
            if len(kwargs.kwargs.obs_stack_keys) > 0:
                num_features = self.env_observation_shapes[kwargs.kwargs.obs_stack_keys[0]][-1]
                tokenizers.append(
                    mlp([num_features] + list(kwargs.kwargs.hidden_sizes) + [self.token_embedding_size], nn.SiLU, output_activation=nn.SiLU).to(self.device)
                )
                kwargs_list.append(kwargs.kwargs)
        return tokenizers, kwargs_list
    
    def create_pos_embeddings(self):
        if 'sinusoidal' in self.position_embedding_type:
            position_embs = SinusoidalPositionalEncoding(self.token_embedding_size)
        elif 'parameter' in self.position_embedding_type:
            position_embs = nn.Parameter(torch.zeros(1, self.window_size*self.num_obs_tokens, self.token_embedding_size))
        elif 'embedding' in self.position_embedding_type:
            position_embs = nn.Embedding(self.window_size*self.num_obs_tokens, self.token_embedding_size)
        else:
            position_embs = None
        return position_embs

    def create_readout_embeddings(self, readout_kwargs):
        readout_emb = nn.ModuleList()
        total_readout_tokens = 0
        for _, n_tokens in readout_kwargs.items():
            total_readout_tokens += n_tokens
            readout_emb.append(nn.Embedding(self.window_size*n_tokens, self.token_embedding_size).to(self.device))
        return readout_emb, total_readout_tokens

    def reset_parameters(self):

        # Linear layers
        params = [self.pi_observation_tokenizers, self.q1_observation_tokenizers, self.q1_observation_tokenizers]
        if self.early_q_action_condn:
            params.append(self.q1_action_tokenizers)
            params.append(self.q2_action_tokenizers)
        for layer in itertools.chain(*params):
            if hasattr(layer, 'weight'): # filter out activations
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias)

        # Embedding layers for readouts
        readout_params = []  
        if self.use_q_readout_tokens:
            readout_params.append(self.readout_pos_emb_Q1)
            readout_params.append(self.readout_pos_emb_Q2)
        if self.use_pi_readout_tokens:
            readout_params.append(self.readout_pos_emb_pi)
        for emb in itertools.chain(*readout_params):
            nn.init.normal_(emb.weight)
        
    def get_q1_parameters(self):
        params = [self.q1_observation_tokenizers.parameters(), self.Q1_transformer.parameters(), self.value_head1.parameters()]
        if self.early_q_action_condn:
            params.append(self.q1_action_tokenizers.parameters())
        if self.use_q_readout_tokens:
            params.append(self.readout_pos_emb_Q1.parameters())
        return itertools.chain(*params)
    
    def get_q2_parameters(self):
        params = [self.q2_observation_tokenizers.parameters(), self.Q2_transformer.parameters(), self.value_head2.parameters()]
        if self.early_q_action_condn:
            params.append(self.q2_action_tokenizers.parameters())
        if self.use_q_readout_tokens:
            params.append(self.readout_pos_emb_Q2.parameters())
        return itertools.chain(*params)

    def get_q_parameters(self):
        return itertools.chain(self.get_q1_parameters(), self.get_q2_parameters())

    def get_pi_parameters(self):
        params = [self.pi_observation_tokenizers.parameters(), self.pi_transformer.parameters(), self.action_head.parameters()]
        if self.use_pi_readout_tokens:
            params.append(self.readout_pos_emb_pi.parameters())
        return itertools.chain(*params)
    
    def freeze_q_params(self):
        for p in self.get_q_parameters():
            p.requires_grad = False
    
    def unfreeze_q_params(self):
        for p in self.get_q_parameters():
            p.requires_grad = True

    def tokenize_inputs(self, obs, tokenizers, tokenizer_kwargs):
        all_tokens = []
        for tokenizer, tokenizer_kwargs in zip(tokenizers, tokenizer_kwargs):
            for obs_key in tokenizer_kwargs.obs_stack_keys:
                tokens = tokenizer(obs[obs_key])
                if len(self.env_observation_shapes[obs_key]) == 1: 
                    tokens = tokens.unsqueeze(2) # [batch_size, window_size, num_tokens, token_embedding_size]
                all_tokens.append(tokens) 
        
        return torch.cat(all_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]
    
    def add_position_embeds(self, obs_tokens, position_embs):
        if 'parameter' in self.position_embedding_type:
            embedding = position_embs
        elif 'embedding' in self.position_embedding_type or 'sinusoidal' in self.position_embedding_type:
            embedding = position_embs(torch.arange(self.num_obs_tokens*self.window_size).to(self.device)) # Use only the timesteps we receive as input
            embedding = embedding.reshape(1, *obs_tokens.shape[1:])
        else:
            embedding = torch.zeros(1, *obs_tokens.shape[1:], device=self.device)
        
        obs_tokens += torch.broadcast_to(embedding, obs_tokens.shape)
        return obs_tokens

    def get_readout_tokens(self, readout_embs, batch_size, horizon, kwargs):
        readout_tokens = []
        readouts = list(kwargs.keys())
        total_readout_tokens = 0
        for readout_emb, readout_name in zip(readout_embs, readouts):
            n_tokens_for_readout = kwargs[readout_name]
            total_readout_tokens += n_tokens_for_readout
            tokens = torch.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size),
                device=self.device
            )
            embedding = readout_emb(torch.arange(n_tokens_for_readout*horizon).to(self.device)) # Use only the timesteps we receive as input
            embedding = embedding.reshape(1, horizon, n_tokens_for_readout, self.token_embedding_size)
            tokens += torch.broadcast_to(embedding, tokens.shape)
            readout_tokens.append(tokens)
        return torch.cat(readout_tokens, dim=2), total_readout_tokens # [batch_size, window_size, num_tokens, token_embedding_size]

    def forward_action(self, obs, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        batch_size, horizon  = obs[list(obs.keys())[0]].shape[:2]
        all_obs_tokens = self.tokenize_inputs(obs, self.pi_observation_tokenizers, self.pi_obs_tokenizer_kwargs)
        all_obs_tokens = self.add_position_embeds(all_obs_tokens, self.pi_position_embs)

        # get readout tokens
        if self.use_pi_readout_tokens:
            all_readout_tokens, total_readout_tokens = self.get_readout_tokens(self.readout_pos_emb_pi, batch_size, horizon, self.ac_kwargs.readouts_actor)

        # Get transformer outputs
        if self.pi_transformer.attention_type == 'SA':
            input_tokens = all_obs_tokens
            if self.use_pi_readout_tokens: # tokens are added at the end
                input_tokens = torch.cat([input_tokens, all_readout_tokens], dim=2)
            input_tokens = einops.rearrange(
                input_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            transformer_outputs = self.pi_transformer(
                input_tokens, cond=None, attention_mask=None
            )
        else:
            raise NotImplementedError('Attention type not implemented!')
        
        transformer_outputs = einops.rearrange(
            transformer_outputs,
            "batch (horizon n_tokens) d -> batch horizon n_tokens d",
            horizon=horizon,
        )
        
        if self.use_pi_readout_tokens:
            state_tokens = transformer_outputs[:,:,-total_readout_tokens:,:].mean(axis=2) # (batch_size, window_size, embedding_size)
        else:
            state_tokens = transformer_outputs.mean(axis=2) # (batch_size, window_size, embedding_size)
        
        return self.action_head.forward_emb(state_tokens) # (batch_size, window_size, pred_horizon, embedding_size)
    
    def _forward_value(
        self, 
        obs, 
        action, 
        value_idx=1,
        train=True
    ):
        batch_size, horizon  = obs[list(obs.keys())[0]].shape[:2]
        if value_idx == 1:
            q_observation_tokenizers = self.q1_observation_tokenizers
            q_obs_tokenizer_kwargs = self.q1_obs_tokenizer_kwargs
            Q_transformer = self.Q1_transformer
            value_head = self.value_head1
            position_embs = self.q1_position_embs
        elif value_idx == 2:
            q_observation_tokenizers = self.q2_observation_tokenizers
            q_obs_tokenizer_kwargs = self.q2_obs_tokenizer_kwargs
            Q_transformer = self.Q2_transformer
            value_head = self.value_head2
            position_embs = self.q2_position_embs
        else:
            raise NotImplementedError('value_idx invalid!')
        all_obs_tokens = self.tokenize_inputs(obs, q_observation_tokenizers, q_obs_tokenizer_kwargs) # Get tokens for observations
        all_obs_tokens = self.add_position_embeds(all_obs_tokens, position_embs)

        # Get tokens for actions
        if self.early_q_action_condn:
            obs['action'] = action
            if value_idx == 1:
                q_action_tokenizers = self.q1_action_tokenizers
                q_action_tokenizer_kwargs = self.q1_action_tokenizer_kwargs
            elif value_idx == 2:
                q_action_tokenizers = self.q2_action_tokenizers
                q_action_tokenizer_kwargs = self.q2_action_tokenizer_kwargs
            else:
                raise NotImplementedError('value_idx invalid!')
            all_action_tokens = self.tokenize_inputs(obs, q_action_tokenizers, q_action_tokenizer_kwargs)

        # get readout tokens
        if self.use_q_readout_tokens:
            if value_idx == 1:
                readout_pos_emb_Q = self.readout_pos_emb_Q1
            elif value_idx == 2:
                readout_pos_emb_Q = self.readout_pos_emb_Q2
            else:
                raise NotImplementedError('value_idx invalid!')
            all_readout_tokens, total_readout_tokens = self.get_readout_tokens(readout_pos_emb_Q, batch_size, horizon, self.ac_kwargs.readouts_critic)

        # Get transformer outputs
        if Q_transformer.attention_type == 'SA':
            input_tokens = all_obs_tokens
            if self.early_q_action_condn:
                input_tokens = torch.cat([input_tokens, all_action_tokens], dim=2)
            if self.use_q_readout_tokens: # tokens are added at the end
                input_tokens = torch.cat([input_tokens, all_readout_tokens], dim=2)
            
            input_tokens = einops.rearrange(
                input_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            transformer_outputs = Q_transformer(
                input_tokens, cond=None, attention_mask=None
            )
        elif Q_transformer.attention_type == 'AdaLN' or Q_transformer.attention_type == 'CA':
            assert self.early_q_action_condn, 'Nothing to condition on!'
            input_tokens = einops.rearrange(
                all_obs_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            cond_tokens = einops.rearrange(
                all_action_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            transformer_outputs = Q_transformer(
                input_tokens, cond=cond_tokens, attention_mask=None
            )
            # TODO: figure out how to condition on readout tokens
        else:
            raise NotImplementedError('Attention type not implemented!')
        
        transformer_outputs = einops.rearrange(
            transformer_outputs,
            "batch (horizon n_tokens) d -> batch horizon n_tokens d",
            horizon=horizon,
        )

        if self.use_q_readout_tokens:
            value_tokens = transformer_outputs[:,:,-total_readout_tokens:,:].mean(axis=2) # (batch_size, window_size, embedding_size)
        else:
            value_tokens = transformer_outputs.mean(axis=2)
        
        if self.late_q_action_condn: 
            value_tokens = torch.cat([value_tokens, action], dim=-1) 
        return value_head.forward_emb(value_tokens)
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        q1 = self._forward_value(obs, action, value_idx=1, train=True)
        q2 = self._forward_value(obs, action, value_idx=2, train=True)
        return q1, q2

    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[0][:,-1,0,:].cpu().numpy()
        
    def value(self, obs):
        action_pred, logp_action = self.forward_action(obs)
        q1_policy, q2_policy = self.forward_value(obs, action_pred[:,:,0,:])
        outputs = {
            'action': action_pred[:,-1,0,:],
            'logp_action': logp_action[:,-1,0],
            'q1_policy': q1_policy[:,-1,0,0],
            'q2_policy': q2_policy[:,-1,0,0],
        }
        return outputs
    
    def action_value(self, obs, action):
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q1_sample, q2_sample = self.forward_value(obs, action)
        return q1_sample[:,-1,0,0], q2_sample[:,-1,0,0]
    
    def pi(self, obs):
        action_pred, logp_action = self.forward_action(obs)
        return action_pred[:,-1,0,:], logp_action[:,-1,0]
    
    def create_optimizers(self, optimizer_cfg):
        # Set up optimizers and schedulers for policy and q-function
        self.q_optimizer = AdamW(self.get_q_parameters(), lr=optimizer_cfg.q_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)

        self.pi_optimizer = AdamW(self.get_pi_parameters(), lr=optimizer_cfg.pi_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)

        if optimizer_cfg.scheduler.warmup_restarts:
            self.q_scheduler = CosineAnnealingWarmRestarts(
                self.q_optimizer, 
                T_0=optimizer_cfg.scheduler.T_0, 
                T_mult=optimizer_cfg.scheduler.T_mult, 
                eta_min=optimizer_cfg.scheduler.min_lr)
            self.pi_scheduler = CosineAnnealingWarmRestarts(
                self.pi_optimizer, 
                T_0=optimizer_cfg.scheduler.T_0, 
                T_mult=optimizer_cfg.scheduler.T_mult, 
                eta_min=optimizer_cfg.scheduler.min_lr)
        else:
            self.q_scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.q_optimizer)
            self.pi_scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.pi_optimizer)