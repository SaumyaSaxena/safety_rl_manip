import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from .utils import mlp
import itertools

from timm.scheduler.scheduler_factory import create_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW

from safety_rl_manip.models.encoders.octo_transformer import OctoTransformer
from safety_rl_manip.models.encoders.tokenizers import *
from safety_rl_manip.models.encoders.action_heads import *
from safety_rl_manip.models.encoders.transformer import Transformer, SinusoidalPositionalEncoding
import einops, time

class MLPModePrediction(nn.Module):
    def __init__(self, obs_dim, 
            n_modes, 
            hidden_sizes=(256,256),
            activation=nn.ReLU,
        ):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [n_modes]
        self.model = mlp(pi_sizes, activation)

    def forward(self, obs):
        return self.model(obs)

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_min, act_max):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_min = act_min
        self.act_max = act_max

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        y = (self.pi(obs) + 1.)/2.*(self.act_max-self.act_min) + self.act_min
        return y

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
            activation=nn.ReLU,
        ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_min = torch.from_numpy(action_space.low).to(device)
        act_max = torch.from_numpy(action_space.high).to(device)

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_min, act_max)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
        
class TransformerActorCritic(nn.Module):

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

        self.heads = nn.ModuleList()
        # Action head
        self.heads.append(eval(ac_kwargs.heads.action.name)(
            **ac_kwargs.heads.action.kwargs,
            action_dim=action_dim,
            min_action=min_action,
            max_action=max_action,
            device=device))
        # Value head
        self.heads.append(eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + action_dim,
            device=device))
        

    def get_q_parameters(self):
        return self.heads[1].parameters()
    
    def encode_obs(self, obs, train=True):
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)

        transformer_outputs = self.octo_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        return transformer_outputs
    
    def forward_action(self, transformer_outputs, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        return self.heads[0](transformer_outputs, train=train)
    
    def forward_value(self, transformer_outputs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        return self.heads[1](transformer_outputs, action, train=train)
    
    def forward(self, obs, action, train=True, verbose=False):
        transformer_outputs = self.encode_obs(obs, train=train)

        q_sample = self.forward_value(transformer_outputs, action, train=train)

        action_pred = self.forward_action(transformer_outputs, train=train)

        q_policy = self.forward_value(transformer_outputs, action_pred[:,:,0,:], train=train) # TODO: considering the action at current time step
        outputs = {
            'action': action_pred[:,-1,0,:],
            'q_sample': q_sample[:,-1,0,0],
            'q_policy': q_policy[:,-1,0,0],
        }
        return outputs
    
    def create_optimizers(self, optimizer_cfg):
        # Set up optimizers and schedulers for policy and q-function
        self.optimizer = AdamW(self.parameters(), lr=optimizer_cfg.lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)
        self.scheduler, _ = create_scheduler(optimizer_cfg.scheduler, self.optimizer)

    def act(self, obs):
        with torch.no_grad():
            transformer_outputs = self.encode_obs(obs, train=False)
            return self.forward_action(transformer_outputs, train=False)[:,-1,0,:].cpu().numpy()
    
    def value(self, obs):
        with torch.no_grad():
            transformer_outputs = self.encode_obs(obs, train=False)
            action_pred = self.forward_action(transformer_outputs, train=False)
            q_policy = self.forward_value(transformer_outputs, action_pred[:,:,0,:], train=False)
            outputs = {
                'action': action_pred[:,-1,0,:],
                'q_policy': q_policy[:,-1,0,0],
            }
        return outputs

class TransformerIndepActorCritic(nn.Module):

    def __init__(
            self, 
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = dict(),
        ):
        super().__init__()

        self.window_size = ac_kwargs.window_size
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
        
        self.Q_transformer = OctoTransformer(
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
            action_dim=action_dim,
            min_action=min_action,
            max_action=max_action,
            device=device)
        
        # Value head
        self.value_head = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + action_dim,
            device=device)
        
    def get_q_parameters(self):
        return itertools.chain(self.value_head.parameters(), self.Q_transformer.parameters())

    def get_pi_parameters(self):
        return itertools.chain(self.action_head.parameters(), self.pi_transformer.parameters())
    
    def forward_action(self, obs, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)

        transformer_outputs = self.pi_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )

        return self.action_head(transformer_outputs, train=train)
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)

        transformer_outputs = self.Q_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        return self.value_head(transformer_outputs, action, train=train)

    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[:,-1,0,:].cpu().numpy()
        
    def value(self, obs):
        action_pred = self.forward_action(obs, train=False)
        q_policy = self.forward_value(obs, action_pred[:,:,0,:], train=False)
        outputs = {
            'action': action_pred[:,-1,0,:],
            'q_policy': q_policy[:,-1,0,0],
        }
        return outputs
    
    def action_value(self, obs, action):

        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q_sample = self.forward_value(obs, action)
        return q_sample[:,-1,0,0]
    
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


class TransformerIndepAdaLNActorCritic(nn.Module):

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
        
        self.Q_transformer = OctoTransformer(
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
        
        # Value head
        self.value_head = eval(ac_kwargs.heads.value.name)(
            **ac_kwargs.heads.value.kwargs,
            embedding_size=ac_kwargs.token_embedding_size + self.action_dim,
            device=device)
        
    def get_q_parameters(self):
        return itertools.chain(self.value_head.parameters(), self.Q_transformer.parameters())

    def get_pi_parameters(self):
        return itertools.chain(self.action_head.parameters(), self.pi_transformer.parameters())
    
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

        return self.action_head(transformer_outputs, train=train)
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        batch_size = obs[list(obs.keys())[0]].shape[0]
        pad_mask = torch.ones((batch_size, self.window_size), dtype=bool).to(device=self.device)
        obs['action'] = action
        transformer_outputs = self.Q_transformer(
            obs, tasks={}, pad_mask=pad_mask, train=train
        )
        return self.value_head(transformer_outputs, action, train=train)

    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[:,-1,0,:].cpu().numpy()
        
    def value(self, obs):
        action_pred = self.forward_action(obs, train=False)
        q_policy = self.forward_value(obs, action_pred[:,:,0,:], train=False)
        outputs = {
            'action': action_pred[:,-1,0,:],
            'q_policy': q_policy[:,-1,0,0],
        }
        return outputs
    
    def action_value(self, obs, action):

        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q_sample = self.forward_value(obs, action)
        return q_sample[:,-1,0,0]
    
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


class TransformerIndepActorCriticSS(nn.Module):

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

        # create observation dense layers (tokenizers) for Q network (NOTE: no tokenizers, aussuming all low-dim inputs)
        self.q_observation_tokenizers, self.q_obs_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.observation_tokenizers)
        self.q_position_embs = self.create_pos_embeddings()

        if self.early_q_action_condn: # # create action tokenizers for Q network for 'SA' or 'CA' with the action tokens
            self.env_observation_shapes['action'] = (self.action_dim,)
            self.num_action_tokens = self.find_num_obs_tokens(ac_kwargs.action_tokenizers)
            self.q_action_tokenizers, self.q_action_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.action_tokenizers)
        if self.use_q_readout_tokens: # Add readout tokens for Q network
            self.readout_pos_emb_Q, self.num_readout_tokens_Q = self.create_readout_embeddings(ac_kwargs.readouts_critic)

            num_input_tokens = self.num_obs_tokens + self.num_action_tokens if self.early_q_action_condn else self.num_obs_tokens
            self.readout_mask_Q = self.create_readout_mask(num_input_tokens, self.num_readout_tokens_Q)
        self.Q_transformer = Transformer(**ac_kwargs.Q_transformer_kwargs, device=device).to(device)


        # create observation tokenizers for pi network
        self.pi_observation_tokenizers, self.pi_obs_tokenizer_kwargs = self.create_tokenizers(ac_kwargs.observation_tokenizers)
        self.pi_position_embs = self.create_pos_embeddings()
        if self.use_pi_readout_tokens: # Add readout tokens for pi network
            self.readout_pos_emb_pi, self.num_readout_tokens_pi = self.create_readout_embeddings(ac_kwargs.readouts_actor)
            self.readout_mask_pi = self.create_readout_mask(self.num_obs_tokens, self.num_readout_tokens_pi)
        self.pi_transformer = Transformer(**ac_kwargs.pi_transformer_kwargs, device=device).to(device)

        # Value head
        embedding_size = self.token_embedding_size + self.action_dim if self.late_q_action_condn else self.token_embedding_size
        self.value_head = eval(ac_kwargs.heads.value.name)(
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
    
    def create_readout_mask(self, num_obs_tokens, num_readout_tokens):
        mask = torch.ones((num_obs_tokens+num_readout_tokens, num_obs_tokens+num_readout_tokens)).to(self.device)
        mask[:, -num_readout_tokens:] = 0
        return mask
    
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
        params = [self.pi_observation_tokenizers, self.q_observation_tokenizers]
        if self.early_q_action_condn:
            params.append(self.q_action_tokenizers)
        for layer in itertools.chain(*params):
            if hasattr(layer, 'weight'): # filter out activations
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias)

        # Embedding layers for readouts
        readout_params = []  
        if self.use_q_readout_tokens:
            readout_params.append(self.readout_pos_emb_Q)
        if self.use_pi_readout_tokens:
            readout_params.append(self.readout_pos_emb_pi)
        for emb in itertools.chain(*readout_params):
            nn.init.normal_(emb.weight)
        

    def get_q_parameters(self):
        params = [self.q_observation_tokenizers.parameters(), self.Q_transformer.parameters(), self.value_head.parameters()]
        if self.early_q_action_condn:
            params.append(self.q_action_tokenizers.parameters())
        if self.use_q_readout_tokens:
            params.append(self.readout_pos_emb_Q.parameters())
        return itertools.chain(*params)

    def get_pi_parameters(self):
        params = [self.pi_observation_tokenizers.parameters(), self.pi_transformer.parameters(), self.action_head.parameters()]
        if self.use_pi_readout_tokens:
            params.append(self.readout_pos_emb_pi.parameters())
        return itertools.chain(*params)
    
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
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        batch_size, horizon  = obs[list(obs.keys())[0]].shape[:2]

        all_obs_tokens = self.tokenize_inputs(obs, self.q_observation_tokenizers, self.q_obs_tokenizer_kwargs) # Get tokens for observations
        all_obs_tokens = self.add_position_embeds(all_obs_tokens, self.q_position_embs)
        
        # Get tokens for actions
        if self.early_q_action_condn:
            obs['action'] = action
            all_action_tokens = self.tokenize_inputs(obs, self.q_action_tokenizers, self.q_action_tokenizer_kwargs)
        
        # get readout tokens
        if self.use_q_readout_tokens:
            all_readout_tokens, total_readout_tokens = self.get_readout_tokens(self.readout_pos_emb_Q, batch_size, horizon, self.ac_kwargs.readouts_critic)
        
        # Get transformer outputs
        if self.Q_transformer.attention_type == 'SA':
            input_tokens = all_obs_tokens
            if self.early_q_action_condn:
                input_tokens = torch.cat([input_tokens, all_action_tokens], dim=2)
            if self.use_q_readout_tokens: # tokens are added at the end
                input_tokens = torch.cat([input_tokens, all_readout_tokens], dim=2)
            
            input_tokens = einops.rearrange(
                input_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            transformer_outputs = self.Q_transformer(
                input_tokens, cond=None, attention_mask=None
            )
        elif self.Q_transformer.attention_type == 'AdaLN' or self.Q_transformer.attention_type == 'CA':
            assert self.early_q_action_condn, 'Nothing to condition on!'
            input_tokens = einops.rearrange(
                all_obs_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            cond_tokens = einops.rearrange(
                all_action_tokens,
                "batch horizon n_tokens d -> batch (horizon n_tokens) d",
            )
            transformer_outputs = self.Q_transformer(
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
        values = self.value_head.forward_emb(value_tokens)
        return values

    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[:,-1,0,:].cpu().numpy()
        
    def value(self, obs):
        action_pred = self.forward_action(obs)
        q_policy = self.forward_value(obs, action_pred[:,:,0,:])
        outputs = {
            'action': action_pred[:,-1,0,:],
            'q_policy': q_policy[:,-1,0,0],
        }
        return outputs
    
    def action_value(self, obs, action):
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        q_sample = self.forward_value(obs, action)
        return q_sample[:,-1,0,0]
    
    def create_optimizers(self, optimizer_cfg):
        # Set up optimizers and schedulers for policy and q-function
        self.q_optimizer = AdamW(self.get_q_parameters(), lr=optimizer_cfg.q_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)

        self.pi_optimizer = AdamW(self.get_pi_parameters(), lr=optimizer_cfg.pi_lr, 
            eps=optimizer_cfg.AdamW.eps,
            weight_decay=optimizer_cfg.AdamW.weight_decay)

        if optimizer_cfg.q_scheduler.warmup_restarts:
            self.q_scheduler = CosineAnnealingWarmRestarts(
                self.q_optimizer, 
                T_0=optimizer_cfg.q_scheduler.T_0, 
                T_mult=optimizer_cfg.q_scheduler.T_mult, 
                eta_min=optimizer_cfg.q_scheduler.min_lr)
        if optimizer_cfg.pi_scheduler.warmup_restarts:
            self.pi_scheduler = CosineAnnealingWarmRestarts(
                self.pi_optimizer, 
                T_0=optimizer_cfg.pi_scheduler.T_0, 
                T_mult=optimizer_cfg.pi_scheduler.T_mult, 
                eta_min=optimizer_cfg.pi_scheduler.min_lr)
        else:
            self.q_scheduler, _ = create_scheduler(optimizer_cfg.q_scheduler, self.q_optimizer)
            self.pi_scheduler, _ = create_scheduler(optimizer_cfg.pi_scheduler, self.pi_optimizer)