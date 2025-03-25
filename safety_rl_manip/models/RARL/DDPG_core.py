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
from safety_rl_manip.models.encoders.transformer import Transformer
import einops

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
        self.action_dim = action_space.shape[0]
        self.min_action = torch.from_numpy(action_space.low).to(device)
        self.max_action = torch.from_numpy(action_space.high).to(device)
        self.device = device

        self.early_q_action_condn = 'early' in ac_kwargs.q_action_condn_type
        self.late_q_action_condn = 'late' in ac_kwargs.q_action_condn_type
        self.use_q_readout_tokens = ac_kwargs.Q_transformer_kwargs.transformer_output_type == 'cls'
        self.use_pi_readout_tokens = ac_kwargs.pi_transformer_kwargs.transformer_output_type == 'cls'

        # create observation dense laters (tokenizers) for Q network (NOTE: no tokenizers, aussuming all low-dim inputs)
        self.q_observation_tokenizers, self.q_obs_tokenizer_kwargs = nn.ModuleList(), []
        for obs_tokenizer, tokenizer_kwargs in ac_kwargs.observation_tokenizers.items():
            if len(tokenizer_kwargs.kwargs.obs_stack_keys) > 0:
                num_features = env_observation_shapes[tokenizer_kwargs.kwargs.obs_stack_keys[0]][-1]
                self.q_observation_tokenizers.append(
                    mlp([num_features] + list(tokenizer_kwargs.kwargs.hidden_sizes) + [ac_kwargs.token_embedding_size], nn.GELU, output_activation=nn.GELU).to(device)
                )
                self.q_obs_tokenizer_kwargs.append(tokenizer_kwargs.kwargs)

        if self.early_q_action_condn: # 'SA' or 'CA' with the action tokens
            # create action tokenizers for Q network
            env_observation_shapes['action'] = (self.action_dim,)
            self.q_action_tokenizers, self.q_action_tokenizer_kwargs = nn.ModuleList(), []
            for act_tokenizer, tokenizer_kwargs in ac_kwargs.action_tokenizers.items():
                if len(tokenizer_kwargs.kwargs.obs_stack_keys) > 0:
                    self.q_action_tokenizers.append(
                        mlp([self.action_dim] + list(tokenizer_kwargs.kwargs.hidden_sizes) + [ac_kwargs.token_embedding_size], nn.GELU, output_activation=nn.GELU).to(device)
                    )
                    self.q_action_tokenizer_kwargs.append(tokenizer_kwargs.kwargs)

        if self.use_q_readout_tokens:
            # Add readout tokens for Q network
            self.readout_pos_emb_Q = nn.ModuleList()
            for _, n_tokens in ac_kwargs.readouts_critic.items():
                self.readout_pos_emb_Q.append(nn.Embedding(ac_kwargs.window_size*n_tokens, ac_kwargs.token_embedding_size).to(device))

        self.Q_transformer = Transformer(**ac_kwargs.Q_transformer_kwargs, device=device).to(device)

        # create observation tokenizers for pi network
        self.pi_observation_tokenizers, self.pi_obs_tokenizer_kwargs = nn.ModuleList(), []
        for obs_tokenizer, tokenizer_kwargs in ac_kwargs.observation_tokenizers.items():
            if len(tokenizer_kwargs.kwargs.obs_stack_keys) > 0:
                num_features = env_observation_shapes[tokenizer_kwargs.kwargs.obs_stack_keys[0]][-1]
                self.pi_observation_tokenizers.append(
                    nn.Linear(
                        in_features=num_features, 
                        out_features=ac_kwargs.token_embedding_size
                    ).to(device)
                )
                self.pi_obs_tokenizer_kwargs.append(tokenizer_kwargs.kwargs)

        if self.use_pi_readout_tokens:
            # Add readout tokens for pi network
            self.readout_pos_emb_pi = nn.ModuleList()
            for _, n_tokens in ac_kwargs.readouts_actor.items():
                self.readout_pos_emb_pi.append(nn.Embedding(ac_kwargs.window_size*n_tokens, ac_kwargs.token_embedding_size).to(device))
        
        self.pi_transformer = Transformer(**ac_kwargs.pi_transformer_kwargs, device=device).to(device)

        # Value head
        if self.late_q_action_condn: 
            embedding_size=ac_kwargs.token_embedding_size + self.action_dim
        else:
            embedding_size=ac_kwargs.token_embedding_size

        self.value_head = mlp([embedding_size] + list(ac_kwargs.heads.value.kwargs.hidden_sizes) + [1*self.pred_horizon], nn.SiLU).to(device)

        # Action head
        self.action_head = mlp([ac_kwargs.token_embedding_size] + list(ac_kwargs.heads.action.kwargs.hidden_sizes) + [self.action_dim*self.pred_horizon], nn.SiLU, output_activation=nn.Tanh).to(device)
        
        self.reset_parameters()
    
    def reset_parameters(self):

        # Linear layers
        params = [self.pi_observation_tokenizers, self.q_observation_tokenizers, self.value_head, self.action_head]
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
    
    def forward_action(self, obs, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, action_dim]
        '''
        batch_size, horizon  = obs[list(obs.keys())[0]].shape[:2]

        obs_tokens = []
        for tokenizer, tokenizer_kwargs in zip(self.pi_observation_tokenizers, self.pi_obs_tokenizer_kwargs):
            for obs_key in tokenizer_kwargs.obs_stack_keys:
                tokens = tokenizer(obs[obs_key])
                if len(self.env_observation_shapes[obs_key]) == 1: 
                    tokens = tokens.unsqueeze(2) # [batch_size, window_size, num_tokens, token_embedding_size]
                obs_tokens.append(tokens) 
        
        all_obs_tokens = torch.cat(obs_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]

        # get readout tokens
        if self.use_pi_readout_tokens:
            readout_tokens = []
            total_readout_tokens = 0
            readouts = list(self.ac_kwargs.readouts_actor.keys())
            for readout_emb, readout_name in zip(self.readout_pos_emb_pi, readouts):
                n_tokens_for_readout = self.ac_kwargs.readouts_actor[readout_name]
                total_readout_tokens += n_tokens_for_readout
                tokens = torch.zeros(
                    (batch_size, horizon, n_tokens_for_readout, self.ac_kwargs.token_embedding_size),
                    device=self.device
                )
                embedding = readout_emb(torch.arange(n_tokens_for_readout*horizon).to(self.device)) # Use only the timesteps we receive as input
                embedding = embedding.reshape(1, horizon, n_tokens_for_readout, self.ac_kwargs.token_embedding_size)
                tokens += torch.broadcast_to(embedding, tokens.shape)
                readout_tokens.append(tokens)
            all_readout_tokens = torch.cat(readout_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]

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
        
        actions = self.action_head(state_tokens)
        actions = rearrange(
            actions, "b w (p a) -> b w p a", p=self.pred_horizon, a=self.action_dim
        )
        actions = (actions + 1) / 2 * (self.max_action - self.min_action) + self.min_action

        return actions
    
    def forward_value(self, obs, action, train=True):
        '''
            Output shape: [batch_size, window_size, pred_horizon, value_dim]
        '''
        batch_size, horizon  = obs[list(obs.keys())[0]].shape[:2]

        # Get tokens for observations
        obs_tokens = []
        for tokenizer, tokenizer_kwargs in zip(self.q_observation_tokenizers, self.q_obs_tokenizer_kwargs):
            for obs_key in tokenizer_kwargs.obs_stack_keys:
                tokens = tokenizer(obs[obs_key])
                if len(self.env_observation_shapes[obs_key]) == 1: 
                    tokens = tokens.unsqueeze(2) # [batch_size, window_size, num_tokens, token_embedding_size]
                obs_tokens.append(tokens) 
        
        all_obs_tokens = torch.cat(obs_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]

        # Get tokens for actions
        if self.early_q_action_condn:
            obs['action'] = action
            action_tokens = []
            for tokenizer, tokenizer_kwargs in zip(self.q_action_tokenizers, self.q_action_tokenizer_kwargs):
                for obs_key in tokenizer_kwargs.obs_stack_keys:
                    tokens = tokenizer(obs[obs_key])
                    if len(self.env_observation_shapes[obs_key]) == 1: 
                        tokens = tokens.unsqueeze(2) # [batch_size, window_size, num_tokens, token_embedding_size]
                    action_tokens.append(tokens)

            all_action_tokens = torch.cat(action_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]
        
        # get readout tokens
        if self.use_q_readout_tokens:
            readout_tokens = []
            readouts = list(self.ac_kwargs.readouts_critic.keys())
            total_readout_tokens = 0
            for readout_emb, readout_name in zip(self.readout_pos_emb_Q, readouts):
                n_tokens_for_readout = self.ac_kwargs.readouts_critic[readout_name]
                total_readout_tokens += n_tokens_for_readout
                tokens = torch.zeros(
                    (batch_size, horizon, n_tokens_for_readout, self.ac_kwargs.token_embedding_size),
                    device=self.device
                )
                embedding = readout_emb(torch.arange(n_tokens_for_readout*horizon).to(self.device)) # Use only the timesteps we receive as input
                embedding = embedding.reshape(1, horizon, n_tokens_for_readout, self.ac_kwargs.token_embedding_size)
                tokens += torch.broadcast_to(embedding, tokens.shape)
                readout_tokens.append(tokens)
            all_readout_tokens = torch.cat(readout_tokens, dim=2) # [batch_size, window_size, num_tokens, token_embedding_size]
        
        
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
            transformer_outputs = self.transformer(
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
        values = self.value_head(value_tokens)
        values = rearrange(
            values, "b w (p a) -> b w p a", p=self.pred_horizon, a=1
        )
        return values

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