import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from .utils import mlp

from timm.scheduler.scheduler_factory import create_scheduler
from torch.optim import AdamW

from safety_rl_manip.models.encoders.octo_transformer import OctoTransformer
from safety_rl_manip.models.encoders.tokenizers import *
from safety_rl_manip.models.encoders.action_heads import *


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