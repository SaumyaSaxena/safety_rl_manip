import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from .SAC_core import SACTransformerIndepActorCriticSS

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class TD3TransformerIndepActorCriticSS(SACTransformerIndepActorCriticSS):
    def __init__(
            self, 
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = dict(),
        ):
        super().__init__(
            env_observation_shapes, 
            action_space,
            device,
            ac_kwargs = ac_kwargs
        )
    
    def act(self, obs):
        with torch.no_grad():
            return self.forward_action(obs, train=False)[:,-1,0,:].cpu().numpy()
    
    def value(self, obs):
        action_pred = self.forward_action(obs)
        q1_policy, q2_policy = self.forward_value(obs, action_pred[:,:,0,:])
        outputs = {
            'action': action_pred[:,-1,0,:],
            'q1_policy': q1_policy[:,-1,0,0],
            'q2_policy': q2_policy[:,-1,0,0],
        }
        return outputs

    def pi(self, obs):
        action_pred = self.forward_action(obs)
        return action_pred[:,-1,0,:]