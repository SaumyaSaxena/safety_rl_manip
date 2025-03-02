import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from .utils import mlp

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
        
class ActorCritic(nn.Module):

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