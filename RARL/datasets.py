import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class AnalyticalExpertDataset(Dataset):
    def __init__(
            self, 
            env,
            data_frac=1.,
            filename='', 
        ):
        self.n = env.n
        self.m = env.m
        self.u_min = env.u_min
        self.u_max = env.u_max

        if np.char.endswith(filename,'.pkl'):
            with open(filename, 'rb') as f:
                expert_trajs = pickle.load(f)
            self.xt, self.ut, self.xtp1, self.utp1 = [], [], [], []
            for traj in expert_trajs:
                self.xt.append(traj[:-2,:self.n])
                self.ut.append(traj[:-2,self.n:])
                self.xtp1.append(traj[1:-1,:self.n])
                self.utp1.append(traj[1:-1,self.n:])
            self.xt = np.concatenate(self.xt, axis=0)
            self.ut = np.concatenate(self.ut, axis=0)
            self.xtp1 = np.concatenate(self.xtp1, axis=0)
            self.utp1 = np.concatenate(self.utp1, axis=0)
            
        elif np.char.endswith(filename,'.npz'):
            ## Loading expert data
            data = np.load(filename) # shape (samples, T, n+m, 1)
            expert_trajs = data['train_expert_trajs']

            # Don't include terminal states
            self.xt = np.concatenate(expert_trajs[:,:-2,:self.n,0], axis=0) # t in [0, T-2]
            self.ut = np.concatenate(expert_trajs[:,:-2,self.n:,0], axis=0) # t in [0, T-2]
            self.xtp1 = np.concatenate(expert_trajs[:,1:-1,:self.n,0], axis=0) # t in [1, T-1]
            self.utp1 = np.concatenate(expert_trajs[:,1:-1,self.n:,0], axis=0) # t in [1, T-1]

        fail, self.gx_expert = env.check_failure(self.xt)
        success, self.lx_expert = env.check_success(self.xt)
        self.rt = env.get_cost(self.lx_expert, self.gx_expert, success, fail)
        self.donet = env.get_done(self.xt, success, fail)

        self.xt = torch.from_numpy(self.xt).float()
        self.ut = torch.from_numpy(self.ut).float()
        self.xtp1 = torch.from_numpy(self.xtp1).float()
        self.utp1 = torch.from_numpy(self.utp1).float()
        self.lx_expert = torch.from_numpy(self.lx_expert).float()
        self.gx_expert = torch.from_numpy(self.gx_expert).float()
        self.rt = torch.from_numpy(self.rt).float()
        self.donet = torch.from_numpy(self.donet).float()

        self.num_expert_samples = self.xt.shape[0]

        print("Total Number of expert samples: ", self.num_expert_samples)
        self.num_expert_samples = int(self.num_expert_samples*data_frac)
        print("Number of expert samples used: ", self.num_expert_samples)

    def __len__(self):
        assert self.xt.shape[0] == self.xtp1.shape[0], "Shapes for data incorrect"
        return self.num_expert_samples

    def __getitem__(self, idx):
        # sample expert
        sample = {
            'xt': self.xt[idx],
            'ut': self.ut[idx],
            'xtp1': self.xtp1[idx],
            'utp1': self.utp1[idx],
            'lx': self.lx_expert[idx, None],
            'gx': self.gx_expert[idx, None],
            'rt': self.rt[idx, None],
            'done': self.donet[idx, None],
        }

        return sample
    
class AnalyticalTerminalDataset(Dataset):
    def __init__(
            self,
            env,
            num_samples=1e+6,
        ):
        self.num_samples = int(num_samples)

        self.n = env.n
        self.m = env.m
        #Sampling both x and u
        len = env.n + env.m
        low = np.append(env.low, env.u_min)
        high = np.append(env.high, env.u_max)

        self.x_u = np.random.uniform(low=low, high=high, size=(self.num_samples, len))

        self.lx = torch.from_numpy(env.target_margin(self.x_u[:, :env.n])).float()
        self.gx = torch.from_numpy(env.safety_margin(self.x_u[:, :env.n])).float()

        self.x_u = torch.from_numpy(self.x_u).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # x = (self.x_max - self.x_min) * torch.rand(4) + self.x_min
        # lx, gx = terminal_costs_Pickup1D(x[None,:], thresh=self.thresh, goal=self.goal, task=self.task)
        sample = {
            'xt': self.x_u[idx, :self.n],
            'ut': self.x_u[idx, self.n:],
            'lx': self.lx[idx, None],
            'gx': self.gx[idx, None],
        }
        return sample


class AnalyticalMixedDataset(Dataset):
    def __init__(
            self, 
            env,
            filename='', 
            num_terminal_samples=1e+6
        ):
        self.n = env.n
        self.m = env.m
        self.u_min = env.u_min
        self.u_max = env.u_max

        if np.char.endswith(filename,'.pkl'):
            with open(filename, 'rb') as f:
                expert_trajs = pickle.load(f)
            self.xt, self.ut, self.xtp1, self.utp1 = [], [], [], []
            for traj in expert_trajs:
                self.xt.append(traj[:-2,:self.n])
                self.ut.append(traj[:-2,self.n:])
                self.xtp1.append(traj[1:-1,:self.n])
                self.utp1.append(traj[1:-1,self.n:])
            self.xt = np.concatenate(self.xt, axis=0)
            self.ut = np.concatenate(self.ut, axis=0)
            self.xtp1 = np.concatenate(self.xtp1, axis=0)
            self.utp1 = np.concatenate(self.utp1, axis=0)
            
        elif np.char.endswith(filename,'.npz'):
            ## Loading expert data
            data = np.load(filename) # shape (samples, T, n+m, 1)
            expert_trajs = data['train_expert_trajs']

            # Don't include terminal states
            self.xt = np.concatenate(expert_trajs[:,:-2,:self.n,0], axis=0) # t in [0, T-2]
            self.ut = np.concatenate(expert_trajs[:,:-2,self.n:,0], axis=0) # t in [0, T-2]
            self.xtp1 = np.concatenate(expert_trajs[:,1:-1,:self.n,0], axis=0) # t in [1, T-1]
            self.utp1 = np.concatenate(expert_trajs[:,1:-1,self.n:,0], axis=0) # t in [1, T-1]

        fail, self.gx_expert = env.check_failure(self.xt)
        success, self.lx_expert = env.check_success(self.xt)
        self.rt = env.get_cost(self.lx_expert, self.gx_expert, success, fail)
        self.donet = env.get_done(self.xt, success, fail)

        self.xt = torch.from_numpy(self.xt).float()
        self.ut = torch.from_numpy(self.ut).float()
        self.xtp1 = torch.from_numpy(self.xtp1).float()
        self.utp1 = torch.from_numpy(self.utp1).float()
        self.lx_expert = torch.from_numpy(self.lx_expert).float()
        self.gx_expert = torch.from_numpy(self.gx_expert).float()
        self.rt = torch.from_numpy(self.rt).float()
        self.donet = torch.from_numpy(self.donet).float()

        self.num_expert_samples = self.xt.shape[0]

        ## Terminal data collection: 
        _num_reach_avoid_samples = 0
        self.x_terminal, self.lx_terminal, self.gx_terminal, self.rx_terminal = [], [], [], []
        while _num_reach_avoid_samples < num_terminal_samples:
            _x_samples = np.random.uniform(
                low=env.low, 
                high=env.high, 
                size=(int(num_terminal_samples), env.n)
            )
            success, _lx_samples = env.check_success(_x_samples)
            fail, _gx_samples = env.check_failure(_x_samples)
            _rx_samples = env.get_cost(_lx_samples, _gx_samples, success, fail)

            reach_avoid_set = np.logical_and(success, (not fail))

            self.x_terminal.append(_x_samples[reach_avoid_set])
            self.lx_terminal.append(_lx_samples[reach_avoid_set])
            self.gx_terminal.append(_gx_samples[reach_avoid_set])
            self.rx_terminal.append(_rx_samples[reach_avoid_set])
            _num_reach_avoid_samples += self.x_terminal[-1].shape[0]

        self.x_terminal = torch.from_numpy(np.concatenate(self.x_terminal, axis=0)).float()
        self.lx_terminal = torch.from_numpy(np.concatenate(self.lx_terminal, axis=0)).float()
        self.gx_terminal = torch.from_numpy(np.concatenate(self.gx_terminal, axis=0)).float()
        self.rx_terminal = torch.from_numpy(np.concatenate(self.rx_terminal, axis=0)).float()

        self.num_terminal_samples = self.x_terminal.shape[0]
        print("Number of expert samples: ", self.num_expert_samples)
        print("Number of terminal samples: ", self.num_terminal_samples)

    def __len__(self):
        assert self.xt.shape[0] == self.xtp1.shape[0], "Shapes for data incorrect"
        return self.num_expert_samples + int(self.num_terminal_samples)

    def __getitem__(self, idx):
        if idx < self.num_expert_samples:
            # sample expert
            sample = {
                'xt': self.xt[idx],
                'ut': self.ut[idx],
                'xtp1': self.xtp1[idx],
                'utp1': self.utp1[idx],
                'lx': self.lx_expert[idx, None],
                'gx': self.gx_expert[idx, None],
                'rt': self.rt[idx, None],
                'done': self.donet[idx, None], # not done
            }
        else:
            _idx = idx - self.num_expert_samples
            sample = {
                'xt': self.x_terminal[_idx],
                'ut': torch.from_numpy(np.random.uniform(low=self.u_min, high=self.u_max)).float(), # dummy
                'xtp1': torch.zeros((self.n)).float(),  # dummy
                'utp1': torch.from_numpy(np.random.uniform(low=self.u_min, high=self.u_max)).float(), # dummy
                'lx': self.lx_terminal[_idx, None],
                'gx': self.gx_terminal[_idx, None],
                'rt': self.rx_terminal[_idx, None],
                'done': torch.ones((1)).float() # done
            }
        return sample