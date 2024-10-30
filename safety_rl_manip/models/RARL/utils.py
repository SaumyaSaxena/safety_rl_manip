"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module provides some utils functions for reinforcement learning
algortihms and some general save and load functions.
"""

import os
import glob
import pickle
import torch
import wandb
import numpy as np
import torch.nn as nn
import random

import heapq
from heapq import heappush, heappop


def soft_update(target, source, tau):
  """Uses soft_update method to update the target network.

  Args:
      target (toch.nn.Module): target network in double deep Q-network.
      source (toch.nn.Module): Q-network in double deep Q-network.
      tau (float): the ratio of the weights in the target.
  """
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def save_model(model, step, logs_path, types, MAX_MODEL, success=0., config=None, debug=False):
  """Saves the weights of the model.

  Args:
      model (toch.nn.Module): the model to be saved.
      step (int): the number of updates so far.
      logs_path (str): the path to save the weights.
      types (str): the decorater of the file name.
      MAX_MODEL (int): the maximum number of models to be saved.
  """
  start = len(types) + 1
  os.makedirs(logs_path, exist_ok=True)
  model_list = glob.glob(os.path.join(logs_path, "*.pth"))
  if len(model_list) > MAX_MODEL - 1:
    min_step = min([int(li.split("/")[-1][start:-4]) for li in model_list])
    os.remove(os.path.join(logs_path, "{}-{}.pth".format(types, min_step)))
  save_file_name = os.path.join(logs_path, f"{types}_step_{step}_success_{success:.2f}.pth")
  
  if config is not None:
    torch.save(
      obj={
          "state_dict": model.state_dict(),
          "config": config,
          "step": step,
      },
      f=save_file_name,
  )
  else:
    torch.save(model.state_dict(), save_file_name)

  if not debug:
    wandb.save(save_file_name, base_path=os.path.join(logs_path, '..'))
  print("  => Save {} after [{}] updates".format(save_file_name, step))


def save_obj(obj, filename):
  """Saves the object into a pickle file.

  Args:
      obj (object): the object to be saved.
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  """Loads the object and return the object.

  Args:
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "rb") as f:
    return pickle.load(f)

def calc_false_pos_neg_rate(pred_v, GT_v):
  pred_success = pred_v > 0.
  GT_success = GT_v < 0. # env considers V(x)<0 as success

  FP = np.sum(np.logical_and((GT_success == False), (pred_success == True)))
  FN = np.sum(np.logical_and((GT_success == True), (pred_success == False)))

  TP = np.sum(np.logical_and((GT_success == True), (pred_success == True)))
  TN = np.sum(np.logical_and((GT_success == False), (pred_success == False)))

  false_pos_rate = FP/(FP+TN)
  false_neg_rate = FN/(FN+TP)

  return false_pos_rate, false_neg_rate

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

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class TopKLogger:
  def __init__(self, k: int):
    self.max_to_keep = k
    self.checkpoint_queue = []
    
  def push(self, ckpt: str, success: float):
    # NOTE: We have a min heap
    if len(self.checkpoint_queue) < self.max_to_keep:
      heappush(self.checkpoint_queue, (success, ckpt))
      return True
    else:
      curr_min_success, _ = self.checkpoint_queue[0]
      if curr_min_success < success:
        heappop(self.checkpoint_queue)
        heappush(self.checkpoint_queue, (success, ckpt))
        return True
      else:
        return False
  
  def best_ckpt(self):
    success, ckpt = heapq.nlargest(1,self.checkpoint_queue)[0]
    return ckpt
  

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for DDPG agents.
  """

  def __init__(self, obs_dim, act_dim, size, device):
    self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.lx_buf = np.zeros(size, dtype=np.float32)
    self.gx_buf = np.zeros(size, dtype=np.float32)
    self.device = device
    self.ptr, self.size, self.max_size = 0, 0, size
    self.ptr_start = 0

  def store(self, obs, act, rew, next_obs, done, lx, gx):
    self.obs_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.lx_buf[self.ptr] = lx
    self.gx_buf[self.ptr] = gx
    self.ptr = self.ptr_start + (self.ptr+1-self.ptr_start) % (self.max_size-self.ptr_start)
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    batch = dict(obs=self.obs_buf[idxs],
                  obs2=self.obs2_buf[idxs],
                  act=self.act_buf[idxs],
                  rew=self.rew_buf[idxs],
                  done=self.done_buf[idxs],
                  lx=self.lx_buf[idxs],
                  gx=self.gx_buf[idxs])
    return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

class ReplayBufferSwitching(ReplayBuffer):
  """
  FIFO experience replay buffer for DDPG switching agents.
  """
  def __init__(self, obs_dim, act_dim, n_modes, size, device):
    super().__init__(obs_dim, act_dim, size, device)

    self.rew_buf = np.zeros(combined_shape(size, n_modes), dtype=np.float32)
    self.done_buf = np.zeros(combined_shape(size, n_modes), dtype=np.float32)
    self.lx_buf = np.zeros(combined_shape(size, n_modes), dtype=np.float32)
    self.gx_buf = np.zeros(combined_shape(size, n_modes), dtype=np.float32)