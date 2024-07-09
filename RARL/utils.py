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
  pred_success = pred_v < 0.
  GT_success = GT_v < 0.

  FP = np.sum((GT_success == False) and (pred_success == True))
  FN = np.sum((GT_success == True) and (pred_success == False))

  TP = np.sum((GT_success == True) and (pred_success == True))
  TN = np.sum((GT_success == False) and (pred_success == False))

  false_pos_rate = FP/(FP+TN)
  false_neg_rate = FN/(FN+TP)

  return false_pos_rate, false_neg_rate

