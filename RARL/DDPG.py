from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import Adam
import gym
import time
import json
from . import DDPG_core
import wandb
import random
from tqdm import trange
from .utils import calc_false_pos_neg_rate, TopKLogger
from RARL.datasets import *

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(DDPG_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(DDPG_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(DDPG_core.combined_shape(size, act_dim), dtype=np.float32)
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


class DDPG(torch.nn.Module):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    def __init__(
        self, env_name, device, train_cfg=None, env_cfg=None,
        outFolder='', debug=False,
    ):
        super().__init__()
        self.env_name = env_name
        self.device = device
        self.train_cfg = train_cfg
        self.env_cfg = env_cfg

        self.mode = train_cfg.mode
        self.seed = train_cfg.seed
        self.steps_per_epoch = train_cfg.steps_per_epoch
        self.epochs = train_cfg.epochs
        self.replay_size = int(train_cfg.replay_size)
        self.gamma = train_cfg.gamma
        self.polyak = train_cfg.polyak
        self.pi_lr = train_cfg.pi_lr
        self.q_lr = train_cfg.q_lr
        self.batch_size = train_cfg.batch_size
        self.start_steps = train_cfg.start_steps
        self.update_after = train_cfg.update_after
        self.update_every = train_cfg.update_every
        self.act_noise = train_cfg.act_noise
        self.noise_decay = train_cfg.get('noise_decay', 1.)
        self.num_test_episodes = train_cfg.num_test_episodes
        self.max_ep_len = train_cfg.max_ep_len
        self.model_save_freq = train_cfg.model_save_freq
        self.plot_save_freq = train_cfg.plot_save_freq
        self.warmup = train_cfg.warmup
        self.warmup_cfg = train_cfg.warmup_cfg

        # Gamma scheduler
        self.gamma_list = np.ones(self.epochs)
        self.gamma_list[:train_cfg.gamma_warmup_epochs] = np.linspace(self.gamma, 1., train_cfg.gamma_warmup_epochs)

        self.outFolder = outFolder
        self.debug = debug

        self.set_seed(self.seed)

        self.env = gym.make(env_name, device=device, cfg=env_cfg)
        self.test_env = gym.make(env_name, device=device, cfg=env_cfg)

        self.figureFolder = os.path.join(outFolder, 'figure')
        os.makedirs(self.figureFolder, exist_ok=True)
        self.env.plot_env(save_dir=self.figureFolder)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!

        # Create actor-critic module and target networks
        self.ac = DDPG_core.MLPActorCritic(
            self.env.observation_space, 
            self.env.action_space,
            device,
            **train_cfg.ac_kwargs
        ).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device=device)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)

        self.MSELoss = torch.nn.MSELoss()

        self.modelFolder = os.path.join(outFolder, "model")
        os.makedirs(self.modelFolder, exist_ok=True)

        self.topk_logger = TopKLogger(train_cfg.save_top_k)

    def set_seed(self, seed):
        self.seed_val = seed
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
        random.seed(self.seed_val)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def create_dataloader(self):
        if self.warmup_cfg.warmup_type == 'warmupQ_terminal_all_states':
            dataset =  AnalyticalTerminalDataset(
                self.test_env, self.warmup_cfg.warmupQ_terminal_all_states.num_terminal_samples)
        elif self.warmup_cfg.warmup_type == 'warmup_pi':
            dataset =  AnalyticalExpertDataset(
                self.test_env, filename=self.warmup_cfg.warmup_pi.expert_data_loc)
        elif self.warmup_cfg.warmup_type == 'warmupQ_expert':
            dataset =  AnalyticalMixedDataset(
                self.test_env, 
                filename=self.warmup_cfg.warmupQ_expert.expert_data_loc, 
                num_terminal_samples=self.warmup_cfg.warmupQ_expert.num_mixed_terminal_samples)
        else:
            raise NotImplementedError('Dataset type not implemented.')

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.warmup_cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=8,
        )
    
    def warmupQ_terminal_update(self, batch):
        for p in self.ac.pi.parameters():
            p.requires_grad = False
        Vx = self.ac.q(batch['xt'], batch['ut'])
        loss = self.MSELoss(Vx, torch.min(batch['lx'], batch['gx']).flatten())

        loss.backward()
        self.warmup_q_optimizer.step()
        for p in self.ac.pi.parameters():
            p.requires_grad = True
        loss_info = {'warmupQ_terminal_q_loss': loss.item()}
        return loss_info

    def warmupQ_expert_update(self, batch):
        for p in self.ac.pi.parameters():
            p.requires_grad = False
        
        q = self.ac.q(batch['xt'], batch['ut'])
        q_pi_targ = self.ac_targ.q(batch['xtp1'], batch['utp1'])
        
        r , done, lx, gx = batch['rt'].flatten(), batch['done'].flatten(), batch['lx'].flatten(), batch['gx'].flatten()
        # WarmupQ
        if self.mode == 'lagrange':
            backup = r + self.gamma * (1 - done) * q_pi_targ
        elif self.mode == 'RA':
            r = torch.min(lx, gx)
            V_non_terminal = torch.min(
                gx,
                torch.max(lx, q_pi_targ),
            )
            
            terminal = done * r
            non_terminal = (1 - done) * (r*(1 - self.gamma) + V_non_terminal * self.gamma)
            backup = terminal + non_terminal
        else:
            raise NotImplementedError("Mode not implemented.")
        
        # loss = ((q - backup)**2).mean()
        loss = self.MSELoss(q, backup)
        loss.backward()
        self.warmup_q_optimizer.step()
        loss_info = {'warmupQ_expert_q_loss': loss.item()}
        
        # Warmup pi
        if self.warmup_cfg.warmupQ_expert.update_pi:
            # Freeze q, unfreeze pi
            for p in self.ac.pi.parameters():
                p.requires_grad = True
            for p in self.ac.q.parameters():
                p.requires_grad = False

            loss_pi = self.warmup_pi_loss(batch, update_type=self.warmup_cfg.warmupQ_expert.update_pi_type)
            loss_pi.backward()
            self.warmup_pi_optimizer.step()
            loss_info.update({f'warmupQ_expert_pi_loss': loss_pi.item()})

            for p in self.ac.q.parameters():
                p.requires_grad = True
        
        return loss_info

    def warmup_pi_loss(self, batch, update_type='expert'):
        if update_type == 'expert':
            # loss = torch.norm(batch['ut'], self.ac.pi(batch['xt']))
            loss = self.MSELoss(self.ac.pi(batch['xt']), batch['ut'])
        elif update_type == 'maxQ':
            q_pi = self.ac.q(batch['xt'], self.ac.pi(batch['xt']))
            loss = -q_pi.mean()
        else:
            raise NotImplementedError('Pi update type not implemented.')
        return loss

    def warmup_pi_update(self, batch):
        for p in self.ac.pi.parameters():
            p.requires_grad = True
        for p in self.ac.q.parameters():
            p.requires_grad = False

        loss_pi = self.warmup_pi_loss(batch, update_type='expert')
        loss_pi.backward()
        self.warmup_pi_optimizer.step()
        loss_info = {f'warmup_pi_loss': loss_pi.item()}

        for p in self.ac.q.parameters():
            p.requires_grad = True

        return loss_info
        
    def warmup_update(self, epoch, step, batch):
        self.warmup_q_optimizer.zero_grad()
        self.warmup_pi_optimizer.zero_grad()

        if self.warmup_cfg.warmup_type == 'warmupQ_terminal_all_states':
            loss_info = self.warmupQ_terminal_update(batch)
            
        elif self.warmup_cfg.warmup_type == 'warmupQ_expert':
            loss_info = self.warmupQ_expert_update(batch)

        elif self.warmup_cfg.warmup_type == 'warmup_pi':
            loss_info = self.warmup_pi_update(batch)
        else:
            raise NotImplementedError('Dataset type not implemented.')

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if not self.debug: 
            loss_info.update({f'warmup_epoch': epoch})
            loss_info.update({'optim/warmup_q_lr': self.warmup_q_optimizer.param_groups[0]['lr']})
            loss_info.update({'optim/warmup_pi_lr': self.warmup_pi_optimizer.param_groups[0]['lr']})
            wandb.log(loss_info, step=step)
    
    def initQ(self):

        self.warmup_q_optimizer = Adam(self.ac.q.parameters(), lr=self.warmup_cfg.warmup_q_lr)
        self.warmup_pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.warmup_cfg.warmup_pi_lr)

        self.create_dataloader()
        step = 0
        for epoch in trange(self.warmup_cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                self.warmup_update(epoch, step, batch)
                step += 1
            if (epoch % self.plot_save_freq == 0):
                # Plot value function
                v = self.get_value_fn()
                self.test_env.plot_value_fn(v.detach().cpu().numpy(),
                    self.test_env.grid_x,
                    target_T=self.test_env.target_T,
                    obstacle_T=self.test_env.obstacle_T,
                    save_dir=self.figureFolder, 
                    name=f'warmup_epoch_{epoch}')
                
        print(f"Warmup: {self.warmup_cfg.warmup_type} complete.")
        
        return step
    
    def add_expert_to_buffer(self):
        dataset =  AnalyticalExpertDataset(
            self.test_env, 
            data_frac=self.train_cfg.expert_data_frac, 
            filename=self.train_cfg.expert_data_loc)
        expert_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=8,
        )
        print("Adding expert demonstrations to buffer")
        for batch_idx, batch in enumerate(expert_dataloader):
            o, a, r = batch['xt'].flatten(), batch['ut'].flatten(), batch['rt'].flatten()
            o2, d = batch['xtp1'].flatten(), batch['done'].flatten()
            lx, gx  = batch['lx'].flatten(), batch['gx'].flatten()
            d = d > 0.  # bool
            self.replay_buffer.store(o, a, r, o2, d, lx, gx)

        self.replay_buffer.ptr_start = len(dataset)
        assert len(dataset) < self.replay_buffer.max_size, "Size of expert dataset if greater than the max buffer storage allowed."
        print("DONE Adding expert demonstrations to buffer")

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        lx, gx = data['lx'], data['gx']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            if self.mode == 'RA':
                backup = torch.zeros(q.shape).float().to(self.device)
                done = d > 0.
                not_done = torch.logical_not(done)

                non_terminal = torch.min(
                    gx[not_done],
                    torch.max(lx[not_done], q_pi_targ[not_done]),
                )
                terminal = torch.min(lx, gx)
                # normal state
                backup[not_done] = non_terminal * self.gamma + terminal[not_done] * (1 - self.gamma)
                backup[done] = terminal[done]

            else:
                backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return loss_q.item(), loss_pi.item(), loss_info

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.env.action_space.low, self.env.action_space.high)

    def test_agent(self, epoch):
        avg_test_return, avg_test_ep_len = 0. , 0.
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            avg_test_return += ep_ret
            avg_test_ep_len += ep_len
        avg_test_return = avg_test_return/self.num_test_episodes
        avg_test_ep_len = avg_test_ep_len/self.num_test_episodes

        trajs = []
        for init_s in self.test_env.visual_initial_states:
            rollout = []
            o, d, ep_ret, ep_len = self.test_env.reset(init_s), False, 0, 0
            rollout.append(o)
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                rollout.append(o)
                ep_len += 1
            trajs.append(np.array(rollout))

        if (epoch % self.plot_save_freq == 0):
            # Plot value function
            v = self.get_value_fn()
            self.test_env.plot_value_fn(v.detach().cpu().numpy(),
                self.test_env.grid_x,
                target_T=self.test_env.target_T,
                obstacle_T=self.test_env.obstacle_T,
                trajs=trajs, 
                save_dir=self.figureFolder, 
                name=f'epoch_{epoch}')

        return avg_test_return, avg_test_ep_len


    def learn(self, start_step=0):
        if self.train_cfg.add_expert_to_buffer:
            self.add_expert_to_buffer()

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs + start_step
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_num = 0

        # Main loop: collect experience in env and update/log each epoch
        for t in trange(start_step, total_steps):
            # torch.cuda.empty_cache()
            if not self.debug:
                log_dict = {}
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > self.start_steps:
                a = self.get_action(o, self.act_noise)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d, info['l_x'], info['g_x'])

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                ep_num += 1
                if not self.debug:
                    log_dict.update({f'Episode return': ep_ret})
                    log_dict.update({f'Episode length': ep_len})
                    log_dict.update({f'Num episodes': ep_num})
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    loss_q, loss_pi, loss_info = self.update(data=batch)
                    if not self.debug:
                        log_dict.update({f'loss_q': loss_q})
                        log_dict.update({f'loss_pi': loss_pi})
                        log_dict.update(loss_info)

            # End of epoch handling
            if (t+1-start_step) % self.steps_per_epoch == 0:
                epoch = (t+1-start_step) // self.steps_per_epoch

                self.gamma = self.gamma_list[min(epoch, self.epochs-1)]
                self.act_noise = self.act_noise*self.noise_decay

                # Test the performance of the deterministic version of the agent.
                print(f"Testing at epoch: {epoch}")
                avg_test_return, avg_test_ep_len = self.test_agent(epoch)
                
                # Save model
                if (epoch % self.model_save_freq == 0) or (epoch == self.epochs):
                    print(f"Saving model at epoch: {epoch}")
                    save_file_name = os.path.join(self.modelFolder, f"step_{t}_test_return_{avg_test_return:.2f}.pth")
                    status = self.topk_logger.push(save_file_name, avg_test_return)
                    if status:
                        torch.save(
                            obj={
                                "state_dict": self.state_dict(),
                                "env_name": self.env_name,
                                "train_cfg": self.train_cfg,
                                "env_cfg": self.env_cfg,
                                "pi_optimizer_state": self.pi_optimizer.state_dict(),
                                "q_optimizer_state": self.q_optimizer.state_dict(),
                                "epoch": epoch,
                            },
                            f=save_file_name,
                        )
                        if not self.debug:
                            wandb.save(save_file_name, base_path=os.path.join(self.modelFolder, '..'))

                    if not self.debug:
                        log_dict.update({f'Epoch': epoch})
                        log_dict.update({f'gamma': self.gamma})
                        log_dict.update({f'act_noise': self.act_noise})
                        log_dict.update({f'Avg Test Return': avg_test_return})
                        log_dict.update({f'Avg Test Episode len': avg_test_ep_len})
                        log_dict.update({f'Time': time.time()-start_time})

            if not self.debug:
                wandb.log(log_dict, step=t)
        
        # Eval best checkpoint so far
        ckpt = torch.load(self.topk_logger.best_ckpt(), map_location=self.device)
        self.eval(ckpt, debug=False) # uploading stuff to wandb
    
    def get_value_fn(self):
        # v = torch.zeros(self.test_env.grid_x_flat.shape[:-1]).to(self.device)
        with torch.no_grad():
            v = self.ac_targ.q(
                self.test_env.grid_x_flat, 
                self.ac_targ.pi(self.test_env.grid_x_flat))
            v = v.reshape(*self.test_env.grid_x.shape[:-1])
        return v

    def eval(self, ckpt=None, debug=True):
        if ckpt is not None:
            self.load_state_dict(ckpt["state_dict"])
        GT_value_fn, GT_grid_x, GT_target_T, GT_obstacle_T = self.test_env.get_GT_value_fn()
        GT_value_fn_flat = GT_value_fn.flatten()
        GT_grid_flat = torch.from_numpy(GT_grid_x.reshape(-1, GT_grid_x.shape[-1])).float().to(self.device)

        pred_value_fn_flat = self.ac_targ.q(
            GT_grid_flat, 
            self.ac_targ.pi(GT_grid_flat)).detach().cpu().numpy()
        pred_value_fn = pred_value_fn_flat.reshape(*GT_grid_x.shape[:-1])

        false_pos_rate, false_neg_rate = calc_false_pos_neg_rate(pred_value_fn_flat, GT_value_fn_flat)

        # Save results
        results = {
            'false_pos_rate': false_pos_rate,
            'false_neg_rate': false_neg_rate,
        }
        fname = os.path.join(self.outFolder, 'results.json')
        with open(fname, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saving FPR={false_pos_rate}; FNR={false_neg_rate} results to: {str(fname)}")

        if not debug:
            wandb.run.summary["FPR"] = false_pos_rate
            wandb.run.summary["FNR"] = false_neg_rate

        trajs = []
        for init_s in self.test_env.visual_initial_states:
            rollout = []
            o, d, ep_ret, ep_len = self.test_env.reset(init_s), False, 0, 0
            rollout.append(o)
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                rollout.append(o)
                ep_len += 1
            trajs.append(np.array(rollout))
        
        #Plotting value functions
        self.test_env.plot_value_fn(
            GT_value_fn,
            GT_grid_x,
            target_T=GT_target_T,
            obstacle_T=GT_obstacle_T,
            save_dir=self.figureFolder,
            name=f'GT_value_fn_inf_horiron',
            debug=debug)
        
        self.test_env.plot_value_fn(
            pred_value_fn,
            GT_grid_x,
            target_T=GT_target_T,
            obstacle_T=GT_obstacle_T,
            trajs=trajs,
            save_dir=self.figureFolder,
            name=f'Pred_value_fn_inf_horizon',
            debug=debug)

        print(f"Saving predicted and GT value fn plots to: {str(self.figureFolder)}")
