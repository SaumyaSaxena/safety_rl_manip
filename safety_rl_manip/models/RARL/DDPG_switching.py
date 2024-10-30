from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import AdamW
import torch.nn.functional as F

import gym
import time
import json
from .DDPG_core import MLPActorCritic, MLPModePrediction
import wandb
from tqdm import trange
from .utils import calc_false_pos_neg_rate, TopKLogger, ReplayBufferSwitching, set_seed
from timm.scheduler.scheduler_factory import create_scheduler
from .datasets import *
import imageio


class DDPGSwitching(torch.nn.Module):
    def __init__(
        self, env_name, device, train_cfg=None, eval_cfg=None,
        env_cfg=None, outFolder='', debug=False,
    ):
        super().__init__()
        self.env_name = env_name
        self.device = device
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
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
        self.max_ep_len = train_cfg.max_ep_len
        self.model_save_freq = train_cfg.model_save_freq
        self.plot_save_freq = train_cfg.plot_save_freq
        self.warmup = train_cfg.warmup
        self.warmup_cfg = train_cfg.warmup_cfg

        # Gamma scheduler
        if train_cfg.schedule_gamma:
            self.gamma_list = np.ones(self.epochs)
            self.gamma_list[:train_cfg.gamma_warmup_epochs] = np.linspace(self.gamma, 0.9999, train_cfg.gamma_warmup_epochs)

        self.outFolder = outFolder
        self.debug = debug

        set_seed(self.seed)

        self.env = gym.make(env_name, device=device, cfg=env_cfg)
        self.test_env = gym.make(env_name, device=device, cfg=env_cfg)

        self.figureFolder = os.path.join(outFolder, 'figure')
        os.makedirs(self.figureFolder, exist_ok=True)
        if self.eval_cfg.eval_value_fn:
            self.env.plot_env(save_dir=self.figureFolder)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.n_modes = self.env.n_modes

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, 
            self.env.action_space,
            device,
            **train_cfg.ac_kwargs
        ).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        self.mode_pred = MLPModePrediction(
            self.env.n_all,
            self.n_modes,
            **train_cfg.ac_kwargs
        ).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBufferSwitching(obs_dim=self.env.n_all, act_dim=self.act_dim, n_modes=self.n_modes, size=self.replay_size, device=device)

        # Set up optimizers and schedulers for policy and q-function
        self.pi_optimizer = AdamW(self.ac.pi.parameters(), lr=self.pi_lr, 
            eps=self.train_cfg.AdamW.eps,
            weight_decay=self.train_cfg.AdamW.weight_decay)
        self.pi_scheduler, _ = create_scheduler(self.train_cfg.scheduler, self.pi_optimizer)
        self.q_optimizer = AdamW(self.ac.q.parameters(), lr=self.q_lr,
            eps=self.train_cfg.AdamW.eps,
            weight_decay=self.train_cfg.AdamW.weight_decay)
        self.q_scheduler, _ = create_scheduler(self.train_cfg.scheduler, self.q_optimizer)

        self.MSELoss = torch.nn.MSELoss()

        self.modelFolder = os.path.join(outFolder, "model")
        os.makedirs(self.modelFolder, exist_ok=True)

        self.topk_logger = TopKLogger(train_cfg.save_top_k)

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
        o_rel, *_ = self.get_relevant_state(batch['xt'])
        Vx = self.ac.q(o_rel, batch['ut'])
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
        
        o_rel = self.get_relevant_state(batch['xt'])
        q = self.ac.q(o_rel, batch['ut'])

        otp1_rel = self.get_relevant_state(batch['xtp1'])
        q_pi_targ = self.ac_targ.q(otp1_rel, batch['utp1'])
        
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
            o_rel = self.get_relevant_state(batch['xt'])
            loss = self.MSELoss(self.ac.pi(o_rel), batch['ut'])
        elif update_type == 'maxQ':
            o_rel = self.get_relevant_state(batch['xt'])
            q_pi = self.ac.q(o_rel, self.ac.pi(o_rel))
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

        self.warmup_q_optimizer = AdamW(self.ac.q.parameters(), lr=self.warmup_cfg.warmup_q_lr)
        self.warmup_pi_optimizer = AdamW(self.ac.pi.parameters(), lr=self.warmup_cfg.warmup_pi_lr)

        self.create_dataloader()
        step = 0
        for epoch in trange(self.warmup_cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                self.warmup_update(epoch, step, batch)
                step += 1
            if (epoch % self.plot_save_freq == 0) and self.eval_cfg.eval_value_fn:
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

        o_rel, r_rel, lx_rel, gx_rel, d_rel = self.get_relevant_state(o, r=r, lx=lx, gx=gx, d=d)
        q = self.ac.q(o_rel,a)

        # Bellman backup for Q function
        with torch.no_grad():
            o2_rel, *_ = self.get_relevant_state(o2)
            q_pi_targ = self.ac_targ.q(o2_rel, self.ac_targ.pi(o2_rel))
            if self.mode == 'RA':
                backup = torch.zeros(q.shape).float().to(self.device)
                non_terminal = torch.min(
                    gx_rel,
                    torch.max(lx_rel, q_pi_targ),
                )
                terminal = torch.min(lx_rel, gx_rel)
                backup = non_terminal * self.gamma + terminal * (1 - self.gamma)

            else:
                backup = r_rel + self.gamma * (1 - d_rel) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = self.train_cfg.scale_q_loss * ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        o_rel, *_ = self.get_relevant_state(o)
        q_pi = self.ac.q(o_rel, self.ac.pi(o_rel))
        return -q_pi.mean()

    def update(self, data, epoch):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.q_scheduler.step(epoch)

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.pi_scheduler.step(epoch)

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

    def get_relevant_state(self, o, r=None, lx=None, gx=None, d=None):
        logits = self.mode_pred(o)
        gumbel_output = F.gumbel_softmax(logits, tau=1, hard=True)

        o_split = torch.split(o, self.env.dim, dim=-1)
        o_stacked = torch.stack(o_split[1:], dim=-2) # shape (B, n_modes, dim)
        o_relevant_obs = torch.sum(gumbel_output.unsqueeze(-1) * o_stacked, dim=-2)
        o_rel = torch.concatenate([o_split[0], o_relevant_obs],axis=-1)

        r_rel, lx_rel, gx_rel, d_rel = None, None, None, None
        if r is not None:
            r_rel = torch.min(r, dim=-1)[0]
            lx_rel = torch.min(lx, dim=-1)[0]
            gx_rel = torch.min(gx, dim=-1)[0]
            d_rel = torch.sum(d, dim=-1)
        return o_rel, r_rel, lx_rel, gx_rel, d_rel

    def get_action(self, o, noise_scale):
        o_rel, *_ = self.get_relevant_state(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        a = self.ac.act(torch.as_tensor(o_rel, dtype=torch.float32).to(self.device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.env.action_space.low, self.env.action_space.high)

    def test_agent(self, epoch):

        test_info = self.rollout_episodes(self.eval_cfg.num_test_episodes)
        # Plot learned value function and overlay trajs on it
        if (epoch % self.plot_save_freq == 0):
            trajs=[]
            # trajs = self.do_visualization_rollouts(save_rollout_gifs=self.eval_cfg.save_rollout_gifs)
            if self.eval_cfg.test_value_fn:
                # Plot value function
                v = self.get_value_fn()
                self.test_env.plot_value_fn(v.detach().cpu().numpy(),
                    self.test_env.grid_x,
                    target_T=self.test_env.target_T,
                    obstacle_T=self.test_env.obstacle_T,
                    trajs=trajs, 
                    save_dir=self.figureFolder, 
                    name=f'epoch_{epoch}')

        return test_info

    def rollout_episodes(self, num_episodes):
        FP, FN, TP, TN, num_pred_success, num_gt_success = 0, 0, 0, 0, 0, 0
        avg_return, avg_ep_len = 0. , 0.
        for i in trange(num_episodes, desc="Testing"):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            d_rel = np.any(d)

            o_rel, *_ = self.get_relevant_state(torch.from_numpy(o).float().to(self.device))
            pred_v = self.ac_targ.q(
                o_rel, 
                self.ac_targ.pi(o_rel)).detach().cpu().numpy()
            pred_success = pred_v > 0.
            
            gt_success = False
            while not(d_rel or (ep_len == self.max_ep_len)):
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                d_rel = np.any(d)
                r_rel = np.min(r)

                ep_ret += r_rel
                fail, _ = self.test_env.check_failure(o.reshape(1,self.test_env.n_all))
                succ, _ = self.test_env.check_success(o.reshape(1,self.test_env.n_all))
                not_fail = np.logical_not(fail)
                gt_success = np.logical_or(np.all(np.logical_and(not_fail,succ)), gt_success)
                ep_len += 1
            avg_return += ep_ret
            avg_ep_len += ep_len
            num_pred_success += pred_success
            num_gt_success += gt_success

            FP += np.sum(np.logical_and((gt_success == False), (pred_success == True)))
            FN += np.sum(np.logical_and((gt_success == True), (pred_success == False)))
            TP += np.sum(np.logical_and((gt_success == True), (pred_success == True)))
            TN += np.sum(np.logical_and((gt_success == False), (pred_success == False)))
        false_pos_rate = FP/(FP+TN)
        false_neg_rate = FN/(FN+TP)
        avg_return = avg_return/num_episodes
        avg_ep_len = avg_ep_len/num_episodes

        info = {
            'Average_return': avg_return,
            'Average_episode_len': avg_ep_len,
            'False_positive_rate': false_pos_rate,
            'False_negative_rate': false_neg_rate,
            'Total_num_episodes': float(num_episodes),
            'FP': float(FP),
            'FN': float(FN),
            'TP': float(TP),
            'TN': float(TN),
            'num_pred_success': float(num_pred_success),
            'num_gt_success': float(num_gt_success),
        }
        return info

    def learn(self, start_step=0, ckpt=None):
        if ckpt is not None:
            self.load_state_dict(ckpt["state_dict"])

        if self.train_cfg.add_expert_to_buffer:
            self.add_expert_to_buffer()

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs + start_step
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_num = 0

        # Main loop: collect experience in env and update/log each epoch
        epoch=0
        for t in trange(start_step, total_steps, desc="Training"):
            # torch.cuda.empty_cache()
            self.env.epoch = epoch
            self.test_env.epoch = epoch

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
            
            #TODO(saumya): should d_rel be np.any or np.all?
            d_rel = np.any(d)
            r_rel = np.min(r)

            ep_ret += r_rel
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            
            d_rel = False if ep_len==self.max_ep_len else d_rel

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d, info['l_x'], info['g_x'])

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d_rel or (ep_len == self.max_ep_len):
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
                    loss_q, loss_pi, loss_info = self.update(batch, epoch)
                    if not self.debug:
                        log_dict.update({f'loss_q': loss_q})
                        log_dict.update({f'loss_pi': loss_pi})
                        log_dict.update(loss_info)

            # End of epoch handling
            if (t+1-start_step) % self.steps_per_epoch == 0:
                epoch = (t+1-start_step) // self.steps_per_epoch

                if self.train_cfg.schedule_gamma:
                    self.gamma = self.gamma_list[min(epoch, self.epochs-1)]
                if self.train_cfg.schedule_noise:
                    self.act_noise = self.act_noise*self.noise_decay

                # Test the performance of the deterministic version of the agent.
                print(f"Testing at epoch: {epoch}")
                test_info = self.test_agent(epoch)
                avg_test_return = test_info['Average_return']

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

                        log_dict.update(test_info)

                        log_dict.update({f'Time': time.time()-start_time})
                        log_dict.update({'optim/q_lr': self.q_optimizer.param_groups[0]['lr']})
                        log_dict.update({'optim/pi_lr': self.pi_optimizer.param_groups[0]['lr']})

            if not self.debug:
                wandb.log(log_dict, step=t)
        
        # Eval best checkpoint so far
        ckpt = torch.load(self.topk_logger.best_ckpt(), map_location=self.device)
        self.eval(ckpt, self.eval_cfg, debug=False) # uploading stuff to wandb
    
    def get_value_fn(self):
        # v = torch.zeros(self.test_env.grid_x_flat.shape[:-1]).to(self.device)
        gridx_rel, *_ = self.get_relevant_state(self.test_env.grid_x_flat)
        with torch.no_grad():
            v = self.ac_targ.q(
                gridx_rel, 
                self.ac_targ.pi(gridx_rel))
            v = v.reshape(*self.test_env.grid_x.shape[:-1])
        return v

    def do_visualization_rollouts(self, save_rollout_gifs=False):
        trajs = []
        for init_s in self.test_env.visual_initial_states:
            rollout, at_all, imgs = [], [], []
            o, d, ep_ret, ep_len = self.test_env.reset(init_s), False, 0, 0
            d_rel = np.any(d)

            rollout.append(o)
            o_rel, *_ = self.get_relevant_state(torch.from_numpy(o).float().to(self.device))
            pred_v = self.ac_targ.q(
                o_rel, 
                self.ac_targ.pi(o_rel)).detach().cpu().numpy()
            
            pred_success = pred_v > 0.
            gt_success = False
            while not(d_rel or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                at = self.get_action(o, 0)
                o, r, d, _ = self.test_env.step(at)
                
                d_rel = np.any(d)

                # ep_ret += r_rel
                fail, _ = self.test_env.check_failure(o.reshape(1,self.test_env.n_all))
                succ, _ = self.test_env.check_success(o.reshape(1,self.test_env.n_all))
                not_fail = np.logical_not(fail)
                gt_success = np.logical_or(np.all(np.logical_and(not_fail,succ)), gt_success)
                
                at_all.append(at)
                rollout.append(o)
                if save_rollout_gifs:
                    self.test_env.renderer.update_scene(self.test_env.data, camera='left_cap3') 
                    img = self.test_env.renderer.render()
                    imgs.append(img)
                ep_len += 1
            trajs.append(np.array(rollout))
            if save_rollout_gifs:
                file_name = f'eval_traj_{self.mode}_{len(trajs)}_pred_succ_{pred_success}_gt_succ_{gt_success}'
                imageio.mimsave(os.path.join(self.figureFolder, f'{file_name}.gif'), 
                    imgs, duration=ep_len*self.test_env.dt)
                self.test_env.plot_trajectory(np.stack(rollout,axis=0), np.stack(at_all,axis=0), os.path.join(self.figureFolder, f'{file_name}.png'))
        return trajs

    def eval(self, ckpt=None, eval_cfg=None, debug=True):
        if ckpt is not None:
            self.load_state_dict(ckpt["state_dict"])
        if eval_cfg is None:
            eval_cfg = self.eval_cfg

        self.env.epoch = self.epochs-1
        self.test_env.epoch = self.epochs-1

        set_seed(eval_cfg.seed)
        trajs = self.do_visualization_rollouts(save_rollout_gifs=eval_cfg.save_rollout_gifs)

        if eval_cfg.eval_value_fn:
            if self.env_cfg.is_GT_value:
                GT_value_fn, GT_grid_x, GT_target_T, GT_obstacle_T = self.test_env.get_GT_value_fn()
                GT_value_fn_flat = GT_value_fn.flatten()
                GT_grid_flat = torch.from_numpy(GT_grid_x.reshape(-1, GT_grid_x.shape[-1])).float().to(self.device)
            else:
                GT_target_T = self.test_env.target_T
                GT_obstacle_T = self.test_env.obstacle_T
                GT_grid_x = self.test_env.grid_x
                GT_grid_flat = torch.from_numpy(GT_grid_x.reshape(-1, GT_grid_x.shape[-1])).float().to(self.device)

            GT_grid_flat_rel, *_ = self.get_relevant_state(GT_grid_flat)
            pred_value_fn_flat = self.ac_targ.q(
                GT_grid_flat_rel, 
                self.ac_targ.pi(GT_grid_flat_rel)).detach().cpu().numpy()
            pred_value_fn = pred_value_fn_flat.reshape(*GT_grid_x.shape[:-1])

            #Plotting value function
            self.test_env.plot_value_fn(
                pred_value_fn,
                GT_grid_x,
                target_T=GT_target_T,
                obstacle_T=GT_obstacle_T,
                trajs=trajs,
                save_dir=self.figureFolder,
                name=f'Pred_value_fn_inf_horizon',
                debug=debug)
            
            #Plotting value function
            logits = self.mode_pred(GT_grid_flat)
            gumbel_output = F.gumbel_softmax(logits, tau=1, hard=True).detach().cpu().numpy()
            mode = np.argmax(gumbel_output, axis=-1).reshape(*GT_grid_x.shape[:-1])
            self.test_env.plot_value_fn(
                mode,
                GT_grid_x,
                target_T=GT_target_T,
                obstacle_T=GT_obstacle_T,
                trajs=trajs,
                save_dir=self.figureFolder,
                name=f'Pred_mode',
                debug=debug)
            if self.env_cfg.is_GT_value:
                false_pos_rate, false_neg_rate = calc_false_pos_neg_rate(pred_value_fn_flat, GT_value_fn_flat)
                            # Save results
                results = {
                    'false_pos_rate': false_pos_rate,
                    'false_neg_rate': false_neg_rate,
                }
                fname = os.path.join(self.outFolder, 'results_compare_to_GT.json')
                with open(fname, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"Saving GT FPR={false_pos_rate}; FNR={false_neg_rate} results to: {str(fname)}")

                #Plotting GT value function
                self.test_env.plot_value_fn(
                    GT_value_fn,
                    GT_grid_x,
                    target_T=GT_target_T,
                    obstacle_T=GT_obstacle_T,
                    save_dir=self.figureFolder,
                    name=f'GT_value_fn_inf_horiron',
                    debug=debug)
        
            print(f"Saving predicted and GT value fn plots to: {str(self.figureFolder)}")
        
        if eval_cfg.eval_safe_rollouts:
            eval_results = self.rollout_episodes(eval_cfg.num_eval_episodes)
            
            fname = os.path.join(self.outFolder, 'results_rollouts.json')
            with open(fname, "w") as f:
                json.dump(eval_results, f, indent=4)

            false_pos_rate = eval_results['False_positive_rate']
            false_neg_rate = eval_results['False_negative_rate']
            print(f"Saving rollouts FPR={false_pos_rate}; FNR={false_neg_rate} results to: {str(fname)}")

        if not debug:
            wandb.run.summary["FPR"] = false_pos_rate
            wandb.run.summary["FNR"] = false_neg_rate