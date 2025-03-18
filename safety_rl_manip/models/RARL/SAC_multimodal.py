from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import AdamW
import gym
import time
import json
from .SAC_core import SACTransformerActorCritic
import wandb
from tqdm import trange
from .utils import calc_false_pos_neg_rate, TopKLogger, ReplayBufferMultimodal, set_seed

from .datasets import *
import imageio
from safety_rl_manip.models.RARL.utils import print_parameters

class SACMultimodal(torch.nn.Module):

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
        self.alpha = train_cfg.alpha
        self.polyak = train_cfg.polyak
        self.batch_size = train_cfg.batch_size
        self.start_steps = train_cfg.start_steps
        self.update_after = train_cfg.update_after
        self.update_every = train_cfg.update_every
        self.update_steps = train_cfg.get('update_steps', 50)
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

        self.act_dim = self.env.action_space.shape[0]

        # Create actor-critic module and target networks
        self.ac = eval(train_cfg.ac_type)(
            self.env.env_observation_shapes, 
            self.env.action_space,
            device,
            ac_kwargs=train_cfg.ac_kwargs[f'{train_cfg.ac_type}'],
        ).to(self.device)

        print_parameters(self.ac)

        # self.ac_targ = deepcopy(self.ac).to(self.device)
        self.ac_targ = eval(train_cfg.ac_type)(
            self.env.env_observation_shapes, 
            self.env.action_space,
            device,
            ac_kwargs=train_cfg.ac_kwargs[f'{train_cfg.ac_type}'],
        ).to(self.device)
        self.ac_targ.load_state_dict(self.ac.state_dict())
        # Check if the parameters are the same
        for param_ac, param_ac_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
            assert torch.equal(param_ac, param_ac_targ)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        print_parameters(self.ac_targ)

        # Experience buffer
        self.replay_buffer = ReplayBufferMultimodal(env_observation_shapes=self.env.env_observation_shapes, act_dim=self.act_dim, size=self.replay_size, device=device)

        # Set up optimizers and schedulers for policy and q-function
        self.ac.create_optimizers(self.train_cfg.optimizer)

        self.MSELoss = torch.nn.MSELoss()

        self.modelFolder = os.path.join(outFolder, "model")
        os.makedirs(self.modelFolder, exist_ok=True)

        self.topk_logger = TopKLogger(train_cfg.save_top_k)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data, outputs):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        lx, gx = data['lx'], data['gx']

        # q = self.ac.q(o,a)
        q1 = outputs['q1_sample']
        q2 = outputs['q2_sample']

        # Bellman backup for Q function
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            outputs_targ = self.ac_targ.action_value(o2, a2)
            q1_pi_targ = outputs_targ['q1_sample']
            q2_pi_targ = outputs_targ['q2_sample']
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            if self.mode == 'RA':
                backup = torch.zeros(q_pi_targ.shape).float().to(self.device)

                q_pi_targ_ent = q_pi_targ - self.alpha * logp_a2
                non_terminal = torch.min(
                    gx,
                    torch.max(lx, q_pi_targ_ent),
                )
                terminal = torch.min(lx, gx)
                backup = non_terminal * self.gamma + terminal * (1 - self.gamma)
            else:
                backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = self.train_cfg.scale_q_loss * ((q1 - backup)**2).mean()
        loss_q2 = self.train_cfg.scale_q_loss * ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                        Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data, outputs):
        q_pi = torch.min(outputs['q1_policy'], outputs['q2_policy'])

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * outputs['logp_action'] - q_pi).mean()

        return loss_pi
    
    def update(self, data, epoch):
        # First run one gradient descent step for Q.
        self.ac.optimizer.zero_grad()

        # Debug start 
        transformer_outputs = self.ac.encode_obs(data['obs'], train=True)
        q1_sample, q2_sample = self.ac.forward_value(transformer_outputs, data['act'], train=True)
        action_pred, logp_action = self.ac.forward_action(transformer_outputs)

        outputs = {
            'action': action_pred[:,-1,0,:],
            'logp_action': logp_action[:,-1,0],
            'q1_sample': q1_sample[:,-1,0,0],
            'q2_sample': q2_sample[:,-1,0,0],
        }
        loss_q, loss_info = self.compute_loss_q(data, outputs)
        loss_q.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)

        self.ac.freeze_q_params()
        q1_policy, q2_policy = self.ac.forward_value(transformer_outputs, action_pred[:,:,0,:], train=True) # TODO: considering the action at current time step
        outputs['q1_policy'] = q1_policy[:,-1,0,0]
        outputs['q2_policy'] = q2_policy[:,-1,0,0]
        loss_pi = self.compute_loss_pi(data, outputs)
        # loss_pi.retain_grad() # for checking gradients
        loss_pi.backward()
        torch.nn.utils.clip_grad_value_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)
        self.ac.unfreeze_q_params()
        # Debug end
        
        # real start
        # start = time.time()
        # outputs = self.ac(data['obs'], data['act'])
        # loss_q, loss_info = self.compute_loss_q(data, outputs)
        # # print(f'Time for forward pass q = {time.time()-start}')

        # # start = time.time()
        # loss_q.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_value_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)
        # # print(f'Time for backward pass q = {time.time()-start}')

        # self.ac.freeze_q_params()

        # # start = time.time()
        # loss_pi = self.compute_loss_pi(data, outputs)
        # # print(f'Time for forward pass pi = {time.time()-start}')

        # # start = time.time()
        # loss_pi.backward()
        # torch.nn.utils.clip_grad_value_(self.ac.parameters(), self.train_cfg.optimizer.clip_grad_norm)
        # # print(f'Time for backward pass pi = {time.time()-start}')

        # self.ac.unfreeze_q_params()

        # real end

        self.ac.optimizer.step()
        self.ac.scheduler.step(epoch)

        # Finally, update target networks by polyak averaging.
        # start = time.time()
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        # print(f'Time for polyak averaging = {time.time()-start}')

        return loss_q.item(), loss_pi.item(), loss_info

    def get_action(self, o, noise_scale):
        # a = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device))
        
        o = self.replay_buffer.get_batch_from_obs(o)
        a = self.ac.act(o)
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
        failure_analysis = {k:0 for k in self.test_env.all_failure_modes+['timeout']}
        for i in trange(num_episodes, desc="Testing"):
            o, d, ep_ret = self.test_env.reset(), False, 0

            with torch.no_grad():
                o = self.replay_buffer.get_batch_from_obs(o)
                outputs = self.ac_targ.value(o)
            # pred_v = self.ac_targ(
            #     torch.from_numpy(o).float().to(self.device), 
            #     self.ac_targ.pi(torch.from_numpy(o).float().to(self.device))).detach().cpu().numpy()
                
            pred_success = outputs['q1_policy'].detach().cpu().numpy() > 0.
            
            gt_success = False
            while not(d or (self.test_env.current_timestep == self.max_ep_len)):
                o, r, d, _ = self.test_env.step(outputs['action'].detach().cpu().numpy())
                ep_ret += r
            avg_return += ep_ret
            avg_ep_len += self.test_env.current_timestep
            num_pred_success += pred_success
            gt_success = self.test_env.failure_mode=='success'
            num_gt_success += gt_success
            if d:
                failure_analysis[self.test_env.failure_mode] += 1
            else:
                failure_analysis['timeout'] += 1

            FP += np.sum(np.logical_and((gt_success == False), (pred_success == True)))
            FN += np.sum(np.logical_and((gt_success == True), (pred_success == False)))
            TP += np.sum(np.logical_and((gt_success == True), (pred_success == True)))
            TN += np.sum(np.logical_and((gt_success == False), (pred_success == False)))
        false_pos_rate = FP/(FP+TN)
        false_neg_rate = FN/(FN+TP)
        avg_return = avg_return/num_episodes
        avg_ep_len = avg_ep_len/num_episodes
        success_rate = num_gt_success/num_episodes
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
            'success_rate': float(success_rate),
            'failure_analysis': failure_analysis
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

        start_epoch = time.time()

        # Main loop: collect experience in env and update/log each epoch
        epoch=0
        for t in trange(start_step, total_steps):
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
                for _ in range(self.update_steps):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    loss_q, loss_pi, loss_info = self.update(batch, epoch)
                    if not self.debug:
                        log_dict.update({f'loss_q': loss_q})
                        log_dict.update({f'loss_pi': loss_pi})
                        log_dict.update(loss_info)
                
            # End of epoch handling
            if (t+1-start_step) % self.steps_per_epoch == 0:
                epoch = (t+1-start_step) // self.steps_per_epoch

                print(f"time for epoch {epoch} = {time.time()-start_epoch}")
                start_epoch = time.time()


                if self.train_cfg.schedule_gamma:
                    self.gamma = self.gamma_list[min(epoch, self.epochs-1)]
                if self.train_cfg.schedule_noise:
                    self.act_noise = self.act_noise*self.noise_decay

                # Test the performance of the deterministic version of the agent.
                print(f"Testing at epoch: {epoch}")

                start = time.time()
                test_info = self.test_agent(epoch)
                print(f"time for testing: {time.time()-start}")

                avg_test_return = test_info['Average_return']
                success_rate = test_info['success_rate']

                # Save model
                if (epoch % self.model_save_freq == 0) or (epoch == self.epochs):
                    print(f"Saving model at epoch: {epoch}")
                    save_file_name = os.path.join(self.modelFolder, f"step_{t}_test_return_{avg_test_return:.2f}_succRate_{success_rate:.2f}.pth")
                    status = self.topk_logger.push(save_file_name, success_rate)
                    if status:
                        start = time.time()
                        torch.save(
                            obj={
                                "state_dict": self.state_dict(),
                                "env_name": self.env_name,
                                "train_cfg": self.train_cfg,
                                "env_cfg": self.env_cfg,
                                "optimizer_state": self.ac.optimizer.state_dict(),
                                "epoch": epoch,
                            },
                            f=save_file_name,
                        )
                        if not self.debug:
                            wandb.save(save_file_name, base_path=os.path.join(self.modelFolder, '..'))
                        print(f'Saving time at epoch {epoch} = {time.time()-start}')
                    if not self.debug:
                        log_dict.update({f'Epoch': epoch})
                        log_dict.update({f'gamma': self.gamma})
                        log_dict.update({f'act_noise': self.act_noise})
                        log_dict.update({f'Time': time.time()-start_time})
                        log_dict.update({'optim/lr': self.ac.optimizer.param_groups[0]['lr']})
                        log_dict.update(test_info)
            if not self.debug:
                wandb.log(log_dict, step=t)
                    
        # Eval best checkpoint so far
        ckpt = torch.load(self.topk_logger.best_ckpt(), map_location=self.device)
        self.eval(ckpt, self.eval_cfg, debug=False) # uploading stuff to wandb
    
    def get_value_fn(self):
        # v = torch.zeros(self.test_env.grid_x_flat.shape[:-1]).to(self.device)
        with torch.no_grad():
            v = self.ac_targ.q(
                self.test_env.grid_x_flat, 
                self.ac_targ.pi(self.test_env.grid_x_flat))
            v = v.reshape(*self.test_env.grid_x.shape[:-1])
        return v

    def do_visualization_rollouts(self, num_visualization_rollouts=20, save_rollout_gifs=False):
        trajs = []
        if self.test_env.visual_initial_states is not None:
            num_visualization_rollouts = len(self.test_env.visual_initial_states)

        for idx in range(num_visualization_rollouts):
            init_s = None
            if self.test_env.visual_initial_states is not None:
                init_s = self.test_env.visual_initial_states[idx]

            rollout, at_all, imgs = [], [], []
            o, d, ep_ret, ep_len = self.test_env.reset(start=init_s), False, 0, 0
            rollout.append(o)

            with torch.no_grad():
                # for k, v in o.items():
                #     o[k] = torch.as_tensor(v, dtype=torch.float32).to(self.device)
                
                o = self.replay_buffer.get_batch_from_obs(o)
                outputs = self.ac_targ.value(o)

            # pred_v = self.ac_targ.q(
            #     torch.from_numpy(o).float().to(self.device), 
            #     self.ac_targ.pi(torch.from_numpy(o).float().to(self.device))).detach().cpu().numpy()
            # pred_success = pred_v > 0.

            pred_success = outputs['q_policy'].detach().cpu().numpy() > 0.

            gt_success = False

            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                at = outputs['action'].detach().cpu().numpy()
                o, r, d, _ = self.test_env.step(at)
                # fail, _ = self.test_env.check_failure(o.reshape(1,self.test_env.n))
                # succ, _ = self.test_env.check_success(o.reshape(1,self.test_env.n))
                # gt_success = np.logical_or(np.logical_and(not fail[0], succ[0]), gt_success)

                gt_success = self.test_env.failure_mode=='success'
                
                at_all.append(at)
                rollout.append(o)
                if save_rollout_gifs:
                    self.test_env.renderer.update_scene(self.test_env.data, camera=self.test_env.front_cam_name) 
                    img = self.test_env.renderer.render()
                    imgs.append(img)
                ep_len += 1
            trajs.append(np.array(rollout))
            if save_rollout_gifs:
                file_name = f'eval_traj_{self.mode}_{len(trajs)}_pred_succ_{pred_success}_gt_succ_{gt_success}_{self.test_env.failure_mode}'
                imageio.mimsave(os.path.join(self.figureFolder, f'{file_name}.gif'), 
                    imgs, duration=ep_len*self.test_env.dt)
                # self.test_env.plot_trajectory(np.stack(rollout,axis=0), np.stack(at_all,axis=0), os.path.join(self.figureFolder, f'{file_name}.png'))
        return trajs

    def eval(self, ckpt=None, eval_cfg=None, debug=True):
        if ckpt is not None:
            self.load_state_dict(ckpt["state_dict"])
        if eval_cfg is None:
            eval_cfg = self.eval_cfg

        self.env.epoch = self.epochs-1
        self.test_env.epoch = self.epochs-1

        set_seed(eval_cfg.seed)
        trajs = self.do_visualization_rollouts(num_visualization_rollouts=eval_cfg.num_visualization_rollouts, save_rollout_gifs=eval_cfg.save_rollout_gifs)

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

            pred_value_fn_flat = self.ac_targ.q(
                GT_grid_flat, 
                self.ac_targ.pi(GT_grid_flat)).detach().cpu().numpy()
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