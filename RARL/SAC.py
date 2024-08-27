from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import AdamW
import gym
import time, os, json
from tqdm import trange
import wandb

from .utils import count_vars, set_seed, TopKLogger, ReplayBuffer, calc_false_pos_neg_rate
from timm.scheduler.scheduler_factory import create_scheduler
from .SAC_core import MLPActorCritic

class SAC(torch.nn.Module):
    
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

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
        self.alpha = train_cfg.alpha
        self.polyak = train_cfg.polyak
        self.steps_per_epoch = train_cfg.steps_per_epoch
        self.epochs = train_cfg.epochs
        self.max_ep_len = train_cfg.max_ep_len
        self.update_after = train_cfg.update_after
        self.update_every = train_cfg.update_every
        self.num_test_episodes = train_cfg.num_test_episodes

        
        set_seed(train_cfg.seed)
        # Gamma scheduler
        self.gamma = train_cfg.gamma
        if train_cfg.schedule_gamma:
            self.gamma_list = np.ones(self.epochs)
            self.gamma_list[:train_cfg.gamma_warmup_epochs] = np.linspace(self.gamma, 0.9999, train_cfg.gamma_warmup_epochs)

        self.outFolder = outFolder
        self.debug = debug

        self.env = gym.make(env_name, device=device, cfg=env_cfg)
        self.test_env = gym.make(env_name, device=device, cfg=env_cfg)

        self.figureFolder = os.path.join(outFolder, 'figure')
        os.makedirs(self.figureFolder, exist_ok=True)
        self.env.plot_env(save_dir=self.figureFolder)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, 
            self.env.action_space,
            device,
            **train_cfg.ac_kwargs
        ).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=int(train_cfg.replay_size), device=device)

        # Set up optimizers for policy and q-function
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

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)


    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        lx, gx = data['lx'], data['gx']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            if self.mode == 'RA':
                backup = torch.zeros(q_pi_targ.shape).float().to(self.device)
                done = d > 0.
                not_done = torch.logical_not(done)

                q_pi_targ_ent = q_pi_targ - self.alpha * logp_a2

                non_terminal = torch.min(
                    gx[not_done],
                    torch.max(lx[not_done], q_pi_targ_ent[not_done]),
                )
                terminal = torch.min(lx, gx)
                # normal state
                backup[not_done] = non_terminal * self.gamma + terminal[not_done] * (1 - self.gamma)
                backup[done] = terminal[done]
            else:
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data, epoch):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.q_scheduler.step(epoch)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.pi_scheduler.step(epoch)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q.item(), loss_pi.item(), q_info, pi_info
    
    
    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                      deterministic)

    def test_agent(self, epoch):
        avg_test_return, avg_test_ep_len = 0. , 0.
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            avg_test_return += ep_ret
            avg_test_ep_len += ep_len
        avg_test_return = avg_test_return/self.num_test_episodes
        avg_test_ep_len = avg_test_ep_len/self.num_test_episodes
        
        
        # Rollout trajs for visualization
        trajs = []
        for init_s in self.test_env.visual_initial_states:
            rollout = []
            o, d, ep_ret, ep_len = self.test_env.reset(init_s), False, 0, 0
            rollout.append(o)
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                rollout.append(o)
                ep_len += 1
            trajs.append(np.array(rollout))

        # Plot learned value function and overlay trajs on it
        if (epoch % self.train_cfg.plot_save_freq == 0):
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

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs + start_step
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_num = 0

        # Main loop: collect experience in env and update/log each epoch
        epoch=0
        for t in trange(start_step, total_steps):

            # torch.cuda.empty_cache()
            if not self.debug:
                log_dict = {}
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.train_cfg.start_steps:
                a = self.get_action(o)
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
                    batch = self.replay_buffer.sample_batch(self.train_cfg.batch_size)
                    loss_q, loss_pi, q_info, pi_info = self.update(batch, epoch)
                    if not self.debug:
                        log_dict.update({f'loss_q': loss_q})
                        log_dict.update({f'loss_pi': loss_pi})
                        log_dict.update(q_info)
                        log_dict.update(pi_info)

            # End of epoch handling
            if (t+1-start_step) % self.steps_per_epoch == 0:
                epoch = (t+1-start_step) // self.steps_per_epoch
                
                if self.train_cfg.schedule_gamma:
                    self.gamma = self.gamma_list[min(epoch, self.epochs-1)]

                # Test the performance of the deterministic version of the agent.
                print(f"Testing at epoch: {epoch}")
                avg_test_return, avg_test_ep_len = self.test_agent(epoch)

                # Save model
                if (epoch % self.train_cfg.model_save_freq == 0) or (epoch == self.epochs):
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
                        log_dict.update({f'Avg Test Return': avg_test_return})
                        log_dict.update({f'Avg Test Episode len': avg_test_ep_len})
                        log_dict.update({f'Time': time.time()-start_time})
                        log_dict.update({'optim/q_lr': self.q_optimizer.param_groups[0]['lr']})
                        log_dict.update({'optim/pi_lr': self.pi_optimizer.param_groups[0]['lr']})

            if not self.debug:
                wandb.log(log_dict, step=t)

        # Eval best checkpoint so far
        ckpt = torch.load(self.topk_logger.best_ckpt(), map_location=self.device)
        self.eval(ckpt, debug=False) # uploading stuff to wandb
    
    def get_value_fn(self):
        # v = torch.zeros(self.test_env.grid_x_flat.shape[:-1]).to(self.device)
        with torch.no_grad():
            a, logp_pi = self.ac_targ.pi(self.test_env.grid_x_flat)
            v1 = self.ac_targ.q1(self.test_env.grid_x_flat, a)
            v2 = self.ac_targ.q2(self.test_env.grid_x_flat, a)

            v = torch.min(v1, v2).reshape(*self.test_env.grid_x.shape[:-1])
        return v
    
    def eval(self, ckpt=None, debug=True):
        if ckpt is not None:
            self.load_state_dict(ckpt["state_dict"])
        GT_value_fn, GT_grid_x, GT_target_T, GT_obstacle_T = self.test_env.get_GT_value_fn()
        GT_value_fn_flat = GT_value_fn.flatten()
        GT_grid_flat = torch.from_numpy(GT_grid_x.reshape(-1, GT_grid_x.shape[-1])).float().to(self.device)

        a, logp_pi = self.ac_targ.pi(GT_grid_flat)
        v1 = self.ac_targ.q1(GT_grid_flat, a)
        v2 = self.ac_targ.q2(GT_grid_flat, a)
        pred_value_fn_flat = torch.min(v1, v2).detach().cpu().numpy()

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
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
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
