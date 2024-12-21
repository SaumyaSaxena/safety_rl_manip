import os
import numpy as np
import torch
from pathlib import Path
import logging
import wandb
import time, json
from omegaconf import OmegaConf
from tqdm import trange
import gym
import imageio

from safety_rl_manip.models.RARL.DDPG import DDPG

logger = logging.getLogger(__name__)


def rollout_episodes(agent1, agent2, test_env, num_episodes, device, eval_path, policy_type='switching', save_rollout_gifs=False):
    figureFolder = os.path.join(eval_path, 'figure')
    
    FP1, FN1, TP1, TN1, num_pred_success1, num_gt_success = 0, 0, 0, 0, 0, 0
    FP2, FN2, TP2, TN2, num_pred_success2 = 0, 0, 0, 0, 0

    avg_return, avg_ep_len = 0. , 0.
    for i in trange(num_episodes, desc="Testing"):
        imgs, rollout, at_all = [], [], []
        sample_state = test_env.sample_initial_state(1)[0]
        o, d, ep_ret, ep_len = test_env.reset(sample_state), False, 0, 0
        rollout.append(o)
        o1, d1, ep_ret1, ep_len1 = agent1.test_env.reset(sample_state), False, 0, 0

        sample_state2 = sample_state.copy()
        sample_state2[12:15] = sample_state[18:21]
        o2, d2, ep_ret2, ep_len2 = agent2.test_env.reset(sample_state2), False, 0, 0
        

        pred_v1 = agent1.ac_targ.q(
            torch.from_numpy(o1).float().to(device), 
            agent1.ac_targ.pi(torch.from_numpy(o1).float().to(device))).detach().cpu().numpy()
        pred_success1 = pred_v1 > 0.

        pred_v2 = agent2.ac_targ.q(
            torch.from_numpy(o2).float().to(device), 
            agent2.ac_targ.pi(torch.from_numpy(o2).float().to(device))).detach().cpu().numpy()
        pred_success2 = pred_v2 > 0.
        
        gt_success = False
        while not(d or (ep_len == agent1.max_ep_len)):
            if policy_type=='switching':
                if test_env.relevant_obj == 'block_obsA':
                    action = agent1.get_action(o1, 0)
                elif test_env.relevant_obj == 'block_obsB':
                    action = agent2.get_action(o2, 0)
                else:
                    raise NotImplementedError
            elif policy_type=='agent1':
                action = agent1.get_action(o1, 0)
            elif policy_type=='agent2':
                action = agent2.get_action(o2, 0)
            else:
                raise NotImplementedError
            if save_rollout_gifs:
                test_env.renderer.update_scene(test_env.data, camera='left_cap3') 
                img = test_env.renderer.render()
                imgs.append(img)
            o, r, d, _ = test_env.step(action)
            o1, r1, d1, _ = agent1.test_env.step(action)
            o2, r2, d2, _ = agent2.test_env.step(action)

            at_all.append(action)
            rollout.append(o)
            ep_ret += r
            fail, _ = test_env.check_failure(o.reshape(1,test_env.n))
            succ, _ = test_env.check_success(o.reshape(1,test_env.n))
            gt_success = np.logical_or(np.logical_and(not fail[0], succ[0]), gt_success)
            ep_len += 1

        if save_rollout_gifs:
            file_name = f'eval_traj_{i}_predSucc1_{pred_success1}_predSucc2_{pred_success2}_gt_succ_{gt_success}'
            imageio.mimsave(os.path.join(figureFolder, f'{file_name}.gif'), 
                imgs, duration=ep_len*test_env.dt)
            test_env.plot_trajectory(np.stack(rollout,axis=0), np.stack(at_all,axis=0), os.path.join(figureFolder, f'{file_name}.png'))
        
        avg_return += ep_ret
        avg_ep_len += ep_len
        num_pred_success1 += pred_success1
        num_pred_success2 += pred_success2
        num_gt_success += gt_success

        FP1 += np.sum(np.logical_and((gt_success == False), (pred_success1 == True)))
        FN1 += np.sum(np.logical_and((gt_success == True), (pred_success1 == False)))
        TP1 += np.sum(np.logical_and((gt_success == True), (pred_success1 == True)))
        TN1 += np.sum(np.logical_and((gt_success == False), (pred_success1 == False)))

        FP2 += np.sum(np.logical_and((gt_success == False), (pred_success2 == True)))
        FN2 += np.sum(np.logical_and((gt_success == True), (pred_success2 == False)))
        TP2 += np.sum(np.logical_and((gt_success == True), (pred_success2 == True)))
        TN2 += np.sum(np.logical_and((gt_success == False), (pred_success2 == False)))
    false_pos_rate1 = FP1/(FP1+TN1)
    false_neg_rate1 = FN1/(FN1+TP1)

    false_pos_rate2 = FP2/(FP2+TN2)
    false_neg_rate2 = FN2/(FN2+TP2)

    avg_return = avg_return/num_episodes
    avg_ep_len = avg_ep_len/num_episodes

    info = {
        'Average_return': avg_return,
        'Average_episode_len': avg_ep_len,
        'False_positive_rate1': false_pos_rate1,
        'False_negative_rate1': false_neg_rate1,
        'False_positive_rate2': false_pos_rate2,
        'False_negative_rate2': false_neg_rate2,
        'Total_num_episodes': float(num_episodes),
        'FP1': float(FP1),
        'FN1': float(FN1),
        'TP1': float(TP1),
        'TN1': float(TN1),
        'FP2': float(FP2),
        'FN2': float(FN2),
        'TP2': float(TP2),
        'TN2': float(TN2),
        'num_pred_success1': float(num_pred_success1),
        'num_pred_success2': float(num_pred_success2),
        'num_gt_success': float(num_gt_success),
    }
    return info


def main():
    cfg = OmegaConf.load('cfg/eval/eval_switching_rl.yaml')

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")

    eval_cfg = cfg.eval_cfg
    
    device = (
        torch.device("cuda", eval_cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    eval_path = Path('outputs/evals')
    model_path = os.path.join(eval_path, 'ckpts')

    # Load checkpoints
    ckpt_file1 = wandb.restore(
        str(eval_cfg.wandb_load.file1), 
        run_path=eval_cfg.wandb_load.run_path1,
        root=str(model_path), replace=True,
    )
    ckpt1 = torch.load(ckpt_file1.name, map_location=device)
    
    ckpt_file2 = wandb.restore(
        str(eval_cfg.wandb_load.file2), 
        run_path=eval_cfg.wandb_load.run_path2,
        root=str(model_path), replace=True,
    )
    ckpt2 = torch.load(ckpt_file2.name, map_location=device)

    env_name = ckpt1['env_name']
    mode = ckpt1['train_cfg']['mode']
    algo_name = ckpt1['train_cfg']['algo_name']

    time_str = time_str + '_' + ckpt1['train_cfg']['run_variant'] + '_' + ckpt2['train_cfg']['run_variant'] + '_' + eval_cfg['policy_type']

    eval_path = os.path.join(eval_path, f'{env_name}_{algo_name}_{mode}', time_str)

    agent1 = eval(algo_name)(
        env_name, device, train_cfg=ckpt1['train_cfg'], eval_cfg=eval_cfg,
        env_cfg=ckpt1['env_cfg'],
        outFolder=eval_path, debug=cfg.debug
    )
    agent1.load_state_dict(ckpt1["state_dict"])

    agent2 = eval(algo_name)(
        env_name, device, train_cfg=ckpt2['train_cfg'], eval_cfg=eval_cfg,
        env_cfg=ckpt2['env_cfg'],
        outFolder=eval_path, debug=cfg.debug
    )
    agent2.load_state_dict(ckpt2["state_dict"])

    env_cfg = ckpt2['env_cfg'].copy()
    env_cfg['block_obsA']['active'] = True
    env_cfg['block_obsB']['active'] = True
    test_env = gym.make(env_name, device=device, cfg=env_cfg)
    eval_results = rollout_episodes(
        agent1, 
        agent2, 
        test_env, 
        eval_cfg.num_eval_episodes, 
        device, 
        eval_path, 
        policy_type=eval_cfg.policy_type,
        save_rollout_gifs=False
    )

    fname = os.path.join(eval_path, 'results_rollouts.json')
    with open(fname, "w") as f:
        json.dump(eval_results, f, indent=4)

    false_pos_rate = eval_results['False_positive_rate2']
    false_neg_rate = eval_results['False_negative_rate2']
    print(f"Saving rollouts FPR={false_pos_rate}; FNR={false_neg_rate} results to: {str(fname)}")

    eval_results = rollout_episodes(
        agent1, 
        agent2, 
        test_env, 
        eval_cfg.num_rollout_gifs, 
        device, 
        eval_path, 
        policy_type=eval_cfg.policy_type,
        save_rollout_gifs=eval_cfg.save_rollout_gifs
    )


if __name__ == "__main__":
    main()