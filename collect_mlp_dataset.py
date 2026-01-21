#!/usr/bin/env python3
"""
采集 MLP Policy 执行轨迹的数据集
用于后续 Transformer 监督学习
"""
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse


def collect_trajectories(
    runner,
    num_episodes: int = 1000,
    save_path: str = "./mlp_trajectories.pkl",
    max_steps_per_episode: int = None,
):
    """
    使用训练好的 MLP policy 采集轨迹数据
    
    Args:
        runner: OnPolicyRunnerMM 实例（已加载 MLP checkpoint）
        num_episodes: 采集的 episode 数量
        save_path: 保存路径
        max_steps_per_episode: 每个 episode 最大步数（None 则使用环境默认）
    
    Returns:
        dataset: {
            'observations': List[np.ndarray],  # 每个 episode 的 obs 序列 [T, obs_dim]
            'ref_observations': List[Tuple[np.ndarray, np.ndarray]],  # (ref_obs, ref_mask) 序列 [T, ref_seq, ref_dim]
            'actions': List[np.ndarray],  # 每个 episode 的 action 序列 [T, act_dim]
            'rewards': List[np.ndarray],  # 每个 episode 的 reward 序列 [T]
            'dones': List[np.ndarray],  # 每个 episode 的 done 标志 [T]
        }
    """
    
    env = runner.env
    policy = runner.get_inference_policy(device=runner.device)
    
    # 重置环境
    env.reset()
    obs, extras = env.get_observations()
    ref_obs_tuple, ref_extras = env.get_reference_observations()
    
    obs = obs.to(runner.device)
    if ref_obs_tuple is not None:
        ref_obs_tuple = tuple(ref_obs.to(runner.device) for ref_obs in ref_obs_tuple)
    
    # 数据存储
    dataset = {
        'observations': [],
        'ref_observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'episode_lengths': [],
        'episode_rewards': [],
    }
    
    # 当前 episode 的缓存
    current_obs = []
    current_ref_obs = []
    current_actions = []
    current_rewards = []
    current_dones = []
    
    num_envs = env.num_envs
    episode_counts = torch.zeros(num_envs, dtype=torch.long, device=runner.device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=runner.device)
    episode_rewards_sum = torch.zeros(num_envs, dtype=torch.float, device=runner.device)
    
    max_steps = max_steps_per_episode if max_steps_per_episode else env.max_episode_length
    
    print(f"开始采集 {num_episodes} 个 episodes 的数据...")
    print(f"并行环境数: {num_envs}")
    print(f"每个 episode 最大步数: {max_steps}")
    
    pbar = tqdm(total=num_episodes, desc="采集进度")
    
    with torch.no_grad():
        step_count = 0
        while episode_counts.sum() < num_episodes:
            # 获取 action
            if runner.empirical_normalization:
                obs_norm = runner.obs_normalizer(obs)
                if ref_obs_tuple is not None and runner.ref_obs_normalizer is not None:
                    ref_obs_norm = (runner.ref_obs_normalizer(ref_obs_tuple[0]), ref_obs_tuple[1])
                else:
                    ref_obs_norm = ref_obs_tuple
                actions = policy(obs_norm, ref_obs_norm)
            else:
                actions = policy(obs, ref_obs_tuple)
            
            # 存储当前状态
            current_obs.append(obs.cpu().numpy())
            if ref_obs_tuple is not None:
                current_ref_obs.append((ref_obs_tuple[0].cpu().numpy(), ref_obs_tuple[1].cpu().numpy()))
            else:
                current_ref_obs.append(None)
            current_actions.append(actions.cpu().numpy())
            
            # 执行 action
            obs, ref_obs_tuple, rewards, dones, infos = env.step(actions.to(env.device))
            
            # 移动到正确的设备
            obs = obs.to(runner.device)
            if ref_obs_tuple is not None:
                ref_obs_tuple = tuple(ref_obs.to(runner.device) for ref_obs in ref_obs_tuple)
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            
            current_rewards.append(rewards.cpu().numpy())
            current_dones.append(dones.cpu().numpy())
            
            episode_steps += 1
            episode_rewards_sum += rewards
            
            # 检查哪些环境 episode 结束
            done_indices = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
            
            if len(done_indices) > 0:
                for env_idx in done_indices:
                    env_idx = env_idx.item()
                    
                    # 只保存还没达到目标数量的 episode
                    if episode_counts[env_idx] < num_episodes // num_envs + 1:
                        # 提取该环境的完整轨迹
                        ep_len = episode_steps[env_idx].item()
                        
                        obs_traj = np.stack([step[env_idx] for step in current_obs[-ep_len:]], axis=0)
                        actions_traj = np.stack([step[env_idx] for step in current_actions[-ep_len:]], axis=0)
                        rewards_traj = np.stack([step[env_idx] for step in current_rewards[-ep_len:]], axis=0)
                        dones_traj = np.stack([step[env_idx] for step in current_dones[-ep_len:]], axis=0)
                        
                        if current_ref_obs[-1] is not None:
                            ref_obs_traj = (
                                np.stack([step[0][env_idx] for step in current_ref_obs[-ep_len:]], axis=0),
                                np.stack([step[1][env_idx] for step in current_ref_obs[-ep_len:]], axis=0)
                            )
                        else:
                            ref_obs_traj = None
                        
                        dataset['observations'].append(obs_traj)
                        dataset['ref_observations'].append(ref_obs_traj)
                        dataset['actions'].append(actions_traj)
                        dataset['rewards'].append(rewards_traj)
                        dataset['dones'].append(dones_traj)
                        dataset['episode_lengths'].append(ep_len)
                        dataset['episode_rewards'].append(episode_rewards_sum[env_idx].item())
                        
                        episode_counts[env_idx] += 1
                        pbar.update(1)
                    
                    # 重置该环境的计数
                    episode_steps[env_idx] = 0
                    episode_rewards_sum[env_idx] = 0
            
            step_count += 1
            
            # 防止无限循环
            if step_count > num_episodes * max_steps * 2:
                print("警告: 达到最大步数限制，提前结束采集")
                break
    
    pbar.close()
    
    # 统计信息
    actual_episodes = len(dataset['observations'])
    print(f"\n采集完成!")
    print(f"实际采集 episodes: {actual_episodes}")
    print(f"平均 episode 长度: {np.mean(dataset['episode_lengths']):.2f} ± {np.std(dataset['episode_lengths']):.2f}")
    print(f"平均 episode reward: {np.mean(dataset['episode_rewards']):.2f} ± {np.std(dataset['episode_rewards']):.2f}")
    print(f"Episode 长度范围: [{np.min(dataset['episode_lengths'])}, {np.max(dataset['episode_lengths'])}]")
    
    # 保存数据
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n数据集已保存到: {save_path}")
    print(f"文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    return dataset


def load_dataset(path: str):
    """加载采集的数据集"""
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def analyze_dataset(dataset):
    """分析数据集统计信息"""
    print("=" * 80)
    print("数据集分析")
    print("=" * 80)
    
    num_episodes = len(dataset['observations'])
    total_steps = sum(dataset['episode_lengths'])
    
    print(f"Episodes 数量: {num_episodes}")
    print(f"总步数: {total_steps}")
    print(f"平均 episode 长度: {np.mean(dataset['episode_lengths']):.2f}")
    print(f"平均 episode reward: {np.mean(dataset['episode_rewards']):.2f}")
    
    # 分析 observation 维度
    obs_shape = dataset['observations'][0].shape
    action_shape = dataset['actions'][0].shape
    
    print(f"\nObservation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    
    if dataset['ref_observations'][0] is not None:
        ref_obs_shape = dataset['ref_observations'][0][0].shape
        ref_mask_shape = dataset['ref_observations'][0][1].shape
        print(f"Reference observation shape: {ref_obs_shape}")
        print(f"Reference mask shape: {ref_mask_shape}")
    
    # 分析 action 分布
    all_actions = np.concatenate(dataset['actions'], axis=0)
    print(f"\nAction 统计:")
    print(f"  Mean: {np.mean(all_actions, axis=0)}")
    print(f"  Std: {np.std(all_actions, axis=0)}")
    print(f"  Min: {np.min(all_actions, axis=0)}")
    print(f"  Max: {np.max(all_actions, axis=0)}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="采集 MLP Policy 轨迹数据")
    parser.add_argument("--checkpoint", type=str, required=True, help="MLP checkpoint 路径")
    parser.add_argument("--num_episodes", type=int, default=1000, help="采集的 episode 数量")
    parser.add_argument("--save_path", type=str, default="./mlp_trajectories.pkl", help="保存路径")
    parser.add_argument("--max_steps", type=int, default=None, help="每个 episode 最大步数")
    parser.add_argument("--analyze", action="store_true", help="采集后分析数据集")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MLP Policy 数据采集工具")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"目标 Episodes: {args.num_episodes}")
    print(f"保存路径: {args.save_path}")
    print("=" * 80)
    
    # 注意: 这里需要你根据实际情况创建 runner 和加载 checkpoint
    # 示例代码:
    # from rsl_rl.runners import OnPolicyRunnerMM
    # runner = OnPolicyRunnerMM(env, train_cfg, log_dir, device)
    # runner.load(args.checkpoint)
    # 
    # dataset = collect_trajectories(
    #     runner=runner,
    #     num_episodes=args.num_episodes,
    #     save_path=args.save_path,
    #     max_steps_per_episode=args.max_steps,
    # )
    # 
    # if args.analyze:
    #     analyze_dataset(dataset)
    
    print("\n⚠️  请在实际项目中配置 runner 和环境后使用此脚本")
    print("参考上面的注释代码进行集成")
