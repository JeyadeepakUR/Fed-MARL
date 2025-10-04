"""
Demo script showcasing integration of urban environment with QMIX training.
"""

import os
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from core.envs.urban_grid_env import UrbanGridEnv
from core.agents.qmix_model import QMIX
from core.agents.policy_runner import PolicyRunner
from core.training.train_central import train_qmix

def visualize_episode(env, model, device, max_steps=200):
    """Run and visualize one episode with the trained model."""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    while not done and step < max_steps:
        # Get actions from model
        with torch.no_grad():
            obs_tensors = {
                i: torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                for i in range(env.num_drones)
            }
            q_values = model.get_q_values(obs_tensors)
            actions = {i: q_values[i].argmax(dim=1).item() for i in range(env.num_drones)}
        
        # Execute actions
        actions_array = np.array([actions[i] for i in range(env.num_drones)])
        obs, reward, terminated, truncated, info = env.step(actions_array)
        total_reward += reward
        done = terminated or truncated
        
        # Render
        env.render()
        step += 1
        
        # Print metrics
        print(f"\rStep {step} | Reward: {reward:.2f} | Coverage: {info['coverage']:.2%} | "
              f"Collisions: {info['collisions']} | No-fly violations: {info['no_fly_violations']}", 
              end="")
    
    print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
    return total_reward, info['coverage']

def main():
    # Load configs
    config_path = "configs/base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/urban_swarm_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create environment
    env = UrbanGridEnv(
        num_drones=10,  # Start with 10 drones
        grid_size=[2000, 2000, 500],  # 2km x 2km area
        cell_size=10.0,  # 10m resolution
        building_density=0.2,
        enable_weather=True,
        enable_comm_constraints=True,
        render_mode="human"
    )
    
    # Train model
    print("\nStarting QMIX training...")
    model_path = train_qmix(config_path)
    
    # Load best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    num_agents = env.num_drones
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim = obs_dim * num_agents
    
    model = QMIX(
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        mixing_embed_dim=config['mixing_embed_dim'],
        hypernet_embed=config['hypernet_embed_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate and visualize
    print("\nRunning visualization with trained model...")
    eval_episodes = 5
    returns = []
    coverages = []
    
    for episode in range(eval_episodes):
        print(f"\nEpisode {episode + 1}/{eval_episodes}")
        episode_return, coverage = visualize_episode(env, model, device)
        returns.append(episode_return)
        coverages.append(coverage)
    
    # Print final metrics
    print("\nEvaluation Results:")
    print(f"Average Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Average Coverage: {np.mean(coverages):.2%} ± {np.std(coverages):.2%}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(returns)
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    
    plt.subplot(1, 2, 2)
    plt.plot(coverages)
    plt.title("Coverage Rates")
    plt.xlabel("Episode")
    plt.ylabel("Coverage")
    
    plt.tight_layout()
    plt.savefig(exp_dir / "evaluation_results.png")
    plt.close()
    
    # Save metrics
    metrics = {
        'returns': returns,
        'coverages': coverages,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_coverage': np.mean(coverages),
        'std_coverage': np.std(coverages)
    }
    
    with open(exp_dir / "evaluation_metrics.yaml", 'w') as f:
        yaml.dump(metrics, f)
    
    env.close()
    print(f"\nExperiment completed. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()