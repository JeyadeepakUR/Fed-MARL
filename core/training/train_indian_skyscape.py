"""
Training script for Indian Skyscape Environment
Optimized for drone swarm coordination in realistic Indian urban layout
"""

import sys
from pathlib import Path
import time
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.envs.indian_skyscape_env import IndianSkyscapeEnv
from core.agents.qmix_model import QMIXAgent


def train_indian_skyscape(
    n_episodes: int = 500,
    max_steps_per_episode: int = 300,
    num_drones: int = 10,
    checkpoint_dir: str = "indian_skyscape_checkpoints",
    verbose: bool = True
):
    """
    Train QMIX on Indian Skyscape Environment
    
    Args:
        n_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        num_drones: Number of drones in swarm
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print training progress
    """
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Indian Skyscape Environment
    env = IndianSkyscapeEnv(
        render_mode=None,
        num_drones=num_drones,
        grid_size=[5000, 5000, 300],
        cell_size=20.0,  # 20m resolution for easier navigation
        reward_structure={
            'coverage': 15.0,  # High coverage reward
            'collision_penalty': -1.0,  # Light collision penalty
            'no_fly_zone_penalty': -0.5,  # Very light no-fly penalty
            'exploration_reward': 3.0  # Reward for exploring new areas
        }
    )
    
    # Get environment info
    obs_dict, _ = env.reset()
    obs_dim = obs_dict[0].shape[0]
    action_dim = env.action_space.n
    
    print(f"\n{'='*80}")
    print(f"INDIAN SKYSCAPE QMIX TRAINING")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Number of agents (drones): {num_drones}")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Environment: Indian Skyscape (realistic urban layout)")
    print(f"  Training mode: Single-Step Updates")
    print(f"\nSuccess Criteria:")
    print(f"  Target collision rate: <5%")
    print(f"  Target average reward: >50")
    print(f"  Target coverage: >40%")
    print(f"{'='*80}\n")
    
    # Create QMIX Agent
    agent = QMIXAgent(
        n_agents=num_drones,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        mixing_hidden_dim=64,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        buffer_capacity=10000,
        batch_size=32,
        target_update_freq=100,
        grad_clip=5.0,
        use_replay_buffer=False,  # Single-step for faster learning
        device=None  # Auto-detect
    )
    
    print(f"[QMIXAgent] Initialized on {agent.device}")
    print("Starting training...\n")
    
    # Training metrics
    episode_rewards = []
    episode_coverages = []
    episode_collisions = []
    episode_explorations = []
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        obs_dict, _ = env.reset()
        episode_reward = 0
        episode_coverage = 0
        episode_collision = 0
        episode_exploration = 0
        
        for step in range(max_steps_per_episode):
            # Get actions from agent
            obs = [obs_dict[i] for i in range(num_drones)]
            actions = agent.get_actions(obs)
            
            # Step environment
            next_obs_dict, rewards, done, truncated, info = env.step(actions)
            next_obs = [next_obs_dict[i] for i in range(num_drones)]
            
            # Store experience and train
            agent.store_experience(obs, actions, rewards, next_obs, [done] * num_drones)
            loss = agent.train_step()
            
            # Update metrics
            episode_reward += np.mean(rewards)
            episode_coverage = info['coverage']
            episode_collision = info['collisions']
            episode_exploration = info['exploration']
            
            obs_dict = next_obs_dict
            
            if done or truncated:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_coverages.append(episode_coverage)
        episode_collisions.append(episode_collision)
        episode_explorations.append(episode_exploration)
        
        # Print progress
        if verbose and (episode + 1) % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_coverage = np.mean(episode_coverages[-25:])
            avg_collision = np.mean(episode_collisions[-25:])
            avg_exploration = np.mean(episode_explorations[-25:])
            
            print(f"Episode {episode + 1:3d}/{n_episodes} | "
                  f"Reward: {avg_reward:6.1f} | "
                  f"Coverage: {avg_coverage:5.1%} | "
                  f"Collisions: {avg_collision:4.1f} | "
                  f"Exploration: {avg_exploration:5.1%} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % 50 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"qmix_episode_{episode + 1}.pth"
            agent.save_model(str(checkpoint_path))
            
            # Evaluate performance
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluation at Episode {episode + 1}")
                print(f"{'='*60}")
                print(f"  Average Reward: {avg_reward:.2f} (target: >50)")
                print(f"  Coverage: {avg_coverage:.1%} (target: >40%)")
                print(f"  Collisions: {avg_collision:.1f} (target: <5)")
                print(f"  Exploration: {avg_exploration:.1%}")
                
                # Check success criteria
                success = (avg_reward > 50 and avg_coverage > 0.40 and avg_collision < 5)
                if success:
                    print(f"  ✓ SUCCESS CRITERIA MET!")
                else:
                    print(f"  ⚠ Still working towards success criteria")
                print(f"{'='*60}\n")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = {
        'avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
        'avg_coverage': np.mean(episode_coverages[-50:]) if len(episode_coverages) >= 50 else np.mean(episode_coverages),
        'avg_collisions': np.mean(episode_collisions[-50:]) if len(episode_collisions) >= 50 else np.mean(episode_collisions),
        'avg_exploration': np.mean(episode_explorations[-50:]) if len(episode_explorations) >= 50 else np.mean(episode_explorations),
        'training_time': training_time
    }
    
    # Save final model
    final_path = Path(checkpoint_dir) / "qmix_final.pth"
    agent.save_model(str(final_path))
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Final Performance:")
    print(f"  Average Reward: {final_metrics['avg_reward']:.2f} (target: >50)")
    print(f"  Coverage: {final_metrics['avg_coverage']:.1%} (target: >40%)")
    print(f"  Collisions: {final_metrics['avg_collisions']:.1f} (target: <5)")
    print(f"  Exploration: {final_metrics['avg_exploration']:.1%}")
    print(f"  Training Time: {training_time/60:.1f} minutes")
    
    # Check final success
    success = (final_metrics['avg_reward'] > 50 and 
              final_metrics['avg_coverage'] > 0.40 and 
              final_metrics['avg_collisions'] < 5)
    
    if success:
        print(f"\n🎉 SUCCESS! All criteria met!")
    else:
        print(f"\n⚠️  Training incomplete - consider more episodes or hyperparameter tuning")
    
    print(f"\n[QMIXAgent] Saved final checkpoint to {final_path}")
    print(f"{'='*80}\n")
    
    env.close()
    
    return agent, final_metrics


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    INDIAN SKYSCAPE QMIX TRAINING                            ║
    ║                    Drone Swarm Coordination in Indian Urban                  ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Train on Indian Skyscape
    agent, metrics = train_indian_skyscape(
        n_episodes=500,
        max_steps_per_episode=300,
        num_drones=10,
        checkpoint_dir="indian_skyscape_checkpoints",
        verbose=True
    )
    
    print(f"Training completed successfully!")
    print(f"Final model saved to: indian_skyscape_checkpoints/qmix_final.pth")
