"""
Module 2 Test Script: MARL Core (QMIX) Verification

Tests Module 2 implementation against success criteria:
- Train QMIX on small swarm (10 drones)
- Achieve <10% collision rate in 20-drone simulation
- Average reward >0.5
- Model saves correctly
- Integration with Module 1 environment works

Usage:
    python tests/test_module2.py --quick  # Quick test (10 episodes)
    python tests/test_module2.py --full   # Full test (100 episodes)
"""

import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.envs.urban_grid_env import UrbanGridEnv
from core.agents.qmix_model import QMIX, ReplayBuffer
from core.agents.policy_runner import PolicyRunner


class Module2Tester:
    """Comprehensive tester for Module 2: MARL Core."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
    
    def test_qmix_architecture(self):
        """Test 1: Verify QMIX architecture is correctly implemented."""
        print("\n" + "="*60)
        print("TEST 1: QMIX Architecture")
        print("="*60)
        
        try:
            # Create dummy environment to get dimensions
            env = UrbanGridEnv(num_drones=10, render_mode=None)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            num_agents = env.num_drones
            state_dim = obs_dim * num_agents
            
            print(f"✓ Environment dimensions:")
            print(f"  - Observation dim: {obs_dim}")
            print(f"  - Action dim: {action_dim}")
            print(f"  - Number of agents: {num_agents}")
            print(f"  - State dim: {state_dim}")
            
            # Create QMIX model
            model = QMIX(
                num_agents=num_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=128,
                mixing_embed_dim=32,
                hypernet_embed=64,
                num_layers=2
            ).to(self.device)
            
            print(f"\n✓ QMIX model created successfully")
            print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass
            observations, _ = env.reset()
            obs_tensors = {
                i: torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                for i in range(num_agents)
            }
            state = torch.FloatTensor(np.concatenate([observations[i] for i in range(num_agents)])).unsqueeze(0).to(self.device)
            
            q_tot, agent_qs = model(obs_tensors, state)
            
            print(f"\n✓ Forward pass successful:")
            print(f"  - Q_tot shape: {q_tot.shape}")
            print(f"  - Agent Q-values: {len(agent_qs)} agents")
            
            # Test action selection
            q_values = model.get_q_values(obs_tensors)
            actions = {i: q_values[i].argmax(dim=1).item() for i in range(num_agents)}
            
            print(f"\n✓ Action selection works:")
            print(f"  - Sample actions: {list(actions.values())[:5]}")
            
            env.close()
            self.results['test_architecture'] = True
            print("\n✓ TEST 1 PASSED")
            
        except Exception as e:
            print(f"\n✗ TEST 1 FAILED: {str(e)}")
            self.results['test_architecture'] = False
            raise
    
    def test_replay_buffer(self):
        """Test 2: Verify replay buffer functionality."""
        print("\n" + "="*60)
        print("TEST 2: Replay Buffer")
        print("="*60)
        
        try:
            # Create buffer
            buffer = ReplayBuffer(
                capacity=1000,
                num_agents=10,
                obs_dim=100,
                state_dim=1000
            )
            
            print(f"✓ Replay buffer created (capacity: 1000)")
            
            # Add experiences
            for _ in range(50):
                obs = {i: np.random.randn(100) for i in range(10)}
                actions = {i: np.random.randint(0, 5) for i in range(10)}
                reward = np.random.randn()
                next_obs = {i: np.random.randn(100) for i in range(10)}
                state = np.random.randn(1000)
                next_state = np.random.randn(1000)
                done = False
                
                buffer.push(obs, actions, reward, next_obs, state, next_state, done)
            
            print(f"✓ Added 50 experiences, buffer size: {len(buffer)}")
            
            # Sample batch
            batch = buffer.sample(32)
            obs_batch, act_batch, rew_batch, next_obs_batch, state_batch, next_state_batch, done_batch = batch
            
            print(f"\n✓ Sampled batch of 32:")
            print(f"  - Observation batch: {obs_batch[0].shape}")
            print(f"  - Actions batch: {act_batch[0].shape}")
            print(f"  - Rewards batch: {rew_batch.shape}")
            
            self.results['test_replay_buffer'] = True
            print("\n✓ TEST 2 PASSED")
            
        except Exception as e:
            print(f"\n✗ TEST 2 FAILED: {str(e)}")
            self.results['test_replay_buffer'] = False
            raise
    
    def test_small_swarm_training(self, num_episodes=10):
        """Test 3: Train on small swarm (10 drones, quick test)."""
        print("\n" + "="*60)
        print(f"TEST 3: Small Swarm Training ({num_episodes} episodes)")
        print("="*60)
        
        try:
            # Create environment with 10 drones
            env = UrbanGridEnv(
                num_drones=10,
                grid_size=[2000, 2000],  # Smaller grid for faster training
                obstacle_density=0.15,
                enable_no_fly_zones=False,  # Disable for quick test
                enable_comm_constraints=True,
                enable_randomization=False,  # Disable for reproducibility
                render_mode=None
            )
            
            # Get dimensions
            num_agents = env.num_drones
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            state_dim = obs_dim * num_agents
            
            print(f"✓ Environment created: {num_agents} drones")
            
            # Create model
            model = QMIX(
                num_agents=num_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=128,
                mixing_embed_dim=32,
                hypernet_embed=64,
                num_layers=2
            ).to(self.device)
            
            target_model = QMIX(
                num_agents=num_agents,
                obs_dim=obs_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=128,
                mixing_embed_dim=32,
                hypernet_embed=64,
                num_layers=2
            ).to(self.device)
            
            target_model.load_state_dict(model.state_dict())
            target_model.eval()
            
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            # Replay buffer
            replay_buffer = ReplayBuffer(
                capacity=5000,
                num_agents=num_agents,
                obs_dim=obs_dim,
                state_dim=state_dim
            )
            
            # Training parameters
            batch_size = 32
            gamma = 0.99
            epsilon = 1.0
            epsilon_decay = 0.95
            epsilon_min = 0.1
            
            # Training loop
            episode_rewards = []
            episode_collisions = []
            episode_coverages = []
            
            print(f"\n✓ Starting training...")
            
            for episode in tqdm(range(num_episodes), desc="Training"):
                observations, _ = env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                
                while not done and steps < 200:  # Limit steps for quick test
                    # Select actions
                    actions = {}
                    with torch.no_grad():
                        obs_tensors = {
                            i: torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                            for i in range(num_agents)
                        }
                        q_values = model.get_q_values(obs_tensors)
                        
                        for i in range(num_agents):
                            if np.random.random() < epsilon:
                                actions[i] = np.random.randint(action_dim)
                            else:
                                actions[i] = q_values[i].argmax(dim=1).item()
                    
                    # Execute actions
                    actions_array = np.array([actions[i] for i in range(num_agents)])
                    next_observations, reward, terminated, truncated, info = env.step(actions_array)
                    done = terminated or truncated
                    
                    # Create state
                    state = np.concatenate([observations[i] for i in range(num_agents)])
                    next_state = np.concatenate([next_observations[i] for i in range(num_agents)])
                    
                    # Store experience
                    replay_buffer.push(
                        observations, actions, reward, next_observations,
                        state, next_state, done
                    )
                    
                    episode_reward += reward
                    observations = next_observations
                    steps += 1
                    
                    # Train
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size)
                        obs_batch, act_batch, rew_batch, next_obs_batch, state_batch, next_state_batch, done_batch = batch
                        
                        # Move to device
                        obs_batch = {i: obs_batch[i].to(self.device) for i in range(num_agents)}
                        act_batch = {i: act_batch[i].to(self.device) for i in range(num_agents)}
                        rew_batch = rew_batch.to(self.device)
                        next_obs_batch = {i: next_obs_batch[i].to(self.device) for i in range(num_agents)}
                        state_batch = state_batch.to(self.device)
                        next_state_batch = next_state_batch.to(self.device)
                        done_batch = done_batch.to(self.device)
                        
                        # Current Q-values
                        q_tot, _ = model(obs_batch, state_batch, act_batch)
                        
                        # Target Q-values
                        with torch.no_grad():
                            target_q_tot, _ = target_model(next_obs_batch, next_state_batch)
                            target_q = rew_batch + gamma * target_q_tot.squeeze() * (~done_batch)
                        
                        # Loss
                        loss = torch.nn.functional.mse_loss(q_tot.squeeze(), target_q)
                        
                        # Optimize
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                
                # Decay epsilon
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                
                # Log metrics
                episode_rewards.append(episode_reward)
                episode_collisions.append(info.get('collisions', 0))
                episode_coverages.append(info.get('coverage', 0))
                
                # Update target network
                if episode % 5 == 0:
                    target_model.load_state_dict(model.state_dict())
            
            env.close()
            
            # Print results
            avg_reward = np.mean(episode_rewards[-5:])  # Last 5 episodes
            avg_collisions = np.mean(episode_collisions[-5:])
            avg_coverage = np.mean(episode_coverages[-5:])
            collision_rate = avg_collisions / num_agents
            
            print(f"\n✓ Training complete!")
            print(f"  - Average reward (last 5 eps): {avg_reward:.2f}")
            print(f"  - Average collisions: {avg_collisions:.1f}")
            print(f"  - Collision rate: {collision_rate:.1%}")
            print(f"  - Average coverage: {avg_coverage:.1%}")
            
            # Save results
            self.results['small_swarm_avg_reward'] = avg_reward
            self.results['small_swarm_collision_rate'] = collision_rate
            self.results['small_swarm_coverage'] = avg_coverage
            self.results['test_small_swarm_training'] = True
            
            # Plot training curves
            self._plot_training_curves(episode_rewards, episode_collisions, episode_coverages)
            
            print("\n✓ TEST 3 PASSED")
            
            return model
            
        except Exception as e:
            print(f"\n✗ TEST 3 FAILED: {str(e)}")
            self.results['test_small_swarm_training'] = False
            raise
    
    def test_20_drone_evaluation(self, model=None):
        """Test 4: Evaluate on 20-drone swarm (Module 2 success criteria)."""
        print("\n" + "="*60)
        print("TEST 4: 20-Drone Evaluation")
        print("="*60)
        
        try:
            # Create 20-drone environment
            env = UrbanGridEnv(
                num_drones=20,
                grid_size=[5000, 5000],
                obstacle_density=0.20,
                enable_no_fly_zones=True,
                enable_comm_constraints=True,
                enable_randomization=True,
                render_mode=None
            )
            
            print(f"✓ Environment created: 20 drones")
            
            if model is None:
                # Create fresh model if none provided
                num_agents = env.num_drones
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                state_dim = obs_dim * num_agents
                
                model = QMIX(
                    num_agents=num_agents,
                    obs_dim=obs_dim,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=128
                ).to(self.device)
            
            # Run evaluation episodes
            num_eval_episodes = 5
            eval_rewards = []
            eval_collisions = []
            eval_collision_rates = []
            eval_coverages = []
            
            print(f"\n✓ Running {num_eval_episodes} evaluation episodes...")
            
            for ep in tqdm(range(num_eval_episodes), desc="Evaluating"):
                observations, _ = env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                
                while not done and steps < 500:
                    # Select actions (greedy)
                    with torch.no_grad():
                        obs_tensors = {
                            i: torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                            for i in range(env.num_drones)
                        }
                        q_values = model.get_q_values(obs_tensors)
                        actions = {i: q_values[i].argmax(dim=1).item() for i in range(env.num_drones)}
                    
                    actions_array = np.array([actions[i] for i in range(env.num_drones)])
                    observations, reward, terminated, truncated, info = env.step(actions_array)
                    episode_reward += reward
                    done = terminated or truncated
                    steps += 1
                
                eval_rewards.append(episode_reward)
                eval_collisions.append(info.get('collisions', 0))
                collision_rate = info.get('collisions', 0) / 20.0  # 20 drones
                eval_collision_rates.append(collision_rate)
                eval_coverages.append(info.get('coverage', 0))
            
            env.close()
            
            # Calculate metrics
            avg_reward = np.mean(eval_rewards)
            avg_collision_rate = np.mean(eval_collision_rates)
            avg_coverage = np.mean(eval_coverages)
            
            print(f"\n✓ Evaluation Results:")
            print(f"  - Average reward: {avg_reward:.2f}")
            print(f"  - Average collision rate: {avg_collision_rate:.1%}")
            print(f"  - Average coverage: {avg_coverage:.1%}")
            
            # Check success criteria
            reward_criterion = avg_reward > 0.5
            collision_criterion = avg_collision_rate < 0.10  # <10%
            
            print(f"\n✓ Success Criteria:")
            print(f"  - Average reward >0.5: {'✓ PASS' if reward_criterion else '✗ FAIL'} ({avg_reward:.2f})")
            print(f"  - Collision rate <10%: {'✓ PASS' if collision_criterion else '✗ FAIL'} ({avg_collision_rate:.1%})")
            
            self.results['20_drone_avg_reward'] = avg_reward
            self.results['20_drone_collision_rate'] = avg_collision_rate
            self.results['20_drone_coverage'] = avg_coverage
            self.results['meets_reward_criterion'] = reward_criterion
            self.results['meets_collision_criterion'] = collision_criterion
            self.results['test_20_drone_evaluation'] = reward_criterion and collision_criterion
            
            if reward_criterion and collision_criterion:
                print("\n✓ TEST 4 PASSED - Module 2 Success Criteria Met!")
            else:
                print("\n⚠ TEST 4 COMPLETED - Some criteria not met (expected for untrained model)")
            
        except Exception as e:
            print(f"\n✗ TEST 4 FAILED: {str(e)}")
            self.results['test_20_drone_evaluation'] = False
            raise
    
    def test_model_save_load(self):
        """Test 5: Verify model can be saved and loaded."""
        print("\n" + "="*60)
        print("TEST 5: Model Save/Load")
        print("="*60)
        
        try:
            # Create model
            model = QMIX(
                num_agents=10,
                obs_dim=100,
                state_dim=1000,
                action_dim=5,
                hidden_dim=128
            ).to(self.device)
            
            # Save model
            save_path = Path("tests/test_model.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'num_agents': 10,
                    'obs_dim': 100,
                    'state_dim': 1000,
                    'action_dim': 5
                }
            }, save_path)
            
            print(f"✓ Model saved to {save_path}")
            
            # Load model
            checkpoint = torch.load(save_path, map_location=self.device)
            new_model = QMIX(
                num_agents=checkpoint['config']['num_agents'],
                obs_dim=checkpoint['config']['obs_dim'],
                state_dim=checkpoint['config']['state_dim'],
                action_dim=checkpoint['config']['action_dim'],
                hidden_dim=128
            ).to(self.device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✓ Model loaded successfully")
            
            # Verify weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                if not torch.allclose(p1, p2):
                    raise ValueError("Loaded weights don't match!")
            
            print(f"✓ Weights verified")
            
            # Cleanup
            save_path.unlink()
            
            self.results['test_model_save_load'] = True
            print("\n✓ TEST 5 PASSED")
            
        except Exception as e:
            print(f"\n✗ TEST 5 FAILED: {str(e)}")
            self.results['test_model_save_load'] = False
            raise
    
    def _plot_training_curves(self, rewards, collisions, coverages):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(rewards)
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)
        
        axes[1].plot(collisions)
        axes[1].set_title('Episode Collisions')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Collisions')
        axes[1].grid(True)
        
        axes[2].plot(coverages)
        axes[2].set_title('Episode Coverage')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Coverage %')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path("tests/module2_training_curves.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training curves saved to {save_path}")
        plt.close()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("MODULE 2 TEST SUMMARY")
        print("="*60)
        
        total_tests = 5
        passed_tests = sum(1 for v in self.results.values() if isinstance(v, bool) and v)
        
        print(f"\nTests Passed: {passed_tests}/{total_tests}")
        print(f"\nDetailed Results:")
        for key, value in self.results.items():
            if isinstance(value, bool):
                status = "✓ PASS" if value else "✗ FAIL"
                print(f"  {key}: {status}")
            else:
                print(f"  {key}: {value}")
        
        # Module 2 Success Criteria Check
        print(f"\n" + "="*60)
        print("MODULE 2 SUCCESS CRITERIA")
        print("="*60)
        
        if 'meets_reward_criterion' in self.results and 'meets_collision_criterion' in self.results:
            reward_ok = self.results['meets_reward_criterion']
            collision_ok = self.results['meets_collision_criterion']
            
            print(f"✓ Trained model achieves <10% collision rate: {'YES' if collision_ok else 'NO'}")
            print(f"✓ Average reward >0.5: {'YES' if reward_ok else 'NO'}")
            
            if reward_ok and collision_ok:
                print(f"\n{'='*60}")
                print("MODULE 2: COMPLETE ✓")
                print("="*60)
            else:
                print(f"\n⚠ Note: For full success criteria, train longer (100+ episodes)")
        
        return passed_tests == total_tests


def main():
    parser = argparse.ArgumentParser(description='Module 2: MARL Core Tests')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 episodes)')
    parser.add_argument('--full', action='store_true', help='Full test (100 episodes)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run tests on')
    args = parser.parse_args()
    
    # Determine number of episodes
    num_episodes = 100 if args.full else 10
    
    print("="*60)
    print("MODULE 2: MARL CORE (QMIX) - COMPREHENSIVE TEST")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Training episodes: {num_episodes}")
    print(f"Test mode: {'FULL' if args.full else 'QUICK'}")
    
    # Create tester
    tester = Module2Tester(device=args.device)
    
    # Run tests
    try:
        tester.test_qmix_architecture()
        tester.test_replay_buffer()
        model = tester.test_small_swarm_training(num_episodes=num_episodes)
        tester.test_20_drone_evaluation(model=model)
        tester.test_model_save_load()
        
    except Exception as e:
        print(f"\n✗ Testing stopped due to error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    success = tester.print_summary()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
