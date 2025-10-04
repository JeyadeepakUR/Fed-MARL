"""
QMIX Testing and Validation Script

This script provides comprehensive testing utilities for the QMIX implementation.
Use this to validate your setup before running full training.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.envs.urban_grid_env import UrbanGridEnv
from core.agents.qmix_model import QMIXAgent


def test_environment():
    """Test 1: Verify environment is working correctly."""
    print("\n" + "="*80)
    print("TEST 1: Environment Validation")
    print("="*80)
    
    try:
        env = UrbanGridEnv(
            render_mode=None,
            num_drones=5,
            grid_size=[1000, 1000, 300],
            cell_size=10.0
        )
        
        obs_dict, info = env.reset()
        print("✓ Environment created successfully")
        print(f"  Number of drones: {env.num_drones}")
        print(f"  Observation shape: {obs_dict[0].shape}")
        print(f"  Action space: {env.action_space}")
        
        # Test one step
        actions = [env.action_space.sample() for _ in range(env.num_drones)]
        obs_dict, rewards, done, truncated, info = env.step(actions)
        
        print("✓ Environment step executed successfully")
        print(f"  Rewards shape: {len(rewards)}")
        print(f"  Info keys: {list(info.keys())}")
        
        env.close()
        print("\nTest 1: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}\n")
        return False


def test_agent_initialization():
    """Test 2: Verify QMIX agent initializes correctly."""
    print("\n" + "="*80)
    print("TEST 2: Agent Initialization")
    print("="*80)
    
    try:
        env = UrbanGridEnv(num_drones=5, grid_size=[1000, 1000, 300])
        obs_dict, _ = env.reset()
        obs_dim = obs_dict[0].shape[0]
        
        agent = QMIXAgent(
            n_agents=5,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=128,
            use_replay_buffer=False
        )
        
        print("✓ Agent initialized successfully")
        print(f"  Number of Q-networks: {len(agent.q_networks)}")
        print(f"  State dimension: {agent.state_dim}")
        print(f"  Epsilon: {agent.epsilon}")
        
        # Test action selection
        obs = [obs_dict[i] for i in range(5)]
        actions = [agent.get_action(obs[i], i) for i in range(5)]
        
        print("✓ Action selection working")
        print(f"  Sample actions: {actions}")
        
        env.close()
        print("\nTest 2: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_single_step_training():
    """Test 3: Verify single-step training works."""
    print("\n" + "="*80)
    print("TEST 3: Single-Step Training")
    print("="*80)
    
    try:
        env = UrbanGridEnv(num_drones=3, grid_size=[500, 500, 200])
        obs_dict, _ = env.reset()
        obs_dim = obs_dict[0].shape[0]
        
        agent = QMIXAgent(
            n_agents=3,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=64,
            use_replay_buffer=False
        )
        
        obs = [obs_dict[i] for i in range(3)]
        actions = [agent.get_action(obs[i], i) for i in range(3)]
        next_obs_dict, rewards, done, truncated, info = env.step(actions)
        next_obs = [next_obs_dict[i] for i in range(3)]
        dones = [done] * 3
        
        # Single training step
        loss = agent.train_step(obs, actions, rewards, next_obs, dones)
        
        print("✓ Single-step training executed")
        print(f"  Loss: {loss:.4f}")
        print(f"  Epsilon after update: {agent.epsilon:.4f}")
        
        env.close()
        print("\nTest 3: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer_training():
    """Test 4: Verify replay buffer training works."""
    print("\n" + "="*80)
    print("TEST 4: Replay Buffer Training")
    print("="*80)
    
    try:
        env = UrbanGridEnv(num_drones=3, grid_size=[500, 500, 200])
        obs_dict, _ = env.reset()
        obs_dim = obs_dict[0].shape[0]
        
        agent = QMIXAgent(
            n_agents=3,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=64,
            use_replay_buffer=True,
            batch_size=8,
            buffer_capacity=100
        )
        
        # Collect some experiences
        for step in range(20):
            obs = [obs_dict[i] for i in range(3)]
            actions = [agent.get_action(obs[i], i) for i in range(3)]
            next_obs_dict, rewards, done, truncated, info = env.step(actions)
            next_obs = [next_obs_dict[i] for i in range(3)]
            dones = [done] * 3
            
            agent.store_experience(obs, actions, rewards, next_obs, dones)
            obs_dict = next_obs_dict
            
            if done or truncated:
                obs_dict, _ = env.reset()
        
        print(f"✓ Collected {len(agent.replay_buffer)} experiences")
        
        # Train from buffer
        loss = agent.train_step()
        
        print("✓ Replay buffer training executed")
        print(f"  Loss: {loss:.4f}")
        print(f"  Buffer size: {len(agent.replay_buffer)}")
        
        env.close()
        print("\nTest 4: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_shape_consistency():
    """Test 5: Verify tensor shapes throughout forward pass."""
    print("\n" + "="*80)
    print("TEST 5: Shape Consistency")
    print("="*80)
    
    try:
        n_agents = 5
        obs_dim = 100
        action_dim = 7
        batch_size = 4
        
        agent = QMIXAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_replay_buffer=False
        )
        
        # Test Q-network forward pass
        test_obs = torch.randn(batch_size, obs_dim)
        q_values = agent.q_networks[0](test_obs)
        assert q_values.shape == (batch_size, action_dim), \
            f"Q-network output shape mismatch: {q_values.shape} vs expected ({batch_size}, {action_dim})"
        print(f"✓ Q-network shape: {test_obs.shape} -> {q_values.shape}")
        
        # Test mixer forward pass
        test_q_values = torch.randn(batch_size, n_agents)
        test_states = torch.randn(batch_size, obs_dim * n_agents)
        mixed_q = agent.mixer(test_q_values, test_states)
        assert mixed_q.shape == (batch_size, 1), \
            f"Mixer output shape mismatch: {mixed_q.shape} vs expected ({batch_size}, 1)"
        print(f"✓ Mixer shape: Q({test_q_values.shape}) + State({test_states.shape}) -> {mixed_q.shape}")
        
        print("\nTest 5: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test 6: Verify model save/load functionality."""
    print("\n" + "="*80)
    print("TEST 6: Model Save/Load")
    print("="*80)
    
    try:
        # Create and train agent briefly
        env = UrbanGridEnv(num_drones=3, grid_size=[500, 500, 200])
        obs_dict, _ = env.reset()
        obs_dim = obs_dict[0].shape[0]
        
        agent1 = QMIXAgent(
            n_agents=3,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=64,
            use_replay_buffer=False
        )
        
        # Do a few training steps
        for _ in range(5):
            obs = [obs_dict[i] for i in range(3)]
            actions = [agent1.get_action(obs[i], i) for i in range(3)]
            next_obs_dict, rewards, done, truncated, info = env.step(actions)
            next_obs = [next_obs_dict[i] for i in range(3)]
            agent1.train_step(obs, actions, rewards, next_obs, [done]*3)
            obs_dict = next_obs_dict
        
        # Save model
        test_path = "test_checkpoint.pth"
        agent1.save_model(test_path)
        print(f"✓ Model saved to {test_path}")
        
        # Create new agent and load
        agent2 = QMIXAgent(
            n_agents=3,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=64,
            use_replay_buffer=False
        )
        agent2.load_model(test_path)
        print(f"✓ Model loaded from {test_path}")
        
        # Compare parameters
        params1 = list(agent1.q_networks[0].parameters())[0].data
        params2 = list(agent2.q_networks[0].parameters())[0].data
        
        assert torch.allclose(params1, params2), "Loaded parameters don't match!"
        print("✓ Parameters match after load")
        
        # Clean up
        Path(test_path).unlink()
        env.close()
        
        print("\nTest 6: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """Test 7: Run a mini training loop (10 episodes) with updated settings."""
    print("\n" + "="*80)
    print("TEST 7: Mini Training Loop (10 episodes)")
    print("="*80)
    
    try:
        env = UrbanGridEnv(
            render_mode=None,
            num_drones=5,
            grid_size=[1000, 1000, 300],
            enable_comm_constraints=True
        )
        
        obs_dict, _ = env.reset()
        obs_dim = obs_dict[0].shape[0]
        
        agent = QMIXAgent(
            n_agents=5,
            obs_dim=obs_dim,
            action_dim=7,
            hidden_dim=256,
            mixing_hidden_dim=64,
            use_replay_buffer=True,
            buffer_capacity=5000,  # Smaller for test
            batch_size=64,
            epsilon_end=0.2,
            epsilon_decay=0.9995,
            target_update_freq=50
        )
        
        episode_rewards = []
        episode_coverages = []
        
        for episode in range(10):
            obs_dict, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not done and not truncated and step < 200:  # Match train_central.py
                obs = [obs_dict[i] for i in range(5)]
                actions = agent.get_actions(obs)
                next_obs_dict, rewards, done, truncated, info = env.step(actions)
                next_obs = [next_obs_dict[i] for i in range(5)]
                
                agent.store_experience(obs, actions, rewards, next_obs, [done] * 5)
                loss = agent.train_step()
                episode_reward += sum(rewards)  # Use sum for individual rewards
                obs_dict = next_obs_dict
                step += 1
            
            episode_rewards.append(episode_reward)
            episode_coverages.append(info['coverage'])
            
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                avg_coverage = np.mean(episode_coverages[-5:])
                print(f"  Episode {episode + 1}/10: Avg Reward = {avg_reward:.2f}, "
                      f"Epsilon = {agent.epsilon:.3f}, "
                      f"Coverage = {avg_coverage:.1%}, "
                      f"Collisions: {info['collisions']}, "
                      f"Violations: {info['no_fly_violations']}")
        
        print("\n✓ Mini training completed")
        print(f"  Final average reward: {np.mean(episode_rewards[-5:]):.2f}")
        print(f"  Reward trend: {episode_rewards[0]:.2f} -> {episode_rewards[-1]:.2f}")
        print(f"  Final coverage: {episode_coverages[-1]:.1%}")
        
        env.close()
        print("\nTest 7: PASSED ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test 7 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("QMIX VALIDATION TEST SUITE")
    print("="*80)
    print("\nRunning comprehensive tests to validate QMIX implementation...")
    
    tests = [
        ("Environment Validation", test_environment),
        ("Agent Initialization", test_agent_initialization),
        ("Single-Step Training", test_single_step_training),
        ("Replay Buffer Training", test_replay_buffer_training),
        ("Shape Consistency", test_shape_consistency),
        ("Model Save/Load", test_save_load),
        ("Mini Training Loop", test_mini_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    print("\n" + "="*80)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Your QMIX implementation is ready for training.")
        print("\nNext steps:")
        print("  1. Run: python -m core.training.train_central")
        print("  2. Monitor training metrics and adjust hyperparameters if needed")
        print("  3. Evaluate on 20-drone swarm after initial training")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Please fix issues before training.")
        print("\nTroubleshooting:")
        print("  - Check error messages above")
        print("  - Verify all dependencies are installed")
        print("  - Ensure environment is properly configured")
    
    print("="*80 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
