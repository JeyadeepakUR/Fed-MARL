"""
Integration tests for urban swarm simulation with QMIX.
"""

import pytest
import torch
import numpy as np
import yaml
from pathlib import Path

from core.envs.urban_grid_env import UrbanGridEnv
from core.agents.qmix_model import QMIX
from core.agents.policy_runner import PolicyRunner

def test_model_env_compatibility():
    """Test that model and environment work together."""
    # Create environment
    env = UrbanGridEnv(
        num_drones=5,
        grid_size=[1000, 1000, 300],
        cell_size=10.0,
        enable_weather=False,
        enable_comm_constraints=False
    )
    
    # Initialize model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim = obs_dim * env.num_drones
    
    model = QMIX(
        num_agents=env.num_drones,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        mixing_embed_dim=32,
        hypernet_embed=64
    )
    
    # Test forward pass
    obs, _ = env.reset()
    obs_tensors = {
        i: torch.FloatTensor(obs[i]).unsqueeze(0)
        for i in range(env.num_drones)
    }
    state = torch.FloatTensor(np.concatenate([obs[i] for i in range(env.num_drones)])).unsqueeze(0)
    
    q_tot, q_values = model(obs_tensors, state)
    
    # Verify outputs
    assert q_tot.shape == (1, 1), "Mixed Q-value shape incorrect"
    assert len(q_values) == env.num_drones, "Wrong number of agent Q-values"
    for i in range(env.num_drones):
        assert q_values[i].shape == (1, action_dim), f"Agent {i} Q-values shape incorrect"

def test_policy_execution():
    """Test that policy can be executed in environment."""
    # Create environment
    env = UrbanGridEnv(
        num_drones=3,
        grid_size=[500, 500, 200],
        cell_size=10.0,
        enable_weather=False,
        enable_comm_constraints=False
    )
    
    # Initialize model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim = obs_dim * env.num_drones
    
    model = QMIX(
        num_agents=env.num_drones,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        mixing_embed_dim=32,
        hypernet_embed=64
    )
    
    # Create policy runner
    runner = PolicyRunner(model)
    
    # Run episode
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 50
    
    while not done and steps < max_steps:
        actions = runner.select_actions(obs, deterministic=True)
        actions_array = np.array([actions[i] for i in range(env.num_drones)])
        obs, reward, terminated, truncated, info = env.step(actions_array)
        done = terminated or truncated
        steps += 1
    
    assert steps > 0, "Episode didn't run"
    assert 'coverage' in info, "Missing coverage metric"
    assert 'collisions' in info, "Missing collision metric"

def test_training_step():
    """Test that training step works."""
    # Load config
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = UrbanGridEnv(
        num_drones=3,
        grid_size=[500, 500, 200],
        cell_size=10.0,
        enable_weather=False,
        enable_comm_constraints=False
    )
    
    # Initialize model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim = obs_dim * env.num_drones
    
    model = QMIX(
        num_agents=env.num_drones,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        mixing_embed_dim=config['mixing_embed_dim'],
        hypernet_embed=config['hypernet_embed_dim']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Run a few training steps
    obs, _ = env.reset()
    initial_params = [param.clone() for param in model.parameters()]
    
    for _ in range(5):
        # Get actions
        obs_tensors = {
            i: torch.FloatTensor(obs[i]).unsqueeze(0)
            for i in range(env.num_drones)
        }
        state = torch.FloatTensor(np.concatenate([obs[i] for i in range(env.num_drones)])).unsqueeze(0)
        
        q_tot, q_values = model(obs_tensors, state)
        actions = {i: q_values[i].argmax(dim=1).item() for i in range(env.num_drones)}
        
        # Step environment
        actions_array = np.array([actions[i] for i in range(env.num_drones)])
        next_obs, reward, terminated, truncated, _ = env.step(actions_array)
        
        # Training step
        next_obs_tensors = {
            i: torch.FloatTensor(next_obs[i]).unsqueeze(0)
            for i in range(env.num_drones)
        }
        next_state = torch.FloatTensor(np.concatenate([next_obs[i] for i in range(env.num_drones)])).unsqueeze(0)
        
        with torch.no_grad():
            target_q_tot, _ = model(next_obs_tensors, next_state)
            target = reward + config['gamma'] * target_q_tot
        
        loss = torch.nn.functional.mse_loss(q_tot, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        obs = next_obs
    
    # Verify parameters changed
    final_params = [param.clone() for param in model.parameters()]
    assert any(not torch.allclose(p1, p2) for p1, p2 in zip(initial_params, final_params)), \
        "Model parameters did not update"

if __name__ == "__main__":
    pytest.main(["-v", __file__])