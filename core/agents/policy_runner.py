"""
Policy runner for testing and deployment of trained agents.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from core.agents.qmix_model import QMIX


class PolicyRunner:
    """Handles policy execution and evaluation."""
    
    def __init__(
        self,
        model: QMIX,
        device: str = 'cpu',
        epsilon: float = 0.0
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.to(device)
        self.model.eval()
    
    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        deterministic: bool = True
    ) -> Dict[int, int]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dict of observations for each agent
            deterministic: If True, use greedy policy. Otherwise epsilon-greedy.
        
        Returns:
            Dict of selected actions
        """
        with torch.no_grad():
            # Convert observations to tensors
            obs_tensors = {
                i: torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                for i, obs in observations.items()
            }
            
            # Get Q-values
            q_values = self.model.get_q_values(obs_tensors)
            
            actions = {}
            for i in range(self.model.num_agents):
                if not deterministic and np.random.random() < self.epsilon:
                    # Random action
                    actions[i] = np.random.randint(self.model.action_dim)
                else:
                    # Greedy action
                    actions[i] = q_values[i].argmax(dim=1).item()
        
        return actions
    
    def evaluate_episode(self, env, render: bool = False) -> Dict:
        """
        Run one evaluation episode.
        
        Args:
            env: Environment instance
            render: Whether to render the episode
        
        Returns:
            Dict with episode statistics
        """
        observations, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Select actions
            actions = self.select_actions(observations, deterministic=True)
            
            # Take actions
            actions_array = np.array([actions[i] for i in range(self.model.num_agents)])
            observations, reward, terminated, truncated, info = env.step(actions_array)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        return {
            'episode_reward': episode_reward,
            'steps': steps,
            'coverage': info.get('coverage', 0.0),
            'collisions': info.get('collisions', 0)
        }
    
    def evaluate(self, env, num_episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            env: Environment instance
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
        
        Returns:
            Dict with aggregated statistics
        """
        results = []
        
        for episode in range(num_episodes):
            episode_result = self.evaluate_episode(env, render=render)
            results.append(episode_result)
        
        # Aggregate results
        avg_reward = np.mean([r['episode_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])
        avg_collisions = np.mean([r['collisions'] for r in results])
        
        return {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_coverage': avg_coverage,
            'avg_collisions': avg_collisions,
            'std_reward': np.std([r['episode_reward'] for r in results]),
            'num_episodes': num_episodes,
            'results': results
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def set_epsilon(self, epsilon: float):
        """Update exploration rate."""
        self.epsilon = epsilon
