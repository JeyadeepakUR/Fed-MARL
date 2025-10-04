"""
Stabilized QMIX Agent module for federated-MARL-drone prototype.
Replaces previous agent file with practical fixes:
- simpler Q-networks
- robust normalization
- stable replay sampling
- correct target shapes
- safer exploration schedule
- robust gradient clipping
"""

import random
import math
from typing import List
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['obs', 'actions', 'rewards', 'next_obs', 'dones'])


class ReplayBuffer:
    """Simple uniform replay buffer for multi-agent QMIX."""
    def __init__(self, capacity: int, num_agents: int, obs_dim: int):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.obs_dim = int(obs_dim)
        self.position = 0
        self.size = 0

        self.observations = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_agents), dtype=np.int64)
        self.rewards = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_agents), dtype=np.float32)

    def push(self, obs, actions, rewards, next_obs, dones):
        obs_array = np.array([np.array(o, dtype=np.float32) for o in obs])
        next_obs_array = np.array([np.array(o, dtype=np.float32) for o in next_obs])
        actions_array = np.array(actions, dtype=np.int64)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array([float(d) for d in dones], dtype=np.float32)

        self.observations[self.position] = obs_array
        self.actions[self.position] = actions_array
        self.rewards[self.position] = rewards_array
        self.next_observations[self.position] = next_obs_array
        self.dones[self.position] = dones_array

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self.size > 0, "Replay buffer is empty"
        batch_size = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.observations[indices]).float(),
            torch.from_numpy(self.actions[indices]).long(),
            torch.from_numpy(self.rewards[indices]).float(),
            torch.from_numpy(self.next_observations[indices]).float(),
            torch.from_numpy(self.dones[indices]).float()
        )

    def __len__(self):
        return int(self.size)


# Simple, stable per-agent Q-network
class DroneQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Accepts [batch, input_dim] or [input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# QMIX Mixer Network (kept but stabilized)
class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # hypernetworks
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        self.hyper_b_1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        q_values: [batch, n_agents]
        states:   [batch, state_dim]
        returns q_tot: [batch, 1]
        """
        batch_size = q_values.size(0)

        # w1: [batch, n_agents * hidden_dim] -> reshape to [batch, n_agents, hidden_dim]
        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        # normalize weight rows to avoid extremes
        w1 = w1 / (w1.sum(dim=1, keepdim=True) + 1e-8)

        b1 = self.hyper_b_1(states).view(batch_size, 1, self.hidden_dim)

        # q_values -> [batch, 1, n_agents]
        agent_qs = q_values.view(batch_size, 1, self.n_agents)
        hidden = torch.bmm(agent_qs, w1) + b1  # [batch, 1, hidden_dim]
        hidden = torch.relu(hidden)

        w_final = torch.abs(self.hyper_w_final(states)).view(batch_size, self.hidden_dim, 1)
        b_final = self.hyper_b_final(states).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + b_final  # [batch, 1, 1]
        q_tot = q_tot.view(batch_size, 1)
        return q_tot


# QMIX Agent (not an nn.Module, but composes modules)
class QMIXAgent:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        mixing_hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,  # Increased from 0.05 to 0.1 for more exploration
        epsilon_decay: float = 0.999,  # Slowed decay from 0.995 to 0.999
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        grad_clip: float = 10.0,
        use_replay_buffer: bool = True,
        device: str = None
    ):
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.mixing_hidden_dim = int(mixing_hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.grad_clip = float(grad_clip)
        self.use_replay_buffer = bool(use_replay_buffer)

        # epsilon schedule
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        # state dim (concatenate per-agent obs)
        self.state_dim = self.obs_dim * self.n_agents

        # device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # networks
        self.q_networks = nn.ModuleList([DroneQNetwork(self.obs_dim, self.hidden_dim, self.action_dim) for _ in range(self.n_agents)]).to(self.device)
        self.target_networks = nn.ModuleList([DroneQNetwork(self.obs_dim, self.hidden_dim, self.action_dim) for _ in range(self.n_agents)]).to(self.device)

        self.mixer = QMixer(self.n_agents, self.state_dim, self.mixing_hidden_dim).to(self.device)
        self.target_mixer = QMixer(self.n_agents, self.state_dim, self.mixing_hidden_dim).to(self.device)

        # optimizers
        all_q_params = []
        for net in self.q_networks:
            all_q_params += list(net.parameters())
        self.q_optimizer = optim.Adam(all_q_params, lr=lr, eps=1e-8)
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr, eps=1e-8)

        # lr schedulers
        self.q_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.q_optimizer, factor=0.5, patience=1000, min_lr=1e-6)
        self.mixer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.mixer_optimizer, factor=0.5, patience=1000, min_lr=1e-6)

        # buffer
        if self.use_replay_buffer:
            self.replay_buffer = ReplayBuffer(buffer_capacity, self.n_agents, self.obs_dim)

        self.training_steps = 0
        self.episode_count = 0

        # sync targets
        self._sync_targets()

        # safety mask parameters
        self.safety_threshold = 0.5  # tune in training script if needed
        self.large_neg = -1e6

        print(f"[QMIXAgent] Initialized on {self.device} | agents={self.n_agents} obs_dim={self.obs_dim} act_dim={self.action_dim}")

    def _sync_targets(self):
        for q, tq in zip(self.q_networks, self.target_networks):
            tq.load_state_dict(q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations in a stable, batch-aware manner.
        Accepts shapes:
         - [obs_dim] -> returns [1, obs_dim]
         - [n_agents, obs_dim] -> returns [n_agents, obs_dim]
         - [batch, n_agents, obs_dim] -> returns same shape
        """
        # Convert to float tensor on device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        orig_shape = tuple(obs.shape)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)  # [1,1,obs_dim]
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0)  # [1,n_agents,obs_dim]

        # Standardize per-sample (per agent) to avoid global dominance
        mean = obs.mean(dim=-1, keepdim=True)
        std = obs.std(dim=-1, keepdim=True) + 1e-8
        normalized = (obs - mean) / std

        # Optionally emphasize first few safety features (if known)
        # Avoid hard-coded slicing; user can set safety indices externally if needed
        if normalized.size(-1) >= 6:
            # amplify position/velocity part modestly
            normalized[..., :6] = normalized[..., :6] * 1.25

        # reshape back
        if len(orig_shape) == 1:
            return normalized.squeeze(0).squeeze(0)  # [obs_dim]
        if len(orig_shape) == 2:
            return normalized.squeeze(0)  # [n_agents, obs_dim]
        return normalized  # [batch, n_agents, obs_dim]

    def get_action(self, obs, agent_id: int):
        """Single-agent greedy/eps action (compatibility helper)."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_n = self._normalize_obs(obs_t).squeeze(0)  # [1, obs_dim] -> [obs_dim]
            q = self.q_networks[agent_id](obs_n.unsqueeze(0))  # [1, action_dim]
            return int(q.argmax(dim=-1).item())

    def get_actions(self, observations: List, explore: bool = True):
        """
        Vectorized action selection for all agents. observations: list(len=n_agents) of obs vectors
        """
        actions = []
        obs_stack = torch.tensor(np.stack(observations, axis=0), dtype=torch.float32, device=self.device)  # [n_agents, obs_dim]
        obs_norm = self._normalize_obs(obs_stack)  # [n_agents, obs_dim]

        # Compute Q for all agents in batch
        with torch.no_grad():
            q_values = []
            for i in range(self.n_agents):
                qv = self.q_networks[i](obs_norm[i].unsqueeze(0))  # [1, action_dim]
                q_values.append(qv.squeeze(0))
            q_values = torch.stack(q_values, dim=0)  # [n_agents, action_dim]

        for i in range(self.n_agents):
            # simple safety heuristic: if any of the first few observations exceed threshold -> conservative
            nearby_obstacle = False
            try:
                nearby_obstacle = bool((obs_norm[i, :6].abs() > self.safety_threshold).any().item())
            except Exception:
                nearby_obstacle = False

            eps = self.epsilon * (0.5 if nearby_obstacle else 1.0) if explore else 0.0
            if random.random() < eps:
                if nearby_obstacle:
                    action = random.randrange(0, max(2, self.action_dim))  # prefer lower-index safe actions
                else:
                    action = random.randrange(self.action_dim)
            else:
                # apply safety mask by setting risky actions to very negative value
                qv = q_values[i].clone()
                if nearby_obstacle:
                    # mask upper half actions as risky (assumption; tune according to action semantics)
                    mask_start = max(1, int(self.action_dim * 0.6))
                    qv[mask_start:] = self.large_neg
                action = int(qv.argmax().item())
            actions.append(action)
        return actions

    def store_experience(self, obs, actions, rewards, next_obs, dones):
        if self.use_replay_buffer:
            self.replay_buffer.push(obs, actions, rewards, next_obs, dones)

    def train_step(self, obs=None, actions=None, rewards=None, next_obs=None, dones=None):
        """
        Single training step. If replay buffer is used, uses a sampled batch.
        Returns scalar loss or None.
        """
        if self.use_replay_buffer:
            if len(self.replay_buffer) < max(256, self.batch_size):
                return 0.0
            obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.replay_buffer.sample(self.batch_size)
            loss = self._update_networks_batch(obs_b.to(self.device), actions_b.to(self.device),
                                               rewards_b.to(self.device), next_obs_b.to(self.device),
                                               dones_b.to(self.device))
        else:
            # single-step update (rare for QMIX)
            if obs is None:
                return 0.0
            loss = self._update_networks_single(obs, actions, rewards, next_obs, dones)

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self._sync_targets()

        # decay epsilon (geometric)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss

    def _update_networks_single(self, obs, actions, rewards, next_obs, dones):
        # convenience wrapper to use batch update with batch_size=1
        obs_b = torch.from_numpy(np.stack([np.array(obs)], axis=0)).float().to(self.device)  # [1, n_agents, obs_dim]
        actions_b = torch.from_numpy(np.stack([np.array(actions)], axis=0)).long().to(self.device)  # [1, n_agents]
        rewards_b = torch.from_numpy(np.stack([np.array(rewards)], axis=0)).float().to(self.device)  # [1, n_agents]
        next_obs_b = torch.from_numpy(np.stack([np.array(next_obs)], axis=0)).float().to(self.device)
        dones_b = torch.from_numpy(np.stack([np.array(dones)], axis=0)).float().to(self.device)
        return self._update_networks_batch(obs_b, actions_b, rewards_b, next_obs_b, dones_b)

    def _update_networks_batch(self, obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch):
        """
        obs_batch: [batch, n_agents, obs_dim]
        actions_batch: [batch, n_agents]
        rewards_batch: [batch, n_agents]
        next_obs_batch: [batch, n_agents, obs_dim]
        dones_batch: [batch, n_agents]
        """
        batch_size = obs_batch.size(0)

        # Normalize observations
        obs_norm = self._normalize_obs(obs_batch)        # [batch, n_agents, obs_dim]
        next_obs_norm = self._normalize_obs(next_obs_batch)

        # create global state by flattening agent observations
        state = obs_batch.view(batch_size, -1).to(self.device)          # [batch, state_dim]
        next_state = next_obs_batch.view(batch_size, -1).to(self.device)

        # compute current Q chosen values for each agent
        chosen_qs = []
        for agent_id in range(self.n_agents):
            agent_obs = obs_norm[:, agent_id, :]                      # [batch, obs_dim]
            q_vals = self.q_networks[agent_id](agent_obs)            # [batch, action_dim]
            chosen = q_vals.gather(1, actions_batch[:, agent_id].unsqueeze(1)).squeeze(1)  # [batch]
            chosen_qs.append(chosen)
        chosen_qs = torch.stack(chosen_qs, dim=1)  # [batch, n_agents]

        # mix to total Q
        q_tot = self.mixer(chosen_qs, state).squeeze(-1)  # [batch]

        # compute targets
        with torch.no_grad():
            next_max_qs = []
            for agent_id in range(self.n_agents):
                next_agent_obs = next_obs_norm[:, agent_id, :]               # [batch, obs_dim]
                next_q = self.target_networks[agent_id](next_agent_obs)      # [batch, action_dim]
                next_max_qs.append(next_q.max(dim=1)[0])
            next_max_qs = torch.stack(next_max_qs, dim=1)                   # [batch, n_agents]
            target_q_tot = self.target_mixer(next_max_qs, next_state).squeeze(-1)  # [batch]

            # team-level reward: mean across agents
            team_rewards = rewards_batch.mean(dim=1)                        # [batch]
            done_mask = 1.0 - dones_batch.max(dim=1)[0]                     # [batch]

            # normalize rewards to avoid scale issues
            reward_scale = team_rewards.abs().mean().clamp(min=1e-3)
            rewards_norm = team_rewards / reward_scale

            targets = rewards_norm + self.gamma * target_q_tot * done_mask  # [batch]

        # MSE loss between current q_tot and targets
        loss = nn.MSELoss()(q_tot, targets)

        # backprop through per-agent networks and mixer
        self.q_optimizer.zero_grad()
        self.mixer_optimizer.zero_grad()
        loss.backward()

        # clip gradients across all parameter groups
        total_params = []
        for net in self.q_networks:
            total_params.extend(list(net.parameters()))
        total_params.extend(list(self.mixer.parameters()))
        torch.nn.utils.clip_grad_norm_(total_params, self.grad_clip)

        self.q_optimizer.step()
        self.mixer_optimizer.step()

        # scheduler step (optional: can be driven by external val loss in training script)
        # self.q_scheduler.step(loss.item())
        # self.mixer_scheduler.step(loss.item())

        return loss.item()

    def save_model(self, path: str):
        ckpt = {
            'q_networks': [net.state_dict() for net in self.q_networks],
            'target_q_networks': [net.state_dict() for net in self.target_networks],
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'mixer_optimizer': self.mixer_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count
        }
        torch.save(ckpt, path)
        print(f"[QMIXAgent] Saved checkpoint to {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(ckpt['q_networks'][i])
        for i, net in enumerate(self.target_networks):
            net.load_state_dict(ckpt['target_q_networks'][i])
        self.mixer.load_state_dict(ckpt['mixer'])
        self.target_mixer.load_state_dict(ckpt['target_mixer'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer'])
        self.mixer_optimizer.load_state_dict(ckpt['mixer_optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.training_steps = ckpt.get('training_steps', 0)
        self.episode_count = ckpt.get('episode_count', 0)
        print(f"[QMIXAgent] Loaded checkpoint from {path}")

    def get_stats(self):
        return {
            'epsilon': float(self.epsilon),
            'training_steps': int(self.training_steps),
            'episode_count': int(self.episode_count),
            'buffer_size': int(len(self.replay_buffer)) if self.use_replay_buffer else 0
        }