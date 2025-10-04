"""
Indian Skyscape Environment - Designed specifically for drone navigation
Realistic Indian urban layout with proper flight corridors and open spaces
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from collections import deque

class IndianSkyscapeEnv(gym.Env):
    """
    Indian Skyscape Environment for Drone Swarm Coordination
    
    Design Philosophy:
    - Realistic Indian urban layout with open spaces
    - Clear flight corridors between building clusters
    - Proper altitude zones for drone navigation
    - Minimal obstacles in airspace
    - Realistic building patterns (temples, markets, residential areas)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # Action constants
    ACTION_FORWARD = 0   # +y
    ACTION_BACKWARD = 1  # -y
    ACTION_LEFT = 2      # -x
    ACTION_RIGHT = 3     # +x
    ACTION_ASCEND = 4    # +z
    ACTION_DESCEND = 5   # -z
    ACTION_HOVER = 6     # no movement
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__()
        
        # Load configuration
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = kwargs
        
        # Environment parameters - 5km² Indian skyscape
        self.grid_size = np.array(config.get('grid_size', [5000, 5000, 300]), dtype=np.float64)
        self.cell_size = config.get('cell_size', 20.0)  # 20m resolution for easier navigation
        self.grid_cells = tuple((self.grid_size / self.cell_size).astype(int))
        
        # Indian urban environment parameters
        self.max_building_height = config.get('max_building_height', 150.0)  # Realistic Indian heights
        self.min_building_height = config.get('min_building_height', 15.0)
        
        # Drone configuration
        self.num_drones = config.get('num_drones', 10)
        self.obs_radius = config.get('obs_radius', 100.0)
        self.comm_range = config.get('comm_range', 500.0)
        self.max_steps = config.get('mission_timeout', 1000)
        
        # Mission parameters
        self.mission_type = config.get('mission_type', 'coverage')
        self.coverage_target = config.get('coverage_target', 0.80)  # 80% coverage target
        
        # Physics
        self.max_velocity = config.get('max_velocity', 20.0)  # 20 m/s for drones
        self.max_vertical_velocity = config.get('max_vertical_velocity', 8.0)
        self.dt = config.get('dt', 1.0)
        
        # Rewards - Optimized for drone learning
        reward_config = config.get('reward_structure', {})
        self.reward_coverage = reward_config.get('coverage', 10.0)  # High coverage reward
        self.reward_collision = reward_config.get('collision_penalty', -2.0)  # Light collision penalty
        self.reward_no_fly_zone = reward_config.get('no_fly_zone_penalty', -1.0)  # Very light no-fly penalty
        self.reward_energy = reward_config.get('energy_penalty', -0.01)
        self.reward_goal = reward_config.get('goal_reward', 100.0)
        self.reward_exploration = reward_config.get('exploration_reward', 2.0)  # Reward for exploring new areas
        
        # Define action space
        self.action_space = spaces.Discrete(7)
        self.action_map = {
            self.ACTION_FORWARD: np.array([0, self.cell_size, 0], dtype=np.float64),
            self.ACTION_BACKWARD: np.array([0, -self.cell_size, 0], dtype=np.float64),
            self.ACTION_LEFT: np.array([-self.cell_size, 0, 0], dtype=np.float64),
            self.ACTION_RIGHT: np.array([self.cell_size, 0, 0], dtype=np.float64),
            self.ACTION_ASCEND: np.array([0, 0, self.cell_size/2], dtype=np.float64),
            self.ACTION_DESCEND: np.array([0, 0, -self.cell_size/2], dtype=np.float64),
            self.ACTION_HOVER: np.array([0, 0, 0], dtype=np.float64)
        }
        
        # Define observation space
        obs_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.drone_positions = None
        self.drone_velocities = None
        self.obstacle_map = None
        self.coverage_map = None
        self.explored_areas = None  # Track explored areas for exploration rewards
        self.steps = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.no_fly_violations = 0
        
        # Rendering
        self.render_mode = config.get('render_mode', None)
        self.fig = None
        self.ax = None
        self.trajectory_history = {i: [] for i in range(self.num_drones)}
    
    def _get_observation_dim(self) -> int:
        """Calculate observation dimension."""
        dim = 0
        dim += 6  # position (3) + velocity (3)
        local_grid_size = int(self.obs_radius / self.cell_size) * 2 + 1
        dim += local_grid_size * local_grid_size  # Local obstacle map
        dim += (self.num_drones - 1) * 3  # Teammate positions
        dim += 1  # Coverage percentage
        dim += 1  # Distance to nearest obstacle
        return dim
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Generate Indian skyscape
        self.obstacle_map = self._generate_indian_skyscape()
        
        # Initialize drone positions in safe areas
        self.drone_positions = self._initialize_safe_positions()
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float64)
        
        # Initialize coverage and exploration tracking
        self.coverage_map = np.zeros(self.grid_cells[:2], dtype=bool)
        self.explored_areas = np.zeros(self.grid_cells[:2], dtype=bool)
        
        self.steps = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.no_fly_violations = 0
        self._previous_coverage = 0
        
        self.trajectory_history = {i: [self.drone_positions[i].copy()] for i in range(self.num_drones)}
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _generate_indian_skyscape(self) -> np.ndarray:
        """Generate realistic Indian urban skyscape with open flight corridors."""
        # Initialize 3D obstacle map
        obstacle_map = np.zeros(self.grid_cells, dtype=bool)
        building_heights = np.zeros(self.grid_cells[:2], dtype=np.float64)
        
        # Create Indian urban zones
        center_x, center_y = self.grid_cells[0] // 2, self.grid_cells[1] // 2
        
        # 1. Temple Complex (center) - Medium height, clustered
        temple_size = 8
        temple_height = 60.0
        for dx in range(-temple_size//2, temple_size//2):
            for dy in range(-temple_size//2, temple_size//2):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]:
                    height_cells = int(temple_height / self.cell_size)
                    building_heights[x, y] = temple_height
                    obstacle_map[x, y, :height_cells] = True
        
        # 2. Market Area (north) - Low height, dense
        market_x, market_y = center_x, center_y - 20
        market_size = 12
        for dx in range(-market_size//2, market_size//2):
            for dy in range(-market_size//2, market_size//2):
                x, y = market_x + dx, market_y + dy
                if 0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]:
                    if np.random.random() < 0.7:  # 70% building density
                        height = np.random.uniform(20.0, 40.0)
                        height_cells = int(height / self.cell_size)
                        building_heights[x, y] = height
                        obstacle_map[x, y, :height_cells] = True
        
        # 3. Residential Area (south) - Low height, sparse
        residential_x, residential_y = center_x, center_y + 20
        residential_size = 15
        for dx in range(-residential_size//2, residential_size//2):
            for dy in range(-residential_size//2, residential_size//2):
                x, y = residential_x + dx, residential_y + dy
                if 0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]:
                    if np.random.random() < 0.3:  # 30% building density
                        height = np.random.uniform(15.0, 30.0)
                        height_cells = int(height / self.cell_size)
                        building_heights[x, y] = height
                        obstacle_map[x, y, :height_cells] = True
        
        # 4. Commercial Area (east) - High height, clustered
        commercial_x, commercial_y = center_x + 20, center_y
        commercial_size = 10
        for dx in range(-commercial_size//2, commercial_size//2):
            for dy in range(-commercial_size//2, commercial_size//2):
                x, y = commercial_x + dx, commercial_y + dy
                if 0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]:
                    if np.random.random() < 0.6:  # 60% building density
                        height = np.random.uniform(40.0, 120.0)
                        height_cells = int(height / self.cell_size)
                        building_heights[x, y] = height
                        obstacle_map[x, y, :height_cells] = True
        
        # 5. Industrial Area (west) - Medium height, sparse
        industrial_x, industrial_y = center_x - 20, center_y
        industrial_size = 12
        for dx in range(-industrial_size//2, industrial_size//2):
            for dy in range(-industrial_size//2, industrial_size//2):
                x, y = industrial_x + dx, industrial_y + dy
                if 0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]:
                    if np.random.random() < 0.4:  # 40% building density
                        height = np.random.uniform(25.0, 80.0)
                        height_cells = int(height / self.cell_size)
                        building_heights[x, y] = height
                        obstacle_map[x, y, :height_cells] = True
        
        # 6. Add some scattered buildings
        num_scattered = 20
        for _ in range(num_scattered):
            x = np.random.randint(0, self.grid_cells[0])
            y = np.random.randint(0, self.grid_cells[1])
            if not obstacle_map[x, y, 0]:  # Only if no building there
                height = np.random.uniform(20.0, 60.0)
                height_cells = int(height / self.cell_size)
                building_heights[x, y] = height
                obstacle_map[x, y, :height_cells] = True
        
        # Store building heights for visualization
        self.building_heights = building_heights
        
        return obstacle_map
    
    def _initialize_safe_positions(self) -> np.ndarray:
        """Initialize drone positions in safe, open areas."""
        positions = []
        min_dist = 100.0  # Minimum distance between drones
        init_height = 80.0  # Initial height above most buildings
        
        for _ in range(self.num_drones):
            valid_pos = False
            attempts = 0
            max_attempts = 1000
            
            while not valid_pos and attempts < max_attempts:
                attempts += 1
                # Try positions away from center (where buildings are)
                if attempts < 500:
                    # Try corners first
                    corner = np.random.choice(['nw', 'ne', 'sw', 'se'])
                    if corner == 'nw':
                        pos_2d = np.array([100, 100])
                    elif corner == 'ne':
                        pos_2d = np.array([self.grid_size[0] - 100, 100])
                    elif corner == 'sw':
                        pos_2d = np.array([100, self.grid_size[1] - 100])
                    else:  # se
                        pos_2d = np.array([self.grid_size[0] - 100, self.grid_size[1] - 100])
                else:
                    # Random position
                    pos_2d = np.random.uniform(200, self.grid_size[:2] - 200, size=2)
                
                pos = np.append(pos_2d, init_height)
                
                # Check if position is safe
                grid_pos = (pos_2d / self.cell_size).astype(int)
                grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
                height_cell = int(init_height / self.cell_size)
                height_cell = min(height_cell, self.grid_cells[2] - 1)
                
                if height_cell < self.grid_cells[2]:
                    if not self.obstacle_map[grid_pos[0], grid_pos[1], height_cell]:
                        if len(positions) == 0:
                            valid_pos = True
                        else:
                            min_dist_current = np.min(np.linalg.norm(np.array(positions) - pos, axis=1))
                            if min_dist_current > min_dist:
                                valid_pos = True
            
            if valid_pos:
                positions.append(pos)
            else:
                # Fallback: force position at edge
                pos_2d = np.array([100, 100])
                pos = np.append(pos_2d, init_height)
                positions.append(pos)
        
        return np.array(positions, dtype=np.float64)
    
    def step(self, actions: Union[np.ndarray, List]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.steps += 1
        
        if isinstance(actions, list):
            actions = np.array(actions)
        
        self._update_positions(actions)
        self._update_coverage()
        
        rewards = self._calculate_reward()
        self.episode_reward += np.mean(rewards)
        
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _update_positions(self, actions: np.ndarray):
        """Update drone positions with simple collision handling."""
        for i in range(self.num_drones):
            velocity = self.action_map[actions[i]].astype(np.float64)
            new_pos = self.drone_positions[i] + velocity * self.dt
            
            # Boundary checking
            new_pos = np.clip(new_pos, 
                             [50, 50, 50], 
                             [self.grid_size[0] - 50, self.grid_size[1] - 50, self.grid_size[2] - 50])
            
            # Simple collision check
            grid_pos = (new_pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            height_cell = int(new_pos[2] / self.cell_size)
            height_cell = min(height_cell, self.grid_cells[2] - 1)
            
            if height_cell < self.grid_cells[2]:
                if not self.obstacle_map[grid_pos[0], grid_pos[1], height_cell]:
                    # Safe position
                    self.drone_positions[i] = new_pos
                    self.drone_velocities[i] = velocity
                    self.trajectory_history[i].append(new_pos.copy())
                else:
                    # Collision - simple recovery: move up
                    self.drone_positions[i][2] = min(self.drone_positions[i][2] + 20, self.grid_size[2] - 50)
                    self.drone_velocities[i] = np.zeros(3)
                    self.collision_count += 1
            else:
                # Above grid - safe
                self.drone_positions[i] = new_pos
                self.drone_velocities[i] = velocity
                self.trajectory_history[i].append(new_pos.copy())
    
    def _update_coverage(self):
        """Update coverage map based on current drone positions."""
        for pos in self.drone_positions:
            grid_pos = (pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            
            # Coverage radius
            radius_cells = int(self.obs_radius / self.cell_size)
            
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    x, y = grid_pos[0] + dx, grid_pos[1] + dy
                    if (0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]):
                        dist = np.sqrt(dx**2 + dy**2) * self.cell_size
                        if dist <= self.obs_radius:
                            self.coverage_map[x, y] = True
                            self.explored_areas[x, y] = True
    
    def _calculate_reward(self) -> np.ndarray:
        """Calculate rewards for each agent."""
        rewards = np.zeros(self.num_drones)
        
        # Coverage reward
        current_coverage = np.sum(self.coverage_map)
        if not hasattr(self, '_previous_coverage'):
            self._previous_coverage = 0
        
        coverage_increase = current_coverage - self._previous_coverage
        self._previous_coverage = current_coverage
        
        if coverage_increase > 0:
            rewards += (coverage_increase / self.coverage_map.size) * self.reward_coverage * 100
        
        # Exploration reward
        for i, pos in enumerate(self.drone_positions):
            grid_pos = (pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            
            if not self.explored_areas[grid_pos[0], grid_pos[1]]:
                rewards[i] += self.reward_exploration
        
        # Energy penalty
        energy_used = np.linalg.norm(self.drone_velocities, axis=1)
        rewards += energy_used * self.reward_energy
        
        # Goal achievement bonus
        coverage_ratio = current_coverage / self.coverage_map.size
        if coverage_ratio >= self.coverage_target:
            rewards += self.reward_goal
        
        return rewards
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
        return coverage_ratio >= self.coverage_target
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all drones."""
        observations = {}
        
        for i in range(self.num_drones):
            obs_parts = []
            
            # Own position and velocity
            own_pos = self.drone_positions[i] / self.grid_size
            own_vel = self.drone_velocities[i] / self.max_velocity
            obs_parts.extend([own_pos, own_vel])
            
            # Local obstacle map
            local_map = self._get_local_map(i)
            obs_parts.append(local_map.flatten())
            
            # Teammate positions
            teammate_obs = self._get_teammate_observations(i)
            obs_parts.append(teammate_obs)
            
            # Coverage ratio
            coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
            obs_parts.append(np.array([coverage_ratio]))
            
            # Distance to nearest obstacle
            nearest_obstacle_dist = self._get_nearest_obstacle_distance(i)
            obs_parts.append(np.array([nearest_obstacle_dist]))
            
            observations[i] = np.concatenate(obs_parts).astype(np.float32)
        
        return observations
    
    def _get_local_map(self, drone_idx: int) -> np.ndarray:
        """Get local obstacle map around drone."""
        pos = self.drone_positions[drone_idx]
        grid_pos = (pos / self.cell_size).astype(int)
        radius_cells = int(self.obs_radius / self.cell_size)
        
        local_map = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1))
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                x, y = grid_pos[0] + dx, grid_pos[1] + dy
                if (0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]):
                    # Check if there's an obstacle at drone's height
                    height_cell = int(pos[2] / self.cell_size)
                    height_cell = min(height_cell, self.grid_cells[2] - 1)
                    
                    if height_cell < self.grid_cells[2]:
                        local_map[dx + radius_cells, dy + radius_cells] = self.obstacle_map[x, y, height_cell]
        
        return local_map
    
    def _get_teammate_observations(self, drone_idx: int) -> np.ndarray:
        """Get relative positions of teammates."""
        own_pos = self.drone_positions[drone_idx]
        teammate_obs = []
        
        for j in range(self.num_drones):
            if j != drone_idx:
                other_pos = self.drone_positions[j]
                relative_pos = (other_pos - own_pos) / self.comm_range
                teammate_obs.extend(relative_pos)
        
        return np.array(teammate_obs)
    
    def _get_nearest_obstacle_distance(self, drone_idx: int) -> float:
        """Get distance to nearest obstacle."""
        pos = self.drone_positions[drone_idx]
        grid_pos = (pos[:2] / self.cell_size).astype(int)
        grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
        
        min_dist = float('inf')
        search_radius = 10  # Search in 10x10 grid around drone
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                x, y = grid_pos[0] + dx, grid_pos[1] + dy
                if (0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]):
                    height_cell = int(pos[2] / self.cell_size)
                    height_cell = min(height_cell, self.grid_cells[2] - 1)
                    
                    if height_cell < self.grid_cells[2]:
                        if self.obstacle_map[x, y, height_cell]:
                            dist = np.sqrt(dx**2 + dy**2) * self.cell_size
                            min_dist = min(min_dist, dist)
        
        return min_dist / self.obs_radius if min_dist != float('inf') else 1.0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
        exploration_ratio = np.sum(self.explored_areas) / self.explored_areas.size
        
        return {
            'coverage': coverage_ratio,
            'exploration': exploration_ratio,
            'steps': self.steps,
            'episode_reward': self.episode_reward,
            'num_drones': self.num_drones,
            'collisions': self.collision_count,
            'no_fly_violations': self.no_fly_violations
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_matplotlib()
            plt.draw()
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_matplotlib(self):
        """Render using Matplotlib with zone labels."""
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 10))
            self.ax = self.fig.add_subplot(111)
        
        self.ax.clear()
        
        # Set viewing limits
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        
        # Add zone labels
        center_x, center_y = self.grid_size[0] / 2, self.grid_size[1] / 2
        
        # Temple Complex (center)
        self.ax.text(center_x, center_y, '🏛️ TEMPLE\nCOMPLEX', ha='center', va='center', 
                    fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
        
        # Market Area (north)
        self.ax.text(center_x, center_y - 400, '🛒 MARKET\nAREA', ha='center', va='center',
                    fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
        # Residential Area (south)
        self.ax.text(center_x, center_y + 400, '🏠 RESIDENTIAL\nAREA', ha='center', va='center',
                    fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Commercial Area (east)
        self.ax.text(center_x + 400, center_y, '🏢 COMMERCIAL\nAREA', ha='center', va='center',
                    fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Industrial Area (west)
        self.ax.text(center_x - 400, center_y, '🏭 INDUSTRIAL\nAREA', ha='center', va='center',
                    fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        # Render buildings with height-based colors
        building_mask = self.building_heights > 0
        x_pos = np.where(building_mask)[0] * self.cell_size
        y_pos = np.where(building_mask)[1] * self.cell_size
        heights = self.building_heights[building_mask]
        
        if len(x_pos) > 0:
            # Color buildings by height
            colors = plt.cm.plasma(heights / np.max(heights))
            self.ax.scatter(x_pos, y_pos, c=colors, s=80, alpha=0.8, edgecolors='black', linewidths=0.5, label='Buildings')
            
            # Add height legend
            for i, height in enumerate([20, 60, 100, 150]):
                color = plt.cm.plasma(height / np.max(heights))
                self.ax.scatter([], [], c=[color], s=80, alpha=0.8, label=f'{height}m')
        
        # Render coverage
        coverage_points = np.argwhere(self.coverage_map)
        if len(coverage_points) > 0:
            coverage_coords = coverage_points * self.cell_size
            self.ax.scatter(coverage_coords[:, 0], coverage_coords[:, 1],
                           c='lightgreen', s=15, alpha=0.4, marker='s', label='Covered Areas')
        
        # Render drones with better visualization
        colors = cm.rainbow(np.linspace(0, 1, self.num_drones))
        for i in range(self.num_drones):
            pos = self.drone_positions[i]
            # Drone body
            self.ax.scatter(pos[0], pos[1], c=[colors[i]], s=150, edgecolors='black', linewidths=2, 
                           marker='o', label=f'Drone {i}' if i < 3 else '')
            
            # Altitude indicator
            self.ax.text(pos[0], pos[1] + 50, f'D{i}\n{pos[2]:.0f}m', ha='center', va='bottom', 
                        fontsize=8, weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Movement trail (last 5 positions)
            if len(self.trajectory_history[i]) > 1:
                trail = np.array(self.trajectory_history[i][-5:])
                self.ax.plot(trail[:, 0], trail[:, 1], color=colors[i], alpha=0.5, linewidth=2)
        
        # Add stats panel
        coverage_pct = np.sum(self.coverage_map) / self.coverage_map.size
        exploration_pct = np.sum(self.explored_areas) / self.explored_areas.size
        
        stats_text = (f'Step: {self.steps}\n'
                     f'Coverage: {coverage_pct:.1%}\n'
                     f'Exploration: {exploration_pct:.1%}\n'
                     f'Drones: {self.num_drones}\n'
                     f'Collisions: {self.collision_count}')
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Title and labels
        self.ax.set_title('🇮🇳 Indian Skyscape Environment - Drone Swarm Simulation', 
                         fontsize=14, weight='bold', pad=20)
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        # Grid for better readability
        self.ax.grid(True, alpha=0.3)
        
        # Legend (limit to avoid clutter)
        handles, labels = self.ax.get_legend_handles_labels()
        if len(handles) > 8:  # Limit legend items
            handles, labels = handles[:8], labels[:8]
        self.ax.legend(handles, labels, loc='upper right', fontsize=8)
        
        plt.pause(0.001)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array."""
        img = np.ones((*self.grid_cells[:2], 3), dtype=np.uint8) * 255
        
        # Buildings in gray
        building_mask = self.building_heights > 0
        img[building_mask] = [100, 100, 100]
        
        # Coverage in green
        img[self.coverage_map] = [200, 255, 200]
        
        # Drones in blue
        for pos in self.drone_positions:
            grid_pos = (pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            img[tuple(grid_pos)] = [0, 0, 255]
        
        return img
    
    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    
    # Enable interactive mode for matplotlib
    plt.ion()
    
    # Create Indian Skyscape Environment with visualization
    env = IndianSkyscapeEnv(
        render_mode="human",
        num_drones=10,  # More drones for better visualization
        grid_size=[5000, 5000, 300],  # Full size
        cell_size=20.0
    )
    
    obs, info = env.reset()
    print(f"\n🏛️  Indian Skyscape Environment Visualization")
    print(f"Environment created: {env.num_drones} drones")
    print(f"Observation shape: {obs[0].shape}")
    print(f"Initial coverage: {info['coverage']:.1%}")
    print(f"Grid size: {env.grid_size}")
    print(f"Cell size: {env.cell_size}m")
    print(f"\n🎯 Environment Zones:")
    print(f"  🏛️  Temple Complex (center)")
    print(f"  🛒 Market Area (north)")
    print(f"  🏠 Residential Area (south)")
    print(f"  🏢 Commercial Area (east)")
    print(f"  🏭 Industrial Area (west)")
    print(f"\n🚁 Drone positions:")
    for i, pos in enumerate(env.drone_positions):
        print(f"  Drone {i}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})m")
    
    print(f"\n🎮 Starting simulation...")
    print(f"Close the visualization window to stop")
    
    done = False
    truncated = False
    step_count = 0
    max_steps = 200  # Longer simulation for better visualization
    collision_count = 0
    
    try:
        while not done and not truncated and step_count < max_steps:
            # Random actions for each drone
            actions = np.random.randint(0, env.action_space.n, size=env.num_drones)
            
            # Step the environment
            obs, rewards, done, truncated, info = env.step(actions)
            
            # Render the current state
            env.render()
            
            # Print status every 20 steps
            if step_count % 20 == 0:
                print(f"\n📍 Step {step_count}")
                print(f"  Coverage: {info['coverage']:.1%}")
                print(f"  Exploration: {info['exploration']:.1%}")
                print(f"  Episode Reward: {np.mean(rewards):.2f}")
                print(f"  Collisions: {info['collisions']}")
                print(f"  Active Drones: {info['num_drones']}")
            
            step_count += 1
            time.sleep(0.1)  # Slower for better observation
            
            # Check if window was closed
            if not plt.get_fignums():
                print("Window closed by user.")
                break
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print(f"\n🏁 Simulation finished.")
        print(f"Final Coverage: {info['coverage']:.1%}")
        print(f"Final Exploration: {info['exploration']:.1%}")
        print(f"Total Steps: {step_count}")
        print(f"Collisions: {info['collisions']}")
        
        env.close()
        plt.ioff()  # Disable interactive mode
        
        print(f"\n✅ Indian Skyscape Environment Test Complete!")
        print(f"The environment is ready for training with QMIX!")
