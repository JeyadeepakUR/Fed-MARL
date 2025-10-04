import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from collections import deque

class UrbanGridEnv(gym.Env):
    """
    Multi-agent environment for drone swarm coordination in urban grid.
    
    Observation: Each drone observes:
        - Its own position (x, y)
        - Its own velocity (vx, vy)
        - Local obstacle map (grid around drone)
        - Teammate positions within comm range
        - Mission-specific info (coverage, targets)
        - Communication status (latency, packet_loss)
    
    Action: Each drone can:
        - Move in 4 directions + hover (discrete): UP, DOWN, LEFT, RIGHT, HOVER
    
    Reward: Based on:
        - Coverage increase
        - Collision avoidance
        - Energy efficiency
        - Task completion
        - No-fly zone compliance
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # Action constants
    ACTION_FORWARD = 0  # +y
    ACTION_BACKWARD = 1  # -y
    ACTION_LEFT = 2     # -x
    ACTION_RIGHT = 3    # +x
    ACTION_ASCEND = 4   # +z
    ACTION_DESCEND = 5  # -z
    ACTION_HOVER = 6    # no movement
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = kwargs
        
        # Environment parameters - 5km² grid with height (3D space)
        self.grid_size = np.array(config.get('grid_size', [5000, 5000, 500]), dtype=np.float64)  # meters (5km x 5km x 500m)
        self.cell_size = config.get('cell_size', 10.0)  # 10m resolution
        self.grid_cells = tuple((self.grid_size / self.cell_size).astype(int))  # Keep as tuple for consistency
        
        # Indian urban environment parameters
        self.max_building_height = config.get('max_building_height', 200.0)  # meters (typical Indian high-rise)
        self.min_building_height = config.get('min_building_height', 10.0)   # meters (typical Indian low-rise)
        self.building_density = config.get('building_density', 0.15)  # 15% overall coverage
        self.commercial_density = 0.4  # Higher density in commercial zones
        self.residential_density = 0.2  # Medium density in residential areas
        self.suburban_density = 0.1   # Lower density in suburban areas
        self.air_corridor_width = config.get('air_corridor_width', 100.0)  # meters (wider corridors)
        self.air_corridor_spacing = config.get('air_corridor_spacing', 300.0)  # meters
        
        # Weather conditions
        self.enable_weather = config.get('enable_weather', True)
        self.wind_speed = np.zeros(3)  # 3D wind vector
        self.wind_direction = 0.0  # radians
        self.turbulence_intensity = 0.0  # 0-1 scale
        self.visibility = 1.0  # 0-1 scale (affects sensor readings)
        
        # Drone configuration (20-50 drones)
        self.num_drones = config.get('num_drones', 20)
        self.max_drones = config.get('max_drones', 50)
        self.obs_radius = config.get('obs_radius', 100.0)  # 100m observation radius
        self.comm_range = config.get('comm_range', 500.0)  # 500m 5G communication range
        self.max_steps = config.get('mission_timeout', 1000)
        
        # Mission parameters
        self.mission_type = config.get('mission_type', 'coverage')
        self.coverage_target = config.get('coverage_target', 0.90)  # 90% coverage
        
        # Action/observation types
        self.action_type = 'discrete'  # UP, DOWN, LEFT, RIGHT, ASCEND, DESCEND, HOVER
        self.obs_type = config.get('obs_type', 'local')
        
        # Physics - optimized for better learning
        self.max_velocity = config.get('max_velocity', 15.0)  # 15 m/s (realistic for commercial drones)
        self.max_vertical_velocity = config.get('max_vertical_velocity', 5.0)  # 5 m/s vertical speed
        self.dt = config.get('dt', 1.0)  # Adjusted time step for noticeable movement
        
        # Obstacles (reduced density for easier learning)
        self.obstacle_density = config.get('obstacle_density', 0.12)  # Reduced from 0.20 to 0.12
        self.min_obstacle_density = 0.08  # Reduced from 0.10
        self.max_obstacle_density = 0.18  # Reduced from 0.30
        
        # No-fly zones (Drone Rules 2021 compliance)
        self.enable_no_fly_zones = config.get('enable_no_fly_zones', True)
        self.no_fly_zones = []  # List of (center, radius) tuples
        self.num_no_fly_zones = config.get('num_no_fly_zones', 1)  # Reduced from 2 to 1
        
        # Communication constraints (5G simulation)
        self.enable_comm_constraints = config.get('enable_comm_constraints', True)
        self.base_latency = config.get('base_latency', 0.050)  # 50ms base latency
        self.latency_variance = config.get('latency_variance', 0.150)  # 0-200ms range
        self.packet_loss_rate = config.get('packet_loss_rate', 0.10)  # 10% packet loss
        self.action_delay_buffer = deque(maxlen=3)  # Buffer for delayed actions
        
        # Domain randomization (sim-to-real gap mitigation)
        self.enable_randomization = config.get('enable_randomization', True)
        self.wind_noise_std = config.get('wind_noise_std', 0.05)  # 10% wind effect
        self.sensor_noise_std = config.get('sensor_noise_std', 0.05)  # 5% sensor noise
        self.dynamics_noise_std = config.get('dynamics_noise_std', 0.015)  # 3% dynamics noise
        
        # Rewards - Strict collision aversion for proper learning
        reward_config = config.get('reward_structure', {})
        self.reward_coverage = reward_config.get('coverage', 5.0)  # Increased from 1.0 to 5.0
        self.reward_collision = reward_config.get('collision_penalty', 0.0)  # No penalty for collisions
        self.reward_no_fly_zone = reward_config.get('no_fly_zone_penalty', 0.0)  # No no-fly penalty
        self.reward_energy = reward_config.get('energy_penalty', -0.01)
        self.reward_goal = reward_config.get('goal_reward', 50.0)
        self.reward_collision_loop = -10.0  # Minimal penalty for collision loops
        
        # Define action space - Discrete: FORWARD, BACKWARD, LEFT, RIGHT, ASCEND, DESCEND, HOVER
        self.action_space = spaces.Discrete(7)
        self.action_map = {
            self.ACTION_FORWARD: np.array([0, self.cell_size, 0], dtype=np.float64),       # Forward (+y)
            self.ACTION_BACKWARD: np.array([0, -self.cell_size, 0], dtype=np.float64),     # Backward (-y)
            self.ACTION_LEFT: np.array([-self.cell_size, 0, 0], dtype=np.float64),         # Left (-x)
            self.ACTION_RIGHT: np.array([self.cell_size, 0, 0], dtype=np.float64),         # Right (+x)
            self.ACTION_ASCEND: np.array([0, 0, self.cell_size/2], dtype=np.float64),      # Up (+z)
            self.ACTION_DESCEND: np.array([0, 0, -self.cell_size/2], dtype=np.float64),    # Down (-z)
            self.ACTION_HOVER: np.array([0, 0, 0], dtype=np.float64)                       # Hover
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
        self.steps = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.no_fly_violations = 0
        self.drone_collision_counts = None  # Track collisions per drone
        
        # Communication state
        self.packet_loss_mask = None
        self.latencies = None
        
        # Rendering
        self.render_mode = config.get('render_mode', None)
        self.fig = None
        self.ax = None
        self.trajectory_history = {i: [] for i in range(self.num_drones)}
    
    def _get_observation_dim(self) -> int:
        """Calculate observation dimension based on configuration."""
        dim = 0
        dim += 6  # position (2) + velocity (2) + latency (1) + packet_loss (1)
        local_grid_size = int(self.obs_radius / self.cell_size) * 2 + 1
        dim += local_grid_size * local_grid_size
        dim += (self.num_drones - 1) * 4  # teammates
        dim += 1  # no-fly zone distance
        dim += 1  # coverage percentage
        return dim
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state with weather conditions."""
        super().reset(seed=seed)
        
        if self.enable_randomization:
            self.obstacle_density = np.random.uniform(self.min_obstacle_density, self.max_obstacle_density)
        
        # Generate urban environment
        self.obstacle_map = self._generate_obstacles()
        
        if self.enable_no_fly_zones:
            self.no_fly_zones = self._generate_no_fly_zones()
        
        # Initialize drone positions in 3D
        self.drone_positions = self._initialize_positions()
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float64)  # 3D velocities
        
        # Initialize weather conditions
        if self.enable_weather:
            # Generate realistic wind conditions
            self.wind_speed = np.random.normal(0, 5, size=3)  # Mean wind speed 5 m/s
            self.wind_direction = np.random.uniform(0, 2*np.pi)  # Random wind direction
            self.turbulence_intensity = np.random.uniform(0, 0.3)  # 0-30% turbulence
            self.visibility = np.clip(np.random.normal(0.8, 0.2), 0.3, 1.0)  # Visibility conditions
        self.coverage_map = np.zeros(self.grid_cells, dtype=bool)
        self.packet_loss_mask = np.zeros(self.num_drones, dtype=bool)
        self.latencies = np.ones(self.num_drones, dtype=np.float64) * self.base_latency
        
        self.steps = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.no_fly_violations = 0  # Reset violations
        self._previous_coverage = 0  # Reset coverage tracking
        self.drone_collision_counts = np.zeros(self.num_drones)  # Reset collision counts
        
        self.trajectory_history = {i: [self.drone_positions[i].copy()] for i in range(self.num_drones)}
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def _generate_obstacles(self) -> np.ndarray:
        """Generate realistic Indian urban environment with buildings and air corridors."""
        # Initialize 3D obstacle map (x, y, z)
        obstacle_map = np.zeros(self.grid_cells, dtype=bool)
        building_heights = np.zeros(self.grid_cells[:2], dtype=np.float64)
        
        # Define zones (commercial, residential, suburban)
        center_x, center_y = self.grid_cells[0] // 2, self.grid_cells[1] // 2
        
        # Generate buildings for each zone
        for x in range(self.grid_cells[0]):
            for y in range(self.grid_cells[1]):
                # Calculate distance from city center
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                dist_normalized = dist_from_center / (min(center_x, center_y))
                
                # Determine zone and building probability
                if dist_normalized < 0.3:  # Commercial district (CBD)
                    building_prob = self.commercial_density
                    high_rise_prob = 0.15  # 15% chance of high-rise in CBD
                    max_height = self.max_building_height
                elif dist_normalized < 0.6:  # Residential zone
                    building_prob = self.residential_density
                    high_rise_prob = 0.05  # 5% chance of high-rise
                    max_height = self.max_building_height * 0.7
                else:  # Suburban area
                    building_prob = self.suburban_density
                    high_rise_prob = 0.01  # 1% chance of high-rise
                    max_height = self.max_building_height * 0.5
                
                # Add random variation to prevent sharp zone boundaries
                building_prob *= np.random.uniform(0.8, 1.2)
                
                # Determine if we should place a building
                if np.random.random() < building_prob:
                    # Determine building height based on zone
                    if np.random.random() < high_rise_prob:
                        height = np.random.uniform(50.0, max_height)
                    else:
                        height = np.random.uniform(self.min_building_height, 30.0)  # Typical 2-3 story buildings
                    
                    # Add some clustering effect
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.grid_cells[0] and 
                                0 <= ny < self.grid_cells[1] and
                                np.random.random() < 0.3):  # 30% chance of adjacent building
                                
                                # Adjacent buildings have similar but slightly different heights
                                adj_height = height * np.random.uniform(0.7, 1.3)
                                height_cells = int(adj_height / self.cell_size)
                                building_heights[nx, ny] = adj_height
                                obstacle_map[nx, ny, :height_cells] = True
                    
                    # Place the main building
                    height_cells = int(height / self.cell_size)
                    building_heights[x, y] = height
                    obstacle_map[x, y, :height_cells] = True
        
        # Add open spaces (parks, grounds, etc.)
        num_open_spaces = int(self.grid_cells[0] * self.grid_cells[1] * 0.005)
        for _ in range(num_open_spaces):
            center = np.random.randint(0, [self.grid_cells[0], self.grid_cells[1]], size=2)
            radius = np.random.randint(5, 15)
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = center[0] + dx, center[1] + dy
                    if (0 <= x < self.grid_cells[0] and 
                        0 <= y < self.grid_cells[1] and
                        dx*dx + dy*dy <= radius*radius):
                        # Clear any buildings in this area
                        building_heights[x, y] = 0
                        obstacle_map[x, y, :] = False
        
        # Add a few landmarks (temples, monuments, etc.)
        num_landmarks = 3
        for _ in range(num_landmarks):
            x = np.random.randint(0, self.grid_cells[0])
            y = np.random.randint(0, self.grid_cells[1])
            
            # Temples and monuments are usually medium height
            height = np.random.uniform(30.0, 60.0)
            height_cells = int(height / self.cell_size)
            
            # Create a distinctive shape (larger base, tapered top)
            for h in range(height_cells):
                radius = max(1, int(3 * (1 - h/height_cells)))
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_cells[0] and 
                            0 <= ny < self.grid_cells[1]):
                            building_heights[nx, ny] = height
                            obstacle_map[nx, ny, h] = True
        
        # Generate air corridors
        corridor_spacing_cells = int(self.air_corridor_spacing / self.cell_size)
        corridor_width_cells = int(self.air_corridor_width / self.cell_size)
        
        # Horizontal corridors
        for y in range(0, self.grid_cells[1], corridor_spacing_cells):
            corridor_height = int(100 / self.cell_size)  # 100m high corridors
            y_start = max(0, y - corridor_width_cells//2)
            y_end = min(self.grid_cells[1], y + corridor_width_cells//2)
            obstacle_map[:, y_start:y_end, corridor_height] = False  # Clear corridor
        
        # Vertical corridors
        for x in range(0, self.grid_cells[0], corridor_spacing_cells):
            corridor_height = int(150 / self.cell_size)  # 150m high corridors
            x_start = max(0, x - corridor_width_cells//2)
            x_end = min(self.grid_cells[0], x + corridor_width_cells//2)
            obstacle_map[x_start:x_end, :, corridor_height] = False  # Clear corridor
        
        # Store building heights for visualization
        self.building_heights = building_heights
        
        return obstacle_map
    
    def _generate_no_fly_zones(self) -> List[Tuple[np.ndarray, float]]:
        """Generate no-fly zones based on Drone Rules 2021."""
        no_fly_zones = []
        
        for _ in range(self.num_no_fly_zones):
            # Only use x,y coordinates for center position
            center = np.random.uniform(0, self.grid_size[:2], size=2)
            zone_type = np.random.choice(['airport', 'government', 'military'])
            if zone_type == 'airport':
                radius = min(500, min(self.grid_size)/10)  # Adjust for small grid
            elif zone_type == 'government':
                radius = 100
            else:
                radius = 400
            
            no_fly_zones.append((center, radius))
        
        return no_fly_zones
    
    def _initialize_positions(self) -> np.ndarray:
        """Initialize drone positions in 3D space avoiding obstacles and no-fly zones."""
        positions = []
        max_attempts = 1000
        min_dist = min(50.0, min(self.grid_size[:2])/self.num_drones)  # Adjust for small grid
        init_height = 100.0  # Initial height for drones
        
        for _ in range(self.num_drones):
            valid_pos = False
            attempts = 0
            
            while not valid_pos and attempts < max_attempts:
                attempts += 1
                pos_2d = np.random.uniform(0, self.grid_size[:2], size=2)
                pos = np.append(pos_2d, init_height)  # Add height component
                
                grid_pos = (pos_2d / self.cell_size).astype(int)
                grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)  # Fix for tuple
                height_cell = int(init_height / self.cell_size)
                height_cell = min(height_cell, self.grid_cells[2] - 1)
                
                # Check if there's an obstacle at any height up to the drone's position
                if self.obstacle_map[grid_pos[0], grid_pos[1], :height_cell].any():
                    continue
                
                if self._is_in_no_fly_zone(pos):
                    continue
                
                if len(positions) == 0:
                    valid_pos = True
                else:
                    min_dist_current = np.min(np.linalg.norm(np.array(positions) - pos, axis=1))
                    if min_dist_current > min_dist:
                        valid_pos = True
            
            if valid_pos:
                positions.append(pos)
            else:
                pos_2d = np.random.uniform(0, self.grid_size[:2], size=2)
                pos = np.append(pos_2d, init_height)
                positions.append(pos)
        
        positions = np.array(positions, dtype=np.float64)
        
        if self.enable_randomization:
            noise = np.random.normal(0, 10.0, size=positions.shape)
            noise[:, 2] *= 0.2  # Reduce vertical noise
            positions += noise
            positions = np.clip(positions, [0, 0, 50], self.grid_size)  # Minimum height of 50m
        
        return positions
    
    def _is_in_no_fly_zone(self, position: np.ndarray) -> bool:
        """Check if position is within any no-fly zone, considering height restrictions."""
        if not self.enable_no_fly_zones:
            return False
        
        for center, radius in self.no_fly_zones:
            # Check horizontal distance only (cylindrical no-fly zones)
            horizontal_dist = np.linalg.norm(position[:2] - center)
            if horizontal_dist < radius:
                # Height-based rules
                if position[2] < 150:  # No-fly below 150m near restricted areas
                    return True
                elif horizontal_dist < radius/2:  # Stricter height restriction near center
                    return position[2] < 300  # No-fly below 300m in core restricted area
        return False
    
    def step(self, actions: Union[np.ndarray, List]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment with communication constraints."""
        self.steps += 1
        
        # Convert actions to numpy array if needed
        if isinstance(actions, list):
            actions = np.array(actions)
        
        if self.enable_comm_constraints:
            actions = self._apply_communication_constraints(actions)
        
        self._update_positions(actions)
        self._update_coverage()
        
        rewards = self._calculate_reward()
        self.episode_reward += np.mean(rewards)  # Track mean reward for episode
        
        terminated = self._check_termination()
        truncated = self.steps >= self.max_steps
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _apply_communication_constraints(self, actions: np.ndarray) -> np.ndarray:
        """Apply 5G communication constraints: latency and packet loss."""
        modified_actions = actions.copy()
        
        self.packet_loss_mask = np.random.random(self.num_drones) < self.packet_loss_rate
        modified_actions[self.packet_loss_mask] = self.ACTION_HOVER
        
        self.latencies = self.base_latency + np.random.uniform(0, self.latency_variance, size=self.num_drones)
        
        delay_mask = np.random.random(self.num_drones) < 0.10
        if len(self.action_delay_buffer) > 0:
            prev_actions = self.action_delay_buffer[-1]
            modified_actions[delay_mask] = prev_actions[delay_mask]
        
        self.action_delay_buffer.append(actions.copy())
        
        return modified_actions
    
    def _update_positions(self, actions: np.ndarray):
        """Update drone positions based on actions with domain randomization."""
        for i in range(self.num_drones):
            velocity = self.action_map[actions[i]].astype(np.float64) * 1.5  # Increased from 0.8 to 1.5 for better coverage
            
            if self.enable_randomization:
                # 3D wind effects
                wind_effect = np.random.normal(0, self.wind_noise_std * self.max_velocity*0.5, size=3)
                wind_effect[2] *= 0.5  # Reduce vertical wind effect
                velocity += wind_effect
                
                # 3D dynamics noise
                dynamics_noise = np.random.normal(0, self.dynamics_noise_std * self.max_velocity*0.5, size=3)
                dynamics_noise[2] *= 0.3  # Reduce vertical dynamics noise
                velocity += dynamics_noise
                
                # Add influence of current wind conditions
                velocity += self.wind_speed * self.dt * 0.1  # Scale wind influence
            
            self.drone_velocities[i] = velocity
            new_pos = self.drone_positions[i] + velocity * self.dt
            
            # Better boundary handling with safety margins
            safety_margin = 10.0  # 10m safety margin from boundaries
            new_pos = np.clip(new_pos, 
                             [safety_margin, safety_margin, 50.0],  # Minimum height 50m
                             [self.grid_size[0] - safety_margin, 
                              self.grid_size[1] - safety_margin, 
                              self.grid_size[2] - safety_margin])
            
            # Implement robust building collision detection
            old_pos = self.drone_positions[i]
            movement_vector = new_pos - old_pos
            distance = np.linalg.norm(movement_vector)
            
            # Reduced collision checks for performance
            num_checks = max(5, int(distance / (self.cell_size * 2.0)))  # Fewer checks
            path_points = np.linspace(old_pos, new_pos, num=num_checks)
            
            # Add safety margin around drone
            safety_margin = self.cell_size * 0.5  # 0.5 cell safety buffer
            collision = False
            collision_point = None
            
            for point in path_points:
                # Simplified collision check for performance - only check center point
                check_point = point
                
                # Convert to grid coordinates
                grid_pos = (check_point[:2] / self.cell_size).astype(int)
                height_cell = int(check_point[2] / self.cell_size)
                
                if (0 <= grid_pos[0] < self.grid_cells[0] and 
                    0 <= grid_pos[1] < self.grid_cells[1] and 
                    0 <= height_cell < self.grid_cells[2]):
                    
                    # Check for building collision
                    if self.obstacle_map[grid_pos[0], grid_pos[1], height_cell]:
                        collision = True
                        collision_point = point
                        break
            
            if collision:
                # Track collision count per drone
                self.drone_collision_counts[i] += 1
                
                # Reduced logging for performance - only print every 10th collision
                if self.steps % 10 == 0:
                    print(f"Building collision detected for drone {i} at position {collision_point}")
                
                # If drone has too many collisions, simple emergency teleport
                if self.drone_collision_counts[i] > 10:  # Lower threshold
                    # Simple emergency teleport to guaranteed safe position
                    new_pos = np.array([
                        self.grid_size[0]/2, 
                        self.grid_size[1]/2, 
                        self.grid_size[2] - 5  # Very high altitude - above all buildings
                    ])
                    self.drone_collision_counts[i] = 0
                    self.drone_velocities[i] = np.zeros(3)  # Stop movement
                    print(f"Drone {i} emergency teleported to center high altitude")
                    # Skip all collision recovery - teleport is final
                    collision = False  # Mark as no collision so position update works
                    continue  # Skip the rest of collision recovery
                
                # Simple collision recovery - just move up
                new_pos = old_pos.copy()
                new_pos[2] = min(old_pos[2] + 30, self.grid_size[2] - 50)  # Move up 30m
            
            in_no_fly = self._is_in_no_fly_zone(new_pos)
            
            if not collision and not in_no_fly:
                self.drone_positions[i] = new_pos
                self.trajectory_history[i].append(new_pos.copy())
            else:
                # If collision detected, stop the drone
                self.drone_velocities[i] = np.zeros(3, dtype=np.float64)
                if collision:
                    print(f"Drone {i} stopped due to building collision")
                if in_no_fly:
                    self.no_fly_violations += 1  # Increment only once per violation
    
    def _update_coverage(self):
        """Update coverage map based on current drone positions in 3D space."""
        for pos in self.drone_positions:
            # Use only x,y coordinates for grid position
            grid_pos = (pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            
            # Coverage radius depends on altitude (higher altitude = wider coverage)
            altitude_factor = 1.0 + (pos[2] / 500.0)  # Increase coverage with height
            effective_radius = self.obs_radius * altitude_factor
            radius_cells = int(effective_radius / self.cell_size)
            
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    x, y = grid_pos[0] + dx, grid_pos[1] + dy
                    if (0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]):
                        dist = np.sqrt(dx**2 + dy**2) * self.cell_size
                        if dist <= effective_radius:
                            # Consider visibility conditions
                            if self.enable_weather:
                                coverage_prob = self.visibility * (1.0 - dist/effective_radius)
                                if np.random.random() < coverage_prob:
                                    self.coverage_map[x, y] = True
                            else:
                                self.coverage_map[x, y] = True
    
    def _calculate_reward(self) -> np.ndarray:
        """Calculate rewards for each agent based on current state."""
        rewards = np.zeros(self.num_drones)
        
        # Track coverage change (only reward NEW coverage)
        current_coverage = np.sum(self.coverage_map)
        if not hasattr(self, '_previous_coverage'):
            self._previous_coverage = 0
        
        coverage_increase = current_coverage - self._previous_coverage
        self._previous_coverage = current_coverage
        
        # Reward only for NEW coverage (scaled by coverage value)
        if coverage_increase > 0:
            rewards += (coverage_increase / self.coverage_map.size) * self.reward_coverage * 100
        
        # Small penalty for staying still (encourage exploration)
        for i in range(self.num_drones):
            if np.linalg.norm(self.drone_velocities[i]) < 0.1:
                rewards[i] -= 0.5  # Penalty for hovering
        
        # Per-agent collision penalties and separation rewards
        collision_penalties = np.zeros(self.num_drones)
        separation_rewards = np.zeros(self.num_drones)
        collisions_this_step = 0
        
        # Apply collision loop penalties (severe learning signal)
        for i in range(self.num_drones):
            if self.drone_collision_counts[i] > 5:
                # Calculate penalty per drone (don't share penalty)
                penalty = min(-200.0, self.reward_collision_loop * self.drone_collision_counts[i])
                collision_penalties[i] += penalty
        
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                vertical_dist = abs(self.drone_positions[i][2] - self.drone_positions[j][2])
                
                # Check both horizontal and vertical separation (relaxed requirements)
                if dist < 20.0 and vertical_dist < 10.0:  # More reasonable separation requirements
                    collision_penalties[i] += self.reward_collision
                    collision_penalties[j] += self.reward_collision
                    collisions_this_step += 1
                    # Reduced logging - only print every 20th warning
                    if self.steps % 20 == 0:
                        print(f"Collision warning: Drones {i} and {j} - Horizontal: {dist:.1f}m, Vertical: {vertical_dist:.1f}m")
                elif dist < 50.0 and vertical_dist < 20.0 and self.steps > 10:
                    # Reduced logging - only print every 50th proximity warning
                    if self.steps % 50 == 0:
                        print(f"Proximity warning: Drones {i} and {j} - Horizontal: {dist:.1f}m, Vertical: {vertical_dist:.1f}m")
                elif dist > 30.0 and vertical_dist > 15.0:
                    # Reward good separation (relaxed requirements)
                    separation_rewards[i] += 0.5
                    separation_rewards[j] += 0.5
        
        rewards += collision_penalties
        rewards += separation_rewards  # Add separation rewards
        
        self.collision_count += collisions_this_step
        
        # Per-agent no-fly zone penalties
        for i, pos in enumerate(self.drone_positions):
            if self._is_in_no_fly_zone(pos):
                rewards[i] += self.reward_no_fly_zone
                self.no_fly_violations += 1  # Ensure violation is counted
        
        # Per-agent energy penalties
        energy_used = np.linalg.norm(self.drone_velocities, axis=1)
        rewards += energy_used * self.reward_energy
        
        # Goal achievement bonus (shared) - calculate coverage ratio
        coverage_ratio = current_coverage / self.coverage_map.size
        if coverage_ratio >= self.coverage_target:
            rewards += self.reward_goal
        
        return rewards
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
        return coverage_ratio >= self.coverage_target
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all drones with sensor noise."""
        observations = {}
        
        for i in range(self.num_drones):
            obs_parts = []
            
            own_pos = self.drone_positions[i] / self.grid_size
            own_vel = self.drone_velocities[i] / self.max_velocity
            
            if self.enable_randomization:
                own_pos += np.random.normal(0, self.sensor_noise_std, size=3)  # 3D noise
                own_vel += np.random.normal(0, self.sensor_noise_std, size=3)  # 3D noise
            
            comm_status = np.array([self.latencies[i], float(self.packet_loss_mask[i])])
            
            obs_parts.extend([own_pos, own_vel, comm_status])
            
            local_map = self._get_local_map(i)
            obs_parts.append(local_map.flatten())
            
            teammate_obs = self._get_teammate_observations(i)
            obs_parts.append(teammate_obs)
            
            no_fly_dist = self._get_nearest_no_fly_distance(i)
            obs_parts.append(np.array([no_fly_dist]))
            
            coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
            obs_parts.append(np.array([coverage_ratio]))
            
            observations[i] = np.concatenate(obs_parts).astype(np.float32)
        
        return observations
    
    def _get_local_map(self, drone_idx: int) -> np.ndarray:
        """Get local obstacle map around drone in 3D."""
        pos = self.drone_positions[drone_idx]
        grid_pos = (pos / self.cell_size).astype(int)
        radius_cells = int(self.obs_radius / self.cell_size)
        height_radius = int(100 / self.cell_size)  # 100m vertical visibility
        
        # Create 3D local map
        local_map = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1, 2 * height_radius + 1))
        
        current_height = grid_pos[2]
        min_z = max(0, current_height - height_radius)
        max_z = min(self.grid_cells[2], current_height + height_radius + 1)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                x, y = grid_pos[0] + dx, grid_pos[1] + dy
                if (0 <= x < self.grid_cells[0] and 0 <= y < self.grid_cells[1]):
                    # Get a vertical slice of obstacles
                    for dz in range(-height_radius, height_radius + 1):
                        z = int(current_height + dz)
                        if 0 <= z < self.grid_cells[2]:
                            local_map[dx + radius_cells, dy + radius_cells, dz + height_radius] = \
                                self.obstacle_map[x, y, z]
        
        # Flatten the 3D map to 1D for the observation space
        return local_map.flatten()
    
    def _get_teammate_observations(self, drone_idx: int) -> np.ndarray:
        """Get relative positions of teammates within comm range."""
        own_pos = self.drone_positions[drone_idx]
        teammate_obs = []
        
        for j in range(self.num_drones):
            if j != drone_idx:
                other_pos = self.drone_positions[j]
                relative_pos = (other_pos - own_pos) / self.comm_range
                dist = np.linalg.norm(other_pos - own_pos)
                in_range = 1.0 if dist <= self.comm_range else 0.0
                other_latency = self.latencies[j]
                teammate_obs.extend([relative_pos[0], relative_pos[1], in_range, other_latency])
        
        return np.array(teammate_obs)
    
    def _get_nearest_no_fly_distance(self, drone_idx: int) -> float:
        """Get distance to nearest no-fly zone (normalized), considering height."""
        if not self.enable_no_fly_zones or len(self.no_fly_zones) == 0:
            return 1.0
        
        pos = self.drone_positions[drone_idx]
        min_dist = float('inf')
        
        for center, radius in self.no_fly_zones:
            # Calculate horizontal distance only
            horizontal_dist = np.linalg.norm(pos[:2] - center) - radius
            
            # Height-based distance calculation
            if pos[2] < 150:  # Below 150m
                height_penalty = (150 - pos[2]) / 150
            else:
                height_penalty = 0
                
            # Combine horizontal and vertical components
            effective_dist = horizontal_dist * (1 - height_penalty)
            min_dist = min(min_dist, effective_dist)
        
        return np.clip(min_dist / np.linalg.norm(self.grid_size[:2]), 0, 1)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        coverage_ratio = np.sum(self.coverage_map) / self.coverage_map.size
        obstacle_hits = self._count_obstacle_collisions()
        
        return {
            'coverage': coverage_ratio,
            'steps': self.steps,
            'episode_reward': self.episode_reward,
            'num_drones': self.num_drones,
            'collisions': self.collision_count,
            'no_fly_violations': self.no_fly_violations,
            'obstacle_hits': obstacle_hits,
            'avg_latency': np.mean(self.latencies) if self.enable_comm_constraints else 0.0,
            'packet_loss_count': np.sum(self.packet_loss_mask),
            'obstacle_hit_rate': obstacle_hits / self.num_drones if self.num_drones > 0 else 0
        }
    
    def _count_obstacle_collisions(self) -> int:
        """Count number of drones currently colliding with obstacles."""
        hits = 0
        for pos in self.drone_positions:
            grid_pos = (pos[:2] / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells[:2]) - 1)
            height_cell = int(pos[2] / self.cell_size)
            height_cell = min(height_cell, self.grid_cells[2] - 1)
            
            if height_cell < self.grid_cells[2]:
                if self.obstacle_map[grid_pos[0], grid_pos[1], height_cell]:
                    hits += 1
        return hits
    
    def render(self):
        """Render the environment using Matplotlib."""
        if self.render_mode == "human":
            self._render_matplotlib()
            plt.draw()
            plt.pause(0.001)  # Small pause to update the plot
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_matplotlib(self):
        """Render 3D urban environment using Matplotlib."""
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(15, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Set viewing limits
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_zlim(0, self.grid_size[2])
        
        # Set good viewing angle for urban environment
        self.ax.view_init(elev=30, azim=45)
        coverage_pct = np.sum(self.coverage_map) / self.coverage_map.size
        self.ax.set_title(f'Urban Skyspace - Step {self.steps} - Coverage: {coverage_pct:.1%}')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Height (meters)')
        
        # Render buildings more efficiently
        building_mask = self.building_heights > 0
        x_pos = np.where(building_mask)[0] * self.cell_size
        y_pos = np.where(building_mask)[1] * self.cell_size
        heights = self.building_heights[building_mask]
        
        if len(x_pos) > 0:  # Only if there are buildings to draw
            # Create building faces all at once
            dx = dy = np.ones_like(x_pos) * self.cell_size
            colors = np.tile([0.7, 0.7, 0.7, 0.5], (len(x_pos), 1))  # Gray with alpha
            
            self.ax.bar3d(
                x_pos, y_pos, np.zeros_like(x_pos),
                dx, dy, heights,
                color=colors,
                shade=True,
                zsort='max'
            )
        
        # Render air corridors
        corridor_spacing = int(self.air_corridor_spacing / self.cell_size)
        corridor_width = int(self.air_corridor_width / self.cell_size)
        
        # Horizontal corridors
        for y in range(0, self.grid_cells[1], corridor_spacing):
            corridor_height = 100  # meters
            self.ax.plot(
                [0, self.grid_size[0]],
                [y * self.cell_size, y * self.cell_size],
                [corridor_height, corridor_height],
                'b--', alpha=0.3, linewidth=2
            )
        
        if self.enable_no_fly_zones:
            # Draw no-fly zones as vertical cylinders
            for idx, (center, radius) in enumerate(self.no_fly_zones):
                # Create points for the cylinder
                z = np.linspace(0, self.grid_size[2], 50)
                theta = np.linspace(0, 2*np.pi, 50)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = radius * np.cos(theta_grid) + center[0]
                y_grid = radius * np.sin(theta_grid) + center[1]
                
                # Plot the cylinder surface
                self.ax.plot_surface(x_grid, y_grid, z_grid, 
                                   color='red', alpha=0.1,
                                   label='No-Fly Zone' if idx == 0 else '')
                
                # Plot top and bottom circles
                theta = np.linspace(0, 2*np.pi, 100)
                x = radius * np.cos(theta) + center[0]
                y = radius * np.sin(theta) + center[1]
                self.ax.plot(x, y, np.zeros_like(x), 'r-', alpha=0.5)
                self.ax.plot(x, y, np.ones_like(x) * self.grid_size[2], 'r-', alpha=0.5)
        
        coverage_points = np.argwhere(self.coverage_map)
        if len(coverage_points) > 0:
            coverage_coords = coverage_points * self.cell_size
            self.ax.scatter(coverage_coords[:, 0], coverage_coords[:, 1],
                           c='lightgreen', s=10, alpha=0.3, marker='s', label='Covered Area')
        
        colors = cm.rainbow(np.linspace(0, 1, self.num_drones))
        for i in range(self.num_drones):
            if len(self.trajectory_history[i]) > 1:
                traj = np.array(self.trajectory_history[i])
                self.ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.5, linewidth=1)
        
        # Render weather conditions (wind vectors)
        if self.enable_weather:
            scale = 1000  # Scale for wind vector visualization
            wind_points = np.array([
                [0, 0, 200],
                [self.grid_size[0], 0, 200],
                [0, self.grid_size[1], 200],
                [self.grid_size[0], self.grid_size[1], 200]
            ])
            
            for point in wind_points:
                self.ax.quiver(
                    point[0], point[1], point[2],
                    self.wind_speed[0] * scale,
                    self.wind_speed[1] * scale,
                    self.wind_speed[2] * scale,
                    color='cyan', alpha=0.3
                )
        
        # Render drones in 3D
        for i, pos in enumerate(self.drone_positions):
            if self.packet_loss_mask[i]:
                color = 'orange'
            elif self._is_in_no_fly_zone(pos):
                color = 'red'
            else:
                color = colors[i]
            
            # Draw drone body
            self.ax.scatter(pos[0], pos[1], pos[2], c=[color], s=100,
                          edgecolors='black', linewidths=2, zorder=5)
            
            # Draw vertical line to ground for better depth perception
            self.ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]],
                        '--', color=color, alpha=0.3)
            
            # Add drone label with altitude
            self.ax.text(pos[0], pos[1], pos[2] + 20,
                        f'D{i}\n{pos[2]:.0f}m',
                        ha='center', fontsize=8)
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        # Add stats text in 3D space
        stats_text = (f'Collisions: {self.collision_count} | '
                     f'No-Fly Violations: {self.no_fly_violations} | '
                     f'Obstacle Hits: {self._count_obstacle_collisions()}/{self.num_drones} | '
                     f'Packet Loss: {np.sum(self.packet_loss_mask)}/{self.num_drones}')
        
        self.ax.text2D(0.02, 0.02, stats_text, transform=self.ax.transAxes,
                      fontsize=10, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.pause(0.001)
        plt.draw()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array."""
        img = np.ones((*self.grid_cells, 3), dtype=np.uint8) * 255
        
        img[self.obstacle_map] = [100, 100, 100]
        img[self.coverage_map] = [200, 255, 200]
        
        for pos in self.drone_positions:
            grid_pos = (pos / self.cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, np.array(self.grid_cells) - 1)  # Fix for tuple
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
    
    # Create environment with specific settings for collision testing
    env = UrbanGridEnv(
        render_mode="human",
        num_drones=5,  # Fewer drones for clearer testing
        enable_comm_constraints=False,  # Disable for testing
        enable_weather=False,  # Disable for testing
        grid_size=[1000, 1000, 300],  # Smaller grid for focused testing
        cell_size=10.0,  # Finer resolution
        building_density=0.3  # Higher density for collision testing
    )
    
    def print_drone_building_status():
        """Print detailed information about drone positions relative to buildings."""
        for i, pos in enumerate(env.drone_positions):
            grid_pos = (pos[:2] / env.cell_size).astype(int)
            height_cell = int(pos[2] / env.cell_size)
            height_cell = min(height_cell, env.grid_cells[2] - 1)
            
            if 0 <= grid_pos[0] < env.grid_cells[0] and 0 <= grid_pos[1] < env.grid_cells[1]:
                has_building = env.obstacle_map[grid_pos[0], grid_pos[1], height_cell]
                building_height = 0
                for h in range(env.grid_cells[2]):
                    if env.obstacle_map[grid_pos[0], grid_pos[1], h]:
                        building_height = h * env.cell_size
                
                print(f"Drone {i}:")
                print(f"  Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                print(f"  Grid Position: {grid_pos}, Height Cell: {height_cell}")
                print(f"  Building Present: {has_building}")
                print(f"  Building Height: {building_height:.1f}m")
                print(f"  Collision Check: {'COLLISION!' if has_building else 'Safe'}\n")
    
    # Reset environment and get initial state
    obs, info = env.reset()
    print("\nEnvironment initialized for collision testing...")
    print(f"Grid size: {env.grid_size}")
    print(f"Number of drones: {env.num_drones}")
    print("\nInitial drone positions and building status:")
    print_drone_building_status()
    
    done = False
    truncated = False
    step_count = 0
    max_steps = 50  # Shorter test for collision focus
    collision_count = 0
    previous_positions = None
    
    try:
        while not done and not truncated and step_count < max_steps:
            # Store previous positions
            previous_positions = env.drone_positions.copy()
            
            # Random actions for each drone
            actions = np.random.randint(0, env.action_space.n, size=env.num_drones)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(actions)
            
            # Check for building collisions
            for i, (prev_pos, curr_pos) in enumerate(zip(previous_positions, env.drone_positions)):
                if np.array_equal(prev_pos, curr_pos) and not np.array_equal(actions[i], env.ACTION_HOVER):
                    # Drone didn't move despite trying to - might be collision
                    grid_pos = (curr_pos[:2] / env.cell_size).astype(int)
                    height_cell = int(curr_pos[2] / env.cell_size)
                    if (0 <= grid_pos[0] < env.grid_cells[0] and 
                        0 <= grid_pos[1] < env.grid_cells[1] and
                        height_cell < env.grid_cells[2]):
                        if env.obstacle_map[grid_pos[0], grid_pos[1], height_cell]:
                            collision_count += 1
                            print(f"\nStep {step_count} - Building Collision Detected!")
                            print(f"Drone {i} attempted to move but hit building")
                            print_drone_building_status()
            
            # Render the current state
            env.render()
            
            # Print status every few steps
            if step_count % 5 == 0:
                print(f"\nStep {step_count}")
                print(f"Total Building Collisions: {collision_count}")
                print(f"Current Positions:")
                print_drone_building_status()
            
            step_count += 1
            time.sleep(0.2)  # Slower for better observation
            
            # Check if window was closed
            if not plt.get_fignums():
                print("Window closed by user.")
                break
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("\nEpisode finished.")
        print(f"Final Coverage: {info['coverage']:.2%}")
        print(f"Total Collisions: {info['collisions']}")
        print(f"No-Fly Zone Violations: {info['no_fly_violations']}")
        print(f"Obstacle Hits: {info['obstacle_hits']}")
        env.close()
        plt.ioff()  # Disable interactive mode