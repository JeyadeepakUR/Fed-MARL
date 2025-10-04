import pytest
import numpy as np
from core.envs.urban_grid_env import UrbanGridEnv

@pytest.fixture
def env():
    return UrbanGridEnv(num_drones=5, render_mode=None, enable_comm_constraints=False)

def test_initialization(env):
    obs, info = env.reset()
    assert env.obstacle_map.shape == env.grid_cells
    assert np.sum(env.obstacle_map) > 0
    assert len(env.drone_positions) == 5
    assert np.all(env.drone_positions >= 0) and np.all(env.drone_positions <= env.grid_size)

def test_step_movement(env):
    obs, _ = env.reset()
    actions = np.ones(env.num_drones, dtype=int) * env.ACTION_FORWARD
    obs_new, _, _, _, _ = env.step(actions)
    # Compare observations properly using numpy's any and not_equal
    assert np.any(np.not_equal(obs_new[0], obs[0]))  # Some movement

def test_no_fly_zone_penalty(env):
    obs, _ = env.reset()
    if not env.no_fly_zones:  # Skip if no no-fly zones defined
        print("Skipping no-fly zone test - no zones defined")
        return
        
    no_fly_center = env.no_fly_zones[0][0]
    env.drone_positions[0] = np.array([no_fly_center[0], no_fly_center[1], 100.0])  # Place in no-fly zone at 100m height
    
    actions = np.ones(env.num_drones, dtype=int) * env.ACTION_HOVER  # Actions for all drones
    _, reward, _, _, info = env.step(actions)
    assert reward < 0  # Penalty applied
    assert info['no_fly_violations'] > 0

def test_coverage_increase(env):
    obs, _ = env.reset()
    initial_coverage = np.sum(env.coverage_map)
    obs, _, _, _, _ = env.step(np.ones(env.num_drones, dtype=int) * env.ACTION_FORWARD)
    env._update_coverage()
    assert np.sum(env.coverage_map) > initial_coverage

def test_building_collisions():
    # Create environment with single drone for easier testing
    env = UrbanGridEnv(num_drones=1, render_mode=None)
    
    # Test multiple collision scenarios
    test_cases = [
        {
            'name': 'Direct collision',
            'building_pos': (5, 5),
            'building_height': 5,
            'start_pos': np.array([3, 5, 3]),
            'action': env.ACTION_RIGHT  # Move towards building
        },
        {
            'name': 'Diagonal collision',
            'building_pos': (7, 7),
            'building_height': 6,
            'start_pos': np.array([5, 5, 4]),
            'action': env.ACTION_FORWARD  # Move diagonally towards building
        },
        {
            'name': 'Vertical collision',
            'building_pos': (5, 5),
            'building_height': 8,
            'start_pos': np.array([5, 5, 2]),
            'action': env.ACTION_ASCEND  # Move up into building
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        env.reset()
        
        # Clear obstacle map and set up single building
        env.obstacle_map[:] = False
        building_pos = case['building_pos']
        building_height = case['building_height']
        env.obstacle_map[building_pos[0], building_pos[1], :building_height] = True
        
        # Position drone
        env.drone_positions[0] = case['start_pos'].copy()
        old_pos = env.drone_positions[0].copy()
        
        # Try to move through building
        for step in range(5):
            obs, reward, done, _, info = env.step(np.array([case['action']]))
            new_pos = env.drone_positions[0]
            
            # Convert position to grid coordinates
            grid_pos = (new_pos[:2] / env.cell_size).astype(int)
            height_cell = int(new_pos[2] / env.cell_size)
            
            # Test 1: Drone should never be inside a building
            assert not env.obstacle_map[
                int(new_pos[0]/env.cell_size), 
                int(new_pos[1]/env.cell_size), 
                int(new_pos[2]/env.cell_size)
            ], f"{case['name']}: Drone entered building space at step {step}"
            
            # Test 2: Check safety margin around buildings
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    adj_x = grid_pos[0] + dx
                    adj_y = grid_pos[1] + dy
                    if (0 <= adj_x < env.grid_cells[0] and 
                        0 <= adj_y < env.grid_cells[1] and
                        env.obstacle_map[adj_x, adj_y, height_cell]):
                        
                        # Calculate actual distance to building
                        building_center = np.array([adj_x + 0.5, adj_y + 0.5]) * env.cell_size
                        distance = np.linalg.norm(new_pos[:2] - building_center)
                        assert distance >= env.cell_size * 0.4, \
                            f"{case['name']}: Drone too close to building at step {step} (distance={distance:.2f})"
            
            # Test 3: Ensure drone made progress unless blocked
            if step == 0:  # Only check first step to account for collision avoidance
                assert np.any(new_pos != old_pos), \
                    f"{case['name']}: Drone did not move from starting position"
        
        print(f"{case['name']}: All collision checks passed")

if __name__ == "__main__":
    pytest.main(["-v", __file__])