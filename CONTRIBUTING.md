# 🤝 Contributing to Fed-MARL

Thank you for your interest in contributing to Fed-MARL! This document provides guidelines and information for contributors.

## 🚀 **Getting Started**

### **Development Setup**

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Fed-MARL.git
   cd Fed-MARL
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv fedenv
   source fedenv/bin/activate  # Linux/Mac
   # or
   fedenv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Run Tests**
   ```bash
   python -m core.training.test_qmix
   pytest tests/
   ```

## 📋 **Contribution Guidelines**

### **Types of Contributions**

- 🐛 **Bug Fixes**: Fix issues in existing code
- ✨ **New Features**: Add new functionality
- 📚 **Documentation**: Improve docs and tutorials
- 🧪 **Tests**: Add or improve test coverage
- 🎨 **Code Quality**: Refactor and optimize code
- 🌍 **Environments**: Create new simulation environments

### **Development Workflow**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific tests
   pytest tests/test_qmix.py
   
   # Run with coverage
   pytest --cov=core tests/
   ```

4. **Code Quality Checks**
   ```bash
   # Format code
   black core/ tests/
   isort core/ tests/
   
   # Lint code
   flake8 core/ tests/
   
   # Type checking
   mypy core/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new environment visualization"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 **Code Standards**

### **Python Style Guide**

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for function signatures
- **Docstrings**: Document all public functions and classes
- **Naming**: Use descriptive variable and function names

### **Example Code Style**

```python
def train_agent(
    env: IndianSkyscapeEnv,
    n_episodes: int = 500,
    learning_rate: float = 5e-4
) -> Tuple[QMIXAgent, Dict[str, float]]:
    """
    Train a QMIX agent in the Indian Skyscape environment.
    
    Args:
        env: The training environment
        n_episodes: Number of training episodes
        learning_rate: Learning rate for optimization
        
    Returns:
        Tuple of trained agent and training metrics
        
    Raises:
        ValueError: If learning_rate is not positive
    """
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    # Implementation here...
    pass
```

### **Commit Message Format**

Use conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add 3D visualization for drone trajectories
fix: resolve collision detection in Indian Skyscape
docs: update README with new training results
test: add integration tests for QMIX agent
```

## 🧪 **Testing Guidelines**

### **Test Structure**

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
├── performance/    # Performance and benchmark tests
└── fixtures/       # Test data and fixtures
```

### **Writing Tests**

```python
import pytest
from core.envs.indian_skyscape_env import IndianSkyscapeEnv

def test_environment_initialization():
    """Test that environment initializes correctly."""
    env = IndianSkyscapeEnv(num_drones=5)
    obs, info = env.reset()
    
    assert env.num_drones == 5
    assert len(obs) == 5
    assert info['coverage'] == 0.0

@pytest.mark.parametrize("num_drones", [5, 10, 20])
def test_scalability(num_drones):
    """Test environment scalability."""
    env = IndianSkyscapeEnv(num_drones=num_drones)
    obs, _ = env.reset()
    assert len(obs) == num_drones
```

### **Test Coverage**

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include performance benchmarks for critical paths

## 📚 **Documentation Standards**

### **Docstring Format**

Use Google-style docstrings:

```python
def calculate_reward(
    self, 
    coverage_increase: float, 
    collision_penalty: float
) -> float:
    """
    Calculate reward for drone actions.
    
    Args:
        coverage_increase: Amount of new area covered
        collision_penalty: Penalty for collisions
        
    Returns:
        Total reward value
        
    Example:
        >>> reward = calculate_reward(0.1, 0.0)
        >>> print(f"Reward: {reward}")
        Reward: 1.0
    """
    return coverage_increase * 10.0 + collision_penalty
```

### **README Updates**

When adding new features:
- Update the main README.md
- Add usage examples
- Update the feature table
- Include performance metrics if applicable

## 🐛 **Bug Reports**

### **Bug Report Template**

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10]
- Python: [e.g., 3.9.0]
- PyTorch: [e.g., 1.11.0]

**Additional Context**
Any other context about the problem.
```

## ✨ **Feature Requests**

### **Feature Request Template**

```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How would you like this to work?

**Alternatives**
Any alternative solutions considered?

**Additional Context**
Any other context or screenshots.
```

## 🏗️ **Architecture Guidelines**

### **Adding New Environments**

1. **Inherit from Base Class**
   ```python
   class NewEnvironment(gym.Env):
       def __init__(self, **kwargs):
           super().__init__()
           # Implementation
   ```

2. **Implement Required Methods**
   - `reset()`: Initialize environment
   - `step()`: Execute action
   - `render()`: Visualization
   - `close()`: Cleanup

3. **Add Tests**
   ```python
   def test_new_environment():
       env = NewEnvironment()
       # Test implementation
   ```

### **Adding New Agents**

1. **Follow QMIX Pattern**
   ```python
   class NewAgent:
       def __init__(self, **kwargs):
           # Initialize networks
           
       def get_actions(self, observations):
           # Action selection
           
       def train_step(self, **kwargs):
           # Training logic
   ```

2. **Add Integration Tests**
   ```python
   def test_new_agent_training():
       agent = NewAgent()
       # Test training workflow
   ```

## 📊 **Performance Guidelines**

### **Benchmarking**

- Add performance benchmarks for new features
- Compare against existing implementations
- Document performance characteristics

### **Memory Usage**

- Monitor memory usage in training loops
- Use appropriate data types (float32 vs float64)
- Clear unused variables and tensors

### **Optimization**

- Profile code before optimizing
- Use vectorized operations where possible
- Consider GPU acceleration for large computations

## 🎯 **Project Priorities**

### **High Priority**
- 🐛 Bug fixes in core functionality
- 📊 Performance improvements
- 🧪 Test coverage improvements
- 📚 Documentation updates

### **Medium Priority**
- ✨ New environment types
- 🤖 Alternative MARL algorithms
- 📈 Advanced visualization features
- 🔧 Configuration improvements

### **Low Priority**
- 🎨 UI/UX improvements
- 🌍 Localization
- 📱 Mobile support
- 🔌 Plugin architecture

## 📞 **Getting Help**

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: your.email@example.com
- **Discord**: [Join our community](https://discord.gg/fedmarl)

## 🏆 **Recognition**

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- Annual contributor awards

Thank you for contributing to Fed-MARL! 🚁🤖
