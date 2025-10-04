# Fed-MARL: Federated Multi-Agent Reinforcement Learning for Drone Swarms

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**A cutting-edge framework for training drone swarms using QMIX in realistic Indian urban environments**

[Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 🎯 **Project Overview**

Fed-MARL is a sophisticated Multi-Agent Reinforcement Learning (MARL) framework designed specifically for **drone swarm coordination** in realistic urban environments. The project features:

- **Indian Skyscape Environment**: Realistic urban simulation with temple complexes, markets, residential areas, and commercial zones
- **QMIX Algorithm**: State-of-the-art cooperative MARL for multi-drone coordination
- **Federated Learning Ready**: Architecture designed for distributed training across multiple environments
- **Comprehensive Evaluation**: Detailed metrics for coverage, collision avoidance, and swarm coordination

### **Demo Results**
```
✅ Average Reward: 179.81 (target: >50) - EXCEEDED!
✅ Collision Rate: 0.3% (target: <5%) - EXCELLENT!
⚠️  Coverage: 15.3% (target: >40%) - Learning in progress
✅ Training Time: 42.4 minutes for 500 episodes
```

---

## 🏗️ **Architecture**

### **Core Components**

```
fed-marl-drone/
├── core/envs/
│   ├── indian_skyscape_env.py    # Realistic Indian urban environment
│   └── urban_grid_env.py         # Generic urban grid environment
├── core/agents/
│   └── qmix_model.py             # QMIX multi-agent implementation
├── core/training/
│   ├── train_indian_skyscape.py  # Main training script
│   ├── train_central.py          # Central training coordinator
│   └── demo_runner.py            # Quick demo and evaluation
├── experiments/
│   └── train_urban_swarm.py      # Experimental configurations
└── tests/
    ├── test_qmix.py              # Comprehensive validation tests
    └── test_integration.py       # Integration testing
```

### **Environment Features**

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Temple Complex** | Sacred center with tall structures | Realistic cultural context |
| **Market Area** | Dense commercial zones | Complex navigation challenges |
| **Residential** | Mixed-height buildings | Diverse obstacle patterns |
| **Commercial** | High-rise business districts | Vertical navigation training |
| **Industrial** | Large open spaces | Efficient pathfinding practice |

---

## 🚀 **Quick Start**

### **1. Clone and Setup**
```bash
git clone https://github.com/yourusername/Fed-MARL.git
cd Fed-MARL

# Create virtual environment
python -m venv fedenv
source fedenv/bin/activate  # Linux/Mac
# or
fedenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **2. Validate Installation**
```bash
python -m core.training.test_qmix
```

### **3. Visualize Environment**
```bash
# View the Indian Skyscape environment
python core/envs/indian_skyscape_env.py
```

### **4. Quick Training Demo**
```bash
# Train 5 drones for 100 episodes (5-10 minutes)
python -m core.training.demo_runner
```

---

## **Installation**

### **Requirements**
- Python 3.8+
- PyTorch 1.9+
- NumPy, Matplotlib, Gymnasium
- YAML configuration support

### **Full Installation**
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib gymnasium pyyaml

# Optional: For advanced visualization
pip install seaborn plotly

# Optional: For federated learning (future)
pip install flwr
```

### **GPU Support (Recommended)**
```bash
# CUDA-enabled PyTorch (faster training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎮 **Usage**

### **Training Modes**

#### **1. Indian Skyscape Training (Recommended)**
```bash
# Full training: 500 episodes, 10 drones
python -m core.training.train_indian_skyscape

# Custom configuration
python -m core.training.train_indian_skyscape \
    --episodes 1000 \
    --drones 20 \
    --grid-size 5000 5000 300
```

#### **2. Demo Runner (Interactive)**
```bash
# Interactive menu with multiple options
python -m core.training.demo_runner
```

#### **3. Central Training**
```bash
# Centralized training coordinator
python -m core.training.train_central
```

### **Configuration Examples**

#### **Small Scale (Testing)**
```python
env = IndianSkyscapeEnv(
    num_drones=5,
    grid_size=[2000, 2000, 200],
    cell_size=25.0,
    building_density=0.03
)
```

#### **Production Scale**
```python
env = IndianSkyscapeEnv(
    num_drones=20,
    grid_size=[5000, 5000, 300],
    cell_size=20.0,
    building_density=0.05,
    enable_weather=True
)
```

---

## 📊 **Results**

### **Training Performance**

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Average Reward** | >50 | 179.81 | ✅ **EXCEEDED** |
| **Collision Rate** | <5% | 0.3% | ✅ **EXCELLENT** |
| **Coverage** | >40% | 15.3% | ⚠️ **LEARNING** |
| **Training Time** | <2 hours | 42.4 min | ✅ **FAST** |

### **Environment Comparison**

| Environment | Collision Rate | Training Stability | Realism |
|-------------|----------------|-------------------|---------|
| **Urban Grid** | 24.0% | ❌ Unstable | ⭐⭐⭐ |
| **Indian Skyscape** | 0.3% | ✅ Stable | ⭐⭐⭐⭐⭐ |

### **Scalability Results**

- **5 Drones**: 100 episodes in 5-10 minutes
- **10 Drones**: 500 episodes in 42.4 minutes  
- **20 Drones**: Ready for federated training

---

## 🧪 **Testing & Validation**

### **Run All Tests**
```bash
python -m core.training.test_qmix
```

### **Individual Tests**
```bash
# Environment validation
python -c "from core.envs.indian_skyscape_env import IndianSkyscapeEnv; env = IndianSkyscapeEnv(); print('✅ Environment OK')"

# Agent initialization
python -c "from core.agents.qmix_model import QMIXAgent; agent = QMIXAgent(5, 100, 7); print('✅ Agent OK')"
```

### **Performance Benchmarks**
```bash
# Quick performance test
python -c "
import time
from core.envs.indian_skyscape_env import IndianSkyscapeEnv
env = IndianSkyscapeEnv(num_drones=10)
obs, _ = env.reset()
start = time.time()
for _ in range(100):
    actions = [env.action_space.sample() for _ in range(10)]
    obs, _, _, _, _ = env.step(actions)
print(f'✅ 100 steps in {time.time()-start:.2f}s')
"
```

---

## 🔧 **Configuration**

### **Environment Parameters**
```yaml
# configs/indian_skyscape.yaml
grid_size: [5000, 5000, 300]  # meters
cell_size: 20.0               # resolution
num_drones: 10                # swarm size
building_density: 0.05        # obstacle density
max_building_height: 150.0    # meters
air_corridor_width: 150.0     # clear paths
```

### **Training Parameters**
```yaml
# configs/training.yaml
n_episodes: 500
max_steps_per_episode: 300
learning_rate: 5e-4
epsilon_start: 1.0
epsilon_end: 0.1
target_sync_freq: 25
```

### **Reward Structure**
```yaml
# configs/rewards.yaml
coverage: 10.0              # reward for new coverage
collision_penalty: -1.0     # penalty for collisions
exploration_reward: 0.1     # bonus for exploration
energy_penalty: -0.005      # energy efficiency
```

---

## 🎯 **Success Criteria**

### **Module 2 Requirements**
- ✅ **Multi-Agent Coordination**: QMIX implementation
- ✅ **Urban Environment**: Realistic Indian skyscape
- ✅ **Collision Avoidance**: <5% collision rate achieved
- ✅ **Scalability**: 5-20 drone support
- ✅ **Training Stability**: No crashes or infinite loops

### **Performance Targets**
- 🎯 **Coverage**: >40% (currently 15.3% - learning in progress)
- 🎯 **Reward**: >50 (✅ achieved: 179.81)
- 🎯 **Efficiency**: <2 hours training (✅ achieved: 42.4 minutes)

---

## 🚀 **Future Work**

### **Phase 1: Enhanced Learning**
- [ ] **Curriculum Learning**: Start simple, increase complexity
- [ ] **Transfer Learning**: Pre-train on simple environments
- [ ] **Hyperparameter Optimization**: Automated tuning

### **Phase 2: Federated Learning**
- [ ] **Distributed Training**: Multiple environments
- [ ] **Privacy-Preserving**: Secure model aggregation
- [ ] **Edge Computing**: Real-time deployment

### **Phase 3: Real-World Integration**
- [ ] **Hardware-in-the-Loop**: Real drone testing
- [ ] **Sim-to-Real Transfer**: Domain adaptation
- [ ] **Mission Planning**: Task-specific optimization

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/Fed-MARL.git
cd Fed-MARL
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black core/
isort core/
```

### **Contributing Areas**
- 🧪 **Testing**: More comprehensive test coverage
- 📊 **Metrics**: Additional performance measures
- 🎮 **Environments**: New scenario types
- 🤖 **Algorithms**: Alternative MARL methods
- 📚 **Documentation**: Tutorials and examples

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 **Support & Contact**

- **Issues**: [GitHub Issues](https://github.com/yourusername/Fed-MARL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Fed-MARL/discussions)
- **Email**: your.email@example.com

---

## 🙏 **Acknowledgments**

- **QMIX Paper**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
- **Gymnasium**: OpenAI's successor to OpenAI Gym
- **PyTorch**: Deep learning framework
- **Indian Urban Planning**: Realistic environment design inspiration

---

<div align="center">

**🌟 Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/Fed-MARL.svg?style=social&label=Star)](https://github.com/yourusername/Fed-MARL)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/Fed-MARL.svg?style=social&label=Fork)](https://github.com/yourusername/Fed-MARL/fork)

</div>