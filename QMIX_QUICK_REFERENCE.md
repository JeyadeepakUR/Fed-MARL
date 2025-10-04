# QMIX Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Run Validation Tests (ALWAYS DO THIS FIRST!)
```bash
cd F:\thales\federated-marl-drone
python -m core.training.test_qmix
```

### 2. Basic Training (10 drones, 100 episodes)
```bash
python -m core.training.train_central
```

### 3. Custom Training
```python
from core.envs.urban_grid_env import UrbanGridEnv
from core.training.train_central import train_marl

env = UrbanGridEnv(num_drones=10)
agent, metrics = train_marl(env, n_episodes=1000)
```

## 📊 Success Criteria (Module 2 Spec)

| Metric | Target | How to Check |
|--------|--------|--------------|
| Collision Rate | <10% | Check eval output: "Collision Rate: X%" |
| Average Reward | >0.5 | Check eval output: "Average Reward: X.XX" |
| Coverage | ≥90% | Check info: "Coverage: XX%" |

## 🔧 Key Parameters

### Agent Configuration
```python
QMIXAgent(
    n_agents=10,           # Number of drones
    lr=0.0005,             # Learning rate (tune if not converging)
    epsilon_start=1.0,     # Start exploration
    epsilon_end=0.05,      # End exploration
    epsilon_decay=0.995,   # Decay per episode
    hidden_dim=256,        # Network size (reduce if memory issues)
    batch_size=32,         # Batch size (if using replay buffer)
    use_replay_buffer=False  # False = single-step (faster initially)
)
```

### Training Configuration
```python
train_marl(
    n_episodes=1000,           # Total episodes
    max_steps_per_episode=200, # Steps per episode
    eval_frequency=50,         # Evaluate every N episodes
    save_frequency=100,        # Save every N episodes
    use_replay_buffer=False,   # Start False, switch to True later
    target_collision_rate=0.10,
    target_avg_reward=0.5
)
```

## 🐛 Common Fixes

### Problem: Shape Error
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```
**Fix**: Already fixed in updated `qmix_model.py` - ensure you're using the new version

### Problem: Not Learning
```
Average Reward stays around 0 or negative
```
**Fixes**:
1. Check rewards in environment: `print(rewards)`
2. Increase learning rate: `lr=0.001`
3. Reduce epsilon decay: `epsilon_decay=0.99`
4. Verify collision penalty is working

### Problem: Too Many Collisions
```
Collision Rate > 20%
```
**Fixes**:
1. Increase penalty: Edit `urban_grid_env.py`, set `reward_collision = -200.0`
2. Train longer (500+ episodes)
3. Reduce drone speed: Modify `action_map` velocity multiplier
4. Check observation includes collision info

### Problem: Slow Training
```
Training taking too long
```
**Fixes**:
1. Disable rendering: `render_mode=None`
2. Reduce grid size: `grid_size=[2000, 2000, 300]`
3. Reduce drones temporarily: `num_drones=5`
4. Use GPU (if available)
5. Run on Google Colab with GPU

## 📈 Training Progression

### Expected Behavior
```
Episodes 1-100:   High exploration, random actions
                  Collision rate: 30-50%
                  Reward: -10 to 0

Episodes 100-300: Learning basic coordination
                  Collision rate: 15-25%
                  Reward: 0 to 0.3

Episodes 300-500: Improving cooperation
                  Collision rate: 10-15%
                  Reward: 0.3 to 0.5

Episodes 500+:    Near optimal behavior
                  Collision rate: <10%
                  Reward: >0.5
```

### Not Following This Pattern?
- **Stuck at high collisions**: Increase collision penalty
- **Stuck at low reward**: Check reward function, increase coverage reward
- **No improvement**: Reduce learning rate, check for bugs

## 💾 Model Management

### Save Model
```python
agent.save_model("my_model.pth")
```

### Load Model
```python
agent = QMIXAgent(n_agents=10, obs_dim=9307, action_dim=7)
agent.load_model("my_model.pth")
```

### Continue Training
```python
agent.load_model("checkpoints/qmix_episode_500.pth")
# Continue with train_marl() - epsilon and optimizer state preserved
```

## 🎯 Hyperparameter Tuning Guide

### If Loss is Exploding
- Reduce learning rate: `lr=0.0001`
- Enable gradient clipping: `grad_clip=10.0` (already enabled)
- Check for NaN in observations

### If Loss is Flat
- Increase learning rate: `lr=0.001`
- Check if gradients are flowing: Add debug prints
- Verify experiences are diverse (not stuck)

### If Epsilon Decreasing Too Fast
- Reduce decay: `epsilon_decay=0.999`
- Increase end value: `epsilon_end=0.1`

### If Epsilon Decreasing Too Slow
- Increase decay: `epsilon_decay=0.99`
- Reduce end value: `epsilon_end=0.01`

## 📝 Logging Tips

### Enable Detailed Logs
```python
train_marl(env, verbose=True)  # Already default
```

### Check Specific Metrics
```python
# In training loop
print(f"Epsilon: {agent.epsilon}")
print(f"Buffer size: {len(agent.replay_buffer)}")
print(f"Collision info: {info['collisions']}")
```

### Save Training Curves
```python
import matplotlib.pyplot as plt

agent, metrics = train_marl(env)

plt.plot(metrics['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.savefig('training_curve.png')
```

## 🔬 Testing Trained Model

### Quick Evaluation
```python
from core.training.train_central import evaluate_agent

agent.load_model("checkpoints/qmix_best.pth")
metrics = evaluate_agent(env, agent, n_eval_episodes=20)

print(f"Collision Rate: {metrics['collision_rate']*100:.1f}%")
print(f"Average Reward: {metrics['avg_reward']:.2f}")
print(f"Coverage: {metrics['avg_coverage']*100:.1f}%")
```

### Visual Demo
```python
env = UrbanGridEnv(render_mode="human", num_drones=10)
agent.load_model("checkpoints/qmix_best.pth")
agent.epsilon = 0.0  # Pure exploitation

obs_dict, _ = env.reset()
for step in range(200):
    obs = [obs_dict[i] for i in range(10)]
    actions = [agent.get_action(obs[i], i) for i in range(10)]
    obs_dict, rewards, done, truncated, info = env.step(actions)
    env.render()
    if done or truncated:
        break
```

## 🎓 Understanding Output

### Training Log Example
```
Episode  100/1000 | Steps:  5000 | ε: 0.605 | Reward:   -5.23 | Coverage: 45.2% | Collisions: 12.3 (24.6%) | Violations:  0.0 | Loss: 0.5432
```

**Interpretation**:
- `ε: 0.605` - Still exploring (60% random actions)
- `Reward: -5.23` - Negative (lots of collisions early on - normal!)
- `Coverage: 45.2%` - Covering less than half the area
- `Collisions: 12.3 (24.6%)` - 2.46 collisions per drone (too high, will improve)
- `Loss: 0.5432` - TD error (should decrease over time)

### Evaluation Output Example
```
Evaluation at Episode 500
======================================
  Average Reward: 0.67 (target: >0.5)
  Collision Rate: 8.50% (target: <10%)
  Coverage: 92.3%
  No-Fly Violations: 0.0
  ✓ SUCCESS CRITERIA MET!
```

## 🚨 Emergency Fixes

### Training Crashed
```python
# Load last checkpoint and continue
agent = QMIXAgent(n_agents=10, obs_dim=9307, action_dim=7)
agent.load_model("checkpoints/qmix_episode_500.pth")
# Resume training from episode 500
```

### Out of Memory
```python
# Reduce these parameters
agent = QMIXAgent(
    hidden_dim=128,        # Was 256
    batch_size=16,         # Was 32
    buffer_capacity=10000  # Was 50000
)
```

### Can't Find Module
```bash
# Ensure you're in correct directory
cd F:\thales\federated-marl-drone

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .
```

## 📞 Need Help?

1. ✅ Run validation tests: `python -m core.training.test_qmix`
2. ✅ Check this guide for your specific issue
3. ✅ Read error messages carefully
4. ✅ Enable verbose logging: `verbose=True`
5. ✅ Test with smaller config: 5 drones, 50 episodes

## 🎯 Next Steps After Success

Once you achieve <10% collision rate and >0.5 avg reward:

1. **Scale to 20 drones**
   ```python
   env = UrbanGridEnv(num_drones=20)
   agent, metrics = train_marl(env, n_episodes=1000)
   ```

2. **Enable replay buffer** for better stability
   ```python
   train_marl(env, use_replay_buffer=True, batch_size=32)
   ```

3. **Fine-tune for specific scenarios**
   - Dense urban areas (more buildings)
   - High communication constraints
   - Adverse weather conditions

4. **Prepare for Module 3: Federated MARL**
   - Save local models from different edge devices
   - Implement federated averaging
   - Test distributed training

---

**Remember**: Module 2 Success = <10% collisions + >0.5 reward on 20-drone swarm! 🎯
