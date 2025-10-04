# Pre-Training Checklist

## ✅ Before You Start Training

### 1. Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated: `.\fedenv\Scripts\activate`
- [ ] In correct directory: `F:\thales\federated-marl-drone`

### 2. Dependencies Installed
```bash
pip install torch>=2.0.0
pip install numpy>=1.24.0
pip install gymnasium>=0.29.0
pip install matplotlib>=3.7.0
pip install pyyaml>=6.0
```

Verify:
```bash
python -c "import torch; import numpy; import gymnasium; print('✓ All imports successful')"
```

### 3. Files in Place
Check these files exist and are updated:
- [ ] `core/agents/qmix_model.py` (updated with fixes)
- [ ] `core/training/train_central.py` (updated)
- [ ] `core/training/test_qmix.py` (new)
- [ ] `core/training/demo_runner.py` (new)
- [ ] `core/envs/urban_grid_env.py` (existing)

### 4. Run Validation Tests
```bash
python -m core.training.test_qmix
```

Expected output:
```
========================================
TEST SUMMARY
========================================
  Environment Validation............... ✓ PASSED
  Agent Initialization................. ✓ PASSED
  Single-Step Training................. ✓ PASSED
  Replay Buffer Training............... ✓ PASSED
  Shape Consistency.................... ✓ PASSED
  Model Save/Load...................... ✓ PASSED
  Mini Training Loop................... ✓ PASSED

========================================
Results: 7/7 tests passed
========================================

🎉 ALL TESTS PASSED! Your QMIX implementation is ready for training.
```

- [ ] All 7 tests passed

### 5. System Resources
- [ ] **Disk Space**: At least 1GB free (for checkpoints)
- [ ] **RAM**: At least 8GB recommended
- [ ] **CPU**: Multi-core recommended for faster training
- [ ] **GPU** (optional): NVIDIA GPU with CUDA for 10x speedup

Check disk space:
```bash
# Windows
dir F:\thales\federated-marl-drone

# Should see plenty of free space
```

### 6. Training Configuration Decision
Choose your training mode:

**Option A: Quick Validation (Recommended First)**
- [ ] 5 drones, 100 episodes
- [ ] Time: 5-10 minutes
- [ ] Command: `python -m core.training.demo_runner` (choose option 1)

**Option B: Full Module 2 Training**
- [ ] 10 drones, 1000 episodes  
- [ ] Time: 2-3 hours (CPU) or 30-60 min (GPU)
- [ ] Command: `python -m core.training.train_central`

**Option C: Custom Training**
- [ ] Edit parameters in `train_central.py` or create custom script
- [ ] Test with smaller config first

---

## 🚀 Training Execution Checklist

### Before Starting
- [ ] Close unnecessary applications (free up RAM)
- [ ] Ensure computer won't sleep during training
- [ ] Have a text editor open to monitor checkpoint files
- [ ] Know where to find logs and checkpoints

### During Training - What to Watch

#### First 10 Episodes
- [ ] Training starts without errors
- [ ] Episode rewards are being calculated
- [ ] Epsilon is decreasing (starts at 1.0)
- [ ] No NaN or Inf values in loss
- [ ] Coverage is non-zero

**Red Flags**:
- ❌ Loss = NaN or Inf → Stop and check for bugs
- ❌ All rewards = 0 → Check reward function
- ❌ No improvement after 50 episodes → Tune hyperparameters

#### Episodes 100-300
- [ ] Collision rate decreasing (should drop from ~40% to ~20%)
- [ ] Average reward trending upward
- [ ] Epsilon around 0.3-0.6
- [ ] Loss stabilizing (not jumping wildly)
- [ ] Coverage improving

**Expected Progress**:
- Episode 100: Collision rate ~30%, Reward ~-5 to 0
- Episode 200: Collision rate ~20%, Reward ~0 to 0.3
- Episode 300: Collision rate ~15%, Reward ~0.3 to 0.5

#### Episodes 500-1000
- [ ] Collision rate approaching target (<10%)
- [ ] Average reward >0.5
- [ ] Epsilon low (0.05-0.15)
- [ ] Evaluation shows success criteria met
- [ ] Checkpoints being saved

**Target Metrics**:
- Collision rate: <10% ✓
- Average reward: >0.5 ✓
- Coverage: >90% ✓

### After Training

#### Immediate Checks
- [ ] Final evaluation ran successfully
- [ ] Success criteria displayed
- [ ] Best model saved in checkpoints/
- [ ] Final model saved
- [ ] Training time logged

#### Verify Checkpoints
```bash
# Check checkpoint directory
dir checkpoints\

# Should see:
# qmix_best.pth
# qmix_final.pth
# qmix_episode_100.pth, qmix_episode_200.pth, etc.
```

- [ ] At least 3 checkpoint files exist
- [ ] File sizes are reasonable (10-50MB each)

#### Test Loaded Model
```python
from core.agents.qmix_model import QMIXAgent

agent = QMIXAgent(n_agents=10, obs_dim=9307, action_dim=7)
agent.load_model("checkpoints/qmix_best.pth")
print(f"Model loaded: epsilon={agent.epsilon:.3f}")
```

- [ ] Model loads without errors
- [ ] Epsilon value makes sense (should be low, ~0.05)

---

## 📊 Success Validation Checklist

### Module 2 Criteria
Run final validation:
```bash
python -m core.training.train_central
# Wait for evaluation output at end
```

Check results against criteria:

#### Required Metrics
- [ ] **Collision Rate < 10%**: _________% (must be green/pass)
- [ ] **Average Reward > 0.5**: _________ (must be green/pass)
- [ ] **Coverage ≥ 90%**: _________% (target for mission completion)

#### 10-Drone Performance
- [ ] Training completed without crashes
- [ ] Model converged (loss stabilized)
- [ ] No critical errors in logs
- [ ] Success message displayed

#### 20-Drone Scalability Test
Optional but recommended:
```bash
python -m core.training.demo_runner
# Choose option 4: 20-Drone Scalability Test
```

- [ ] 20-drone agent can be created
- [ ] Training progresses without errors
- [ ] Collision rate reasonable (<15% acceptable for 20 drones)
- [ ] System handles increased complexity

---

## 🐛 Troubleshooting Checklist

### If Tests Fail
- [ ] Re-read error message carefully
- [ ] Check QMIX_QUICK_REFERENCE.md for common issues
- [ ] Verify all files are updated versions
- [ ] Try with smaller configuration (3 drones, 20 episodes)
- [ ] Check Python version (should be 3.8+)

### If Training Crashes
- [ ] Check available RAM (close other apps)
- [ ] Reduce batch_size or hidden_dim
- [ ] Check disk space for checkpoints
- [ ] Look for error in traceback
- [ ] Load last checkpoint and resume

### If Not Learning
- [ ] Verify rewards are non-zero
- [ ] Check epsilon is decreasing
- [ ] Increase learning rate temporarily
- [ ] Reduce epsilon_decay (explore more)
- [ ] Check collision detection is working

### If Too Slow
- [ ] Disable rendering (render_mode=None)
- [ ] Reduce grid_size
- [ ] Reduce num_drones temporarily
- [ ] Use GPU if available
- [ ] Close background applications

---

## 📁 File Organization Checklist

### Before Training
Organize your workspace:
```
F:\thales\federated-marl-drone\
├── checkpoints\              (will be created)
├── demo_checkpoints\         (for quick demos)
├── logs\                     (optional, for detailed logs)
└── results\                  (optional, for plots/analysis)
```

Create directories:
```bash
mkdir checkpoints
mkdir demo_checkpoints
mkdir logs
mkdir results
```

- [ ] Directories created
- [ ] Have write permissions

### During Training
Monitor these locations:
- [ ] `checkpoints\` - New .pth files appearing
- [ ] Console output - Training progress
- [ ] Task Manager - RAM/CPU usage

### After Training
Backup important files:
```bash
# Backup best model
copy checkpoints\qmix_best.pth results\qmix_best_10drone.pth

# Backup final model  
copy checkpoints\qmix_final.pth results\qmix_final_10drone.pth
```

- [ ] Models backed up
- [ ] Training logs saved (if any)
- [ ] Results documented

---

## 🎯 Final Pre-Flight Checklist

### Ready to Train?
- [ ] All validation tests passed (7/7)
- [ ] Dependencies installed and verified
- [ ] System resources adequate
- [ ] Training mode selected
- [ ] Checkpoints directory ready
- [ ] Understand what to monitor during training
- [ ] Know success criteria
- [ ] Have troubleshooting guide handy

### Time Estimates Confirmed
- [ ] Quick demo (5 drones): ~10 minutes
- [ ] Full training (10 drones): ~2-3 hours
- [ ] Scalability test (20 drones): ~4-6 hours
- [ ] Computer will be available for full duration

### Launch Commands Ready
```bash
# Step 1: Validate (always do this first)
python -m core.training.test_qmix

# Step 2: Quick demo (optional but recommended)
python -m core.training.demo_runner

# Step 3: Full training
python -m core.training.train_central

# Step 4: Evaluate results
# (automatic at end of training)
```

---

## 🚁 Ready for Takeoff!

If all boxes are checked, you're ready to start training!

### Launch Sequence:
1. ✅ Open terminal in project directory
2. ✅ Activate virtual environment
3. ✅ Run validation tests
4. ✅ Start training command
5. ✅ Monitor progress
6. ✅ Verify results

### Expected Timeline:
```
[0 min]     Launch training
[10 min]    First 100 episodes complete
[30 min]    Learning stabilizes
[1 hour]    Halfway through training
[2-3 hours] Training complete
[+5 min]    Final evaluation and saving
```

### Success Indicators:
- ✓ Training completes without crashes
- ✓ Collision rate < 10%
- ✓ Average reward > 0.5
- ✓ Model checkpoints saved
- ✓ "SUCCESS CRITERIA MET!" message

---

## 📞 Need Help During Training?

### Quick Reference
1. **QMIX_QUICK_REFERENCE.md** - Common issues and fixes
2. **README_QMIX.md** - Full documentation
3. **IMPLEMENTATION_SUMMARY.md** - What was implemented

### Emergency Stops
- Press `Ctrl+C` to stop training gracefully
- Last checkpoint will be saved
- Can resume from last checkpoint

### Key Files to Check
- `core/agents/qmix_model.py` - Agent implementation
- `core/training/train_central.py` - Training script
- `core/envs/urban_grid_env.py` - Environment

---

## ✨ Final Notes

**Remember**: 
- First training run might not meet criteria - that's normal!
- You can always continue training from checkpoints
- Start small (5 drones) then scale up
- GPU makes training 10x faster if available

**Module 2 Goal**: 
Train QMIX agents that achieve <10% collision rate and >0.5 average reward on a 20-drone swarm. You're fully set up to achieve this!

**Good luck with your training!** 🚁🤖✨

---

Last updated: Based on complete QMIX implementation with all fixes applied.
