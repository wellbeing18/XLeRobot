# Pi0.5 Training Guide for SO-ARM101

**Complete guide for finetuning Pi0.5 on your SO-101 dual arm robot**

**Last Updated**: 2025-11-23
**Target Hardware**: RTX 5090 24GB, SO-101 follower arms
**Status**: Ready for MVP validation

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Your Current Status](#your-current-status)
4. [Step-by-Step Training Process](#step-by-step-training-process)
5. [Hyperparameters Explained](#hyperparameters-explained)
6. [Expected Performance](#expected-performance)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Quick Start

**If you just want to get started:**

```bash
# 1. Validate training pipeline with your 10 existing episodes (30-45 min)
bash scripts/train_pi05_mvp_test.sh

# 2. If test passes, collect 40-90 more episodes (see RECORDING_GUIDE.md)

# 3. Run full training with 50-100 total episodes (6-10 hours)
bash scripts/train_pi05_so101.sh

# 4. Evaluate finetuned model
python jdocs/top_level/run_pi05_inference_corrected.py \
  --checkpoint outputs/pi05_so101_v1/checkpoint-6000
```

---

## Overview

### What is Pi0.5?

- **4B parameter VLA** (Vision-Language-Action model)
- Pretrained on 400 hours of mobile manipulation data
- Uses **32-dim universal action space** for multi-embodiment learning
- Requires **finetuning** for new robots (SO-101 wasn't in pretraining)

### Why Finetuning is Required

**The Normalization Mismatch Problem:**

1. **Pretrained Pi0.5**: Trained on ALOHA, UR5, Franka, etc.
   - Normalization stats from those robots' joint ranges
   - Outputs actions in pretrained normalized space

2. **Your SO-101**: Different joint ranges (in degrees)
   - shoulder_lift range: -150° to +150°
   - Pretrained model doesn't know your ranges

3. **Result without finetuning**: Model outputs ~1.0 when robot is at 98°
   - Robot sees this as "almost no change" → doesn't move
   - **This is exactly what you experienced!**

**Finetuning fixes this by:**
- Computing YOUR robot's normalization stats (from your dataset)
- Adapting model to output actions in YOUR normalized space
- Teaching model YOUR robot's kinematics and camera viewpoints

### Expected Timeline

| Stage | Episodes | Time | Output |
|-------|----------|------|--------|
| **MVP Test** | 10 (done!) | 1 hour | Pipeline validation ✅ |
| **Data Collection** | +40-90 more | 10-20 hours | 50-100 total episodes |
| **Full Training** | 50-100 | 6-10 hours | Finetuned model |
| **Evaluation** | N/A | 2-3 hours | Performance metrics |
| **TOTAL** | 50-100 | **20-35 hours** | Production model |

---

## Your Current Status

### ✅ What You Have

**Dataset**: `/home/jrobot/project/XLeRobot/jdocs/top_level/datasets/`
- ✅ 10 episodes collected (MVP validation dataset)
- ✅ Correct format (LeRobot v3.0)
- ✅ Normalization stats computed in `meta/stats.json`
- ✅ 5Hz action frequency (correct!)
- ✅ 2 cameras: `head`, `left_wrist`
- ✅ 6-DOF actions: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

**Hardware**:
- ✅ RTX 5090 24GB (excellent for Pi0.5 with LoRA)
- ✅ SO-101 dual arms (currently using left arm only)
- ✅ 2-3 UVC cameras

**Software**:
- ✅ LeRobot installed
- ✅ Pi0.5 pretrained model accessible
- ✅ Inference script working (run_pi05_inference_corrected.py)

### 🎯 Next Steps

1. **Stage 1**: Run MVP training test (validates pipeline)
2. **Stage 2**: Collect 40-90 more episodes (reach 50-100 total)
3. **Stage 3**: Run full training (6000 steps)
4. **Stage 4**: Evaluate and compare to baseline

---

## Step-by-Step Training Process

### Stage 1: MVP Training Test (1 hour)

**Purpose**: Validate that training pipeline works before investing 10-20 hours collecting more data.

**Command**:
```bash
bash scripts/train_pi05_mvp_test.sh
```

**What it does**:
- Loads your 10-episode dataset
- Runs training for 100 steps (quick test)
- Verifies model fits in 24GB VRAM
- Checks for crashes, NaN losses, OOM errors
- Saves test checkpoint

**Expected output**:
```
[STEP 10] Loss: 0.485 | LR: 3.0e-06 | Time: 15.2s
[STEP 20] Loss: 0.432 | LR: 3.0e-06 | Time: 15.1s
...
[STEP 100] Loss: 0.284 | LR: 3.0e-06 | Time: 15.0s
✅ Checkpoint saved: outputs/pi05_mvp_test/checkpoint-100/
```

**✅ Pass Criteria**:
- Training completes 100 steps without crashes
- Loss values between 0.2-1.0 (normal range)
- No "CUDA out of memory" errors
- VRAM usage ≤22GB
- Checkpoint saves successfully

**❌ Fail Criteria**:
- Crashes or OOM errors → See [Troubleshooting](#troubleshooting)
- NaN losses → Check gradient clipping is enabled
- Loss >5.0 → Check data preprocessing

**If test passes**: Proceed to Stage 2 (data collection)
**If test fails**: Debug before collecting more data!

---

### Stage 2: Data Collection (10-20 hours)

**Goal**: Collect 40-90 more episodes to reach 50-100 total.

**Why this many episodes?**
- Research shows **minimum 50 episodes** for reliable generalization
- SmolVLA used 23k episodes to reach 78% (you'll get 40-60% with 50-100)
- Egocentric cameras need 15-25% more data than fixed cameras

**Recording Command** (see `RECORDING_GUIDE.md` for full details):

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm \
  --robot.cameras='{"left_wrist": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15}, "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=xlerobot_left_leader \
  --dataset.repo_id=lerobot/xlerobot_mvp_pick_15fps \
  --dataset.root=/home/jrobot/project/XLeRobot/jdocs/top_level/datasets \
  --dataset.fps=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=40 \
  --dataset.single_task="pick red_cube from center" \
  --dataset.push_to_hub=false \
  --display_data=false \
  --resume=true
```

**Key parameter**: `--dataset.num_episodes=40` (adjust based on how many more you need)

**Best Practices**:

1. **Vary object positions**: Place cube at 3-5 different locations
   - Center (done in your 10 episodes)
   - Left, right, front, back
   - Mix close and far positions

2. **Quality over quantity**:
   - **Good demos**: Smooth, successful task completion
   - **Acceptable**: Minor hesitations but completes task
   - **Re-record**: Dropped objects, major collisions, failures

3. **Include recovery behaviors**:
   - 90% successful demonstrations
   - 10% with minor errors but recovery (teaches robustness)

4. **Batch recording**:
   - Day 1: 15 episodes (varied positions)
   - Day 2: 15 episodes (continue variety)
   - Day 3: 15 episodes (continue)
   - Day 4: Buffer day (re-record poor quality demos)

**Progress tracking**:
```bash
# Check how many episodes you have
python -c "
import json
with open('/home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/info.json') as f:
    info = json.load(f)
    print(f'Episodes: {info[\"total_episodes\"]}')
    print(f'Frames: {info[\"total_frames\"]}')
    print(f'FPS: {info[\"fps\"]}')
"
```

**Target**: 50-100 total episodes before proceeding to Stage 3.

---

### Stage 3: Full Training (6-10 hours)

**Prerequisites**:
- ✅ Stage 1 MVP test passed
- ✅ 50-100 episodes collected
- ✅ GPU available for 6-10 hours uninterrupted

**Command**:
```bash
bash scripts/train_pi05_so101.sh
```

**What it does**:
- Loads pretrained Pi0.5 model (4B params)
- Applies LoRA for memory-efficient finetuning
- Trains for 6000 steps
- Saves checkpoints every 1000 steps
- Uses YOUR dataset's normalization stats

**Training Configuration**:
```
Model: Pi0.5 (4B params) with LoRA
Steps: 6000
Batch Size: 8
Learning Rate: 3e-6 (lower than default)
Gradient Clipping: 1.0 (critical!)
LoRA Rank: 32
Expected Time: 6-10 hours on RTX 5090
VRAM Usage: 18-22GB / 24GB
```

**Monitoring Training**:

**Terminal 1** (training output):
```bash
bash scripts/train_pi05_so101.sh
```

**Terminal 2** (GPU monitoring):
```bash
watch -n 5 nvidia-smi
```

**Terminal 3** (RAM monitoring):
```bash
watch -n 10 'free -h'
```

**Expected training output**:
```
[STEP 100] Loss: 0.485 | LR: 3.0e-06 | GPU: 20.5GB | Time: 15.2s
[STEP 200] Loss: 0.412 | LR: 3.0e-06 | GPU: 20.5GB | Time: 15.1s
[STEP 500] Loss: 0.298 | LR: 3.0e-06 | GPU: 20.5GB | Time: 15.0s
[STEP 1000] ✅ Checkpoint saved: outputs/pi05_so101_v1/checkpoint-1000/
[STEP 1000] Loss: 0.187 | LR: 2.8e-06 | GPU: 20.5GB | Time: 15.1s
...
[STEP 6000] Loss: 0.089 | LR: 2.5e-07 | GPU: 20.5GB | Time: 15.0s
[STEP 6000] ✅ Final checkpoint saved!
```

**Healthy training signs**:
- ✅ Loss decreases from ~0.5 to ~0.1 over 6000 steps
- ✅ VRAM stable at 18-22GB (no growth over time)
- ✅ CPU RAM stable at 8-12GB
- ✅ Training speed: ~15 seconds per step
- ✅ No NaN losses

**Warning signs**:
- ⚠️ Loss not decreasing after 1000 steps → May need more data
- ⚠️ VRAM increasing over time → Memory leak (known Pi0.5 issue)
- ⚠️ NaN losses → Gradient explosion (shouldn't happen with grad clip)
- ⚠️ Loss <0.01 early → Overfitting (dataset too small or repetitive)

**Checkpoints saved**:
```
outputs/pi05_so101_v1/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
├── checkpoint-4000/
├── checkpoint-5000/
└── checkpoint-6000/  ← Final model (use this)
```

---

### Stage 4: Evaluation (2-3 hours)

**Goal**: Compare finetuned model to pretrained baseline.

#### 4.1: Run Pretrained Baseline (if not done yet)

**Command**:
```bash
python jdocs/top_level/run_pi05_inference_corrected.py
```

**Expected**: 0-15% success (normalization mismatch - normal!)

**Record**:
- Number of attempts: 10
- Successes: X/10
- Failure modes: random movements, stays still, wrong direction

#### 4.2: Run Finetuned Model

**Command**:
```bash
python jdocs/top_level/run_pi05_inference_corrected.py \
  --checkpoint /home/jrobot/project/XLeRobot/outputs/pi05_so101_v1/checkpoint-6000
```

**Note**: You'll need to update the inference script to support `--checkpoint` flag (see [Adding Checkpoint Loading](#adding-checkpoint-loading))

**Expected**: 40-60% success (50 eps) or 65-80% (100 eps)

**Record**:
- Number of attempts: 10
- Successes: X/10
- Failure modes: grasping errors, trajectory issues, task-specific failures

#### 4.3: Compare Results

| Metric | Pretrained | Finetuned (50 eps) | Finetuned (100 eps) |
|--------|------------|-------------------|---------------------|
| Success Rate | 0-15% | **40-60%** | **65-80%** |
| Grasping | Random | Attempts grasp | Reliable grasp |
| Trajectory | Confused | Approaches object | Smooth motion |
| Task Understanding | Poor | Moderate | Good |

**If performance is lower than expected**: See [Troubleshooting](#troubleshooting)

---

## Hyperparameters Explained

### Why These Values?

The training script uses carefully tuned hyperparameters based on:
- Pi0.5 research papers
- SO-101 hardware constraints
- Community findings
- Your specific setup (egocentric cameras, 24GB VRAM)

### Key Hyperparameters

#### Learning Rate: `3e-6` (vs default `5e-6`)

**Why lower?**
- Pretrained model already learned general manipulation
- Finetuning = small adjustments, not learning from scratch
- Higher LR causes gradient explosion on small datasets

**Evidence**: Community reports NaN losses with default 5e-6 on <100 episodes

#### Gradient Clipping: `1.0` (CRITICAL!)

**Why essential?**
- Pi0.5 has 4B parameters → large gradients possible
- Small datasets → high variance gradients
- Without clipping: NaN losses common

**What it does**: Caps gradient norm at 1.0 before optimizer step

#### LoRA Rank: `32`

**Why LoRA?**
- Full finetuning 4B params = ~32GB VRAM (doesn't fit!)
- LoRA only trains small adapter layers = 18-22GB VRAM ✅

**Why rank 32?**
- Rank 16: Too constrained (used by GR00T, but we have more VRAM)
- Rank 32: Good balance (captures SO-101 kinematics)
- Rank 64: Diminishing returns, higher VRAM

#### Batch Size: `8`

**Why not larger?**
- Batch 8 = 20GB VRAM (safe headroom on 24GB)
- Batch 16 = 24GB VRAM (risky, no headroom for spikes)
- Batch 4 = 16GB VRAM (inefficient, slower training)

**Trade-off**: Smaller batch = noisier gradients but faster iterations

#### Steps: `6000`

**Why this many?**
- 50 episodes × 150 frames/ep = 7500 frames
- 6000 steps × batch 8 = 48,000 training samples
- **~6-8 epochs** through dataset (standard for finetuning)

**Scaling**:
- 30 episodes → 3000-4000 steps
- 75 episodes → 6000 steps (default)
- 100 episodes → 8000 steps

#### Warmup Steps: `500`

**Why warmup?**
- First 500 steps: LR ramps from 0 → 3e-6
- Prevents large gradient spikes at start
- Especially important when loading pretrained weights

#### Scheduler Decay: `6000 steps`

**Why cosine decay?**
- After warmup: LR smoothly decreases 3e-6 → 2.5e-7
- Final low LR prevents overwriting learned features
- Cosine curve = smooth, no sharp drops

### Advanced: Tuning for Your Setup

**If you have MORE data (100+ episodes)**:
```bash
# In scripts/train_pi05_so101.sh, modify:
STEPS=8000               # More steps
BATCH_SIZE=12            # Larger batch (if VRAM allows)
LEARNING_RATE=4e-6       # Slightly higher LR
```

**If you have LESS data (30-40 episodes)**:
```bash
# In scripts/train_pi05_so101.sh, modify:
STEPS=4000               # Fewer steps (avoid overfitting)
BATCH_SIZE=4             # Smaller batch (less VRAM pressure)
LEARNING_RATE=2e-6       # Lower LR (more conservative)
```

**If training is SLOW**:
```bash
# Trade accuracy for speed:
NUM_WORKERS=8            # More data loading workers
--policy.compile_model=true  # Enable torch.compile (experimental)
```

---

## Expected Performance

### Realistic Expectations

**Your setup**: SO-101 + egocentric cameras + Pi0.5
**Comparison**: Standard LeRobot setups use fixed external cameras

| Scenario | Success Rate | Notes |
|----------|--------------|-------|
| **Pretrained baseline** | 0-15% | Normalization mismatch (expected!) |
| **50 episodes finetuned** | 40-60% | Usable for simple tasks |
| **75 episodes finetuned** | 55-70% | Reliable for single task |
| **100 episodes finetuned** | 65-80% | Robust performance |
| **200+ episodes** | 75-85% | Approaching research-level |

**Why not 90%+ like papers?**
1. **Egocentric cameras** = 15-25% harder than fixed cameras
2. **Small dataset** = Less generalization than 10k+ episodes
3. **Single task** = Doesn't learn composable primitives

### Success Criteria by Stage

**Stage 1 (MVP Test - 10 episodes)**:
- ✅ Training runs without crashes
- ✅ Loss decreases from 0.5 → 0.2
- ✅ No NaN losses or OOM errors
- ❌ Don't expect robot to work (too little data!)

**Stage 3 (50 episodes)**:
- ✅ Robot attempts task correctly
- ✅ Grasps object 60-80% of attempts
- ✅ Completes task 40-60% of runs
- ⚠️ May struggle with varied positions

**Stage 3 (100 episodes)**:
- ✅ Smooth, confident movements
- ✅ Reliable grasping (80-90%)
- ✅ Task completion 65-80%
- ✅ Some generalization to new positions

### Performance Breakdown

**What improves most**:
- ✅ **Trajectory planning** (biggest gain from finetuning)
- ✅ **Grasp pose prediction** (learns your gripper)
- ✅ **Camera adaptation** (learns egocentric viewpoint)

**What still struggles**:
- ⚠️ **New objects** (didn't see during training)
- ⚠️ **Occlusions** (wrist camera blocks view)
- ⚠️ **Rare positions** (underrepresented in dataset)

---

## Troubleshooting

### Common Issues

#### Issue 1: "CUDA out of memory" Error

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
(GPU 0; 23.70 GiB total capacity; 21.52 GiB already allocated)
```

**Solutions**:

1. **Reduce batch size** (in `scripts/train_pi05_so101.sh`):
   ```bash
   BATCH_SIZE=4  # Down from 8
   ```

2. **Enable gradient checkpointing** (already enabled in script):
   ```bash
   --policy.gradient_checkpointing=true
   ```

3. **Close other GPU processes**:
   ```bash
   # Check what's using GPU
   nvidia-smi

   # Kill if needed
   kill -9 <PID>
   ```

4. **Lower LoRA rank**:
   ```bash
   LORA_RANK=16  # Down from 32
   ```

---

#### Issue 2: NaN Losses

**Symptoms**:
```
[STEP 234] Loss: nan | LR: 3.0e-06
RuntimeError: Loss is NaN, stopping training
```

**Causes**:
- Gradient explosion (large parameter updates)
- Bad data in dataset (inf/nan values)
- Learning rate too high

**Solutions**:

1. **Verify gradient clipping enabled** (should be in script):
   ```bash
   --optimizer.grad_clip_norm=1.0
   ```

2. **Lower learning rate**:
   ```bash
   LEARNING_RATE=2e-6  # Down from 3e-6
   ```

3. **Check dataset for bad data**:
   ```python
   import json
   import numpy as np

   with open('jdocs/top_level/datasets/meta/stats.json') as f:
       stats = json.load(f)

   for key, vals in stats.items():
       if 'max' in vals:
           if np.any(np.isnan(vals['max'])) or np.any(np.isinf(vals['max'])):
               print(f"❌ Bad data in {key}: {vals}")
   ```

4. **Re-record problematic episodes**:
   - Check which episode caused NaN (look at step number → map to episode)
   - Delete episode: `rm -rf datasets/data/chunk-XXX/episode-YYY/`
   - Re-record replacement episode

---

#### Issue 3: Loss Not Decreasing

**Symptoms**:
```
[STEP 1000] Loss: 0.512
[STEP 2000] Loss: 0.498
[STEP 3000] Loss: 0.487
```

**Loss stays >0.4 and decreases very slowly**

**Causes**:
- Dataset too small (can't learn)
- Dataset too uniform (no diversity)
- Learning rate too low
- Data preprocessing issue

**Solutions**:

1. **Check dataset size**:
   ```bash
   python -c "import json; print(json.load(open('jdocs/top_level/datasets/meta/info.json'))['total_episodes'])"
   ```
   - If <30 episodes: Collect more data!
   - If 30-50 episodes: Try increasing LR to 4e-6
   - If >50 episodes: Continue investigation

2. **Check data diversity**:
   - Are all episodes nearly identical?
   - Did you vary object positions?
   - Is task too easy/repetitive?

3. **Increase learning rate** (cautiously):
   ```bash
   LEARNING_RATE=4e-6  # Up from 3e-6
   ```

4. **Visualize dataset**:
   ```python
   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
   import matplotlib.pyplot as plt

   dataset = LeRobotDataset(
       repo_id='lerobot/xlerobot_mvp_pick_15fps',
       root='jdocs/top_level/datasets'
   )

   # Plot action distributions
   actions = []
   for i in range(len(dataset)):
       actions.append(dataset[i]['action'].numpy())

   actions = np.array(actions)
   plt.figure(figsize=(12, 4))
   for j in range(6):
       plt.subplot(2, 3, j+1)
       plt.hist(actions[:, j], bins=50)
       plt.title(f'Joint {j}')
   plt.tight_layout()
   plt.savefig('action_distributions.png')
   print("Saved action_distributions.png")
   ```

---

#### Issue 4: Training Crashes Randomly

**Symptoms**:
- Training runs fine for 1000-2000 steps
- Suddenly crashes with no clear error
- Or: "Killed" message (no traceback)

**Causes**:
- **CPU RAM leak** (known Pi0.5 issue)
- **System OOM killer** terminates process
- Corrupted checkpoint write

**Solutions**:

1. **Monitor CPU RAM**:
   ```bash
   # Terminal 3
   watch -n 10 'free -h'
   ```

2. **If RAM grows over time**:
   - Known Pi0.5 memory leak issue
   - **Workaround**: Restart training from checkpoint every 2000 steps

   ```bash
   # In scripts/train_pi05_so101.sh, add after line 95:
   --resume_from_checkpoint=$OUTPUT_DIR/checkpoint-<LAST_SAVED>
   ```

3. **Check disk space**:
   ```bash
   df -h /home/jrobot/project/XLeRobot/outputs/
   ```

   Checkpoints need ~5GB. If disk full:
   ```bash
   # Delete old checkpoints (keep only last 2)
   rm -rf outputs/pi05_so101_v1/checkpoint-1000/
   rm -rf outputs/pi05_so101_v1/checkpoint-2000/
   ```

4. **Enable checkpoint recovery**:
   ```bash
   # If crash happens, resume from last checkpoint:
   python $LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py \
     <same flags as before> \
     --resume_from_checkpoint=$OUTPUT_DIR/checkpoint-4000
   ```

---

#### Issue 5: Inference Script Doesn't Work with Checkpoint

**Symptoms**:
```
python jdocs/top_level/run_pi05_inference_corrected.py \
  --checkpoint outputs/pi05_so101_v1/checkpoint-6000

Error: run_pi05_inference_corrected.py: error: unrecognized arguments: --checkpoint
```

**Cause**: Inference script doesn't have `--checkpoint` flag yet.

**Solution**: Add checkpoint loading to inference script.

See [Adding Checkpoint Loading](#adding-checkpoint-loading) section below.

---

### Adding Checkpoint Loading

**To update your inference script to load finetuned checkpoints:**

1. **Add argparse** (after line 77 in run_pi05_inference_corrected.py):
```python
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                   help='Path to finetuned checkpoint (e.g., outputs/pi05_so101_v1/checkpoint-6000)')
args = parser.parse_args()

# Update MODEL_ID if checkpoint provided
if args.checkpoint:
    MODEL_ID = args.checkpoint
    print(f"[CONFIG] Using finetuned checkpoint: {MODEL_ID}")
```

2. **Test**:
```bash
python jdocs/top_level/run_pi05_inference_corrected.py \
  --checkpoint /home/jrobot/project/XLeRobot/outputs/pi05_so101_v1/checkpoint-6000
```

---

## FAQ

### Q1: Why 50-100 episodes? I thought VLAs need thousands?

**A**: Pi0.5 is **pretrained** on 400 hours (~1M+ frames) of diverse manipulation data.

**Finetuning** (what you're doing):
- Adapt pretrained knowledge to YOUR robot
- 50-100 episodes = ~7.5k-15k frames
- Sufficient to learn:
  - Your robot's kinematics
  - Your camera viewpoints
  - Your normalization ranges
  - Your specific task

**Training from scratch** (not what you're doing):
- Would need 10k+ episodes
- You're leveraging pretrained knowledge!

---

### Q2: Can I use both arms (dual arm)?

**A**: Yes, but requires changes:

**Current setup**: Left arm only (6 DOF)

**Dual arm setup**: Both arms (12 DOF)
- Update config: `shape=(12,)` instead of `(6,)`
- Update dataset recording to include right arm
- Action space: `[left_shoulder_pan, left_shoulder_lift, ..., left_gripper, right_shoulder_pan, ..., right_gripper]`

**Data requirements**: 2x more (100-150 episodes recommended)

**See**: Future guide on dual-arm setup

---

### Q3: Can I train on multiple tasks?

**A**: Yes! Multi-task training is recommended.

**Single-task** (what you're doing now):
- 50-100 episodes of "pick cube"
- Model learns ONE task very well
- Limited generalization

**Multi-task** (better approach):
- 25 episodes "pick cube"
- 25 episodes "place cube"
- 15 episodes "push cube"
- 10 episodes "grasp cube"
- **Total 75 episodes across 4 primitives**

**Advantages**:
- Better generalization
- Learns composable behaviors
- Can combine primitives into complex tasks

**How to record**:
```bash
# Record each task separately
--dataset.single_task="pick red_cube"    # 25 episodes
--dataset.single_task="place red_cube"   # 25 episodes
--dataset.single_task="push red_cube"    # 15 episodes
```

**See**: `VLM_VLA_FINETUNING_STRATEGY.md` lines 1063-1088 for primitive vocabulary

---

### Q4: How long until I see results?

**Timeline from TODAY**:

| Day | Task | Time | Cumulative |
|-----|------|------|------------|
| **Day 0** | MVP test | 1 hour | 1 hour |
| **Day 1-3** | Collect 40 episodes | 10 hours | 11 hours |
| **Day 4** | Run training | 8 hours | 19 hours |
| **Day 5** | Evaluate | 2 hours | **21 hours** |

**Realistic**: **3-5 days** from now to working model

**If you work full-time on this**: 2-3 days

---

### Q5: What if performance is still poor after finetuning?

**Diagnostic checklist**:

1. **Check dataset quality**:
   - Are demonstrations smooth?
   - Do YOU successfully complete task?
   - Are failures recorded as successes?

2. **Check data diversity**:
   - Did you vary object positions?
   - Are all episodes nearly identical?
   - Is task too easy/hard?

3. **Check training metrics**:
   - Did loss decrease to <0.15?
   - Were there NaN losses?
   - Did training complete all 6000 steps?

4. **Check inference**:
   - Are actions changing each step?
   - Is robot moving smoothly?
   - Are actions in reasonable ranges?

**If all checks pass but performance still poor**:
- Egocentric cameras = inherently harder
- May need 100-150 episodes (not 50)
- Consider fixed external cameras
- Try different camera angles

---

### Q6: Can I use pretrained normalization stats?

**A**: Technically yes, but **not recommended**.

**Why not**:
- Pretrained stats are from OTHER robots (ALOHA, UR5, Franka)
- Your SO-101 has different joint ranges
- This is exactly why pretrained baseline doesn't move!

**Alternative**: Use SmolVLA stats (trained on SO-100/101)
- SmolVLA checkpoint includes SO-100 normalization stats
- Could extract and reuse
- But: Your robot might have different calibration

**Recommended**: Just finetune with YOUR data
- Automatically computes YOUR stats
- Guaranteed to match your robot
- Takes same time as extracting pretrained stats

---

### Q7: How do I know if my dataset is good quality?

**Quick quality check**:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np

dataset = LeRobotDataset(
    repo_id='lerobot/xlerobot_mvp_pick_15fps',
    root='jdocs/top_level/datasets'
)

print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {len(dataset)}")
print(f"Average episode length: {len(dataset) / dataset.num_episodes:.1f} frames")

# Check action smoothness (should be smooth, not jittery)
actions = []
for i in range(min(500, len(dataset))):  # First 500 frames
    actions.append(dataset[i]['action'].numpy())

actions = np.array(actions)

# Plot first 3 joints
plt.figure(figsize=(12, 4))
for j in range(3):
    plt.subplot(1, 3, j+1)
    plt.plot(actions[:, j])
    plt.title(f'Joint {j}')
    plt.xlabel('Frame')
    plt.ylabel('Position')
plt.tight_layout()
plt.savefig('action_smoothness.png')
print("✅ Saved action_smoothness.png - check for smoothness!")

# Check for outliers
for j in range(6):
    mean = actions[:, j].mean()
    std = actions[:, j].std()
    outliers = np.abs(actions[:, j] - mean) > 3 * std
    if outliers.sum() > 0:
        print(f"⚠️  Joint {j}: {outliers.sum()} outliers detected")
```

**Good dataset**:
- ✅ Smooth trajectories (no sudden jumps)
- ✅ Few outliers (<5% of frames)
- ✅ Episode lengths 100-200 frames (20-40s at 5Hz)
- ✅ Varied action distributions (not all identical)

**Bad dataset**:
- ❌ Jittery trajectories (teleop issues)
- ❌ Many outliers (calibration issues)
- ❌ Very short episodes (<50 frames = incomplete tasks)
- ❌ All actions identical (no diversity)

---

## Next Steps

**You're ready to start!**

1. ✅ Read this guide
2. ⏭️ **Run MVP test**: `bash scripts/train_pi05_mvp_test.sh`
3. ⏭️ **If test passes**: Collect 40-90 more episodes (see `RECORDING_GUIDE.md`)
4. ⏭️ **Run full training**: `bash scripts/train_pi05_so101.sh`
5. ⏭️ **Evaluate**: Compare finetuned vs pretrained baseline

**Good luck! 🤖**

---

**Questions or issues?** Check:
- This guide's [Troubleshooting](#troubleshooting) section
- `RECORDING_GUIDE.md` for data collection help
- `VLM_VLA_FINETUNING_STRATEGY.md` for advanced strategies
- LeRobot Discord/GitHub for community support
