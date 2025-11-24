# Complete LoRA Finetuning Guide for SO-101
## GR00T 1.5 & Pi0.5 Dual-Track Implementation

**Last Updated:** 2025-11-23 (v1.1 - Multi-Task Strategy)
**Robot:** SO-ARM101 Left Arm (6 DOF)
**GPU:** RTX 5090 (24GB VRAM)
**Approach:** Multi-task learning for better generalization
**Target:** 50 episodes MVP (2 tasks) → 100 episodes full (4 tasks)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Collection Strategy](#dataset-collection-strategy)
4. [Environment Setup](#environment-setup)
5. [MVP Stage (500 Steps)](#mvp-stage-500-steps)
6. [Full Training Stage](#full-training-stage)
7. [Evaluation & Deployment](#evaluation--deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Comparison](#performance-comparison)

---

## Overview

This guide covers **two parallel LoRA finetuning tracks**:

| Track | Model | Status | Recommendation |
|-------|-------|--------|----------------|
| **Track 1** | GR00T N1.5 (3B) | ✅ Proven | **PRIMARY** - Use this first |
| **Track 2** | Pi0.5 (4B) | ⚠️ **NOT IMPLEMENTED** | DESIGN STUB - Do not use yet |

### ⚠️ Implementation Status

**Ready to Use:**
- ✅ GR00T N1.5 LoRA training scripts (MVP + Full)
- ✅ Multi-task dataset collection strategy
- ✅ Environment setup guides

**Not Yet Implemented (Design Specs Only):**
- ❌ Pi0.5 LoRA (needs config fields + PEFT wiring in LeRobot)
- ❌ Helper scripts: `record_episodes.py`, `compute_stats.py`, `convert_to_lerobot_format.py`
- ❌ Benchmark scripts: `benchmark_groot.py`, `benchmark_pi05.py`

**Alternative Solutions:**
- For Pi0.5: Use existing LeRobot CLIs (e.g., `lerobot-record`) or treat script names as placeholders
- For now: **Focus on GR00T track** (proven, production-ready)

### Why LoRA?

- **Memory Efficient:** Fits in 24GB VRAM (full finetuning needs 58-70GB)
- **Data Efficient:** Needs 75-100 episodes (full finetuning needs 500+)
- **Fast Training:** 6-8 hours for production model
- **Trainable Params:** ~1% of model (40M params instead of 4B)

### Two-Stage Process

```
Stage 1: MVP (500 steps, 1-2 hours)
└── Validates pipeline works with your dataset

Stage 2: Full Training (6,000-10,000 steps, 6-8 hours)
└── Production model with 75-100 episodes
```

---

## Prerequisites

### Hardware Requirements

- **GPU:** RTX 5090 (24GB VRAM) ✅
- **CPU RAM:** 32GB+ recommended
- **Disk Space:** 50GB free
  - Datasets: ~10-20GB
  - Checkpoints: ~5-10GB per model
  - Model weights: ~10GB

### Software Requirements

- **OS:** Ubuntu 20.04+ / Linux
- **Python:** 3.10
- **CUDA:** 12.1+
- **Conda:** Miniconda/Anaconda

### Verified System

```bash
# Check your system
nvidia-smi  # Should show RTX 5090 with 24GB
python --version  # Should be 3.10
conda --version
```

---

## Dataset Collection Strategy

### Overview: Multi-Task Learning Approach

This guide recommends a **multi-task dataset** from the MVP stage onwards. Benefits:

✅ **Data Efficiency:** Shared skills (reaching, grasping) transfer across tasks
✅ **Better Generalization:** Model learns underlying manipulation skills, not task memorization
✅ **Validates Multi-Task Learning Early:** Know immediately if model handles task variety
✅ **Reusable Data:** All episodes contribute to final production model

### Target Dataset Size & Task Mix

| Stage | Total Episodes | Task Distribution | Training Time | Expected Performance |
|-------|---------------|-------------------|---------------|---------------------|
| **MVP Multi-Task** | 50 | Pick (30) + Push (20) | 1-2 hours | Pipeline validation only |
| **Good Multi-Task** | 75 | Pick (40) + Push (30) + Drawer (15) | 6-8 hours | 50-70% per task |
| **Optimal Multi-Task** | 100 | Pick (40) + Push (30) + Open (15) + Close (15) | 6-8 hours | 65-80% per task |

### Recommended Task Sequence

#### MVP Stage (50 Episodes)
```
Task 1: Pick-and-Place     30 episodes (60%)
└── Core manipulation skill, clear success criteria

Task 2: Push/Slide Object  20 episodes (40%)
└── Shares reaching skills, different contact dynamics
```

**Rationale:**
- 2 tasks keeps setup simple (not overwhelming)
- Push task shares positioning skills with pick-and-place
- Enough data per task for meaningful signal (30 & 20 episodes)
- Tests multi-task learning capability early

#### Full Stage Extension (50 → 100 Episodes)
```
Pick-and-Place:  30 → 40 episodes (+10)
Push/Slide:      20 → 30 episodes (+10)
Open Drawer:     0 → 15 episodes (+15)  [NEW]
Close Drawer:    0 → 15 episodes (+15)  [NEW]
```

**Why add drawer tasks?**
- Different gripper usage (parallel jaw vs pinch)
- Requires precise alignment (complements gross positioning)
- Bidirectional skills (open/close) test generalization

### Alternative: Staged MVP Approach (Lower Risk)

If you want to minimize risk and validate the pipeline quickly first:

**Phase 1: Single-Task Proof (20 episodes, 2-3 days)**
```
Pick-and-Place: 20 episodes
Goal: Prove pipeline works end-to-end
```

**Phase 2: Multi-Task Extension (30 more episodes, 3-4 days)**
```
Pick-and-Place: +15 episodes (total 35)
Push/Slide:     +15 episodes (total 15)
Goal: Validate multi-task learning
```

This gives you an early checkpoint to catch issues before investing in multi-task data collection.

### Dataset Structure & Version Requirements

**IMPORTANT: GR00T vs Pi0.5 Dataset Compatibility**

| Model | Dataset Version | Required Files |
|-------|----------------|----------------|
| **GR00T** | LeRobot **v2** | `meta/info.json`, `meta/modality.json`, `meta/stats.json` |
| **Pi0.5** | LeRobot **v3** | `meta/info.json`, `meta/stats.json` |

**LeRobot v3.0 Format (Pi0.5):**

```
datasets/
├── meta/
│   ├── info.json          # Dataset metadata (includes task labels)
│   └── stats.json         # Normalization statistics
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Pick task
│       ├── episode_000001.parquet  # Pick task
│       ├── episode_000030.parquet  # Push task starts
│       └── ...
└── videos/
    └── chunk-000/
        ├── episode_000000_head.mp4
        ├── episode_000000_left_wrist.mp4
        └── ...
```

**LeRobot v2 Format (GR00T) - Additional Requirements:**

```
datasets/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── modality.json      # ⚠️ REQUIRED for GR00T (camera/state/action mappings)
│   └── stats.json         # Normalization statistics
├── (same structure as v3)
```

**Converting v3 → v2 for GR00T:**

If your dataset is in LeRobot v3 format:

```bash
# 1. Get conversion script from LeRobot PR #2109
#    https://github.com/huggingface/lerobot/pull/2109
#    Copy convert_dataset_v3_to_v2.py to /home/jrobot/project/lerobot/scripts/

# 2. Run conversion from LeRobot repo directory
cd /home/jrobot/project/lerobot
python scripts/convert_dataset_v3_to_v2.py \
    --input /home/jrobot/project/XLeRobot/jdocs/top_level/datasets \
    --output /home/jrobot/project/XLeRobot/jdocs/top_level/datasets_v2

# 3. Update DATASET_PATH in GR00T training scripts to point to datasets_v2
```

**Important:** The conversion script is part of LeRobot (PR #2109), not the XLeRobot project.

**Creating modality.json for GR00T:**

```bash
# 1. Copy template from Isaac-GR00T
cp $ISAAC_GROOT_ROOT/configs/so100_dualcam__modality.json \
   /path/to/datasets/meta/modality.json

# 2. Edit to match your camera keys
# Change camera keys to: "head" and "left_wrist"
# Ensure action/state dimensions match (6 DOF for SO-101)
```

Example `modality.json` structure:
```json
{
  "observation.images.head": {"type": "image", "shape": [3, 224, 224]},
  "observation.images.left_wrist": {"type": "image", "shape": [3, 224, 224]},
  "observation.state": {"type": "state", "shape": [6]},
  "action": {"type": "action", "shape": [6]}
}
```

### Recording Multi-Task Episodes

**See detailed instructions in:** `jdocs/top_level/RECORDING_GUIDE.md`

**MVP Stage - Collect 50 episodes across 2 tasks:**

```bash
# Day 1-3: Record pick-and-place (30 episodes)
python scripts/record_episodes.py \
    --robot-type so101_follower \
    --cameras head left_wrist \
    --fps 5 \
    --task pick_place \
    --output-dir jdocs/top_level/datasets \
    --num-episodes 30

# Day 4-5: Record push/slide (20 episodes)
python scripts/record_episodes.py \
    --robot-type so101_follower \
    --cameras head left_wrist \
    --fps 5 \
    --task push_slide \
    --output-dir jdocs/top_level/datasets \
    --num-episodes 20 \
    --start-episode-idx 30  # Continue from episode 30
```

**Full Stage Extension - Add 50 more episodes:**

```bash
# Add 10 more pick-and-place
python scripts/record_episodes.py \
    --task pick_place \
    --num-episodes 10 \
    --start-episode-idx 50

# Add 10 more push/slide
python scripts/record_episodes.py \
    --task push_slide \
    --num-episodes 10 \
    --start-episode-idx 60

# Add 15 open drawer
python scripts/record_episodes.py \
    --task open_drawer \
    --num-episodes 15 \
    --start-episode-idx 70

# Add 15 close drawer
python scripts/record_episodes.py \
    --task close_drawer \
    --num-episodes 15 \
    --start-episode-idx 85
```

### Data Quality Guidelines

✅ **Good Episodes:**
- Task completed successfully (e.g., object picked and placed)
- Smooth, natural motions
- Good lighting and camera angles
- No collisions or errors

❌ **Bad Episodes:**
- Task failed (object dropped, missed grasp)
- Jerky, unnatural motions
- Poor visibility (shadows, glare)
- Robot errors or safety stops

**Rule of Thumb:** Only keep episodes you would want the robot to imitate.

### Task-Specific Recording Tips

**Pick-and-Place:**
- Vary object positions (left/right, near/far, different heights)
- Vary placement targets (3-5 different target positions)
- Keep objects consistent size/weight for MVP (add variety in full stage)

**Push/Slide:**
- Vary push directions (left, right, forward, backward)
- Vary object starting positions
- Different push distances (short vs long)

**Open/Close Drawer:**
- Vary starting handle position (drawer partially open/closed)
- Different grasp approaches (top, side, centered)
- Full open/close cycles for both tasks

### Verifying Multi-Task Dataset

```bash
# Check dataset info
cat jdocs/top_level/datasets/meta/info.json

# Expected output for MVP (50 episodes):
{
    "robot_type": "so101_follower",
    "total_episodes": 50,
    "total_frames": 7500,
    "fps": 5,
    "tasks": {
        "pick_place": {"episodes": 30, "range": [0, 29]},
        "push_slide": {"episodes": 20, "range": [30, 49]}
    },
    "features": {
        "action": {
            "shape": [6],
            "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                     "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
        },
        "observation.state": {
            "shape": [6]
        },
        "observation.images.head": {
            "shape": [3, 224, 224]
        },
        "observation.images.left_wrist": {
            "shape": [3, 224, 224]
        }
    }
}

# Verify episode distribution
python -c "
import json
with open('jdocs/top_level/datasets/meta/info.json') as f:
    info = json.load(f)
print(f'Total episodes: {info[\"total_episodes\"]}')
for task, data in info.get('tasks', {}).items():
    print(f'  {task}: {data[\"episodes\"]} episodes')
"
```

---

## Environment Setup

### Option 1: GR00T 1.5 (Primary Track)

```bash
# Clone Isaac-GR00T
cd ~
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Create conda environment
conda create -n groot python=3.10
conda activate groot

# Install dependencies
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import peft; print('PEFT installed ✅')"

# Set environment variable (add to ~/.bashrc for persistence)
export ISAAC_GROOT_ROOT=~/Isaac-GR00T
```

### Option 2: Pi0.5 (Secondary Track)

**⚠️ WARNING: Pi0.5 LoRA is NOT IMPLEMENTED in LeRobot yet!**

This track is a **design specification** for future implementation. The Pi0.5 LoRA training scripts will fail until:
1. LoRA fields added to `PI05Config` (use_lora, lora_rank, lora_alpha, lora_dropout)
2. PEFT wrapping implemented in `modeling_pi05.py`
3. CLI flags validated against `lerobot_train.py`

**For now, skip this section and use GR00T (Option 1).**

If you want to implement Pi0.5 LoRA later:

```bash
# LeRobot should already be installed at /home/jrobot/project/lerobot
cd /home/jrobot/project/lerobot

# Activate environment
conda activate lerobot

# Install PEFT for LoRA support
pip install peft

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import peft; print('PEFT installed ✅')"
python -c "from lerobot.policies.pi05 import PI05Config; print('Pi0.5 config exists ✅')"

# Note: This only checks that Pi0.5 exists, NOT that LoRA is implemented
```

### Environment Verification Checklist

- [ ] Python 3.10
- [ ] PyTorch with CUDA support
- [ ] PEFT library installed
- [ ] GPU detected (RTX 5090)
- [ ] Dataset accessible at `/home/jrobot/project/XLeRobot/jdocs/top_level/datasets`

---

## MVP Stage (500 Steps)

**Purpose:** Validate that the training pipeline works with your dataset before committing to full 6-8 hour training.

### MVP Success Criteria

✅ **Must Pass:**
1. Training completes 500 steps without crashes
2. VRAM stays below 22GB throughout
3. Loss decreases from initial value
4. Checkpoint saves successfully

❌ **Not Evaluated:**
- Robot performance (not tested at this stage)
- Task success rate (need more data + full training)

### Track 1: GR00T MVP Test

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_so101_mvp.sh
```

**Configuration:**
- Steps: 500
- Batch Size: 8
- LoRA Rank: 16
- Learning Rate: 1e-4
- Duration: ~1-2 hours
- Expected VRAM: 18-20GB

**Monitor Training:**

Open a second terminal:
```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Watch loss (in training terminal)
# Look for: loss decreasing from ~2.0 to ~0.5-1.0
```

**Expected Output:**
```
Step 100/500 | Loss: 1.245 | VRAM: 19.2GB
Step 200/500 | Loss: 0.892 | VRAM: 19.3GB
Step 300/500 | Loss: 0.734 | VRAM: 19.2GB
Step 400/500 | Loss: 0.621 | VRAM: 19.4GB
Step 500/500 | Loss: 0.548 | VRAM: 19.3GB
✅ Checkpoint saved: outputs/groot_mvp_test/checkpoint-500/
```

### Track 2: Pi0.5 MVP Test (NOT IMPLEMENTED)

**⚠️ This script will fail - Pi0.5 LoRA is not implemented yet!**

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/train_pi05_mvp_lora.sh
# Will exit with error message - use GR00T instead
```

**Why this doesn't work:**
- `PI05Config` does not have LoRA fields
- `modeling_pi05.py` does not wrap model with PEFT
- Script is a design stub for future implementation

**Use GR00T MVP instead** (proven, production-ready)

**Configuration (for reference when implemented):**
- Steps: 500
- Batch Size: 8
- LoRA Rank: 16
- Learning Rate: 3e-6
- Duration: ~1-2 hours
- Expected VRAM: 18-22GB (experimental)

**Monitor Training:**

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Check LoRA was applied correctly
# Look for in logs:
# "trainable params: 40M || all params: 4B || trainable%: 1.00%"
```

**Expected Output:**
```
Applying LoRA to Pi0.5: rank=16, alpha=32, dropout=0.1
Wrapping PaliGemma language model with LoRA adapters...
trainable params: 24M || all params: 2.9B || trainable%: 0.82%
Wrapping Action Expert (Gemma) with LoRA adapters...
trainable params: 16M || all params: 340M || trainable%: 4.70%
✅ LoRA successfully applied to Pi0.5 model

Step 100/500 | Loss: 1.534 | VRAM: 20.1GB
Step 200/500 | Loss: 1.102 | VRAM: 20.2GB
Step 300/500 | Loss: 0.891 | VRAM: 20.1GB
Step 400/500 | Loss: 0.745 | VRAM: 20.3GB
Step 500/500 | Loss: 0.672 | VRAM: 20.2GB
✅ Checkpoint saved: outputs/pi05_mvp_lora/checkpoint-500/
```

### MVP Metrics to Check

| Metric | How to Check | Success Criteria |
|--------|--------------|------------------|
| **Loss Decrease** | Training logs | Initial loss (1.5-2.5) → Final loss (0.5-1.0) |
| **VRAM Usage** | `nvidia-smi` | Peak < 22GB |
| **No Crashes** | Script completion | Reaches step 500/500 |
| **Checkpoint Saved** | `ls outputs/*/checkpoint-500/` | Directory exists with model files |
| **LoRA Applied** | Training logs | "trainable params: ~1%" message |

### MVP Troubleshooting

**Problem: CUDA Out of Memory**
```bash
# Reduce batch size in the script
--batch_size=4  # Instead of 8
```

**Problem: Loss not decreasing**
```
Step 100: Loss 2.1
Step 200: Loss 2.0
Step 300: Loss 2.1  # Not improving!
```
→ Check dataset quality (are episodes labeled correctly?)
→ Check normalization stats exist (`meta/stats.json`)

**Problem: PEFT not found (Pi0.5)**
```bash
pip install peft
```

### After MVP: Decision Point & Multi-Task Analysis

✅ **If MVP PASSED:** Analyze multi-task performance before collecting more data

**Multi-Task Success Scenarios:**

**Scenario A: Both tasks show learning** ✅
```
Pick-and-place loss:  2.0 → 0.6
Push/slide loss:      2.1 → 0.7
```
→ **Action:** Multi-task learning works! Proceed to collect 50 more episodes across 4 tasks

**Scenario B: Primary task learns, secondary struggles** ⚠️
```
Pick-and-place loss:  2.0 → 0.6
Push/slide loss:      2.1 → 1.9 (not improving)
```
→ **Action:** Push task may need more episodes. Try adding 10 more push episodes before full stage

**Scenario C: Neither task learns well** ❌
```
Both tasks: Loss stays > 1.5
```
→ **Action:** Pipeline issue, not multi-task problem. See troubleshooting.

**Key Insight:** Even if push task loss is higher than pick-and-place, as long as BOTH are decreasing, multi-task learning is working. The model is learning shared skills.

❌ **If MVP FAILED:**
- GR00T failed → Check Isaac-GR00T installation, dataset format
- Pi0.5 failed → Fall back to GR00T (proven implementation)
- Both failed → See [Troubleshooting](#troubleshooting) section

---

## Full Training Stage

**Prerequisites:**
- ✅ MVP test passed
- ✅ Multi-task learning validated (both tasks show decreasing loss)
- ✅ 100 episodes collected across 4 tasks
- ✅ Dataset verified with `info.json` showing task distribution

### Track 1: GR00T Full Training

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_so101_full.sh
```

**Configuration:**
- Steps: 10,000
- Batch Size: 16
- LoRA Rank: 16
- Learning Rate: 1e-4
- Duration: ~6-8 hours
- Expected VRAM: 18-20GB
- Checkpoint Frequency: Every 1,000 steps

**Training Timeline:**
```
Hour 0-1:   Steps 0-1,250     | Loss: 1.8 → 0.9
Hour 1-2:   Steps 1,250-2,500 | Loss: 0.9 → 0.6
Hour 2-4:   Steps 2,500-5,000 | Loss: 0.6 → 0.4
Hour 4-6:   Steps 5,000-7,500 | Loss: 0.4 → 0.3
Hour 6-8:   Steps 7,500-10,000| Loss: 0.3 → 0.25
```

**Checkpoints Saved:**
```
outputs/groot_so101_v1/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
├── ...
└── checkpoint-10000/  ← Final model
```

### Track 2: Pi0.5 Full Training (NOT IMPLEMENTED)

**⚠️ This script will fail - Pi0.5 LoRA is not implemented yet!**

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/train_pi05_full_lora.sh
# Will exit with error message - use GR00T instead
```

**Why this doesn't work:**
- `PI05Config` does not have LoRA fields
- `modeling_pi05.py` does not wrap model with PEFT
- Script is a design stub for future implementation

**Use GR00T Full Training instead** (proven, production-ready)

**Configuration (for reference when implemented):**
- Steps: 6,000
- Batch Size: 16
- LoRA Rank: 16
- Learning Rate: 2.5e-5
- Duration: ~6-8 hours
- Expected VRAM: 18-22GB (experimental)
- Checkpoint Frequency: Every 1,000 steps

**Training Timeline:**
```
Hour 0-1:   Steps 0-750       | Loss: 2.1 → 1.1
Hour 1-3:   Steps 750-2,250   | Loss: 1.1 → 0.7
Hour 3-5:   Steps 2,250-4,500 | Loss: 0.7 → 0.5
Hour 5-8:   Steps 4,500-6,000 | Loss: 0.5 → 0.4
```

**Checkpoints Saved:**
```
outputs/pi05_full_lora_v1/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
├── ...
└── checkpoint-6000/  ← Final model
```

### Monitoring Full Training

**Terminal 1: Training Process**
```bash
# Run the training script
bash scripts/train_groot_so101_full.sh
# or
bash scripts/train_pi05_full_lora.sh
```

**Terminal 2: GPU Monitor**
```bash
watch -n 5 nvidia-smi
# Watch for:
# - VRAM usage (should be stable 18-22GB)
# - GPU utilization (should be 95-100%)
# - Temperature (should be < 85°C)
```

**Terminal 3: Loss Monitor**
```bash
# For GR00T
tail -f outputs/groot_so101_v1/logs/train.log | grep "Loss"

# For Pi0.5
tail -f outputs/pi05_full_lora_v1/logs/train.log | grep "Loss"
```

### Full Training Metrics

| Metric | Target | How to Check |
|--------|--------|--------------|
| **Final Loss** | < 0.3 (GR00T), < 0.4 (Pi0.5) | Training logs |
| **VRAM Peak** | < 22GB | `nvidia-smi` |
| **Training Time** | 6-8 hours | Wall clock |
| **Checkpoints Saved** | 6-10 checkpoints | `ls outputs/*/checkpoint-*` |
| **No NaN Loss** | All finite values | Check logs for "NaN" |
| **GPU Utilization** | > 90% | `nvidia-smi` |

### Early Stopping Considerations

You can stop training early if:
1. **Loss plateaus** for 1,000+ steps (no improvement)
2. **Performance sufficient** on robot evaluation (see next section)

Example: If checkpoint-5000 achieves 70% success on robot, checkpoint-10000 may only improve to 75% - diminishing returns.

---

## Evaluation & Deployment

### Quick Evaluation (On Robot)

**Purpose:** Test model performance on real robot before full benchmark.

**GR00T Inference:**
```bash
cd ~/Isaac-GR00T
conda activate groot

python scripts/inference_service.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_so101_v1/checkpoint-10000 \
    --server \
    --port 8000
```

**Pi0.5 Inference:**
```bash
cd /home/jrobot/project/lerobot
conda activate lerobot

python src/lerobot/scripts/lerobot_eval.py \
    --policy.path=/home/jrobot/project/XLeRobot/outputs/pi05_full_lora_v1/checkpoint-6000 \
    --robot.type=so101_follower \
    --robot.cameras=[head,left_wrist]
```

### Manual Testing Protocol (Multi-Task)

**Run 10 test episodes PER TASK:**

1. Set up environment (same as training)
2. Run inference service
3. Test each task separately: 10 trials per task
4. Record outcomes per task

**Success Criteria:**
- ✅ Task completed successfully
- ⚠️ Task completed with minor errors (e.g., object slightly misplaced)
- ❌ Task failed (object dropped, missed grasp, collision)

**Example Results (Multi-Task Model with 100 episodes):**

```
Pick-and-Place (10 trials):
✅✅❌✅✅✅✅❌✅✅
Success Rate: 80% (8/10)

Push/Slide (10 trials):
✅✅✅⚠️✅✅❌✅✅❌
Success Rate: 70% (7/10), Partial: 10% (1/10)

Open Drawer (10 trials):
✅✅❌✅✅✅❌✅✅✅
Success Rate: 80% (8/10)

Close Drawer (10 trials):
✅✅✅✅❌✅✅✅✅❌
Success Rate: 80% (8/10)

Overall Multi-Task Success: 77.5%
```

**Analyzing Multi-Task Performance:**
- **Expected:** Primary task (pick) slightly higher success than others
- **Good sign:** All tasks > 50% success (shows generalization)
- **Concern:** One task < 30% while others > 60% (may need more data for that task)

### Benchmark Evaluation

**Run full evaluation suite (50-100 episodes):**

```bash
# GR00T
python scripts/benchmark_groot.py \
    --model-path outputs/groot_so101_v1/checkpoint-10000 \
    --num-episodes 50 \
    --output-file results/groot_benchmark.json

# Pi0.5 (NOTE: benchmark_pi05.py is not implemented - design stub only)
python scripts/benchmark_pi05.py \
    --model-path outputs/pi05_full_lora_v1/checkpoint-6000 \
    --num-episodes 50 \
    --output-file results/pi05_benchmark.json
```

**Note:** `benchmark_pi05.py` is not yet implemented. Use manual testing protocol for Pi0.5 evaluation.

### Expected Performance (Multi-Task Models)

| Dataset Mix | GR00T Overall | Pi0.5 Overall (Est.) | Notes |
|-------------|---------------|---------------------|-------|
| **50 eps (MVP)** | | | |
| - Pick (30) | 35-45% | 30-40% | Primary task |
| - Push (20) | 25-35% | 20-30% | Secondary task |
| **100 eps (Full)** | | | |
| - Pick (40) | 70-80% | 60-70% | Most data |
| - Push (30) | 60-70% | 50-60% | Good data |
| - Open (15) | 50-60% | 40-50% | New task |
| - Close (15) | 50-60% | 40-50% | New task |
| **Overall Avg** | 65-75% | 55-65% | Multi-task generalization |

**Key Insights:**
- Tasks with more episodes → higher success rates (expected)
- Related tasks (open/close) should have similar performance
- Overall multi-task performance slightly lower than single-task (acceptable tradeoff for generalization)
- Multi-task model may show **zero-shot transfer** to similar unseen tasks

**Note:** Pi0.5 numbers are estimates - experimental implementation.

### Comparing Models

**Comparison Criteria:**

1. **Success Rate** (most important)
2. **Inference Speed** (GR00T typically faster)
3. **Motion Quality** (smoothness, naturalness)
4. **Robustness** (handles variations in environment)
5. **Safety** (no collisions or dangerous motions)

**Decision Matrix:**

| Scenario | Recommendation |
|----------|---------------|
| GR00T: 70%, Pi0.5: 65% | **Deploy GR00T** (higher success + proven) |
| GR00T: 50%, Pi0.5: 60% | **Test both further** (Pi0.5 slightly better) |
| GR00T: 40%, Pi0.5: 40% | **Collect more data** (both underperforming) |
| Both < 30% | **Review data quality** (likely issue) |

### Deployment

**Once satisfied with model:**

```bash
# Copy best checkpoint to deployment location
cp -r outputs/groot_so101_v1/checkpoint-10000 ~/models/production/groot_so101_latest

# Create deployment config
cat > ~/models/production/config.yaml <<EOF
model_name: groot_so101_v1
checkpoint: checkpoint-10000
training_episodes: 100
success_rate: 0.72
last_updated: 2025-11-23
EOF
```

---

## Troubleshooting

### Common Training Issues

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.5 GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   # In training script, change:
   --batch_size=8  # Instead of 16
   ```

2. **Enable gradient checkpointing** (Pi0.5):
   ```bash
   --policy.gradient_checkpointing=true
   ```

3. **Clear GPU cache before training:**
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

4. **Check other processes:**
   ```bash
   nvidia-smi
   # Kill other GPU processes if needed
   ```

#### Issue 2: Loss Not Decreasing

**Symptoms:**
```
Step 100: Loss 2.5
Step 500: Loss 2.4
Step 1000: Loss 2.5  # Stuck!
```

**Possible Causes & Solutions:**

1. **Bad data quality:**
   - Check episodes are labeled correctly
   - Remove failed episodes from dataset
   - Verify normalization stats exist

2. **Learning rate too high/low:**
   ```bash
   # Try different learning rates:
   --optimizer.lr=1e-5  # Lower
   --optimizer.lr=5e-5  # Higher
   ```

3. **Dataset too small:**
   - Need at least 50 episodes for meaningful training
   - Collect more data

4. **Normalization issues:**
   ```bash
   # Regenerate normalization stats
   python scripts/compute_stats.py --dataset-path datasets/
   ```

#### Issue 3: NaN Loss

**Symptoms:**
```
Step 245: Loss 0.8
Step 246: Loss nan
RuntimeError: loss is NaN
```

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   --optimizer.lr=1e-5  # Much lower
   ```

2. **Increase gradient clipping:**
   ```bash
   --optimizer.grad_clip_norm=0.5  # More aggressive
   ```

3. **Check dataset for outliers:**
   ```python
   # Script to check action ranges
   import numpy as np
   import json

   with open('datasets/meta/stats.json') as f:
       stats = json.load(f)

   actions = stats['action']
   print("Action ranges:")
   print(f"Min: {actions['min']}")
   print(f"Max: {actions['max']}")

   # Look for extreme values (>1000, <-1000)
   ```

#### Issue 4: Training Crashes Midway

**Symptoms:**
```
Step 3452/10000 | Loss 0.42
Killed
```

**Possible Causes:**

1. **CPU RAM exhausted:**
   ```bash
   # Check memory usage
   free -h
   # If low, reduce data loading workers
   --num_workers=2  # Instead of 4
   ```

2. **Disk space full:**
   ```bash
   df -h
   # Clean up old checkpoints if needed
   rm -rf outputs/old_experiments/
   ```

3. **Power/thermal issues:**
   - Check GPU temperature: `nvidia-smi`
   - Ensure adequate cooling

4. **Resume from checkpoint:**
   ```bash
   # Most scripts support resuming
   --resume_from=outputs/groot_so101_v1/checkpoint-3000
   ```

### Dataset Issues

#### Issue: Dataset Format Error

**Symptoms:**
```
KeyError: 'observation.images.head'
FileNotFoundError: videos/chunk-000/episode_000000_head.mp4
```

**Solution:** Verify dataset structure:
```bash
cd jdocs/top_level/datasets

# Check structure
tree -L 3

# Expected:
# ├── meta/
# │   ├── info.json
# │   └── stats.json
# ├── data/
# │   └── chunk-000/
# │       ├── episode_000000.parquet
# └── videos/
#     └── chunk-000/
#         ├── episode_000000_head.mp4
#         └── episode_000000_left_wrist.mp4

# Regenerate dataset if needed
python scripts/convert_to_lerobot_format.py \
    --input raw_recordings/ \
    --output datasets/
```

#### Issue: Wrong Action Dimensions

**Symptoms:**
```
RuntimeError: Expected action dimension 32, got 6
```

**Solution:** This is normal - Pi0.5 uses universal 32-dim action space. The model will pad your 6-dim actions automatically. No action needed.

#### Issue: Missing Normalization Stats

**Symptoms:**
```
FileNotFoundError: datasets/meta/stats.json
```

**Solution:**
```bash
python scripts/compute_stats.py \
    --dataset-path jdocs/top_level/datasets \
    --output-file jdocs/top_level/datasets/meta/stats.json
```

### Model Issues

#### Issue: LoRA Not Applied (Pi0.5)

**Symptoms:**
```
Training uses 40GB VRAM (should be 20GB)
No "trainable params: 1%" message in logs
```

**Solution:**
```bash
# Verify PEFT installed
pip install peft

# Check LoRA config (NOTE: This will only work AFTER implementing LoRA in PI05Config)
python -c "
from lerobot.policies.pi05 import PI05Config
config = PI05Config(use_lora=True, lora_rank=16)  # Will fail - use_lora not implemented yet
print(f'LoRA enabled: {config.use_lora}')
"
```

**Note:** This code is illustrative only. It will not work until LoRA fields are added to `PI05Config`.

#### Issue: Checkpoint Loading Error

**Symptoms:**
```
RuntimeError: Error loading checkpoint: size mismatch for paligemma.weight
```

**Solutions:**

1. **Check checkpoint integrity:**
   ```bash
   ls -lh outputs/groot_so101_v1/checkpoint-10000/
   # Should have: pytorch_model.bin or model.safetensors
   ```

2. **Try different checkpoint:**
   ```bash
   # Use earlier checkpoint
   --model-path outputs/groot_so101_v1/checkpoint-9000
   ```

3. **Regenerate from training:**
   ```bash
   # Resume training to regenerate final checkpoint
   bash scripts/train_groot_so101_full.sh --resume_from checkpoint-9000
   ```

### Performance Issues

#### Issue: Slow Inference

**Symptoms:**
- Inference takes >500ms per action (should be <200ms)

**Solutions:**

1. **Enable torch.compile** (Pi0.5):
   ```python
   config = PI05Config(compile_model=True, compile_mode="max-autotune")
   ```

2. **Use FP16/BF16:**
   ```bash
   --policy.dtype=bfloat16
   ```

3. **Reduce batch size** (for single inference):
   ```python
   # In inference code
   batch_size = 1
   ```

#### Issue: Poor Robot Performance Despite Low Loss

**Symptoms:**
- Training loss: 0.25 (good)
- Robot success rate: 15% (poor)

**Possible Causes:**

1. **Normalization mismatch:**
   - Model trained on quantile-normalized actions
   - Robot expects raw joint angles
   - **Solution:** Ensure inference denormalizes actions correctly

2. **Overfitting to training data:**
   - Model memorized specific scenarios
   - **Solution:** Collect more diverse episodes

3. **Sim-to-real gap:**
   - Environment changed between training and deployment
   - **Solution:** Record episodes in deployment environment

4. **Action execution issues:**
   - Actions predicted correctly but not executed properly
   - **Solution:** Check robot control loop, interpolation

---

## Performance Comparison

### GR00T vs Pi0.5: Expected Results

| Metric | GR00T N1.5 | Pi0.5 | Notes |
|--------|------------|-------|-------|
| **Model Size** | 3B params | 4B params | Pi0.5 larger |
| **LoRA Params** | ~30M (1%) | ~40M (1%) | Both efficient |
| **Training Time** | 6-8 hours | 6-8 hours | Similar |
| **VRAM Usage** | 18-20GB | 18-22GB | Pi0.5 slightly higher |
| **Inference Speed** | ~100-150ms | ~150-200ms | GR00T faster |
| **Success Rate (75 eps)** | 50-70% | 40-60% (est.) | GR00T likely better |
| **Success Rate (100 eps)** | 65-80% | 55-75% (est.) | GR00T likely better |
| **Implementation** | ✅ Proven | 🧪 Experimental | GR00T validated |
| **Documentation** | ✅ Extensive | ⚠️ Limited | GR00T better docs |

### Recommendation: Start with GR00T

**Reasons:**
1. ✅ Proven implementation with community validation
2. ✅ Extensive documentation and troubleshooting guides
3. ✅ Likely higher success rates
4. ✅ Faster inference
5. ✅ Better support from NVIDIA/HuggingFace

**When to try Pi0.5:**
- GR00T performance insufficient (<50% success with 100 episodes)
- Specific Pi0.5 features needed (universal action space, PaliGemma vision)
- Research/experimentation goals

### Training Time Breakdown

**GR00T (10,000 steps):**
```
Data loading:        ~10-15%  (0.6-1.2 hours)
Forward pass:        ~40-45%  (2.4-3.6 hours)
Backward pass:       ~25-30%  (1.5-2.4 hours)
Optimizer step:      ~10-15%  (0.6-1.2 hours)
Checkpointing:       ~5%      (0.3 hours)
Total:               6-8 hours
```

**Pi0.5 (6,000 steps):**
```
Data loading:        ~10-15%  (0.6-1.2 hours)
Forward pass:        ~45-50%  (2.7-4.0 hours)
Backward pass:       ~25-30%  (1.5-2.4 hours)
Optimizer step:      ~10-15%  (0.6-1.2 hours)
Checkpointing:       ~5%      (0.3 hours)
Total:               6-8 hours
```

### Cost Analysis

**Single Training Run:**

| Resource | Cost |
|----------|------|
| Electricity (24GB GPU, 8h @ 400W) | ~$0.30-0.50 |
| Human time (setup + monitoring) | 2-3 hours |
| Disk space (checkpoints) | 5-10GB |

**Total Project Cost (MVP + Full Training + Evaluation):**
- GPU time: ~15-20 hours
- Human time: ~10-15 hours
- Electricity: ~$1-2
- Data collection: ~20-30 hours (75-100 episodes @ 15min each)

---

## Quick Reference Commands

### Dataset
```bash
# Record episodes (NOTE: record_episodes.py not implemented - use lerobot-record CLI)
python scripts/record_episodes.py --robot-type so101_follower --num-episodes 100

# Verify dataset
cat jdocs/top_level/datasets/meta/info.json

# Compute normalization stats (NOTE: compute_stats.py not implemented - use LeRobot utilities)
python scripts/compute_stats.py --dataset-path jdocs/top_level/datasets
```

### Training
```bash
# GR00T MVP (1-2 hours) - ✅ PRODUCTION READY
bash scripts/train_groot_so101_mvp.sh

# GR00T Full (6-8 hours) - ✅ PRODUCTION READY
bash scripts/train_groot_so101_full.sh

# Pi0.5 MVP (design stub, exits with error) - ❌ NOT IMPLEMENTED
bash scripts/train_pi05_mvp_lora.sh

# Pi0.5 Full (design stub, exits with error) - ❌ NOT IMPLEMENTED
bash scripts/train_pi05_full_lora.sh
```

### Monitoring
```bash
# GPU usage
watch -n 5 nvidia-smi

# Training logs
tail -f outputs/groot_so101_v1/logs/train.log
tail -f outputs/pi05_full_lora_v1/logs/train.log

# Check checkpoints
ls -lh outputs/*/checkpoint-*/
```

### Evaluation
```bash
# GR00T inference - ✅ PRODUCTION READY
cd ~/Isaac-GR00T && python scripts/inference_service.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_so101_v1/checkpoint-10000

# Pi0.5 inference - ⚠️ Will not work until LoRA implemented
cd /home/jrobot/project/lerobot && python src/lerobot/scripts/lerobot_eval.py \
    --policy.path=/home/jrobot/project/XLeRobot/outputs/pi05_full_lora_v1/checkpoint-6000
```

---

## Timeline Summary

### Week 1: Setup & Multi-Task MVP
- **Day 1:** Environment setup (GR00T + Pi0.5)
- **Day 2-4:** Record 30 pick-and-place episodes (MVP primary task)
- **Day 5-6:** Record 20 push/slide episodes (MVP secondary task)
- **Day 7:** Run GR00T MVP test (1-2 hours), analyze multi-task learning

### Week 2-3: Full Multi-Task Dataset Collection
- **Week 2:** Add 10 pick + 10 push episodes (4-5 days @ 5 episodes/day)
- **Week 3:** Add 15 open + 15 close drawer episodes (6-7 days @ 5 episodes/day)
- **Total added:** 50 episodes (50 → 100 total)
- **Goal:** 4-task diverse dataset with balanced coverage

### Week 4: Full Training & Multi-Task Evaluation
- **Day 1:** Run full training (6-8 hours, overnight)
- **Day 2-3:** Manual testing on robot (10 trials × 4 tasks = 40 trials)
- **Day 4:** Analyze per-task performance, identify weak tasks
- **Day 5:** Optional: Collect 5-10 more episodes for weakest task

### Week 5: Deployment & Iteration
- **Day 1-2:** Benchmark evaluation (comprehensive testing)
- **Day 3:** Performance analysis, compare to baselines
- **Day 4:** Deploy best model to production
- **Day 5:** Monitor real-world multi-task performance

**Total Time: ~5 weeks (multi-task approach with validation)**

### Alternative: Faster Single-Task Timeline

If you prefer single-task MVP (faster validation):

**Week 1:** Record 50 pick-and-place episodes, run MVP test
**Week 2-3:** Add 50 more pick-and-place episodes (single task)
**Week 4:** Full training & evaluation on single task
**Total Time: ~4 weeks**

This is faster but loses multi-task generalization benefits.

---

## Support & Resources

### Documentation
- **This Guide:** `/home/jrobot/project/XLeRobot/jdocs/top_level/lora/COMPLETE_LORA_GUIDE.md`
- **Recording Guide:** `/home/jrobot/project/XLeRobot/jdocs/top_level/RECORDING_GUIDE.md`
- **Pi0.5 Issue Tracker:** `/home/jrobot/project/XLeRobot/jdocs/top_level/finetuning_pi0_5_issue.md`

### External Resources
- **GR00T Documentation:** https://github.com/NVIDIA/Isaac-GR00T/tree/main/getting_started
- **GR00T Tuning Guide:** https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- **LeRobot Documentation:** https://github.com/huggingface/lerobot
- **Pi0.5 Paper:** https://arxiv.org/abs/2410.24164

### Getting Help

**If you encounter issues:**

1. **Check troubleshooting section** in this guide
2. **Review training logs** for error messages
3. **Search GitHub issues:**
   - Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T/issues
   - LeRobot: https://github.com/huggingface/lerobot/issues
4. **Ask in communities:**
   - LeRobot Discord
   - NVIDIA Developer Forums

---

## Changelog

**2025-11-24 (v1.2.1):** Polish updates (GPT-5 follow-up review)
- **GR00T Scripts Polish:**
  - Added DATASET_PATH update reminder (for v3→v2 conversion)
  - Clarified conversion script runs from LeRobot repo directory
  - Added step-by-step conversion instructions in warnings
- **Guide Clarity:**
  - Clarified conversion script location (LeRobot PR #2109, not XLeRobot)
  - Added "design stub only" notes to all Pi0.5 code examples
  - Annotated Quick Reference commands (✅ ready vs ❌ not implemented)
  - Added notes to benchmark and helper script examples

**2025-11-24 (v1.2):** Production-ready fixes (based on GPT-5 review)
- **GR00T Scripts Fixed:**
  - Added dataset v2/modality.json checks (required by GR00T)
  - Added LeRobot v3 → v2 conversion warnings
  - Added VRAM fallback guidance (batch size reduction)
  - Improved error messages for missing files
- **Pi0.5 LoRA Clarified:**
  - Added "NOT IMPLEMENTED" warnings to all Pi0.5 sections
  - Scripts now exit immediately with helpful error messages
  - Fixed CLI flags to match `lerobot_train.py` (when implemented)
  - Marked as design stubs for future work
- **Guide Improvements:**
  - Added implementation status section at top
  - Clarified dataset version requirements (v2 for GR00T, v3 for Pi0.5)
  - Documented modality.json creation process
  - Added helper scripts status (not implemented)
- **Recommendation:** Focus on GR00T track (proven, production-ready)

**2025-11-23 (v1.1):** Multi-task dataset strategy
- Added comprehensive multi-task learning approach
- MVP stage: 50 episodes (30 pick + 20 push)
- Full stage: 100 episodes across 4 tasks
- Task-specific recording tips and variation guidelines
- Multi-task performance analysis and evaluation protocols
- Updated timeline to reflect multi-task data collection (5 weeks)

**2025-11-23 (v1.0):** Initial version
- GR00T 1.5 LoRA implementation (MVP + Full training)
- Pi0.5 LoRA implementation (experimental)
- Complete setup, training, and evaluation guides
- Troubleshooting and performance comparison

---

**🎯 Ready to Start?**

1. ✅ Read [Prerequisites](#prerequisites)
2. ✅ Review [Dataset Collection Strategy](#dataset-collection-strategy) (multi-task approach)
3. ✅ Follow [Environment Setup](#environment-setup)
4. ✅ Record 50 episodes for MVP (30 pick + 20 push)
5. ✅ Run [MVP Stage](#mvp-stage-500-steps) and validate multi-task learning
6. ✅ Collect 50 more episodes → Run [Full Training](#full-training-stage)

**Good luck with your multi-task LoRA finetuning! 🚀**
