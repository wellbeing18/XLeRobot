# Pi0.5 Inference Script - Quick Start Guide

## Overview

The corrected Pi0.5 inference script (`run_pi05_inference_corrected.py`) runs the pretrained pi0.5 model on your dual SO-ARM101 follower arms to establish a baseline BEFORE finetuning.

## Critical Fixes from Original Script

The original `run_pi05_dual_arm_inference.py` had **critical bugs**:

1. ❌ **WRONG IMPORT**: Used `PI0Policy` instead of `PI05Policy` (line 14)
2. ❌ **Loaded dataset stats**: Not needed for pretrained pi0.5
3. ⚠️ **Incomplete logging**: Hard to debug issues

The corrected script fixes all of these issues!

## Prerequisites

### 1. Verify Hardware Connections

**Robot arms:**
```bash
ls -l /dev/ttyACM*
# Expected output:
# /dev/ttyACM2 -> left arm
# /dev/ttyACM3 -> right arm
```

**Cameras:**
```bash
ls -l /dev/video*
# Expected output:
# /dev/video4 -> left wrist camera
# /dev/video6 -> right wrist camera
# /dev/video8 -> head/base camera
```

### 2. Verify Model is Downloaded

```bash
ls -la ~/.cache/huggingface/hub/ | grep pi05
# Should show: models--lerobot--pi05_base
```

✅ **Already confirmed**: Your model is cached and ready!

### 3. Verify Calibration Files

```bash
ls -la ~/.cache/lerobot/calibration/
# Should show:
# follower_left_so101/
# follower_right_so101/
```

## Configuration

Before running, you may need to modify these settings in the script (lines 30-60):

### Camera Indices (if different)
```python
CAMERA_INDICES = {
    "left_wrist_0_rgb": 4,   # Change if /dev/video4 is wrong
    "right_wrist_0_rgb": 6,  # Change if /dev/video6 is wrong
    "base_0_rgb": 8,         # Change if /dev/video8 is wrong
}
```

### Robot Ports (if different)
```python
LEFT_FOLLOWER_PORT = "/dev/ttyACM2"   # Change if different
RIGHT_FOLLOWER_PORT = "/dev/ttyACM3"  # Change if different
```

### Task Description
```python
TASK = "pick up the red cube"  # Modify based on your setup
```

### Test Parameters
```python
MAX_EPISODES = 5                # Number of test episodes
MAX_STEPS_PER_EPISODE = 150     # ~30 seconds per episode at 5Hz
CONTROL_FREQUENCY_HZ = 5        # Matches your recorded dataset
```

## Running the Script

### Step 1: Activate Environment
```bash
conda activate lerobot
```

### Step 2: Navigate to Directory
```bash
cd /home/jrobot/project/XLeRobot/jdocs/top_level
```

### Step 3: Run the Script
```bash
python run_pi05_inference_corrected.py
```

Or make it executable and run directly:
```bash
chmod +x run_pi05_inference_corrected.py
./run_pi05_inference_corrected.py
```

### Step 4: Monitor Output

The script provides **extensive logging** at every step:

```
[CONFIG] Using device: cuda
[IMPORT] ✅ All LeRobot modules imported successfully
[MODEL] ✅ Pi0.5 model loaded in 32.4s
[CAMERA] ✅ Connected to left_wrist_0_rgb
[ROBOT] ✅ Left arm connected
[STEP 0] 📷 left_wrist_0_rgb: (480, 640, 3)
[STEP 0] 🤖 Left arm state keys: ['shoulder_pan_pos', ...]
[STEP 0] 🧠 Model inference: 95.3ms
[STEP 0] ⚡ Left action: {'shoulder_pan_pos': 45.2, ...}
[STEP 0] ✅ Actions sent to both arms
```

### Step 5: Stop Anytime
Press `Ctrl+C` to stop gracefully. The script will safely disconnect all devices.

## What to Expect

### Expected Baseline Performance

Since this is the **pretrained (NOT finetuned)** model:

- **Success rate**: 0-15% ⚠️
- **Behavior**: May move erratically or randomly
- **Reason**: Egocentric cameras are OUT-OF-DISTRIBUTION

**This is NORMAL and EXPECTED!** The pretrained pi0.5 was trained on:
- External fixed cameras (not egocentric moving cameras)
- Different robot setups
- Different environments

### Why Run This?

1. **Establish baseline**: Know starting performance before finetuning
2. **Verify pipeline**: Ensure all hardware/software works
3. **Compare later**: After finetuning, you'll see improvement!

## Recording Results

At the end of each run, the script will prompt you to record:

1. **Success count**: How many episodes completed the task?
2. **Observed behaviors**: What did the arms do?
3. **Success rate**: Estimated percentage

**IMPORTANT**: Save these observations! You'll compare them after finetuning.

## Troubleshooting

### Import Error: Cannot find PI05Policy
```
❌ ImportError: cannot import name 'PI05Policy'
```

**Solution**: Update LeRobot
```bash
cd /home/jrobot/project/lerobot
git pull
pip install -e .
```

### Camera Connection Failed
```
❌ [CAMERA] Camera setup failed
```

**Solution**: Check camera indices
```bash
# Test each camera
v4l2-ctl --device=/dev/video4 --all
v4l2-ctl --device=/dev/video6 --all
v4l2-ctl --device=/dev/video8 --all

# Or use your camera test script
```

### Robot Connection Failed
```
❌ [ROBOT] Robot connection failed
```

**Solution**: Check USB ports
```bash
# Verify ports exist
ls -l /dev/ttyACM*

# Check permissions
sudo chmod 666 /dev/ttyACM2
sudo chmod 666 /dev/ttyACM3
```

### Out of Memory Error
```
❌ CUDA out of memory
```

**Solution**: Pi0.5 needs ~6-8GB GPU memory. Your RTX 5090 has 24GB, so this shouldn't happen. If it does:
```bash
# Check what's using GPU
nvidia-smi

# Kill other processes if needed
```

### Arms Don't Move
Check the logs for:
```
[STEP X] ⚡ Left action: {...}
[STEP X] ⚡ Right action: {...}
```

If you see actions being sent but arms don't move:
1. Check calibration files exist
2. Verify `use_degrees=True` in robot config
3. Check action values are reasonable (0-360 degrees)

## Performance Metrics

The script tracks:
- **Inference time**: Time per model forward pass (expect ~80-100ms)
- **Frequency**: Control loop frequency (target: 5Hz)
- **GPU memory**: Memory usage (expect ~6-8GB)

Example output:
```
✅ Average inference time: 95.3ms
✅ Average frequency: 5.2Hz
✅ Target frequency: 5Hz
✅ Performance: EXCELLENT (meeting target frequency)
```

## Next Steps After Running

1. **Record baseline success rate** (0-15% expected)
2. **Finetune pi0.5** on your 10 recorded episodes:
   - Follow `VLM_VLA_FINETUNING_STRATEGY.md` Week 2 instructions
   - Train for 6000 steps (~6-10 hours)
3. **Run this script again** with finetuned checkpoint
4. **Compare results**:
   - Baseline: 0-15%
   - After finetuning: 40-60% (with egocentric cameras)
   - Improvement: +40-45% 🎉

## Key Differences from SmolVLA Test Script

If you're familiar with `test_smolvla_official.py`, here are the key differences:

| Aspect | SmolVLA Script | Pi0.5 Script |
|--------|----------------|--------------|
| **Import** | `SmolVLAPolicy` | `PI05Policy` ✅ |
| **Model ID** | `lerobot/smolvla_base` | `lerobot/pi05_base` ✅ |
| **Dataset Stats** | ❌ Required (loads dataset) | ✅ NOT needed! |
| **Camera Keys** | `camera1`, `camera2` | `left_wrist_0_rgb`, etc. ✅ |
| **Robot Setup** | Single arm | Dual arms ✅ |
| **Model Size** | 450M params (~2GB GPU) | 4B params (~6-8GB GPU) |
| **Expected Speed** | ~30Hz | ~10-15Hz |

## File Locations

**New corrected script:**
```
/home/jrobot/project/XLeRobot/jdocs/top_level/run_pi05_inference_corrected.py
```

**Original buggy script (kept for reference):**
```
/home/jrobot/project/XLeRobot/jdocs/top_level/run_pi05_dual_arm_inference.py
```

**Cached model:**
```
~/.cache/huggingface/hub/models--lerobot--pi05_base/
```

**Calibration files:**
```
~/.cache/lerobot/calibration/follower_left_so101/
~/.cache/lerobot/calibration/follower_right_so101/
```

## Support

If you encounter issues:

1. Check the **extensive logs** printed by the script
2. Verify hardware connections (cameras, arms)
3. Ensure LeRobot is up to date
4. Review `VLM_VLA_FINETUNING_STRATEGY.md` Section 8 (Known Issues)

## Summary

✅ **Script is ready to run!**
✅ **All imports verified**
✅ **Syntax checked**
✅ **Comprehensive logging included**
✅ **Hardware configuration easy to modify**

**Expected result**: Arms will move (possibly erratically) for ~5 episodes. This establishes your baseline before finetuning. Success rate: 0-15%.

Good luck! 🤖🚀
