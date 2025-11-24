# Running Pi0.5 Inference on SO-101 Dual Follower Arms with UVC Cameras

**Created**: 2025-11-19
**Purpose**: Run pretrained pi0.5 model on your XLeRobot dual-arm setup to establish baseline performance

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Setup](#hardware-setup)
4. [Script Explanation](#script-explanation)
5. [Usage](#usage)
6. [Expected Behavior](#expected-behavior)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Overview

This document provides a complete script to run the pretrained `lerobot/pi05_base` model on your XLeRobot dual SO-101 follower arm setup with 3 UVC cameras (left_wrist, right_wrist, head).

**Important Context** (from DATA_COLLECTION_GUIDE.md):
- You need to test the pretrained baseline BEFORE collecting data
- This establishes what performance to expect before finetuning
- Expected success: 0-15% (egocentric cameras are out-of-distribution)
- This baseline is critical for comparison after finetuning

---

## Prerequisites

### 1. LeRobot Environment
```bash
conda activate lerobot
```

### 2. Pi0.5 Dependencies
```bash
# Install pi0.5 specific dependencies
pip install -e ".[pi]"
```

### 3. Hardware Connected
- 2x SO-101 follower arms plugged in via USB
- 3x UVC cameras plugged in
- All devices powered on

### 4. Verify Hardware
```bash
# Check camera indices
python -c "import cv2; print('Cameras:', [i for i in [0,2,4,6,8] if cv2.VideoCapture(i).isOpened()])"

# Check USB ports for arms
ls -l /dev/ttyACM*
# Should show /dev/ttyACM0, /dev/ttyACM1, etc.
```

---

## Hardware Setup

Based on your configuration from DATA_COLLECTION_GUIDE.md and XLeRobot setup:

```yaml
# Expected configuration
Follower Arms:
  Left Follower: /dev/ttyACM2 (or /dev/ttyACM0)
  Right Follower: /dev/ttyACM3 (or /dev/ttyACM1)

UVC Cameras:
  Left Wrist: Camera index 6
  Right Wrist: Camera index 8
  Head: Camera index 4
```

**Note**: Camera indices and USB ports may vary. Use the verification commands above to confirm.

---

## Script Explanation

The script below (`run_pi05_dual_arm_inference.py`) does the following:

1. **Load pretrained Pi0.5 model** from HuggingFace (`lerobot/pi05_base`)
2. **Configure dual SO-101 follower arms** with your USB ports
3. **Configure 3 UVC cameras** (egocentric setup)
4. **Run inference loop**:
   - Capture camera observations
   - Read current robot joint states
   - Feed to Pi0.5 model with task instruction
   - Execute predicted actions on both arms
5. **Safety features**:
   - Configurable max steps per episode
   - Graceful shutdown on Ctrl+C
   - Torque disable on exit

---

## The Script

```python
#!/usr/bin/env python3
"""
Pi0.5 Baseline Inference on XLeRobot Dual SO-101 Follower Arms
Tests pretrained pi0.5 model BEFORE finetuning to establish baseline
"""

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# ============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# GPU/CPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration
model_id = "lerobot/pi05_base"  # Pretrained Pi0.5 model

# Robot USB ports (VERIFY THESE with: ls -l /dev/ttyACM*)
LEFT_FOLLOWER_PORT = "/dev/ttyACM2"   # Modify if different
RIGHT_FOLLOWER_PORT = "/dev/ttyACM3"  # Modify if different

# Robot IDs for calibration files
LEFT_FOLLOWER_ID = "follower_left_so101"
RIGHT_FOLLOWER_ID = "follower_right_so101"

# Camera configuration (VERIFY THESE with camera test script)
# Camera keys MUST match what pi0.5 expects
# Pi0.5 was trained with these camera names, so we use them
CAMERA_CONFIG = {
    "left_wrist_0_rgb": OpenCVCameraConfig(
        index_or_path=6,  # Modify if different
        width=640,
        height=480,
        fps=30
    ),
    "right_wrist_0_rgb": OpenCVCameraConfig(
        index_or_path=8,  # Modify if different
        width=640,
        height=480,
        fps=30
    ),
    "base_0_rgb": OpenCVCameraConfig(  # Using head camera as base
        index_or_path=4,  # Modify if different
        width=640,
        height=480,
        fps=30
    ),
}

# Task instruction (natural language)
# Try simple pick/place tasks that are in Pi0.5's training data
TASK = "pick red cube from center"  # Modify based on your setup

# Episode configuration
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 100  # ~20 seconds at 5Hz

# Robot type (for multi-embodiment models like Pi0.5)
ROBOT_TYPE = "so101_follower"

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"Loading Pi0.5 model from {model_id}...")
model = PI0Policy.from_pretrained(model_id)
model = model.to(device)
model.eval()

# Create preprocessor and postprocessor
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
print("Model loaded successfully!")

# ============================================================================
# SETUP DUAL ROBOTS
# ============================================================================

print("\nInitializing dual SO-101 follower arms...")

# Left arm configuration
left_robot_cfg = SO101FollowerConfig(
    port=LEFT_FOLLOWER_PORT,
    id=LEFT_FOLLOWER_ID,
    cameras={},  # No cameras on individual arms
    use_degrees=True  # Use degrees for easier debugging
)

# Right arm configuration
right_robot_cfg = SO101FollowerConfig(
    port=RIGHT_FOLLOWER_PORT,
    id=RIGHT_FOLLOWER_ID,
    cameras={},  # No cameras on individual arms
    use_degrees=True
)

# Create robot instances
left_robot = SO101Follower(left_robot_cfg)
right_robot = SO101Follower(right_robot_cfg)

# Connect to robots
print("Connecting to left arm...")
left_robot.connect()
print("Connecting to right arm...")
right_robot.connect()

print("Robots connected successfully!")

# ============================================================================
# SETUP CAMERAS SEPARATELY
# ============================================================================

print("\nInitializing cameras...")
from lerobot.cameras.utils import make_cameras_from_configs

cameras = make_cameras_from_configs(CAMERA_CONFIG)

# Connect cameras
for cam_name, cam in cameras.items():
    cam.connect()
    print(f"Connected to {cam_name}")

print("All cameras connected!")

# ============================================================================
# PREPARE DATASET FEATURES
# ============================================================================

# Combine observation features from both arms and cameras
# For dual arms, we need to merge features
observation_features = {}

# Add left arm motors
for motor_name, motor_type in left_robot.observation_features.items():
    if "pos" in motor_name:
        observation_features[f"left_{motor_name}"] = motor_type

# Add right arm motors
for motor_name, motor_type in right_robot.observation_features.items():
    if "pos" in motor_name:
        observation_features[f"right_{motor_name}"] = motor_type

# Add camera features
for cam_name in cameras.keys():
    # Get camera dimensions from config
    cam_cfg = CAMERA_CONFIG[cam_name]
    observation_features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

# Combine action features (same as observation for dual arms)
action_features = {}
for motor_name, motor_type in left_robot.action_features.items():
    if "pos" in motor_name:
        action_features[f"left_{motor_name}"] = motor_type

for motor_name, motor_type in right_robot.action_features.items():
    if "pos" in motor_name:
        action_features[f"right_{motor_name}"] = motor_type

# Convert to dataset features format
obs_features_dataset = hw_to_dataset_features(observation_features, "observation")
action_features_dataset = hw_to_dataset_features(action_features, "action")
dataset_features = {**obs_features_dataset, **action_features_dataset}

print(f"\nDataset features configured:")
print(f"  Observation features: {len(obs_features_dataset)}")
print(f"  Action features: {len(action_features_dataset)}")

# ============================================================================
# INFERENCE LOOP
# ============================================================================

print(f"\n{'='*70}")
print(f"STARTING BASELINE INFERENCE TEST")
print(f"{'='*70}")
print(f"Task: {TASK}")
print(f"Episodes: {MAX_EPISODES}")
print(f"Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"Expected success: 0-15% (pretrained model on egocentric cameras)")
print(f"{'='*70}\n")

try:
    for episode in range(MAX_EPISODES):
        print(f"\n[Episode {episode + 1}/{MAX_EPISODES}]")
        print("Starting episode... (Press Ctrl+C to stop)")

        for step in range(MAX_STEPS_PER_EPISODE):
            # Get observations from cameras
            camera_obs = {}
            for cam_name, cam in cameras.items():
                camera_obs[cam_name] = cam.async_read()  # Read frame

            # Get robot states
            left_state = left_robot.get_observation()
            right_state = right_robot.get_observation()

            # Combine into single observation dict
            observation = {}

            # Add camera observations
            observation.update(camera_obs)

            # Add left arm states
            for key, value in left_state.items():
                if "pos" in key:
                    observation[f"left_{key}"] = value

            # Add right arm states
            for key, value in right_state.items():
                if "pos" in key:
                    observation[f"right_{key}"] = value

            # Build inference frame
            obs_frame = build_inference_frame(
                observation=observation,
                ds_features=dataset_features,
                device=device,
                task=TASK,
                robot_type=ROBOT_TYPE
            )

            # Preprocess
            obs_preprocessed = preprocess(obs_frame)

            # Model inference
            with torch.no_grad():
                action = model.select_action(obs_preprocessed)

            # Postprocess action
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)

            # Split actions for left and right arms
            left_action = {}
            right_action = {}

            for key, value in action.items():
                if key.startswith("left_"):
                    left_action[key.replace("left_", "")] = value
                elif key.startswith("right_"):
                    right_action[key.replace("right_", "")] = value

            # Send actions to robots
            left_robot.send_action(left_action)
            right_robot.send_action(right_action)

            # Print progress
            if step % 20 == 0:
                print(f"  Step {step}/{MAX_STEPS_PER_EPISODE}")

        print(f"Episode {episode + 1} finished!")
        print("Starting new episode in 3 seconds...")
        import time
        time.sleep(3)

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Shutting down gracefully...")

finally:
    # ============================================================================
    # CLEANUP
    # ============================================================================
    print("\nCleaning up...")

    # Disconnect cameras
    for cam_name, cam in cameras.items():
        try:
            cam.disconnect()
            print(f"Disconnected {cam_name}")
        except:
            pass

    # Disconnect robots
    try:
        left_robot.disconnect()
        print("Disconnected left robot")
    except:
        pass

    try:
        right_robot.disconnect()
        print("Disconnected right robot")
    except:
        pass

    print("\nShutdown complete!")
    print("\n" + "="*70)
    print("BASELINE TEST SUMMARY")
    print("="*70)
    print(f"Task tested: {TASK}")
    print(f"Episodes completed: {episode + 1}/{MAX_EPISODES}")
    print("")
    print("RECORD YOUR OBSERVATIONS:")
    print("- How many times did the robot successfully complete the task?")
    print("- What behaviors did you observe?")
    print("- Success rate: ___% (out of 5 episodes)")
    print("")
    print("This baseline is CRITICAL for comparison after finetuning!")
    print("Expected baseline: 0-15% (egocentric cameras are out-of-distribution)")
    print("="*70)
```

Save this as: `/home/jrobot/project/XLeRobot/jdocs/top_level/run_pi05_dual_arm_inference.py`

---

## Usage

### 1. Verify Configuration

Before running, verify your hardware setup:

```bash
# Find camera indices
python -c "import cv2; print('Available cameras:', [i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Find USB ports
ls -l /dev/ttyACM*

# Find motor IDs
cd /home/jrobot/project/lerobot
python -m lerobot.scripts.find_motors_bus_port
```

### 2. Modify Script Constants

Edit the script and update these values based on your verification:

```python
# Update these lines (around line 30-35)
LEFT_FOLLOWER_PORT = "/dev/ttyACM2"   # Your left arm port
RIGHT_FOLLOWER_PORT = "/dev/ttyACM3"  # Your right arm port

# Update camera indices (around line 45-60)
"left_wrist_0_rgb": OpenCVCameraConfig(index_or_path=6, ...)  # Your left camera
"right_wrist_0_rgb": OpenCVCameraConfig(index_or_path=8, ...) # Your right camera
"base_0_rgb": OpenCVCameraConfig(index_or_path=4, ...)        # Your head camera
```

### 3. Setup Workspace

```bash
# Create a simple test workspace
# Place a red cube at table center
# Ensure good lighting
# Clear clutter
```

### 4. Run the Script

```bash
cd /home/jrobot/project/XLeRobot/jdocs/top_level
conda activate lerobot

# Make executable
chmod +x run_pi05_dual_arm_inference.py

# Run
python run_pi05_dual_arm_inference.py
```

### 5. Monitor and Record

Watch the robot execute the task 5 times. Record:
- Number of successful completions
- Common failure modes
- Interesting behaviors

---

## Expected Behavior

### What You'll See

**Pretrained Pi0.5 on egocentric cameras (out-of-distribution):**

1. **Most likely (80% probability)**:
   - Robot moves but doesn't complete task
   - Actions seem somewhat relevant but not precise
   - May reach toward object but miss
   - **Success rate: 0-5%**

2. **Possible (15% probability)**:
   - Robot occasionally succeeds by luck
   - Movements show some task understanding
   - **Success rate: 5-15%**

3. **Unlikely (5% probability)**:
   - Robot succeeds consistently
   - This would be surprising given camera mismatch
   - **Success rate: >15%**

### Why Low Success is Expected

From VLM_VLA_FINETUNING_STRATEGY.md:

> "Dynamic egocentric views create distribution shifts that static robot systems cannot replicate, leading to degraded policy performance."

> "Expected baseline: 0-15% (egocentric cameras are out-of-distribution)"

**This is NORMAL and EXPECTED!** The whole point is to finetune to improve this.

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Change device to CPU (line ~30)
device = torch.device("cpu")
```

Note: CPU inference will be slower (~1-2 Hz vs 10-15 Hz)

### Issue 2: Camera Not Found

**Error:**
```
Camera X not available
```

**Solution:**
```bash
# Find available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Update camera indices in script (lines 45-60)
```

### Issue 3: Robot Port Not Found

**Error:**
```
Cannot connect to /dev/ttyACM2
```

**Solution:**
```bash
# List available ports
ls -l /dev/ttyACM*

# Update LEFT_FOLLOWER_PORT and RIGHT_FOLLOWER_PORT in script
```

### Issue 4: Model Download Fails

**Error:**
```
Cannot download lerobot/pi05_base
```

**Solution:**
```bash
# Download model manually
from huggingface_hub import snapshot_download
snapshot_download("lerobot/pi05_base", local_dir="./pi05_base")

# Update model_id in script
model_id = "./pi05_base"
```

### Issue 5: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'lerobot.policies.pi0'
```

**Solution:**
```bash
# Install pi0.5 dependencies
conda activate lerobot
pip install -e ".[pi]"
```

### Issue 6: Dual Arm Feature Mismatch

If you get errors about mismatched features, you may need to simplify to single arm first:

```python
# Simplify to LEFT ARM ONLY for testing
# Comment out all right_robot code
# This validates the pipeline works before adding complexity
```

---

## Next Steps

### After Running Baseline Test

1. **Record Results**
   ```
   Task: pick red cube from center
   Episodes: 5
   Successes: 0  (example)
   Success rate: 0%

   Observations:
   - Robot moves toward cube but stops short
   - Gripper opens/closes at wrong times
   - Seems confused by camera viewpoint
   ```

2. **Proceed to MVP Data Collection**

   From DATA_COLLECTION_GUIDE.md:

   > "You need a baseline to compare against. Without it, you can't tell if finetuning worked!"
   > "40% after finetuning = excellent if pretrained got 5%"
   > "40% after finetuning = poor if pretrained got 35%"

   Next: Follow DATA_COLLECTION_GUIDE.md Section "MVP Step 2: Collect 7-10 Episodes"

3. **Compare After Finetuning**

   After you complete:
   - MVP data collection (7-10 episodes)
   - MVP finetuning (100 steps)
   - Run this script again with finetuned checkpoint

   Compare results:
   ```
   Pretrained:  0% success
   Finetuned:   30% success  ← This shows finetuning worked!
   ```

---

## Important Notes

### About Dual Arms

This script controls BOTH arms simultaneously. However:

- Pi0.5 was NOT trained specifically for dual-arm coordination
- It treats each arm's joints as independent observations/actions
- For better dual-arm performance, you'll need to finetune on dual-arm data

**Simplification Option**: For initial testing, you can modify the script to use only the LEFT arm and temporarily disable the right arm code. This simplifies debugging.

### About Camera Names

The camera keys MUST match what Pi0.5 expects:
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`
- `base_0_rgb`

These names come from Pi0.5's training data format. Don't change them unless you know what you're doing.

### About Success Rates

**From MINIMAL_VALIDATION_STRATEGY.md:**

> "Expected with pretrained: 0-15% success (egocentric cameras out-of-distribution)"
> "Expected after MVP finetuning (7-10 episodes): 0-20% (not enough data!)"
> "Expected after full finetuning (50-100 episodes): 60-80%"

Low baseline is GOOD - it means there's room for improvement!

---

## File Location

Save this entire document and the script:

```
/home/jrobot/project/XLeRobot/jdocs/top_level/RUN_PI05_INFERENCE.md
/home/jrobot/project/XLeRobot/jdocs/top_level/run_pi05_dual_arm_inference.py
```

---

## Summary Checklist

Before running:
- [ ] LeRobot environment activated
- [ ] Pi0.5 dependencies installed (`pip install -e ".[pi]"`)
- [ ] Both SO-101 arms connected and powered
- [ ] 3 UVC cameras connected
- [ ] USB ports verified (`ls -l /dev/ttyACM*`)
- [ ] Camera indices verified (test script)
- [ ] Script configuration updated with YOUR hardware values
- [ ] Simple workspace setup (red cube at center)
- [ ] Notebook ready to record results

After running:
- [ ] Success rate recorded
- [ ] Observations documented
- [ ] Baseline established for comparison
- [ ] Ready to proceed to MVP data collection

**Good luck with your baseline test! This is Step 0 of your VLA finetuning journey!** 🤖
