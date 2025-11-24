# XLeRobot Recording Guide

**Purpose**: Record demonstration episodes for Pi0.5 fine-tuning
**Date Created**: 2025-11-22
**Status**: Tested and Working

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the Recording Command](#understanding-the-recording-command)
3. [Common Issues and Solutions](#common-issues-and-solutions)
4. [Recording Best Practices](#recording-best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### **Standard Recording Command (7 episodes)**

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
  --dataset.num_episodes=7 \
  --dataset.single_task="pick red_cube from center" \
  --dataset.push_to_hub=false \
  --display_data=false \
  --resume=true
```

---

## Understanding the Recording Command

### **Key Parameters You Might Want to Change**

| Parameter | What It Does | How to Change It |
|-----------|--------------|------------------|
| `--dataset.num_episodes=7` | Number of episodes to record | Change `7` to desired number (e.g., `1` for testing, `10` for full dataset) |
| `--dataset.episode_time_s=30` | Duration of each episode in seconds | Change `30` to desired length (e.g., `20` for faster tasks, `45` for complex ones) |
| `--dataset.reset_time_s=15` | Time between episodes to reset setup | Change `15` to more/less time as needed |
| `--dataset.single_task="pick red_cube from center"` | Task description | Change to describe your task (e.g., `"place cube in box"`) |
| `--dataset.repo_id=lerobot/xlerobot_mvp_pick_15fps` | Dataset name | Change to organize different tasks/versions |

### **Parameters You Should NOT Change**

| Parameter | Value | Why Keep It |
|-----------|-------|-------------|
| `--dataset.fps=5` | 5 Hz | **CRITICAL** - Pi0.5 expects 5Hz action frequency |
| `fps: 15` (in cameras) | 15 fps | Fixes video glitching issue (see below) |
| `--robot.port=/dev/ttyACM2` | ttyACM2 | Left follower arm port (verify with `ls /dev/ttyACM*`) |
| `--teleop.port=/dev/ttyACM0` | ttyACM0 | Left leader arm port |
| `--dataset.push_to_hub=false` | false | Prevents uploading to HuggingFace during recording |
| `--display_data=false` | false | Disables Rerun viewer (reduces lag) |
| `--resume=true` | true | Required to add episodes to existing dataset |

---

## Common Issues and Solutions

### **Issue 1: Video Glitching/Tearing** ✅ SOLVED

**Symptom**: Bottom third of video has twitching/glitching, appears in different rows each time

**Cause**: USB bandwidth saturation when using USB hub with 2 cameras at 30fps

**Solution**: Reduce camera FPS from 30 to 15

```bash
# BEFORE (glitchy):
--robot.cameras='{"left_wrist": {..., "fps": 30}, "head": {..., "fps": 30}}'

# AFTER (fixed):
--robot.cameras='{"left_wrist": {..., "fps": 15}, "head": {..., "fps": 15}}'
```

**Why this works**:
- You're recording actions at 5Hz anyway
- 15fps camera is still 3x more than needed
- Halves USB bandwidth usage
- **No impact on training quality** - model only uses 5 frames/second

---

### **Issue 2: FileExistsError When Starting Recording** ✅ SOLVED

**Symptom**:
```
FileExistsError: [Errno 17] File exists: '/home/jrobot/project/XLeRobot/jdocs/top_level/datasets'
```

**Cause**: LeRobot tries to create dataset directory with `exist_ok=False`, but directory already exists

**Solution**: Add `--resume=true` flag

```bash
--resume=true
```

**When to use**:
- ✅ Adding episodes to an existing dataset
- ✅ Continuing after a previous recording session
- ❌ Creating a brand new dataset (use different `--dataset.repo_id` instead)

---

### **Issue 3: Follower Arm Moves in Jerky/Chunky Increments**

**Symptom**: Follower arm updates in choppy steps instead of smooth motion during recording

**This is NORMAL and EXPECTED!**

**Why**: Recording at 5Hz means follower only gets updated 5 times per second

**Comparison**:
- Normal teleoperation: 30-60 Hz (smooth)
- Recording: 5 Hz (chunky)

**What to do**:
- Move the leader arm **very slowly** (2-3x slower than normal)
- The chunkiness doesn't affect training quality
- The trained model will still move smoothly during inference

---

### **Issue 4: Resolution Cannot Be Changed**

**Symptom**:
```
RuntimeError: OpenCVCamera(6) failed to set capture_width=480
```

**Cause**: UVC cameras only support specific resolutions (640x480, 320x240, etc.)

**Solution**: Stick with 640x480 or use 320x240 (not recommended - too low quality)

```bash
# Supported:
"width": 640, "height": 480  ✅
"width": 320, "height": 240  ✅ (but very low quality)

# NOT supported:
"width": 480, "height": 360  ❌
```

---

## Recording Best Practices

### **Before Recording**

1. **Verify hardware connections**:
   ```bash
   ls -l /dev/ttyACM*  # Should show ACM0, ACM1, ACM2, ACM3
   ls -l /dev/video*   # Should show video4, video6, video8
   ```

2. **Test with 1 episode first**:
   - Change `--dataset.num_episodes=1`
   - Complete one full recording
   - Check video for glitching/quality
   - Then record full dataset

3. **Prepare workspace**:
   - Consistent lighting
   - Cube at same position for all episodes
   - Clear workspace

### **During Recording**

1. **Move VERY slowly**
   - 2-3x slower than natural speed
   - Target: 15-30 seconds per episode
   - The 5Hz recording makes it feel chunky - that's normal!

2. **Stay consistent**:
   - Same approach strategy every episode
   - Same cube position
   - Same starting arm position

3. **Complete the task**:
   - Don't stop mid-motion
   - Hold final position for 2-3 seconds
   - Wait for "Stop recording" message

### **After Recording**

1. **Check episode count**:
   ```bash
   cat /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/info.json | grep total_episodes
   ```

2. **Check video quality**:
   ```bash
   # Find latest video
   ls -lht /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/videos/observation.images.head/chunk-000/*.mp4 | head -1

   # Play with VLC
   vlc [path_to_video]
   ```

3. **Verify no glitching** in recorded videos

---

## Troubleshooting

### **Cameras not found**

```bash
# Check camera indices
ls -l /dev/video*

# Test cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in [4, 6, 8]])"
```

**Expected**: `[True, True, True]` for cameras 4, 6, 8

### **Motor ports wrong**

```bash
# Find correct ports
python -m lerobot.scripts.lerobot_find_port

# Update command with correct ports
--robot.port=/dev/ttyACM[X]
--teleop.port=/dev/ttyACM[Y]
```

### **Ctrl+C during recording**

**Effect**: Incomplete episode is discarded (not saved)

**Solution**: Let episodes complete fully, or re-record

### **"Overload error" when disconnecting**

**Symptom**:
```
RuntimeError: Failed to write 'Torque_Enable' on id_=6 with '0' after 6 tries. [RxPacketError] Overload error!
```

**This is harmless!** It happens during motor shutdown after recording is complete. All your data is already saved.

---

## Example: Recording Different Numbers of Episodes

### **Test Recording (1 episode)**
```bash
--dataset.num_episodes=1
--dataset.repo_id=lerobot/xlerobot_test
```

### **MVP Recording (7-10 episodes)**
```bash
--dataset.num_episodes=7
--dataset.repo_id=lerobot/xlerobot_mvp_pick
```

### **Full Dataset (50+ episodes)**
```bash
--dataset.num_episodes=50
--dataset.repo_id=lerobot/xlerobot_full_pick
```

---

## File Structure After Recording

```
/home/jrobot/project/XLeRobot/jdocs/top_level/datasets/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet  (robot states & actions)
│       └── file-001.parquet
├── meta/
│   ├── info.json             (dataset metadata)
│   ├── stats.json            (normalization stats)
│   ├── tasks.parquet         (task descriptions)
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  (episode boundaries)
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       ├── file-000.mp4  (head camera video)
    │       └── file-001.mp4
    └── observation.images.left_wrist/
        └── chunk-000/
            ├── file-000.mp4  (left wrist camera video)
            └── file-001.mp4
```

**All files are needed for training!**

---

## Quick Reference

### **Current Working Setup**
- **Robot**: Left arm only (SO-101)
- **Cameras**: Left wrist (video6) + Head (video4) at **15fps**
- **Action frequency**: 5Hz
- **Ports**:
  - Left follower: `/dev/ttyACM2`
  - Left leader: `/dev/ttyACM0`

### **Why 15fps Cameras?**
- Using USB hub causes bandwidth issues at 30fps
- 15fps eliminates video glitching
- Still 3x higher than needed (action recording is 5Hz)
- **No impact on training quality**

### **Why 5Hz Actions?**
- Pi0.5 was trained on 5Hz data
- Standard for robot learning
- Smaller dataset files
- Sufficient for manipulation tasks

---

**Last Updated**: 2025-11-22
**Tested Configuration**: SO-101 left arm, 2 cameras at 15fps, 5Hz actions
