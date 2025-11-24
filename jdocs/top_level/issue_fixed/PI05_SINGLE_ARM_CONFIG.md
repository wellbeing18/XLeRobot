# Pi0.5 Single Arm Configuration - Quick Reference

## ✅ Script Updated for Your Setup

The script `run_pi05_inference_corrected.py` has been configured for:

### Hardware Configuration

**Single Arm:**
- Left follower only: `/dev/ttyACM0`
- Calibration ID: `follower_left_so101`
- Leader arm (ACM2) is NOT used during inference

**Cameras (2 cameras):**
- Head/base camera: `/dev/video4`
- Left wrist camera: `/dev/video6`
- Right wrist camera: NOT used

**Test Parameters:**
- Episodes: **1** (runs once only)
- Duration: ~30 seconds (150 steps at 5Hz)
- Task: "pick up the red cube"

---

## Current Configuration Summary

```python
# Robot
FOLLOWER_PORT = "/dev/ttyACM0"      # Left follower arm
FOLLOWER_ID = "follower_left_so101"

# Cameras
CAMERA_INDICES = {
    "base_0_rgb": 4,         # Head camera (/dev/video4)
    "left_wrist_0_rgb": 6,   # Left wrist camera (/dev/video6)
}

# Test settings
MAX_EPISODES = 1                 # Run once only
MAX_STEPS_PER_EPISODE = 150     # ~30 seconds at 5Hz
TASK = "pick up the red cube"
```

---

## Running the Script

### Prerequisites Check

```bash
# 1. Verify follower port
ls -l /dev/ttyACM0
# Should show left follower arm

# 2. Verify cameras
ls -l /dev/video4  # Head camera
ls -l /dev/video6  # Left wrist camera

# 3. Verify calibration exists
ls ~/.cache/lerobot/calibration/follower_left_so101/
```

### Run Command

```bash
conda activate lerobot
cd /home/jrobot/project/XLeRobot/jdocs/top_level
python run_pi05_inference_corrected.py
```

---

## What to Expect

### Execution Flow

1. **Startup** (~30s): Load model, connect cameras, connect arm
2. **Single episode** (~30s): Arm moves for 150 steps trying to complete task
3. **Summary**: Performance stats and observation prompts

**Total runtime**: ~1 minute

### Expected Behavior

**The left follower arm will:**
- Receive commands from pi0.5 based on camera + joint state
- Move for ~30 seconds
- Behavior may be:
  - ✅ Smooth but incorrect direction
  - ⚠️ Somewhat erratic (camera domain mismatch)
  - ⚠️ Minimal movement (model uncertain)

**Expected success**: 0-15% (pretrained baseline)

---

## Detailed Logs

The script prints comprehensive logs every 10 steps:

```
[STEP 0] 📷 base_0_rgb: (480, 640, 3)
[STEP 0] 📷 left_wrist_0_rgb: (480, 640, 3)
[STEP 0] 🤖 Arm state keys: ['shoulder_pan_pos', 'shoulder_lift_pos', ...]
[STEP 0]     shoulder_pan_pos: 45.2
[STEP 0]     shoulder_lift_pos: 30.1
[STEP 0]     elbow_flex_pos: -60.5
[STEP 0]     wrist_flex_pos: 15.8
[STEP 0]     wrist_roll_pos: 0.0
[STEP 0]     gripper_pos: 50.0
[STEP 0] 📦 Combined observation keys: ['base_0_rgb', 'left_wrist_0_rgb', 'shoulder_pan_pos', ...]
[STEP 0] 🧠 Model inference: 95.3ms
[STEP 0] ⚡ Action to send: {'shoulder_pan_pos': 48.7, ...}
[STEP 0]     shoulder_pan_pos: 48.7
[STEP 0]     shoulder_lift_pos: 32.3
[STEP 0]     ...
[STEP 0] ✅ Action sent to follower arm
[STEP 0] ⏱️  Step time: 203.4ms (avg freq: 5.1Hz)
[STEP 0] 💾 GPU memory: 6.84 GB
```

You can see:
- Camera frames being read
- Joint positions (in degrees)
- Actions being sent (in degrees)
- Inference speed and GPU usage

---

## Key Changes from Original Dual-Arm Script

| Aspect | Original (Dual Arm) | Updated (Single Arm) |
|--------|-------------------|---------------------|
| **Arms** | Left + Right (ACM2, ACM3) | Left only (ACM0) |
| **Cameras** | 3 cameras (both wrists + head) | 2 cameras (left wrist + head) |
| **Episodes** | 5 episodes | 1 episode |
| **Feature prefix** | `left_*` and `right_*` | No prefix (direct) |
| **Action splitting** | Split for dual arms | Direct send (no split) |
| **Runtime** | ~3-4 minutes | ~1 minute |

---

## Troubleshooting

### Port Not Found
```
❌ [ROBOT] Robot connection failed: [Errno 2] No such file or directory: '/dev/ttyACM0'
```

**Solution**: Check actual port
```bash
ls -l /dev/ttyACM*
# Update FOLLOWER_PORT in script if different
```

### Camera Not Found
```
❌ [CAMERA] Camera setup failed
```

**Solution**: Check camera indices
```bash
ls -l /dev/video*
# Update CAMERA_INDICES in script if different
```

### Model Takes Long to Load
```
[MODEL] ⏳ This will load from cache (~30s) or download first time (~3-5 min)
```

This is normal! First run downloads ~3-5GB. Subsequent runs load from cache in ~30s.

---

## After Running

### Record Your Observations

The script will prompt you:

```
📊 RECORD YOUR OBSERVATIONS:
======================================================================
1. Did the episode successfully complete the task?
   Success: [ ] Yes  [ ] No

2. What behaviors did you observe?
   [ ] Arm moved toward objects
   [ ] Arm grasped objects
   [ ] Arm placed objects correctly
   [ ] Arm moved erratically/randomly
   [ ] Arm stayed mostly still
   [ ] Other: ___________________________

3. Overall assessment: ___________________________
```

**Save these notes!** You'll compare them after finetuning.

---

## Next Steps

1. ✅ **Run script** and record baseline behavior
2. **Expected result**: 0-15% success (camera domain mismatch)
3. **Finetune pi0.5** on your recorded dataset
4. **Run again** with finetuned model
5. **Compare**: Expect 40-60% improvement!

---

## Performance Metrics to Watch

The script tracks:

```
✅ Average inference time: 95.3ms
✅ Average frequency: 5.2Hz
✅ Target frequency: 5Hz
✅ Performance: EXCELLENT (meeting target frequency)
```

**Target**: 5Hz control frequency (matches your recorded dataset)

---

## Summary

✅ **Single left arm only** (follower on ACM0)
✅ **2 cameras** (head + left wrist)
✅ **1 episode** (~30 seconds)
✅ **Comprehensive logging** (every 10 steps)
✅ **Ready to run!**

```bash
conda activate lerobot
python run_pi05_inference_corrected.py
```

The script will run once, show detailed logs, and give you a summary of the arm's behavior. This establishes your baseline before finetuning! 🤖
