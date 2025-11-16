# Step 2.1: SmolVLA Pretrained Model Testing - Complete Breakdown

**Status:** Ready to Execute
**Your Progress:** Completed 1.5, 1.7, 1.8 (basic hardware testing)
**Next:** Test pretrained VLA model inference

---

## Executive Summary

**Goal:** Test if the SmolVLA pretrained model can control your robot without any training or data collection.

**What You'll Do:**
1. Load SmolVLA pretrained model (~2-3GB download)
2. Run it on your SO-101 arm with UVC cameras
3. See if it can perform basic pick-and-place tasks
4. Measure success rate (target: 50%+)

**Time Required:** 2-3 hours (including first-time model download)

**Expected Outcome:**
- ✅ If success rate ≥50%: Proceed to Stage 3 (language control)
- ⚠️ If success rate <50%: Try ACT model or collect custom data

---

## Key Differences: Plan Code vs. Your Hardware

### ❌ **What Won't Work from the MVP Plan (Lines 813-883)**

| Issue | Plan Assumption | Your Actual Hardware | Fix |
|-------|----------------|---------------------|-----|
| Camera type | Intel RealSense | USB UVC cameras | Use OpenCV VideoCapture |
| Import paths | Old LeRobot structure | Current version | Updated imports |
| Camera indices | Not specified | 3 cameras at indices 0,2,4 | Explicit indexing |
| Observation format | Assumed correct | Needs validation | Added preprocessing |

### ✅ **What I Fixed in `test_smolvla_inference.py`**

1. **UVC Camera Support** - Uses OpenCV instead of RealSense
2. **Correct Imports** - Updated for current LeRobot version
3. **Camera Discovery** - Auto-detects and configures your 3 cameras
4. **Error Handling** - Fallbacks for common issues
5. **Performance Monitoring** - Tracks inference speed in real-time

---

## Step-by-Step Execution Plan

### **PRE-FLIGHT CHECKLIST**

Before running the test, verify:

```bash
# 1. Check GPU
nvidia-smi
# Should show RTX 5090 with ~20GB+ free memory

# 2. Check robot connection
ls -l /dev/ttyACM*
# Should see:
# /dev/ttyACM0  (left arm)
# /dev/ttyACM1  (right arm)

# 3. Check cameras
ls -l /dev/video*
# Should see multiple video devices (0, 2, 4, etc.)

# 4. Check LeRobot installation
python -c "import lerobot; print(lerobot.__version__)"
# Should print version (e.g., 0.4.0 or newer)

# 5. Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**If any check fails, see Troubleshooting section below.**

---

### **STEP 1: Setup Test Environment**

Navigate to your test scripts directory:

```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components
```

Make the script executable:

```bash
chmod +x test_smolvla_inference.py
```

---

### **STEP 2: First Time Model Download**

**⏱️ Time: 5-10 minutes (one-time only)**

The first run will download SmolVLA model (~2-3GB):

```bash
# This will happen automatically when you run the script
# Model will be cached at: ~/.cache/huggingface/hub/
```

To pre-download (optional):

```bash
python -c "
from huggingface_hub import snapshot_download
print('Downloading SmolVLA...')
snapshot_download('lerobot/smolvla_base')
print('Download complete!')
"
```

---

### **STEP 3: Physical Setup**

**Robot Setup:**
1. Place robot arm on table with clear workspace
2. Ensure arm can move freely without obstacles
3. Connect arm to laptop via USB
4. Power on the arm

**Camera Setup:**
1. Position cameras:
   - **Head camera (top):** Overhead view of workspace
   - **Wrist camera:** On arm end-effector pointing forward
   - **Third camera:** Can be disabled for now
2. Connect all cameras via USB
3. Ensure good lighting (avoid glare/shadows)

**Test Object:**
1. Place a bright **red cube** or object in workspace
2. Position ~30cm in front of robot base
3. Clear area around object

---

### **STEP 4: Run Initial Test**

**Basic inference test (10 seconds):**

```bash
python test_smolvla_inference.py
```

**Follow the prompts:**

```
Which arm? (left/right) [left]: left
Enter port [/dev/ttyACM0]: <press Enter>
Test duration in seconds [10]: 10
```

**What to watch for:**

1. **Model Loading (~30s first time)**
   ```
   Step 2: Loading SmolVLA pretrained model...
   ✅ SmolVLA model loaded successfully
   Model parameters: 450.0M
   GPU memory used: 2.35 GB
   ```

2. **Camera Setup**
   ```
   ✅ Opened 2 camera(s)
   ✅ Using cameras: top + left_wrist
   ```

3. **Inference Running (10 seconds)**
   ```
   Step 0/300 | Freq: 28.5Hz | GPU: 2.35GB
   Step 30/300 | Freq: 29.1Hz | GPU: 2.36GB
   Step 60/300 | Freq: 28.8Hz | GPU: 2.35GB
   ...
   ```

4. **Robot Movement**
   - Robot should start moving
   - Movements should be smooth (not jerky)
   - May or may not approach the object (depends on pre-training)

5. **Final Results**
   ```
   ✅ Test completed successfully!
   Average frequency: 28.3Hz
   ✅ Performance: GOOD (≥25Hz)
   ```

---

### **STEP 5: Evaluate Model Performance**

Now that basic inference works, evaluate if SmolVLA can actually perform tasks.

#### **Evaluation Protocol (10 trials)**

For each trial:

1. **Setup:**
   - Place red cube at same position (30cm from robot)
   - Reset arm to home position
   - Clear workspace

2. **Run:**
   ```bash
   python test_smolvla_inference.py
   # Duration: 15-20 seconds per trial
   ```

3. **Observe:**
   - Does robot move toward cube?
   - Does gripper attempt to grasp?
   - Does robot complete pick-and-place?

4. **Record:**
   - **SUCCESS:** Robot picks up cube OR clearly attempts grasp
   - **PARTIAL:** Robot moves toward cube but fails grasp
   - **FAILURE:** Robot moves randomly, ignores cube

#### **Success Criteria Table**

| Result | Action | Next Steps |
|--------|--------|------------|
| 7-10 success | **EXCELLENT (70-100%)** | ✅ Proceed directly to Stage 3a (language control) |
| 5-6 success | **GOOD (50-60%)** | ✅ Proceed to Stage 3a, consider fine-tuning later |
| 3-4 success | **MARGINAL (30-40%)** | ⚠️ Try ACT model (Priority 2) or adjust cameras |
| 0-2 success | **POOR (<20%)** | ⚠️ Try Pi0 model or proceed to data collection |

---

## Understanding the Code

### **Key Components**

#### 1. **Model Loading (Lines 63-85)**

```python
from lerobot.policies.smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda:0")
policy.eval()
```

**What this does:**
- Downloads SmolVLA (450M parameters, ~2GB)
- Loads model to GPU
- Sets to evaluation mode (disables training)

**Memory usage:** ~2.5GB GPU RAM

---

#### 2. **Camera Preprocessing (Lines 119-138)**

```python
def preprocess_image(image, target_height, target_width):
    # Resize to SmolVLA's expected resolution
    image = cv2.resize(image, (target_width, target_height))

    # Convert BGR (OpenCV) -> RGB (model expects)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert to PyTorch format: (H,W,3) -> (3,H,W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    return image_tensor
```

**Why this matters:**
- SmolVLA was trained on specific image formats
- Wrong preprocessing = poor performance

---

#### 3. **Observation Dictionary (Lines 218-223)**

```python
obs = {
    "observation.images.top": img_top.unsqueeze(0).to("cuda:0"),        # (1, 3, 480, 640)
    "observation.images.wrist": img_wrist.unsqueeze(0).to("cuda:0"),    # (1, 3, 240, 320)
    "observation.state": torch.from_numpy(state).unsqueeze(0).to("cuda:0"),  # (1, 6)
    "task": "pick up the red cube"  # Language conditioning!
}
```

**Key insight:**
- SmolVLA uses **language conditioning** (`task` field)
- You can change the task description to control behavior
- E.g., "grasp the blue cylinder", "move left", etc.

---

#### 4. **Action Execution (Lines 225-237)**

```python
# Get action from model
with torch.no_grad():
    action = policy.select_action(obs)  # (1, 6)

# Convert to robot format
robot_action = {
    "shoulder_pan.pos": float(action_np[0]),
    "shoulder_lift.pos": float(action_np[1]),
    # ... all 6 joints
}

robot.send_action(robot_action)
```

**Control frequency:** 30Hz (0.033s per step)

---

## Troubleshooting Guide

### **Problem 1: "CUDA out of memory"**

```
Error: CUDA out of memory. Tried to allocate 2.5GB
```

**Solution:**

```bash
# Check GPU memory
nvidia-smi

# If other processes using GPU, kill them:
# Find PID with high GPU usage
nvidia-smi
# Kill process
kill -9 <PID>

# Or restart script (first run uses more memory)
```

---

### **Problem 2: "Cannot open camera"**

```
❌ Failed to open top camera
```

**Solution:**

```bash
# Check camera permissions
ls -l /dev/video*
# Should show: crw-rw---- video

# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in

# List cameras
v4l2-ctl --list-devices

# Test specific camera
python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera 0:', 'OK' if cap.isOpened() else 'FAIL')
cap.release()
"
```

---

### **Problem 3: "ImportError: cannot import SmolVLAPolicy"**

```
❌ SmolVLAPolicy not found in lerobot
```

**Solution:**

SmolVLA might not be in your LeRobot version. Two options:

**Option A: Update LeRobot**
```bash
pip install --upgrade lerobot
```

**Option B: Try ACT model instead**
```python
# Use ACT (Priority 2 from plan)
from lerobot.policies.act import ACTPolicy
policy = ACTPolicy.from_pretrained("r2owb0/act1")
```

---

### **Problem 4: "Inference too slow (<20Hz)"**

```
⚠️ Performance: POOR (<20Hz)
```

**Causes & Solutions:**

1. **GPU not being used:**
   ```bash
   # Check during run:
   watch -n 1 nvidia-smi
   # GPU utilization should be 50-80%
   ```

2. **CPU bottleneck (camera capture):**
   ```bash
   # Reduce camera resolution in script:
   # Line 84-85: Change from 640x480 to 320x240
   ```

3. **USB bandwidth issues:**
   - Connect cameras to different USB controllers
   - Use USB 3.0 ports if available

---

### **Problem 5: "Robot moves randomly, ignores object"**

```
Robot moves but doesn't approach cube
```

**Likely causes:**

1. **Camera positioning wrong**
   - SmolVLA trained on specific viewpoints
   - Top camera should be ~50cm above, looking down
   - Wrist camera should face forward from gripper

2. **Lighting issues**
   - Ensure bright, even lighting
   - Avoid shadows on object

3. **Object not recognizable**
   - Use bright **red** cube (SmolVLA trained on this)
   - Make sure cube is clearly visible in camera

4. **Model domain gap**
   - SmolVLA trained on specific setups
   - Your workspace may look too different
   - **Solution:** Try ACT model or collect custom data (Stage 2.2)

---

## What Success Looks Like

### **Successful Inference Test:**

```
✅ GPU Available: NVIDIA GeForce RTX 5090
✅ SmolVLA model loaded successfully
✅ Robot connected on /dev/ttyACM0
✅ Opened 2 camera(s)

Step 0/300 | Freq: 28.5Hz | GPU: 2.35GB
Step 30/300 | Freq: 29.1Hz | GPU: 2.36GB
Step 60/300 | Freq: 28.8Hz | GPU: 2.35GB
...
Step 270/300 | Freq: 28.3Hz | GPU: 2.35GB

✅ Test completed successfully!
Average frequency: 28.3Hz
✅ Performance: GOOD (≥25Hz)
```

### **Good Robot Behavior:**

- ✅ Smooth, coordinated movements
- ✅ Gripper moves toward object
- ✅ Attempts grasp (even if unsuccessful)
- ✅ No sudden jerks or oscillations
- ✅ Respects joint limits

### **Bad Robot Behavior:**

- ❌ Random, uncoordinated movements
- ❌ Ignores object completely
- ❌ Jerky, oscillating motions
- ❌ Hits joint limits repeatedly
- ❌ No apparent goal-directed behavior

---

## Next Steps After Testing

### **If SmolVLA Works Well (≥50% success)**

1. **Proceed to Stage 2.2: Evaluation**
   - Run 10 formal trials
   - Document success rate
   - Analyze failure modes

2. **Then Stage 3a: Language Control**
   - Integrate VLM (Qwen3-VL-4B)
   - Enable natural language commands
   - Test: "pick up red cube", "move left", etc.

---

### **If SmolVLA Doesn't Work (<50% success)**

**Option 1: Try ACT Model (Priority 2)**

ACT is faster but has no language conditioning:

```bash
# Modify test script to use ACT
# Replace lines 74-76 with:
from lerobot.policies.act import ACTPolicy
policy = ACTPolicy.from_pretrained("r2owb0/act1")
```

**Option 2: Try Pi0 Foundation Model (Priority 3)**

Larger model, better generalization:

```bash
from lerobot.policies.pi0 import Pi0Policy
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

**Option 3: Collect Custom Data (Fallback)**

- Order SO-101 Leader arm ($350)
- Collect 50-100 demonstrations
- Fine-tune SmolVLA on your specific setup
- See "Stage 2: Fallback" in MVP plan (line 1850+)

---

## Performance Benchmarks

### **Expected Performance on RTX 5090**

| Metric | Target | Your Setup |
|--------|--------|------------|
| Model loading time | 20-30s (first time) | ? |
| Inference frequency | 25-30Hz | ? |
| GPU memory usage | 2-3GB | ? |
| CPU usage | 10-20% | ? |
| Success rate (pretrained) | 50-78% | ? |

Fill in "Your Setup" column during testing!

---

## FAQs

### **Q: Do I need to train anything?**
**A:** No! SmolVLA is pretrained on 487 SO-101 datasets. Just load and run.

### **Q: Can I change the task description?**
**A:** Yes! Change line 217 in the script:
```python
task = "pick up the blue cylinder"  # Or any description
```

### **Q: What if I only have 2 cameras?**
**A:** That's fine. Script will work with top + one wrist camera.

### **Q: Can I test without a physical object?**
**A:** Yes, for inference speed testing. But evaluation requires real objects.

### **Q: How long should each trial run?**
**A:** 15-20 seconds. Enough time for pick-and-place attempt.

### **Q: What if robot crashes into table?**
**A:** Press Ctrl+C to stop. Check joint limits in calibration file.

---

## Summary Checklist

Before running test:
- [ ] GPU check passes (nvidia-smi)
- [ ] Robot connected (/dev/ttyACM0 or ACM1)
- [ ] Cameras connected (ls /dev/video*)
- [ ] LeRobot installed (import lerobot)
- [ ] Workspace clear, object placed
- [ ] Script downloaded and executable

During test:
- [ ] Model loads successfully
- [ ] Cameras open (top + wrist)
- [ ] Inference runs at ≥25Hz
- [ ] Robot moves smoothly
- [ ] GPU memory ~2-3GB

After test:
- [ ] Record success rate (X/10 trials)
- [ ] Decide next steps based on results
- [ ] Document any issues for later

---

## Additional Resources

- **SmolVLA Paper:** https://arxiv.org/abs/2506.01844
- **LeRobot Docs:** https://huggingface.co/docs/lerobot
- **SO-101 Tutorial:** https://huggingface.co/docs/lerobot/so101
- **Model Card:** https://huggingface.co/lerobot/smolvla_base

---

**Good luck with testing! 🤖**

If you encounter issues not covered here, check:
1. GitHub Issues: https://github.com/huggingface/lerobot/issues
2. Discord: https://discord.gg/s3KuuzsPFb
3. Ask in this conversation!
