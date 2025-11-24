# Comprehensive Data Collection Guide for XLeRobot

**Project**: VLM→VLA Fine-tuning with Pi0.5
**Date**: 2025-11-16
**Last Updated**: 2025-11-19 (Added MVP validation, research-backed warnings, and common mistakes)
**Status**: Ready for Week 1 Data Collection

---

## Table of Contents
1. [Overview](#overview) - **Start here for context**
2. ⭐ **[MVP: Quick Validation](#-mvp-quick-validation-do-this-first)** - **DO THIS FIRST! (2-3 hours)**
   - [Why MVP First?](#why-mvp-first)
   - [MVP Steps 1-4](#mvp-step-1-minimal-setup-30-minutes)
   - [MVP Decision: Pass or Fail?](#mvp-decision-pass-or-fail)
3. **NEW:** [Common Data Collection Mistakes](#common-data-collection-mistakes-research-backed) ⚠️ **Avoid These!**
4. [Pre-Collection Setup (Day 1)](#pre-collection-setup-day-1)
5. [Define Primitive Vocabulary](#define-primitive-vocabulary)
   - **UPDATED:** [Option A: Minimal Set (50 episodes)](#option-a-minimal-validation-set-50-episodes-7-8-hours-collection)
   - **UPDATED:** [Option B: Full Set (75-100 episodes)](#option-b-standard-full-set-75-100-episodes-10-15-hours-collection)
6. [Recording Episodes (Days 2-5)](#recording-episodes-days-2-5)
7. [Demonstration Best Practices](#demonstration-best-practices)
8. [Quality Control](#quality-control)
9. [Diversity Guidelines](#diversity-guidelines)
10. [Daily Collection Schedule](#daily-collection-schedule)
11. [Troubleshooting](#troubleshooting)
12. [Final Checklist](#final-checklist)

**🎯 Quick Navigation:**
- **Never done this before?** → Section 2 (MVP) + Section 3 (Common Mistakes)
- **MVP passed, ready for full collection?** → Section 5 (Primitive Vocabulary)
- **Having issues?** → Section 11 (Troubleshooting)
- **Want detailed validation strategy?** → See [MINIMAL_VALIDATION_STRATEGY.md](MINIMAL_VALIDATION_STRATEGY.md)

---

## Overview

**Goal**: Collect 50-100 high-quality episodes demonstrating primitive actions
**Time**: 5-7 days (20-30 hours total) + **2-3 hours MVP validation FIRST**
**Critical Success Factor**: **Validate pipeline with MVP BEFORE full collection**
**Action Frequency**: **5Hz** (CRITICAL - not 30Hz!)

**⚠️ RESEARCH-BACKED WARNING**:
- Egocentric cameras (your setup) are HARDER than fixed cameras (standard LeRobot)
- Expected success rates: 15-25% lower initially due to embodiment gap
- LeRobot official guidance recommends fixed external cameras
- Your advantage: Pi0.5's 400h mobile manipulation training data

**Why This Matters:**
- Quality data → Better model performance
- Consistent demonstrations → Faster learning
- Diverse scenarios → Better generalization

---

## 🚀 MVP: Quick Validation (DO THIS FIRST!)

**⚠️ CRITICAL: Do this BEFORE collecting all 50-100 episodes!**

**Research-Backed Practice**: OpenVLA, LeRobot, and robotics community all recommend minimal validation FIRST.

> "Best practices recommend first recording just a handful of episodes to confirm data was saved correctly, then running a full small-scale cycle with a small dataset (e.g., 10 episodes)" - LeRobot Community

> OpenVLA docs: "Two key verification steps are recommended: (1) replay actions from a demonstration from your fine-tuning dataset... (2) once you've fine-tuned a model, load it in your inference pipeline and feed images from the fine-tuning dataset to verify you can reproduce the token accuracies"

### Why MVP First?

**Problem**: Collecting 50-100 episodes takes 20-30 hours. What if there's an issue with:
- Data format/compatibility
- Camera setup
- Action frequency (5Hz)
- Fine-tuning pipeline crashes
- Configuration errors

**Solution**: Collect 5-10 episodes, test fine-tuning, validate **technical pipeline** works!

**⚠️ REALISTIC EXPECTATIONS:**
- ✅ **WILL catch**: Technical errors, format issues, config problems
- ❌ **WON'T show**: Actual learning or performance improvement
- ❌ **WON'T validate**: Data quality (need 50+ episodes for that)

**Why this is still valuable:**
- Catch bugs EARLY before investing 20-30 hours
- Verify pipeline runs end-to-end without crashes
- Confirm VRAM usage is acceptable
- Test data format compatibility

**Time Investment:**
- MVP: 2-3 hours (5-10 episodes + quick pipeline test)
- Full collection: 20-30 hours

**Risk Reduction**: Catch TECHNICAL problems early, save 20+ hours of wasted collection!

---

### MVP Step 0: Test Pretrained Pi0.5 Baseline FIRST (30 minutes) ⭐

**⚠️ CRITICAL: Do this BEFORE collecting any data!**

**Why:**
- You need a baseline to compare against
- Without it, you can't tell if finetuning worked!
- 40% after finetuning = excellent if pretrained got 5%
- 40% after finetuning = poor if pretrained got 35%

**How to test pretrained baseline:**

```python
from lerobot.common.policies.pi05.modeling_pi05 import Pi05ForActionPrediction

# Load pretrained Pi0.5 (before any finetuning)
model = Pi05ForActionPrediction.from_pretrained("lerobot/pi05_base")
model.eval().cuda()

# Test on your robot with pick_center task
# Try 5-10 attempts, record success rate
# Expected: 0-15% success (egocentric cameras are out-of-distribution)
```

**Record this baseline!** You'll compare after finetuning.

---

### MVP Step 1: Minimal Setup (30 minutes)

#### Quick Hardware Check

```bash
cd /home/jrobot/project/XLeRobot
conda activate lerobot

# Test cameras (10 seconds)
python -c "import cv2; print('Cameras:', [cv2.VideoCapture(i).isOpened() for i in [0,2,4]])"

# Test motors (10 seconds)
python -c "from lerobot.motors.dynamixel import DynamixelMotorsBus; m = DynamixelMotorsBus(port='/dev/ttyUSB0'); print('Motors: OK')"
```

**If both pass → Continue to recording**
**If either fails → See [Pre-Collection Setup](#pre-collection-setup-day-1) for detailed troubleshooting**

#### Quick Robot Config

```bash
# Create minimal robot config
mkdir -p ~/.cache/lerobot/configs

cat > ~/.cache/lerobot/configs/xlerobot_mvp.yaml << 'EOF'
robot_type: so101

cameras:
  left_wrist:
    index: 6
    fps: 30
    width: 640
    height: 480
  right_wrist:
    index: 8
    fps: 30
    width: 640
    height: 480
  head:
    index: 4
    fps: 30
    width: 640
    height: 480

# Dual-arm leader-follower setup
# NOTE: Verify which port matches which arm by testing!
motors:
  follower_left:
    port: /dev/ttyACM2
    baudrate: 1000000
  follower_right:
    port: /dev/ttyACM3
    baudrate: 1000000
  leader_left:
    port: /dev/ttyACM0
    baudrate: 1000000
  leader_right:
    port: /dev/ttyACM1
    baudrate: 1000000

# CRITICAL: 5Hz action frequency
policy_fps: 5
EOF

echo "✅ MVP config created"
```

#### Workspace Setup (5 minutes)

**You need:**
- ✅ 1 object (red cube)
- ✅ Clear table space (30cm x 30cm minimum)
- ✅ Consistent lighting

**That's it! No elaborate setup for MVP.**

---

### MVP Step 2: Collect 7-10 Episodes (1 hour)

**Choose ONE primitive for MVP: `pick_center`**

**Updated: Collect 7-10 episodes minimum (not 5)**

Why 7-10?
- Need proper train/val split (80/20 → 6 train, 1-2 val if 7-8 episodes)
- 5 episodes too few for meaningful validation split
- 10 episodes provides better training stability

Why pick_center?
- ✅ Most important primitive
- ✅ Tests full pipeline (approach → grasp → lift)
- ✅ If pick works, others will too
- ✅ ONE location keeps it simple for MVP

#### Option A: Manual Recording (Recommended for First-Time)

```bash
# Create MVP collection script
cat > ~/project/XLeRobot/scripts/mvp_collect.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""MVP Data Collection - Single primitive test"""

from lerobot.scripts.control_robot import record_episode

def collect_mvp():
    print("="*70)
    print("🚀 MVP DATA COLLECTION")
    print("="*70)
    print("Task: pick red_cube from center")
    print("Episodes: 5-10")
    print("Duration: 30s per episode")
    print("="*70)

    for ep in range(1, 11):  # Up to 10 episodes
        print(f"\n[Episode {ep}/10]")
        print("✋ Position: Red cube at table center")
        print("✋ Leader arms: Ready position")
        input("⏸️  Press ENTER to record (or Ctrl+C to stop)...")

        try:
            print(f"🎬 Recording episode {ep}...")
            record_episode(
                robot_config="~/.cache/lerobot/configs/xlerobot_mvp.yaml",
                dataset_repo_id="lerobot/xlerobot_mvp_test",
                task="pick red_cube from center",
                fps=5,  # CRITICAL!
                episode_time_s=30,
                warmup_time_s=5,
                episode_index=None,
            )
            print(f"✅ Episode {ep} saved!")

        except KeyboardInterrupt:
            print(f"\n⏹️  Stopped at {ep-1} episodes")
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            retry = input("Retry? (y/n): ")
            if retry.lower() != 'y':
                break

    print("\n" + "="*70)
    print("✅ MVP COLLECTION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    collect_mvp()
ENDOFPYTHON

chmod +x ~/project/XLeRobot/scripts/mvp_collect.py

# Run MVP collection
python ~/project/XLeRobot/scripts/mvp_collect.py
```

#### Recording Tips for MVP

**Demonstration technique:**

```
Phase 1 (0-5s): Start position
- Gripper open
- ~15cm above table
- ~10cm from cube

Phase 2 (5-12s): Approach
- Move slowly toward cube
- Center gripper over cube
- Lower to cube (3-5 seconds)

Phase 3 (12-18s): Grasp
- Close gripper (2 seconds)
- Verify object secure

Phase 4 (18-24s): Lift
- Raise 10cm (3 seconds)
- Hold stable position

Phase 5 (24-30s): Hold
- Maintain position
- Recording ends
```

**Quality for MVP:**
- ✅ Slow and smooth motion
- ✅ Task completes SUCCESSFULLY every time
- ✅ All 3 cameras can see object throughout
- ✅ CONSISTENT strategy (same approach every time)
- ❌ NO failures, NO retries (save for later stages)
- ❌ NO variety in approach angle (keep it identical)

**Target: 7-10 SUCCESSFUL episodes with consistent strategy**

**Why consistency matters even for MVP:**
- Model needs pattern to potentially learn from
- Varied approaches = confusing signal
- Choose ONE strategy (e.g., always from above), repeat it

#### Verify Data

```bash
# Check episodes were saved
python << 'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset("lerobot/xlerobot_mvp_test")
    print(f"✅ Episodes collected: {dataset.num_episodes}")
    print(f"✅ Action frequency: {dataset.meta.fps} Hz")
    print(f"✅ Cameras: {dataset.meta.camera_keys}")

    if dataset.num_episodes >= 7:
        print("\n🎉 READY FOR MVP FINE-TUNING TEST!")
    else:
        print(f"\n⚠️  Need {7 - dataset.num_episodes} more episodes")
except Exception as e:
    print(f"❌ Dataset error: {e}")
    print("Check that episodes were saved correctly")
EOF
```

---

### MVP Step 3: Quick Fine-Tune Test (1 hour)

**Goal**: Verify the fine-tuning pipeline works with your data

#### Create MVP Training Config

```bash
cat > ~/project/XLeRobot/config/train_pi05_mvp.yaml << 'EOF'
# MVP Training Configuration - VERY SHORT
policy:
  type: pi05
  pretrained_path: lerobot/pi05_base
  use_lora: true
  lora_rank: 32
  lora_alpha: 64

dataset:
  repo_id: lerobot/xlerobot_mvp_test

training:
  steps: 100  # Only 100 steps for MVP test!
  batch_size: 4  # Small batch
  lr: 5e-6
  mixed_precision: bf16
  eval_freq: 50
  save_freq: 50

output_directory: outputs/pi05_mvp_test
device: cuda
EOF
```

#### Run MVP Fine-Tuning

```bash
# Start MVP training (should take ~10-15 minutes)
python src/lerobot/scripts/lerobot_train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --dataset.repo_id=lerobot/xlerobot_mvp_test \
  --steps=100 \
  --batch_size=4 \
  --output_directory=outputs/pi05_mvp_test

# Monitor progress
# You should see:
# - Loading Pi0.5 model...
# - Loading dataset...
# - Training step 1/100
# - Training step 50/100
# - Training step 100/100
# - Saving checkpoint...
```

**Expected behavior:**

✅ **TECHNICAL SUCCESS if (no crashes):**
```
Loading Pi0.5 from cache... ✓
Loading dataset lerobot/xlerobot_mvp_test... ✓
Dataset: 5 episodes, XXX frames
Training step 1/100... Loss: X.XXX
Training step 50/100... Loss: X.XXX
Training step 100/100... Loss: X.XXX
Saved checkpoint to outputs/pi05_mvp_test/checkpoint-100
```

**⚠️ CRITICAL EXPECTATION MANAGEMENT:**
- **Loss values DON'T MATTER** with 5-10 episodes!
- Loss might decrease (overfitting to tiny dataset) - IGNORE THIS
- Loss might stay flat - THAT'S NORMAL
- Loss might even increase slightly - ALSO NORMAL
- **This does NOT validate that the model will learn** - you need 50+ episodes for that
- **Purpose**: Catch technical crashes/errors ONLY, not evaluate learning quality

**What "Success" Means for MVP:**
✅ No crashes = SUCCESS (that's all you need!)
❌ Crashes/NaN/OOM = FAILURE (debug before continuing)

❌ **TECHNICAL FAILURE if (crashes/errors):**
```
Error: Camera key mismatch
Error: Action dimension mismatch
Error: FPS mismatch (expected 5, got 30)
Error: CUDA out of memory
```

---

### MVP Step 4: Test Inference (5 minutes)

```bash
# Test that checkpoint loads and runs inference
# (Same as in MINIMAL_VALIDATION_STRATEGY.md)
```

---

### MVP Step 5: Compare vs Pretrained Baseline (10 minutes) ⭐

**Now compare your mini-finetuned model vs the baseline you tested in Step 0:**

```python
# You tested pretrained in Step 0, recorded: X% success
# Now test finetuned model (from Step 3):
# - Load checkpoint: outputs/pi05_mvp_test/checkpoint-100
# - Try 5-10 pick_center attempts
# - Record: Y% success

# Expected with only 7-10 episodes:
# - Finetuned: 0-20% success (not enough data!)
# - This is NORMAL - MVP just tests pipeline
```

**What you're checking:**
- ✅ Model responds to finetuning (not identical to pretrained)
- ✅ Pipeline works end-to-end
- ❌ NOT expecting actual learning with 7-10 episodes!

---

### MVP Step 6: Validate Technical Success (5 minutes)

**Goal**: Verify training completed without crashes (NOT evaluating performance)

#### Check Training Completed

```bash
# Check checkpoint exists
ls -lh outputs/pi05_mvp_test/

# Should see:
# checkpoint-100/
# config.yaml
# training.log
```

#### Check for Errors

```bash
# Look for errors in log
tail -50 outputs/pi05_mvp_test/training.log | grep -i error

# If no output = good! No errors found

# Check training completed
tail -20 outputs/pi05_mvp_test/training.log

# Look for:
# - "Training step 100/100" (completed)
# - "Saved checkpoint" (saved successfully)
# - No error messages
```

**⚠️ Don't worry about loss values!**
- Loss might decrease, stay flat, or even increase
- With 5-10 episodes, loss is meaningless
- You're just checking the pipeline runs without crashing

#### Test Inference (Optional but Recommended)

```bash
# Load fine-tuned model
python << 'EOF'
from lerobot.common.policies.pi05.modeling_pi05 import Pi05ForActionPrediction
import torch

print("Loading MVP fine-tuned model...")
model = Pi05ForActionPrediction.from_pretrained("outputs/pi05_mvp_test/checkpoint-100")
model.eval().cuda()
print("✅ Model loaded successfully!")

# Test forward pass
dummy_images = torch.randn(1, 3, 3, 480, 640).cuda()  # Batch, cameras, channels, H, W
dummy_state = torch.randn(1, 12).cuda()  # Batch, joints
dummy_task = "pick red_cube from center"

try:
    with torch.no_grad():
        output = model(images=dummy_images, state=dummy_state, task=dummy_task)
    print(f"✅ Inference works! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Inference failed: {e}")
EOF
```

---

### MVP Decision: Pass or Fail?

#### ✅ MVP PASSED (Technical Validation) if:

1. **Data Collection:**
   - ✅ 5-10 episodes saved successfully
   - ✅ Action frequency is 5Hz
   - ✅ All 3 cameras captured footage
   - ✅ Dataset loads without errors

2. **Fine-Tuning:**
   - ✅ Training started without errors
   - ✅ Training completed (100 steps)
   - ✅ Checkpoint saved
   - ✅ No crashes or memory errors

3. **Inference:**
   - ✅ Model loads from checkpoint
   - ✅ Forward pass works without errors

**→ PROCEED to full data collection (75-100 episodes)!**

**What you've validated:**
- ✅ Technical pipeline works end-to-end
- ✅ Data format is compatible with Pi0.5
- ✅ Configuration is correct (fps, cameras, actions)
- ✅ VRAM usage is acceptable

**What you HAVEN'T validated yet:**
- ❌ Demonstration quality (need more data)
- ❌ Model performance (need real training)
- ❌ Generalization ability (need diversity)

---

#### ❌ MVP FAILED - Troubleshoot Before Full Collection

**Common Issues:**

**Issue 1: FPS Mismatch**
```
Error: Expected 5Hz, got 30Hz
```
**Fix:**
```bash
# Check robot config
cat ~/.cache/lerobot/configs/xlerobot_mvp.yaml | grep policy_fps
# Should show: policy_fps: 5

# If wrong, fix it:
sed -i 's/policy_fps: 30/policy_fps: 5/' ~/.cache/lerobot/configs/xlerobot_mvp.yaml

# Re-collect all episodes
```

**Issue 2: Camera Key Mismatch**
```
Error: Expected camera keys [...], got [...]
```
**Fix:**
- Check camera names in config match dataset
- Ensure all 3 cameras (left_wrist, right_wrist, head) present

**Issue 3: CUDA Out of Memory**
```
Error: CUDA out of memory
```
**Fix:**
```bash
# Reduce batch size
--batch_size=2  # Instead of 4

# Or use gradient checkpointing
--gradient_checkpointing=true
```

**Issue 4: Data Loading Fails**
```
Error: Dataset not found
```
**Fix:**
- Check dataset was saved: `ls -la lerobot_data/` or check HuggingFace
- Verify repo_id matches between collection and training

---

### MVP Success: Next Steps

**If MVP passed:**

1. ✅ **Delete MVP data** (optional):
   ```bash
   # Clean up test data
   rm -rf lerobot_data/xlerobot_mvp_test/
   # Or keep for reference
   ```

2. ✅ **Proceed to full collection**:
   - Follow [Pre-Collection Setup](#pre-collection-setup-day-1)
   - Collect 75-100 episodes over 5-7 days
   - Use lessons learned from MVP

3. ✅ **What you've TECHNICALLY validated**:
   - Data collection pipeline works ✓
   - Data format is correct ✓
   - Fine-tuning pipeline runs without crashing ✓
   - Configuration is correct (fps, cameras, actions) ✓
   - VRAM usage is acceptable ✓

**Confidence boost**: Technical pipeline works end-to-end! 🎉

**Important**: This does NOT validate demonstration quality or model performance. You need the full 75-100 episodes + real training (6000 steps) to see actual learning.

---

### MVP Lessons Learned

**After MVP, you should know:**

1. **Technical setup works**: Pipeline runs without crashes
2. **VRAM usage**: How much GPU memory is used (~18-22GB expected)
3. **Training speed**: ~6-10 minutes per 100 steps (→ ~6-10 hours for 6000 steps)
4. **Hardware reliability**: Cameras/motors work consistently
5. **Data collection workflow**: Comfortable with recording process

**What you DON'T know yet (and that's OK!):**
- ❌ If demonstrations are good quality (need 50+ episodes)
- ❌ If model will actually learn (need real training)
- ❌ Final performance (need full dataset + evaluation)

**The MVP's purpose**: Catch technical bugs EARLY, not evaluate quality

**Apply these lessons to full collection!**

---

## Common Data Collection Mistakes (Research-Backed)

**From LeRobot documentation, research papers, and community reports:**

### ❌ Mistake 1: Using Wrong Action Frequency

**Problem**: Recording at 30Hz camera FPS instead of 5Hz action frequency
**Impact**: Dataset incompatible with training, must re-collect everything
**How it happens**: Confusing camera FPS (30Hz) with action frequency (5Hz)

**Fix:**
```yaml
# In robot config
policy_fps: 5  # Action frequency (CRITICAL!)
cameras:
  left_wrist:
    fps: 30  # Camera FPS (can be 30Hz)
```

**Validation**: Check dataset after FIRST episode!
```python
dataset = LeRobotDataset("your_dataset")
assert dataset.meta.fps == 5, f"Wrong FPS: {dataset.meta.fps}"
```

### ❌ Mistake 2: Object Not Visible in All Camera Views

**LeRobot Official Guidance**:
> "Ensure the object you are manipulating is visible on the camera... you should be able to do the task yourself by only looking at the camera images."

**Problem**: Object goes out of frame during motion
**Impact**: Model can't learn from what it can't see
**Common with egocentric cameras**: Views change as arm moves

**Fix:**
- Test camera views BEFORE collecting
- Record one episode, review ALL camera angles
- Adjust camera positions if object obscured
- Move SLOWER so cameras can track objects

### ❌ Mistake 3: Too Much Variation Too Quickly

**LeRobot Official Warning**:
> "Avoid adding too much variation too quickly, as it may hinder your results."

**Problem**: Varying objects, positions, lighting, clutter all at once
**Impact**: Model confused, can't learn core skill
**Common trap**: "More variation = better generalization" (FALSE for first dataset!)

**Fix:**
- Start simple: ONE object type, FEW positions, NO clutter
- Add variation GRADUALLY after basic skill works
- Week 1: Simple pick/place with cube
- Week 3+: Add variety if model shows competence

### ❌ Mistake 4: Demonstrations Too Fast

**Community Consensus**: #1 reported mistake in imitation learning

**Problem**: Moving at natural human speed (too fast for robot)
**Impact**: Model learns jerky, unreliable policies
**Egocentric camera issue**: Fast motion creates blur, missing frames

**Fix:**
- Target: 15-30 seconds per episode (not 5-10 seconds!)
- Move 2-3x SLOWER than feels natural
- Pretend moving underwater
- Practice motion 3 times before recording

**Test**: Can you see object clearly in ALL frames? If blurry → too fast!

### ❌ Mistake 5: Inconsistent Strategy Within Primitive

**Research Finding** (Pi0 paper):
> Models learn patterns from consistent demonstrations

**Problem**:
- Episode 1: Approach from above
- Episode 2: Approach from side
- Episode 3: Approach from angle

**Impact**: Model sees 3 different "pick" strategies, learns none well

**Fix:**
- Choose ONE approach strategy
- Practice it 3-5 times
- Use SAME strategy for all episodes of that primitive
- Mark strategy in notes for consistency

### ❌ Mistake 6: Egocentric Camera Challenges (SPECIFIC TO YOUR SETUP)

**Research Warning** (EgoMI, EMMA papers):
> "Dynamic egocentric views create distribution shifts that static robot systems cannot replicate, leading to degraded policy performance."

**Problems specific to egocentric (moving) cameras:**
- Viewpoint changes continuously
- Scale/zoom changes as arm moves
- Background changes frame-to-frame
- Model must learn viewpoint-invariant features (HARDER!)

**Your Setup's Extra Challenges:**
- Wrist cameras move WITH arms (not static)
- Views change every frame (unlike fixed external cameras)
- Standard LeRobot uses fixed cameras (easier learning)

**Mitigations:**
1. **Move VERY SLOWLY** - reduce visual motion blur
2. **Keep head camera still** if possible (one stable view)
3. **Ensure object ALWAYS visible** in at least one camera
4. **High quality demos** - consistency even MORE critical
5. **Consider hybrid**: Add 1-2 fixed external cameras if <40% success

### ❌ Mistake 7: Not Testing Replay Before Scaling

**LeRobot Recommendation**:
> "Test the repeatability of your robot's actions through the replay function before scaling up."

**Problem**: Collect 100 episodes, then discover replay doesn't work
**Impact**: Can't validate if robot can reproduce recorded actions

**Fix:**
```bash
# After FIRST episode, test replay
python scripts/test_replay.py --episode 0

# If replay fails → fix robot/motors BEFORE collecting more
# If replay works → proceed confidently
```

### ❌ Mistake 8: Ignoring Environmental Consistency

**LeRobot Guidance**:
> "Keep cameras fixed and maintain consistent conditions throughout recordings."

**Problem**:
- Lighting changes (morning vs afternoon sun)
- Background objects moved
- Table surface changed
- Different time of day

**Impact**: Model distracted by irrelevant changes, harder learning

**Fix:**
- Record all data SAME time of day
- Close blinds (use artificial lighting)
- Lock down workspace (don't move anything)
- Take "before" photo, ensure "after" matches

---

## Pre-Collection Setup (Day 1)

### Morning: Hardware Verification (2 hours)

#### Step 1: Test Cameras

```bash
# Navigate to project
cd /home/jrobot/project/XLeRobot
conda activate lerobot

# Verify all 3 cameras work
python -c "
import cv2
cameras = [0, 2, 4]  # left_wrist, right_wrist, head
for i in cameras:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f'Camera {i}: ✓ Working ({frame.shape[1]}x{frame.shape[0]})')
        cap.release()
    else:
        print(f'Camera {i}: ✗ NOT WORKING - CHECK CONNECTION')
"
```

**Expected Output:**
```
Camera 0: ✓ Working (640x480)
Camera 2: ✓ Working (640x480)
Camera 4: ✓ Working (640x480)
```

#### Step 2: Test Motors

```bash
# Verify Dynamixel motors
python -c "
from lerobot.motors.dynamixel import DynamixelMotorsBus
try:
    motors = DynamixelMotorsBus(port='/dev/ttyUSB0')  # Adjust port if needed
    positions = motors.read('Present_Position')
    print(f'Motors: ✓ Working ({len(positions)} motors detected)')
    print(f'Current positions: {positions}')
except Exception as e:
    print(f'Motors: ✗ ERROR - {e}')
"
```

**Expected Output:**
```
Motors: ✓ Working (12 motors detected)
Current positions: [0.0, 45.0, -30.0, 15.0, 0.0, 0.0, 0.0, 45.0, -30.0, 15.0, 0.0, 0.0]
```

**If motors fail:**
```bash
# Check USB permissions
ls -l /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0  # If needed

# List available ports
python -m serial.tools.list_ports
```

#### Step 3: Physical Workspace Setup

**Required Materials:**
- ✅ Clear table surface (60cm x 60cm minimum)
- ✅ **Cubes**: 3-5 colored cubes (red, blue, green) - ~5cm size
- ✅ **Targets**: 2-3 target zones (boxes, circles marked on table)
- ✅ Good lighting (consistent, not too bright/dark)
- ✅ Fixed camera positions

**Lighting Tips:**
- Use overhead lighting or natural daylight
- Avoid direct sunlight (creates harsh shadows)
- Keep consistent throughout all sessions
- Test: Can you clearly see object colors in camera feeds?

**Camera Position Verification:**

```bash
# Take test snapshots
python << 'EOF'
import cv2

cameras = {0: 'left_wrist', 2: 'right_wrist', 4: 'head'}
for cam_id, cam_name in cameras.items():
    cap = cv2.VideoCapture(cam_id)
    ret, frame = cap.read()
    cv2.imwrite(f'test_{cam_name}.jpg', frame)
    print(f'Saved test_{cam_name}.jpg')
    cap.release()
print('\nCheck images to verify workspace is fully visible!')
EOF

# View the test images
ls -lh test_*.jpg
```

**What to verify in images:**
- ✅ **Left wrist cam**: Can see gripper + objects within 20cm
- ✅ **Right wrist cam**: Can see gripper + objects within 20cm
- ✅ **Head cam**: Can see entire 60x60cm workspace (bird's eye view)
- ✅ **Focus**: Images are clear, not blurry
- ✅ **Brightness**: Can distinguish object colors clearly

---

## Define Primitive Vocabulary

### Recommended Starter Set - Research-Backed Minimalist Approach

**Updated based on Pi0.5 research (≥15min data minimum) and minimalist validation strategy:**

#### **Option A: Minimal Validation Set (50 episodes, ~7-8 hours collection)**
*Best for first-time finetuning - verify pipeline works before scaling*

| Primitive | Template | Episodes | Duration | Description |
|-----------|----------|----------|----------|-------------|
| **pick_center** | `pick red_cube from center` | 7 | 30s | Grasp cube from center |
| **pick_left** | `pick blue_cube from left` | 7 | 30s | Grasp cube from left area |
| **pick_right** | `pick red_cube from right` | 6 | 30s | Grasp cube from right area |
| **place_box** | `place red_cube at box` | 5 | 30s | Release cube into box |
| **place_left** | `place blue_cube at left_target` | 5 | 30s | Release cube at left target |
| **place_right** | `place red_cube at right_target` | 5 | 30s | Release cube at right target |
| **grasp** | `grasp red_cube` | 8 | 25s | Close gripper on cube |
| **release** | `release object` | 7 | 15s | Open gripper |

**Total: 50 episodes** (~25min of demonstrations = meets Pi0.5 minimum 15min requirement)

**Why this set:**
- ✅ Meets Pi0.5 official minimum (≥15 minutes of data)
- ✅ Spatial diversity (3 positions for pick/place)
- ✅ Enables simple multi-step: "pick then place"
- ✅ Fast to collect (can iterate if needed)
- ✅ Tests if egocentric cameras work at all
- ❌ Omits push/reach (add later if 50-episode model works)

#### **Option B: Standard Full Set (75-100 episodes, ~10-15 hours collection)**
*Recommended after Option A works and you verify pipeline*

| Primitive | Template | Episodes | Duration | Description |
|-----------|----------|----------|----------|-------------|
| **pick_center** | `pick red_cube from center` | 10 | 30s | Grasp cube from center |
| **pick_left** | `pick blue_cube from left` | 10 | 30s | Grasp cube from left area |
| **pick_right** | `pick red_cube from right` | 10 | 30s | Grasp cube from right area |
| **place_box** | `place red_cube at box` | 10 | 30s | Release cube into box |
| **place_left** | `place blue_cube at left_target` | 10 | 30s | Release cube at left target |
| **place_right** | `place red_cube at right_target` | 10 | 30s | Release cube at right target |
| **push** | `push red_cube to target` | 10 | 40s | Slide cube to target |
| **reach** | `reach table_center` | 10 | 20s | Move gripper to location |
| **grasp** | `grasp red_cube` | 10 | 25s | Close gripper on cube |
| **release** | `release object` | 8 | 15s | Open gripper |

**Total: 88 episodes** (~40min of demonstrations = 2.6x Pi0.5 minimum)

### Why These Primitives?

1. **Pick + Place** (60 episodes): Core manipulation skills
   - Most important for general manipulation
   - VLM will compose these for complex tasks

2. **Push** (10 episodes): Non-prehensile manipulation
   - Useful for objects that can't be grasped
   - Adds diversity to action repertoire

3. **Reach + Grasp + Release** (28 episodes): Sub-components
   - Help model learn intermediate states
   - Improve pick/place success rate

4. **Position Variants**: Build generalization
   - Same action, different locations
   - Model learns location-invariant skills

### Create Your Collection Plan

```bash
cat > ~/project/XLeRobot/collection_plan.txt << 'EOF'
WEEK 1 DATA COLLECTION PLAN
============================

DAY 2 (Pick primitives - 30 episodes, ~3-4 hours):
Morning:
  - pick_center: pick red_cube from center (10x)
  - pick_left: pick blue_cube from left (10x)
Afternoon:
  - pick_right: pick red_cube from right (10x)

DAY 3 (Place primitives - 30 episodes, ~3-4 hours):
Morning:
  - place_box: place red_cube at box (10x)
  - place_left: place blue_cube at left_target (10x)
Afternoon:
  - place_right: place red_cube at right_target (10x)

DAY 4 (Push + Reach - 20 episodes, ~2-3 hours):
Morning:
  - push: push red_cube to target (10x)
Afternoon:
  - reach: reach table_center (10x)

DAY 5 (Grasp + Release - 18 episodes, ~2 hours):
Morning:
  - grasp: grasp red_cube (10x)
  - release: release object (8x)

DAY 6-7: Buffer for quality review and re-recording
  - Review all episodes
  - Re-record any episodes with quality issues
  - Aim for 75-100 GOOD episodes total

QUALITY TARGETS:
- Acceptance rate: 75-85%
- If 100 recorded → 75-85 good episodes
- Re-record immediately if issues detected
EOF

cat ~/project/XLeRobot/collection_plan.txt
```

---

## Recording Episodes (Days 2-5)

### Step 1: Configure Robot for LeRobot

```bash
# Create configuration directory
mkdir -p ~/.cache/lerobot/configs

# Create robot configuration file
cat > ~/.cache/lerobot/configs/xlerobot_egocentric.yaml << 'EOF'
# XLeRobot SO-101 Configuration with Egocentric Cameras
robot_type: so101

cameras:
  left_wrist:
    index: 6
    fps: 30
    width: 640
    height: 480
  right_wrist:
    index: 8
    fps: 30
    width: 640
    height: 480
  head:
    index: 4
    fps: 30
    width: 640
    height: 480

# Dual-arm leader-follower setup
# NOTE: Verify which port matches which arm by testing!
motors:
  follower_left:
    port: /dev/ttyACM2
    baudrate: 1000000
  follower_right:
    port: /dev/ttyACM3
    baudrate: 1000000
  leader_left:
    port: /dev/ttyACM0
    baudrate: 1000000
  leader_right:
    port: /dev/ttyACM1
    baudrate: 1000000

# CRITICAL: 5Hz action frequency (NOT 30Hz!)
policy_fps: 5
EOF

echo "Robot configuration saved!"
```

### Step 2: Create Collection Script

```bash
# Create scripts directory
mkdir -p ~/project/XLeRobot/scripts

# Create the collection script
cat > ~/project/XLeRobot/scripts/collect_primitives.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""
Data Collection Script for XLeRobot Primitives
Collects demonstration episodes for Pi0.5 fine-tuning
"""

import argparse
from lerobot.scripts.control_robot import record_episode
from datetime import datetime
import sys

PRIMITIVES = {
    "pick_center": ("pick red_cube from center", 30),
    "pick_left": ("pick blue_cube from left", 30),
    "pick_right": ("pick red_cube from right", 30),
    "place_box": ("place red_cube at box", 30),
    "place_left": ("place blue_cube at left_target", 30),
    "place_right": ("place red_cube at right_target", 30),
    "push": ("push red_cube to target", 40),
    "reach": ("reach table_center", 20),
    "grasp": ("grasp red_cube", 25),
    "release": ("release object", 15),
}

def collect_episodes(primitive_name, count, dataset_repo):
    if primitive_name not in PRIMITIVES:
        print(f"❌ Unknown primitive: {primitive_name}")
        print(f"Available primitives: {', '.join(PRIMITIVES.keys())}")
        return

    task_desc, duration = PRIMITIVES[primitive_name]

    print(f"\n{'='*70}")
    print(f"📹 COLLECTING DATA")
    print(f"{'='*70}")
    print(f"Task: {task_desc}")
    print(f"Episodes: {count}")
    print(f"Duration: {duration}s per episode")
    print(f"Action frequency: 5Hz")
    print(f"Dataset: {dataset_repo}")
    print(f"{'='*70}\n")

    completed = 0
    failed = 0

    for ep in range(1, count + 1):
        print(f"\n{'─'*70}")
        print(f"[Episode {ep}/{count}] '{task_desc}'")
        print(f"{'─'*70}")
        print("✋ Position leader arms and objects")
        print("⏸️  Press ENTER when ready to record...")
        input()

        try:
            print(f"🎬 Recording episode {ep}...")
            record_episode(
                robot_config="~/.cache/lerobot/configs/xlerobot_egocentric.yaml",
                dataset_repo_id=dataset_repo,
                task=task_desc,
                fps=5,  # CRITICAL: 5Hz action frequency!
                episode_time_s=duration,
                warmup_time_s=5,
                episode_index=None,  # Auto-increment
            )
            print(f"✅ Episode {ep} saved!")
            completed += 1

        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user (Ctrl+C)")
            break

        except Exception as e:
            print(f"❌ Episode {ep} failed: {e}")
            failed += 1
            retry = input("🔄 Retry this episode? (y/n): ")
            if retry.lower() == 'y':
                # Don't increment, will retry on next iteration
                continue

    print(f"\n{'='*70}")
    print(f"📊 SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Completed: {completed}/{count} episodes")
    print(f"❌ Failed: {failed} episodes")
    print(f"Task: '{task_desc}'")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Collect demonstration episodes for XLeRobot primitives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 10 pick_center episodes
  python collect_primitives.py --task pick_center --count 10

  # Collect to custom dataset
  python collect_primitives.py --task grasp --count 5 --dataset my_username/my_dataset

Available primitives:
  pick_center, pick_left, pick_right
  place_box, place_left, place_right
  push, reach, grasp, release
        """
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Primitive name (e.g., pick_center, place_box)"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--dataset",
        default="lerobot/xlerobot_primitives_egocentric",
        help="Dataset repository ID (default: lerobot/xlerobot_primitives_egocentric)"
    )

    args = parser.parse_args()

    collect_episodes(args.task, args.count, args.dataset)

if __name__ == "__main__":
    main()
ENDOFPYTHON

# Make executable
chmod +x ~/project/XLeRobot/scripts/collect_primitives.py

echo "✅ Collection script created at ~/project/XLeRobot/scripts/collect_primitives.py"
```

### Step 3: Test Single Episode

**Before collecting all data, test one episode:**

```bash
cd ~/project/XLeRobot
conda activate lerobot

# Test recording ONE episode
python scripts/collect_primitives.py --task pick_center --count 1
```

**What happens:**
1. **Prompt**: "Press ENTER when ready..."
2. **Warmup (5s)**: Get leader arms in position
3. **Beep/Signal**: Recording starts
4. **Demonstrate**: Perform the pick action smoothly
5. **Beep/Signal**: Recording ends (after 30 seconds)
6. **Save**: Episode automatically saved to dataset

**If test succeeds:**
- ✅ Camera feeds captured
- ✅ Motor positions recorded at 5Hz
- ✅ File saved to local directory
- ✅ Ready for full collection!

**If test fails:**
- ❌ Check error message
- ❌ Verify robot config path
- ❌ Check camera/motor connections
- ❌ See [Troubleshooting](#troubleshooting) section

---

## Demonstration Best Practices

### 🌟 Golden Rules for High-Quality Demos

#### Rule 1: SLOW and SMOOTH ⭐⭐⭐

**Target speed**: Move leader arms **2-3x slower than natural human speed**

```
Good Speed:
- Pick action: 15-20 seconds total
- Reach phase: 3-5 seconds
- Grasp phase: 2-3 seconds
- Lift phase: 2-3 seconds

Bad Speed ✗:
- Pick action: 3 seconds (TOO FAST!)
- Slam gripper down
- Jerky sudden movements
```

**Why**: Model learns from your speed. Slow demos → Smooth robot actions

**Tip**: Pretend you're moving underwater - graceful, continuous motion

#### Rule 2: CONSISTENT STRATEGY ⭐⭐⭐

**Same approach every time for the same primitive**

```
Good Consistency:
- Pick always from above (not sometimes from side)
- Always center gripper before descending
- Always same grasp orientation
- Always same retract distance

Bad Consistency ✗:
- Demo 1: Approach from front
- Demo 2: Approach from side
- Demo 3: Approach from diagonal
```

**Why**: Model learns patterns. Variation confuses the learning process.

**Tip**: Choose ONE strategy, practice it 3 times, then record all episodes

#### Rule 3: COMPLETE THE TASK ⭐⭐⭐

**Don't stop until task is fully complete**

```
Good Completion:
- Object grasped AND lifted AND stable
- Object placed AND released AND settled
- End in stable state for 2-3 seconds

Bad Completion ✗:
- Stop recording mid-grasp
- Object falls after release (not stable)
- Recording ends during motion
```

**Why**: Model needs to learn the "done" state

**Tip**: Hold final position for 2-3 seconds before recording ends

#### Rule 4: CONTINUOUS MOTION ⭐⭐

**No pauses or idle periods**

```
Good Motion:
- Smooth continuous movement
- Transitions are gradual
- Constant velocity changes

Bad Motion ✗:
- Stop to "think"
- Pause before grasping
- Idle periods (arm not moving)
```

**Why**: 5Hz expects continuous motion data

**Tip**: Plan the motion mentally before pressing ENTER to record

### Demonstration Technique by Primitive

#### Pick (Most Important - 30 episodes)

**Step-by-step breakdown:**

```
Phase 1: Start Position (0-2s)
- Gripper: Open
- Position: Above table, ~20cm from object
- Orientation: Aligned with object

Phase 2: Approach (2-7s)
- Move: Smooth horizontal approach to object
- Speed: ~5cm/second
- Action: Center gripper over object

Phase 3: Descend (7-10s)
- Move: Lower vertically to object
- Speed: ~3cm/second
- Stop: Just touching object

Phase 4: Grasp (10-12s)
- Action: Close gripper smoothly
- Check: Object secure in gripper
- Hold: Brief pause (0.5s)

Phase 5: Lift (12-15s)
- Move: Raise vertically 10-15cm
- Speed: ~3cm/second
- Check: Object still grasped

Phase 6: Hold (15-18s)
- Position: Stable with object
- Duration: 2-3 seconds
- Purpose: Show "task complete" state

Total: ~18 seconds
```

**Common Pick Mistakes:**

❌ **Too fast** - Slamming down on object
✅ Fix: Take 3-5 seconds to descend

❌ **Jerky approach** - Sudden starts/stops
✅ Fix: Accelerate/decelerate smoothly

❌ **Inconsistent grasp** - Different orientation each time
✅ Fix: Always grasp from same angle

❌ **Object falls** - Gripper not closed enough
✅ Fix: Ensure firm grasp, test before lifting

#### Place (30 episodes)

**Step-by-step breakdown:**

```
Phase 1: Start (0-2s)
- State: Holding object
- Position: Above target area
- Height: ~15cm above target

Phase 2: Approach Target (2-5s)
- Move: Horizontal to target location
- Speed: ~5cm/second
- Align: Center object over target

Phase 3: Descend (5-8s)
- Move: Lower vertically to target
- Speed: ~3cm/second
- Stop: Object touching target surface

Phase 4: Release (8-10s)
- Action: Open gripper smoothly
- Check: Object stable on surface
- Pause: 0.5s after opening

Phase 5: Retract (10-13s)
- Move: Lift gripper 5-10cm
- Speed: ~3cm/second
- Direction: Straight up

Phase 6: Observe (13-15s)
- Position: Gripper clear of object
- Duration: 2 seconds
- Purpose: Verify object is stable

Total: ~15 seconds
```

**Common Place Mistakes:**

❌ **Release too high** - Object falls/bounces
✅ Fix: Release when object is ON surface

❌ **Gripper doesn't clear** - Bumps object after release
✅ Fix: Retract straight up, not sideways

❌ **Object rolls away** - Unstable placement
✅ Fix: Ensure flat placement, wait for settling

#### Push (10 episodes)

**Step-by-step breakdown:**

```
Phase 1: Approach (0-3s)
- Position: Gripper closed, approaching object
- Target: Side of object (not top)
- Distance: Stop 1cm from object

Phase 2: Contact (3-5s)
- Action: Touch side of object gently
- Check: Object not moving yet
- Align: Pushing direction toward target

Phase 3: Push (5-12s)
- Move: Continuous slide toward target
- Speed: ~2cm/second (SLOW!)
- Maintain: Constant contact with object

Phase 4: Arrival (12-14s)
- Position: Object at target location
- Maintain: Brief pressure (1 second)
- Check: Object stable

Phase 5: Retract (14-16s)
- Move: Pull gripper away from object
- Speed: ~3cm/second
- Direction: Opposite of push direction

Total: ~16 seconds
```

**Push Tips:**
- ✅ Push from SIDE, not top
- ✅ Maintain contact throughout
- ✅ Slow, steady motion
- ❌ Don't knock object over
- ❌ Don't lose contact mid-push

#### Reach (10 episodes)

**Simplest primitive - no grasping:**

```
Phase 1: Start (0-1s)
- Position: Gripper anywhere on table
- Gripper: Closed or open (doesn't matter)

Phase 2: Move to Target (1-8s)
- Path: Direct to target location
- Speed: ~5cm/second
- Motion: Smooth, continuous

Phase 3: Arrive (8-10s)
- Position: Gripper at target location
- Hold: Stay for 2 seconds
- Purpose: Show target reached

Total: ~10 seconds
```

#### Grasp (10 episodes)

**Focus on grasp action itself:**

```
Phase 1: Pre-grasp (0-3s)
- Position: Gripper open, near object
- Alignment: Centered on object
- Distance: 2-3cm from object

Phase 2: Approach (3-6s)
- Move: Slowly enclose object
- Action: Close gripper gradually
- Check: Object centered in gripper

Phase 3: Close (6-9s)
- Action: Full gripper closure
- Force: Gentle but secure
- Check: Object secure

Phase 4: Hold (9-12s)
- State: Gripper closed with object
- Duration: 3 seconds
- Purpose: Show successful grasp

Total: ~12 seconds
```

#### Release (8 episodes)

**Focus on release action:**

```
Phase 1: Pre-release (0-2s)
- State: Holding object above surface
- Position: Object near target
- Height: Just touching surface

Phase 2: Release (2-5s)
- Action: Open gripper smoothly
- Speed: Gradual opening (not sudden)
- Check: Object stable on surface

Phase 3: Observe (5-8s)
- State: Gripper open, object released
- Duration: 3 seconds
- Purpose: Show release complete

Total: ~8 seconds
```

### Common Mistakes to Avoid

#### ❌ Mistake 1: Moving Too Fast

**Problem**: 5-second demos don't give model enough data
**Impact**: Model can't learn smooth trajectories
**Fix**: Target 15-30 seconds per episode

**How to slow down:**
- Count "one Mississippi, two Mississippi" during approach
- Use a metronome (60 BPM = 1 second per beat)
- Practice motion 3 times before recording

#### ❌ Mistake 2: Varying Strategy

**Problem**: Demo 1 picks from front, Demo 2 from side
**Impact**: Model gets confused about the "right" approach
**Fix**: Standardize ONE approach for each primitive

**How to be consistent:**
- Write down your strategy (e.g., "always approach from above")
- Practice the same motion path 5 times
- Use visual markers on table for alignment

#### ❌ Mistake 3: Incomplete Task Execution

**Problem**: Stop recording before object settles
**Impact**: Model doesn't learn "task complete" state
**Fix**: Hold final position for 2-3 seconds

**Completion checklist:**
- ✅ Object grasped? (for pick)
- ✅ Object released? (for place)
- ✅ Object stable? (not moving)
- ✅ Held final position? (2-3s)

#### ❌ Mistake 4: Jerky Motion

**Problem**: Sudden starts, stops, direction changes
**Impact**: Model learns jerky behavior
**Fix**: Smooth acceleration/deceleration

**Smoothness tips:**
- Start slow, gradually speed up
- Slow down before stopping
- No sharp corners in motion path
- Pretend moving through honey

#### ❌ Mistake 5: Environmental Changes

**Problem**: Moving background objects, changing lighting
**Impact**: Model distracted by irrelevant changes
**Fix**: Keep workspace static between episodes

**Environmental consistency:**
- ✅ Same lighting all day
- ✅ Same background
- ✅ Same table surface
- ✅ Same camera angles
- ❌ Don't move workspace furniture
- ❌ Don't change time of day (lighting)

---

## Quality Control

### After Each Episode: Immediate Check

**While episode is fresh in memory, verify:**

| Check | Question | Pass/Fail |
|-------|----------|-----------|
| **Completion** | Did task fully complete? | ✅ / ❌ |
| **Smoothness** | Was motion smooth? | ✅ / ❌ |
| **Speed** | Was it slow enough (15-30s)? | ✅ / ❌ |
| **Consistency** | Same strategy as previous? | ✅ / ❌ |
| **Success** | Object in correct final state? | ✅ / ❌ |

**If ANY check fails → Re-record immediately!**

### After Each Session: Playback Review

```bash
# Review today's episodes
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/xlerobot_primitives_egocentric \
  --episode-index -1  # Latest episode

# Or specify specific episode
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/xlerobot_primitives_egocentric \
  --episode-index 42  # Episode 42
```

**What to look for in playback:**
- ✅ All 3 camera views show clear footage
- ✅ Object visible in all frames
- ✅ Motion appears smooth in video
- ✅ Task completes successfully
- ❌ Any camera views blocked/blurry
- ❌ Object out of view
- ❌ Jerky motion visible

### End of Day: Dataset Statistics

```bash
# Check dataset stats
python << 'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/xlerobot_primitives_egocentric")

print("="*70)
print("📊 DATASET STATISTICS")
print("="*70)
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Action frequency: {dataset.meta.fps} Hz (target: 5)")
print(f"Avg episode length: {dataset.num_frames / dataset.num_episodes:.1f} frames")
print(f"Avg duration: {dataset.num_frames / dataset.num_episodes / 5:.1f} seconds")
print(f"Tasks: {len(dataset.meta.tasks)} primitives")
print(f"Camera keys: {dataset.meta.camera_keys}")
print("="*70)
EOF
```

**Target metrics:**
- ✅ Action frequency: **5 Hz** (not 30!)
- ✅ Avg duration: **15-30 seconds**
- ✅ All episodes: **Complete and successful**

### Quality Acceptance Criteria

**Episode is GOOD if:**
- ✅ Task completed successfully (object in final state)
- ✅ Motion was smooth (no jerks)
- ✅ Duration was 15-30 seconds
- ✅ All 3 cameras captured clear footage
- ✅ Object visible in all camera views
- ✅ Consistent with other episodes of same primitive

**Episode is BAD if:**
- ❌ Task failed (object dropped, missed target)
- ❌ Jerky or sudden movements
- ❌ Too fast (<10s) or too slow (>45s)
- ❌ Camera views blocked/blurry
- ❌ Object out of frame
- ❌ Significantly different strategy than others

**Acceptance rate target: 75-85%**
- If you record 100 episodes → expect 75-85 good
- Re-record bad ones immediately
- Don't settle for mediocre quality!

### Re-recording Strategy

**When to re-record:**
1. **Immediately**: If you know episode failed during recording
2. **Same day**: If playback review shows issues
3. **Next morning**: If end-of-day stats reveal problems

**How to re-record:**
```bash
# Continue from where you left off
python scripts/collect_primitives.py --task pick_center --count 3
# This adds 3 more episodes to dataset
```

**Pro tip**: Keep a "re-record list"
```bash
# Create re-record tracking file
cat > ~/project/XLeRobot/rerecord_list.txt << 'EOF'
EPISODES TO RE-RECORD
=====================
Episode 15: pick_center - object dropped
Episode 23: place_box - too fast
Episode 31: push - jerky motion

EOF
```

---

## Diversity Guidelines

### Why Diversity Matters

**Problem**: If all episodes are identical, model overfits
**Solution**: Controlled variation within each primitive

**The Balance:**
- ✅ **Consistency** in strategy/approach
- ✅ **Diversity** in starting conditions

### Object Position Diversity

**For each primitive, vary starting positions:**

#### Pick Primitive (30 episodes total)

```
Position Set 1: Center (10 episodes)
- Object at exact table center
- Same object (red_cube)
- Same final lift height

Position Set 2: Left Offset (10 episodes)
- Object 8-10cm left of center
- Same object (red_cube)
- Same approach strategy

Position Set 3: Right Offset (10 episodes)
- Object 8-10cm right of center
- Same object (red_cube)
- Same approach strategy
```

**Visual layout:**
```
Table view (60cm x 60cm):

┌─────────────────────────────┐
│                             │
│    [cube]                   │  ← Left position
│         [cube]              │  ← Center position
│              [cube]         │  ← Right position
│                             │
└─────────────────────────────┘
```

#### Place Primitive (30 episodes total)

```
Target Set 1: Box Center (10 episodes)
- Place at center of box
- Same release height
- Same retract direction

Target Set 2: Box Left (10 episodes)
- Place at left edge of box
- Same release height
- Same retract direction

Target Set 3: Box Right (10 episodes)
- Place at right edge of box
- Same release height
- Same retract direction
```

### Object Color/Type Diversity (Optional)

**If you have multiple objects:**

```
Pick red_cube (5 episodes)
Pick blue_cube (5 episodes)
Pick green_cube (5 episodes)

Benefits:
- Model learns object-invariant grasping
- Better generalization to new objects
- More robust visual features
```

**If you only have one object type:**
- ✅ That's okay! Position diversity is more important
- ✅ Focus on getting 75-100 high-quality episodes
- ⚠️ Don't sacrifice quality for variety

### Clutter Diversity (Advanced - Optional)

**Progression:**

```
Week 1 (Days 2-3): Empty Table
- Just target object on table
- No distractors
- Learn basic skill

Week 1 (Days 4-5): Light Clutter (if time permits)
- 1-2 distractor objects nearby
- Target object still clearly visible
- Tests object recognition

Future: Medium Clutter
- 3-4 objects on table
- Closer distractors
- More challenging scenarios
```

**Important**: Don't add clutter until basic primitives work!

### Diversity Checklist

For each primitive type, ensure:

- ✅ **3 different starting positions** (left, center, right)
- ✅ **Consistent approach** (same strategy for all)
- ✅ **Same object type** (don't mix unless you have extras)
- ✅ **Same final state** (all episodes end same way)
- ⚠️ **Optional**: Different object colors (if available)
- ⚠️ **Optional**: Light clutter (if basic works)

---

## Daily Collection Schedule

### Example Day 2: Pick Primitives (30 episodes, 3-4 hours)

**Morning Session (9 AM - 12 PM)**

**9:00 - 9:15 AM: Setup**
```bash
# Boot up
cd ~/project/XLeRobot
conda activate lerobot

# Hardware check
python -c "import cv2; [print(f'Cam {i}: OK' if cv2.VideoCapture(i).isOpened() else f'Cam {i}: FAIL') for i in [0,2,4]]"

# Lighting check - same as yesterday?
# Object placement - ready?
```

**9:15 - 10:15 AM: Pick Center (10 episodes)**
```bash
python scripts/collect_primitives.py --task pick_center --count 10

# Tips:
# - Take 2-minute break every 3 episodes
# - Check quality after episode 5
# - Maintain same approach strategy
```

**10:15 - 10:30 AM: Break**
- Stretch
- Review first 10 episodes
- Re-record any failures

**10:30 - 11:30 AM: Pick Left (10 episodes)**
```bash
# Move object 10cm left of center
python scripts/collect_primitives.py --task pick_left --count 10
```

**11:30 AM - 12:00 PM: Quick Review**
```bash
# Check statistics
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('lerobot/xlerobot_primitives_egocentric'); print(f'Episodes so far: {d.num_episodes}')"
```

**Lunch Break: 12:00 - 1:00 PM**

**Afternoon Session (1:00 - 3:00 PM)**

**1:00 - 2:00 PM: Pick Right (10 episodes)**
```bash
# Move object 10cm right of center
python scripts/collect_primitives.py --task pick_right --count 10
```

**2:00 - 2:30 PM: Quality Review**
```bash
# Playback review - watch random episodes
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/xlerobot_primitives_egocentric \
  --episode-index 5

# Check 3-5 random episodes
# Note any quality issues
```

**2:30 - 3:00 PM: Re-record if needed**
- Identify any bad episodes
- Re-record immediately
- Aim for 30 GOOD episodes from Day 2

**End of Day Summary:**
- ✅ 30 pick episodes collected
- ✅ Quality checked
- ✅ Issues re-recorded
- ✅ Ready for Day 3 (place primitives)

---

## Troubleshooting

### Issue 1: "Camera not found" Error

**Symptoms:**
```
Camera 0: ✗ NOT WORKING
```

**Solutions:**

```bash
# Check camera devices
ls -l /dev/video*

# List cameras with v4l2
v4l2-ctl --list-devices

# Test camera manually
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# If still failing:
# 1. Check USB connection
# 2. Try different USB port
# 3. Restart computer
# 4. Check camera permissions
sudo chmod 666 /dev/video*
```

### Issue 2: "Motor communication failed"

**Symptoms:**
```
Motors: ✗ ERROR - Serial port not found
```

**Solutions:**

```bash
# Check USB ports
ls -l /dev/ttyUSB*

# If no /dev/ttyUSB*, check alternatives
ls -l /dev/ttyACM*

# Set permissions
sudo chmod 666 /dev/ttyUSB0

# Test motor communication
python -c "
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
motors = DynamixelMotorsBus(port='/dev/ttyUSB0')
print('Success!')
"
```

### Issue 3: Episodes recorded at 30Hz instead of 5Hz

**Symptoms:**
```
Action frequency: 30 Hz (should be 5)
```

**Solutions:**

```bash
# Check robot config
cat ~/.cache/lerobot/configs/xlerobot_egocentric.yaml | grep fps

# Should show:
# policy_fps: 5

# If wrong, fix config:
sed -i 's/policy_fps: 30/policy_fps: 5/' ~/.cache/lerobot/configs/xlerobot_egocentric.yaml

# Verify fix
cat ~/.cache/lerobot/configs/xlerobot_egocentric.yaml | grep fps
```

**Important**: Re-record all episodes if they were at wrong frequency!

### Issue 4: Jerky/not smooth demonstrations

**Symptoms:**
- Robot motion looks choppy in playback
- Sudden starts/stops

**Solutions:**

**Mental strategies:**
- ✅ Think "underwater motion"
- ✅ Count slowly during approach
- ✅ Practice motion 3 times before recording
- ✅ Use metronome (60 BPM)

**Physical strategies:**
- ✅ Relax your hands/arms
- ✅ Move leader arms slower than feels natural
- ✅ Accelerate/decelerate gradually
- ❌ Don't "think" mid-motion (plan first!)

### Issue 5: Object falls or task fails

**Symptoms:**
- Object drops during pick
- Gripper doesn't grasp properly
- Object rolls away after place

**Solutions:**

**For pick failures:**
- ✅ Ensure gripper is aligned before closing
- ✅ Close gripper slowly (2-3 seconds)
- ✅ Check gripper force - too weak?
- ✅ Object size appropriate for gripper?

**For place failures:**
- ✅ Release when object ON surface (not hovering)
- ✅ Open gripper slowly
- ✅ Retract straight up (not sideways)
- ✅ Ensure surface is flat

### Issue 6: Cameras out of view

**Symptoms:**
- Object not visible in camera feed
- Gripper blocks view

**Solutions:**

```bash
# Re-check camera positions
python << 'EOF'
import cv2
for i in [0, 2, 4]:
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    cv2.imwrite(f'debug_cam_{i}.jpg', frame)
    cap.release()
print('Check debug_cam_*.jpg files')
EOF

# View images
ls debug_cam_*.jpg

# Adjust camera positions if needed
# Re-test until workspace fully visible
```

### Issue 7: Inconsistent lighting

**Symptoms:**
- Some episodes brighter than others
- Shadows change throughout day

**Solutions:**

**Prevention:**
- ✅ Close blinds (use artificial light)
- ✅ Collect all data same time of day
- ✅ Use consistent overhead lighting

**If already inconsistent:**
- ⚠️ Re-record affected episodes
- ⚠️ Or accept slight variation (model may handle it)

### Issue 8: Dataset upload fails

**Symptoms:**
```
Error: Failed to push to hub
```

**Solutions:**

```bash
# Check HuggingFace login
huggingface-cli whoami

# If not logged in:
huggingface-cli login

# Check repo exists
# Go to https://huggingface.co/YOUR_USERNAME
# Create dataset repo if needed

# Or keep data local (fine for Week 2 training):
# Dataset stored in: ~/lerobot_data/ or similar
```

---

## Final Checklist

### Before Week 2 Training

Run this comprehensive check:

```bash
# Final validation script
cat > ~/project/XLeRobot/scripts/validate_dataset.py << 'ENDPYTHON'
#!/usr/bin/env python3
"""Validate dataset is ready for Pi0.5 training"""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import sys

def validate_dataset(repo_id):
    try:
        dataset = LeRobotDataset(repo_id)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

    print("="*70)
    print("🔍 DATASET VALIDATION")
    print("="*70)

    # Check 1: Episode count
    print(f"\n1️⃣  Episode Count")
    print(f"   Total episodes: {dataset.num_episodes}")
    if dataset.num_episodes >= 75:
        print(f"   ✅ PASS (≥75 episodes)")
    else:
        print(f"   ❌ FAIL (need {75 - dataset.num_episodes} more)")

    # Check 2: Action frequency
    print(f"\n2️⃣  Action Frequency")
    print(f"   FPS: {dataset.meta.fps} Hz")
    if dataset.meta.fps == 5:
        print(f"   ✅ PASS (5Hz)")
    else:
        print(f"   ❌ FAIL (should be 5Hz, not {dataset.meta.fps}Hz)")

    # Check 3: Cameras
    print(f"\n3️⃣  Cameras")
    print(f"   Camera keys: {dataset.meta.camera_keys}")
    expected_cameras = ['left_wrist', 'right_wrist', 'head']
    if all(cam in dataset.meta.camera_keys for cam in expected_cameras):
        print(f"   ✅ PASS (all 3 cameras present)")
    else:
        print(f"   ❌ FAIL (missing cameras)")

    # Check 4: Primitives
    print(f"\n4️⃣  Primitive Tasks")
    print(f"   Number of tasks: {len(dataset.meta.tasks)}")
    print(f"   Tasks: {list(dataset.meta.tasks)}")
    if len(dataset.meta.tasks) >= 6:
        print(f"   ✅ PASS (≥6 primitive types)")
    else:
        print(f"   ⚠️  WARNING (fewer than 6 types - acceptable but not ideal)")

    # Check 5: Episode duration
    print(f"\n5️⃣  Episode Duration")
    avg_frames = dataset.num_frames / dataset.num_episodes
    avg_duration = avg_frames / dataset.meta.fps
    print(f"   Avg frames: {avg_frames:.1f}")
    print(f"   Avg duration: {avg_duration:.1f}s")
    if 10 <= avg_duration <= 40:
        print(f"   ✅ PASS (10-40s range)")
    else:
        print(f"   ⚠️  WARNING (outside typical 10-40s range)")

    # Final verdict
    print("\n" + "="*70)
    if (dataset.num_episodes >= 75 and
        dataset.meta.fps == 5 and
        all(cam in dataset.meta.camera_keys for cam in expected_cameras)):
        print("🎉 DATASET READY FOR WEEK 2 TRAINING!")
        print("="*70)
        return True
    else:
        print("⚠️  DATASET NOT READY - Fix issues above")
        print("="*70)
        return False

if __name__ == "__main__":
    repo_id = "lerobot/xlerobot_primitives_egocentric"
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]

    validate_dataset(repo_id)
ENDPYTHON

chmod +x ~/project/XLeRobot/scripts/validate_dataset.py

# Run validation
python ~/project/XLeRobot/scripts/validate_dataset.py
```

### Ready for Week 2 If:

- ✅ **Episodes**: 75-100 good quality episodes
- ✅ **Frequency**: 5Hz action recording
- ✅ **Cameras**: All 3 cameras (left_wrist, right_wrist, head)
- ✅ **Primitives**: 6-8 different primitive types
- ✅ **Quality**: >75% acceptance rate
- ✅ **Duration**: Average 15-30 seconds per episode
- ✅ **Consistency**: Same strategy within each primitive
- ✅ **Completeness**: All tasks fully completed

### Not Ready Yet If:

- ❌ **<75 episodes**: Collect more
- ❌ **Wrong frequency**: Re-record at 5Hz
- ❌ **Missing cameras**: Fix camera setup, re-record
- ❌ **Low quality**: Re-record bad episodes
- ❌ **Too fast/slow**: Adjust demonstration speed

---

## Next Steps After Collection

### Week 2 Preview: Fine-tune Pi0.5

Once dataset is validated:

```bash
# Install Pi0.5 dependencies (if not already done)
pip install -e ".[pi]"

# Start training (Week 2 Day 1)
python src/lerobot/scripts/lerobot_train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --dataset.repo_id=lerobot/xlerobot_primitives_egocentric \
  --steps=6000 \
  --batch_size=8

# Expected training time: 6-10 hours
# Expected VRAM: 18-22GB (with LoRA)
```

### Celebrate! 🎉

**Data collection is the hardest and most important part!**

You've now created:
- ✅ Custom robot dataset
- ✅ High-quality demonstrations
- ✅ Reusable data for multiple VLA models
- ✅ Foundation for VLM→VLA system

**Week 1 Done → Week 2: Let the model learn!** 🤖

---

## Quick Reference Card

### Daily Routine

**Morning:**
```bash
cd ~/project/XLeRobot
conda activate lerobot
# Hardware check (30 sec)
# Collect 10-15 episodes (1-2 hours)
```

**Afternoon:**
```bash
# Collect 10-15 episodes (1-2 hours)
# Quality review (30 min)
# Re-record if needed
```

### Key Commands

```bash
# Collect episodes
python scripts/collect_primitives.py --task TASK_NAME --count N

# Check dataset
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('lerobot/xlerobot_primitives_egocentric'); print(f'Episodes: {d.num_episodes}')"

# Visualize episode
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/xlerobot_primitives_egocentric \
  --episode-index N

# Validate dataset (end of week)
python scripts/validate_dataset.py
```

### Quality Checklist

Every episode must be:
- ✅ Slow (15-30s)
- ✅ Smooth (no jerks)
- ✅ Complete (task finished)
- ✅ Consistent (same strategy)
- ✅ Visible (all cameras clear)

---

**Good luck with data collection! This is the foundation of your VLM→VLA system! 🚀**
