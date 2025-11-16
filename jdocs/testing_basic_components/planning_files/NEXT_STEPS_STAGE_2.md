# Next Steps: Stage 2 Reality-Based Action Plan

**Date:** 2025-11-08
**Status:** Ready to Execute

---

## Table of Contents

- [TL;DR](#tldr)
- [Decision Tree](#decision-tree)
- [Path A: Test r2owb0/act1 (Try This First)](#path-a-test-r2owb0act1-try-this-first)
  - [Time: 1 hour](#time-1-hour)
  - [Prerequisites](#prerequisites)
  - [Step 1: Create Test Script](#step-1-create-test-script)
  - [Step 2: Run Test](#step-2-run-test)
  - [Step 3: Evaluate](#step-3-evaluate)
  - [Decision](#decision)
- [Path B: Test Community Models (Try This Second)](#path-b-test-community-models-try-this-second)
  - [Time: 3 hours](#time-3-hours)
  - [Step 1: Download Top 3 Models](#step-1-download-top-3-models)
  - [Step 2: Test Each Model](#step-2-test-each-model)
  - [Step 3: Pick Best](#step-3-pick-best)
  - [Decision](#decision-1)
- [Path C: Fine-tune SmolVLA (The Realistic Path)](#path-c-fine-tune-smolvla-the-realistic-path)
  - [Time: 1-2 weeks](#time-1-2-weeks)
  - [Step 1: Decide on Teleoperation Method](#step-1-decide-on-teleoperation-method)
  - [Step 2: Collect Demonstrations](#step-2-collect-demonstrations)
  - [Step 3: Verify Dataset](#step-3-verify-dataset)
  - [Step 4: Fine-tune SmolVLA](#step-4-fine-tune-smolvla)
  - [Step 5: Evaluate Fine-tuned Model](#step-5-evaluate-fine-tuned-model)
  - [Step 6: Proceed to Stage 3](#step-6-proceed-to-stage-3)
- [Timeline Comparison](#timeline-comparison)
- [My Recommendation](#my-recommendation)
  - [This Week (Quick Tests)](#this-week-quick-tests)
  - [Next 1-2 Weeks (If Fine-tuning Needed)](#next-1-2-weeks-if-fine-tuning-needed)
  - [Expected Outcome](#expected-outcome)
- [Questions to Ask Yourself](#questions-to-ask-yourself)
  - [Before Starting](#before-starting)
  - [After Quick Tests (Paths A/B)](#after-quick-tests-paths-ab)
- [Summary](#summary)
- [Next Action](#next-action)

---

## TL;DR

Your MVP plan Stage 2 had unrealistic expectations. SmolVLA is pretrained ONLY on SO-100, not SO-101. Here's what to actually do:

**Quick Path (1-2 days):** Test existing models, maybe get lucky
**Realistic Path (1-2 weeks):** Fine-tune on your own data (standard practice)

---

## Decision Tree

```
START HERE
    ↓
Test r2owb0/act1 (1 hour)
    ├─ Works (>60%) → Use for simple tasks, proceed to Stage 3
    └─ Fails (<40%) → Continue ↓

Test Top 3 Community Models (3 hours)
    ├─ One works (>40%) → Use it, proceed to Stage 3
    └─ All fail (<30%) → Continue ↓

Fine-tune SmolVLA (1-2 weeks)
    ├─ Collect 80 episodes (3-7 days)
    ├─ Train model (6-8 hours GPU)
    └─ Achieve 70-90% success → Proceed to Stage 3
```

---

## Path A: Test r2owb0/act1 (Try This First)

### Time: 1 hour

### Prerequisites
- ✅ Your robot already calibrated
- ✅ Cameras working
- ✅ Test object (any small cube/block)

### Step 1: Create Test Script

Create `/home/jrobot/project/XLeRobot/jdocs/testing_basic_components/test_act_r2owb0.py`:

```python
#!/usr/bin/env python3
"""
Test r2owb0/act1 - The only true SO-101 pretrained model
"""

import torch
import time
import numpy as np
from pathlib import Path

def test_act_r2owb0(arm_port="/dev/ttyACM2", arm_id="xlerobot_left_arm",
                    camera_top=0, camera_wrist=6, duration=10):
    """
    Test ACT model trained on SO-101

    This model was trained on 10 episodes of pick-place task.
    It will try to repeat that exact task.
    """

    print("\n" + "="*60)
    print("Testing r2owb0/act1 - SO-101 ACT Model")
    print("="*60)

    device = torch.device("cuda")

    try:
        # Import LeRobot modules
        from lerobot.policies.act import ACTPolicy
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower

        # Load ACT model
        print("\nLoading r2owb0/act1...")
        model_id = "r2owb0/act1"
        model = ACTPolicy.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded")

        # Setup preprocessors
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        # Configure cameras (MATCH TRAINING: top=640x480, wrist=320x240)
        camera_config = {
            "camera1": OpenCVCameraConfig(
                index_or_path=camera_top,
                width=640, height=480, fps=30
            ),
            "camera2": OpenCVCameraConfig(
                index_or_path=camera_wrist,
                width=320, height=240, fps=30  # Note: smaller than top!
            ),
        }

        # Initialize robot
        robot_cfg = SO101FollowerConfig(port=arm_port, id=arm_id, cameras=camera_config)
        robot = SO101Follower(robot_cfg)
        robot.connect()
        print(f"✅ Robot connected: {arm_port}")

        # Setup features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        # Run inference at 50Hz (ACT's training frequency)
        print("\nStarting inference...")
        print("⚠️  This model will try to do pick-place task from training")
        print("⚠️  Watch if movements are task-relevant (not just random)")
        print("="*60 + "\n")

        total_steps = int(duration * 50)  # 50Hz
        robot_type = ""

        for step in range(total_steps):
            step_start = time.time()

            try:
                # Get observation
                obs = robot.get_observation()

                # Build inference frame (no task - ACT has no language conditioning)
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task="",  # ACT doesn't use language
                    robot_type=robot_type
                )

                # Preprocess
                obs_processed = preprocess(obs_frame)

                # Model inference
                with torch.no_grad():
                    action = model.select_action(obs_processed)

                # Postprocess
                action = postprocess(action)

                # Convert to robot action
                action = make_robot_action(action, dataset_features)

                # Send to robot
                robot.send_action(action)

                # Maintain 50Hz
                elapsed = time.time() - step_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

                # Progress
                if step % 50 == 0:
                    freq = 1.0 / elapsed if elapsed > 0 else 0
                    print(f"Step {step}/{total_steps} | {freq:.1f}Hz")

            except Exception as e:
                print(f"⚠️  Error at step {step}: {e}")
                continue

        print("\n" + "="*60)
        print("Test Complete!")
        print("="*60)

        # Cleanup
        robot.disconnect()

        # User evaluation
        print("\n📊 EVALUATION")
        print("-" * 60)
        print("Did the robot:")
        print("  1. Make large, purposeful movements? (not tiny jitters)")
        print("  2. Approach or interact with objects?")
        print("  3. Attempt to grasp something?")
        print("  4. Move toward a target position?")
        print("-" * 60)

        success = input("\nDid it show task-relevant behavior? (y/n): ")

        if success.lower() == 'y':
            print("\n✅ SUCCESS! r2owb0/act1 works on your setup!")
            print("You can use this for simple pick-place tasks.")
            print("\nNext: Proceed to Stage 3a with this model")
        else:
            print("\n❌ Model doesn't work on your setup")
            print("This is expected - camera/lighting may differ from training")
            print("\nNext: Try Path B (community models)")

        return success.lower() == 'y'

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Test r2owb0/act1 - SO-101 ACT Model")
    print("="*60)

    # Get camera indices
    camera_top = int(input("Enter TOP camera index [0]: ") or "0")
    camera_wrist = int(input("Enter WRIST camera index [6]: ") or "6")

    # Get robot config
    arm_choice = input("Which arm? (left/right) [left]: ").strip().lower()
    if arm_choice == 'right':
        arm_port = "/dev/ttyACM3"
        arm_id = "xlerobot_right_arm"
    else:
        arm_port = "/dev/ttyACM2"
        arm_id = "xlerobot_left_arm"

    # Duration
    duration = int(input("Test duration in seconds [10]: ") or "10")

    print("\n⚠️  IMPORTANT: This model expects:")
    print("   - Top camera: 640×480")
    print("   - Wrist camera: 320×240 (smaller!)")
    print("   - Consistent lighting")
    print("   - Will repeat its trained pick-place task")

    input("\nPress Enter to start test...")

    success = test_act_r2owb0(arm_port, arm_id, camera_top, camera_wrist, duration)
```

### Step 2: Run Test

```bash
cd ~/project/XLeRobot
conda activate lerobot

python jdocs/testing_basic_components/test_act_r2owb0.py
```

### Step 3: Evaluate

**Success Indicators:**
- ✅ Large, purposeful movements (not tiny jitters)
- ✅ Approaches objects
- ✅ Opens/closes gripper at appropriate times
- ✅ Moves toward target position

**Failure Indicators:**
- ❌ Only small random movements
- ❌ No object interaction
- ❌ Looks similar to your SmolVLA test

### Decision

- **If successful (>60%):** Use this model, proceed to Stage 3a
- **If fails (<40%):** Continue to Path B

---

## Path B: Test Community Models (Try This Second)

### Time: 3 hours

### Step 1: Download Top 3 Models

```bash
cd ~/project/XLeRobot

# Create models directory
mkdir -p community_models

# Model 1: HuggingFadeUser/my_smolvla (60 likes)
huggingface-cli download HuggingFadeUser/my_smolvla \
  --local-dir ./community_models/huggingfadeuser

# Model 2: jhou/smolvla_pickplace (42 likes)
huggingface-cli download jhou/smolvla_pickplace \
  --local-dir ./community_models/jhou

# Model 3: saood65/my_smolvla (4 likes, recent)
huggingface-cli download saood65/my_smolvla \
  --local-dir ./community_models/saood65
```

### Step 2: Test Each Model

Modify your `test_smolvla_simple.py` to accept model path:

```bash
# Test each model
python jdocs/testing_basic_components/test_smolvla_simple.py \
  --model_path ./community_models/huggingfadeuser \
  --duration 10

# Manually observe and score (0-100)

python jdocs/testing_basic_components/test_smolvla_simple.py \
  --model_path ./community_models/jhou \
  --duration 10

python jdocs/testing_basic_components/test_smolvla_simple.py \
  --model_path ./community_models/saood65 \
  --duration 10
```

### Step 3: Pick Best

Compare the three:
- Which showed most task-relevant movement?
- Which had largest action magnitudes?
- Which moved toward objects?

### Decision

- **If one scores >40%:** Use it for Stage 3
- **If all score <30%:** Continue to Path C (fine-tuning)

---

## Path C: Fine-tune SmolVLA (The Realistic Path)

### Time: 1-2 weeks

This is what the community actually does. It's standard practice, not a failure.

### Step 1: Decide on Teleoperation Method

**Option A: SO-101 Leader Arm (Recommended)**
- Cost: ~$350
- Speed: 3-5 min per episode
- Quality: High
- Delivery: 1-2 weeks
- Order: https://shop.wowrobo.com/products/so-arm101-leader

**Option B: Keyboard Teleoperation (Budget)**
- Cost: $0
- Speed: 10-15 min per episode
- Quality: Medium (harder to be smooth)
- Available: Immediately

### Step 2: Collect Demonstrations

**Target:**
- Episodes: 80 minimum (community-proven)
- Structure: 10 episodes × 8 object positions
- Task: Pick-place (start simple)
- Duration: 10-20 seconds per episode

**With Leader Arm:**
```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=xlerobot_leader \
  --dataset.repo_id=${HF_USER}/xlerobot_pickplace \
  --dataset.fps=30 \
  --dataset.num_episodes=80 \
  --task="pick up cube and place at target"
```

**With Keyboard:**
```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=keyboard \
  --dataset.repo_id=${HF_USER}/xlerobot_pickplace \
  --dataset.fps=30 \
  --dataset.num_episodes=80 \
  --task="pick up cube and place at target"
```

**Data Collection Tips:**
1. **Consistent Setup:**
   - Mark object positions with tape
   - Keep lighting consistent
   - Same camera positions every time

2. **Episode Structure:**
   ```
   Position 1 (front-left):    10 episodes
   Position 2 (front-center):  10 episodes
   Position 3 (front-right):   10 episodes
   Position 4 (middle-left):   10 episodes
   Position 5 (middle-center): 10 episodes
   Position 6 (middle-right):  10 episodes
   Position 7 (back-left):     10 episodes
   Position 8 (back-right):    10 episodes
   Total:                      80 episodes
   ```

3. **Quality Over Quantity:**
   - Only keep successful demonstrations
   - Delete failed attempts
   - Smooth, natural movements

**Time Estimate:**
- Leader arm: 3-5 days (3-5 min/episode, breaks)
- Keyboard: 5-7 days (10-15 min/episode)

### Step 3: Verify Dataset

```bash
# Visualize dataset
python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id=${HF_USER}/xlerobot_pickplace \
  --episode-index=0

# Check stats
python -c "
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('${HF_USER}/xlerobot_pickplace')
print(f'Episodes: {ds.meta.total_episodes}')
print(f'Frames: {ds.meta.total_frames}')
print(f'FPS: {ds.meta.fps}')
print(f'Stats: {ds.meta.stats.keys()}')
"
```

### Step 4: Fine-tune SmolVLA

```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/xlerobot_pickplace \
  --training.num_epochs=20000 \
  --training.batch_size=64 \
  --training.learning_rate=1e-4 \
  --training.save_checkpoint_every_n_epochs=1000 \
  --output_dir=outputs/smolvla_xlerobot
```

**Expected Time:**
- RTX 5090: ~6-8 hours
- Monitor with: `watch -n 1 nvidia-smi`

**Checkpoints:**
- Saved every 1000 epochs
- Best model usually around 15k-20k steps

### Step 5: Evaluate Fine-tuned Model

```python
#!/usr/bin/env python3
"""Evaluate fine-tuned SmolVLA"""

from lerobot.policies.smolvla import SmolVLAPolicy

# Load YOUR fine-tuned model
policy = SmolVLAPolicy.from_pretrained(
    "outputs/smolvla_xlerobot/checkpoints/20000/pretrained_model"
)

# Test 10 trials
success_count = 0
for trial in range(10):
    print(f"\nTrial {trial+1}/10")
    input("Reset scene to trained position, press Enter...")

    # Run inference for 10 seconds
    run_inference(policy, task="pick up cube and place at target", duration=10)

    result = input("Success? (y/n): ")
    if result.lower() == 'y':
        success_count += 1

    print(f"Current success rate: {success_count}/{trial+1}")

print(f"\nFinal success rate: {success_count}/10 = {success_count*10}%")
```

**Expected Results:**
- ✅ 70-90% on trained positions
- ✅ 40-60% on new positions
- ✅ Language conditioning works

### Step 6: Proceed to Stage 3

With 70-90% success, you're ready for:
- Stage 3a: Natural language control
- Stage 3b: Complex multi-step tasks

---

## Timeline Comparison

| Approach | Time | Cost | Success Rate | Ready for Stage 3? |
|----------|------|------|--------------|-------------------|
| **Path A:** r2owb0/act1 | 1 hour | $0 | 60-80%* | ⚠️ Limited (one task) |
| **Path B:** Community | 3 hours | $0 | 40-70%* | ✅ Yes (if works) |
| **Path C:** Fine-tune | 1-2 weeks | $0-350** | 70-90% | ✅ Yes |

*If lucky with camera/setup match
**Leader arm optional but highly recommended

---

## My Recommendation

### This Week (Quick Tests)

**Day 1:**
- Morning: Test r2owb0/act1 (1 hour)
- Afternoon: Download & test community models (3 hours)
- Evening: Make decision

**Day 2:**
- If Path A/B worked: Start Stage 3a planning
- If both failed: Order leader arm, plan data collection

### Next 1-2 Weeks (If Fine-tuning Needed)

**Week 1:**
- Wait for leader arm delivery (or start keyboard teleoperation)
- Setup data collection workspace
- Collect 80 episodes

**Week 2:**
- Fine-tune model (6-8 hours GPU)
- Evaluate performance
- Iterate if needed

### Expected Outcome

**Optimistic (10% chance):**
- r2owb0/act1 or community model works
- Proceed to Stage 3 by Day 2

**Realistic (90% probability):**
- Need fine-tuning
- 1-2 weeks to complete
- Achieve 70-90% success
- Proceed to Stage 3 by Day 15

---

## Questions to Ask Yourself

### Before Starting

1. **Timeline:**
   - Do I have 1-2 weeks for fine-tuning?
   - Or do I need quick results (try paths A/B first)?

2. **Budget:**
   - Can I spend $350 on leader arm?
   - Or must I use keyboard teleoperation?

3. **Goals:**
   - Just proof-of-concept? (40-60% OK)
   - Production-ready? (70-90% needed)

### After Quick Tests (Paths A/B)

1. **If one worked:**
   - Is 40-60% success enough for my use case?
   - Can I accept one-task-only limitation (r2owb0/act1)?

2. **If both failed:**
   - Am I ready to commit to fine-tuning?
   - Do I understand this is standard practice?

---

## Summary

Your situation is **completely normal**:

1. ❌ Pretrained SmolVLA doesn't work on SO-101 → Expected
2. ❌ MVP plan Stage 2 was unrealistic → Now you know
3. ✅ You have working hardware → Great foundation
4. ✅ Multiple paths forward → Choose based on timeline/budget

**The community-proven path is fine-tuning with 80+ episodes.**

You're not behind or doing anything wrong. You just discovered what everyone else learned: pretrained models need adaptation to your specific robot.

---

## Next Action

**RIGHT NOW:**

```bash
cd ~/project/XLeRobot
conda activate lerobot

# Try Path A (1 hour)
python jdocs/testing_basic_components/test_act_r2owb0.py
```

Then decide based on results.

Good luck! 🤖
