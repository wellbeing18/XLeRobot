# Path B: Testing Community Fine-tuned Models

**Quick Guide for Testing jhou/smolvla_pickplace**

---

## 📋 Pre-Test Checklist

### 1. One Arm Only
- ✅ Choose which arm to test (left or right)
- ❌ **Power off the other arm** or move it out of camera view
- Why: Model trained on single-arm setup

### 2. Camera Setup
- ✅ Static head camera (tilted down for workspace view)
- ✅ Wrist camera on chosen arm
- ✅ Both should be accessible (check indices with `ls /dev/video*`)

### 3. Workspace
- ✅ Red cube (or similar small object) on table
- ✅ Clear workspace (no obstacles)
- ✅ Good lighting (consistent)

### 4. Know Your Camera Indices
Your cameras change indices sometimes! Check:
```bash
# Quick check
ls -l /dev/video*

# Or use helper
python jdocs/testing_basic_components/check_my_cameras.py
```

**Current (as of last test):**
- Head camera: `/dev/video4`
- Left wrist: `/dev/video6`
- Right wrist: `/dev/video?` (check if using right arm)

---

## 🚀 Run the Test

### Step 1: Activate Environment

```bash
cd ~/project/XLeRobot
conda activate lerobot
```

### Step 2: Run Test Script

```bash
python jdocs/testing_basic_components/test_community_smolvla.py
```

### Step 3: Answer Prompts

**Model ID:**
```
Enter model ID [jhou/smolvla_pickplace]:
→ Press Enter (use default)
```

**Cameras:**
```
Enter TOP camera index [4]:
→ Enter 4 (or your current head camera index)

Enter WRIST camera index [6]:
→ Enter 6 (or your current wrist camera index)
```

**Robot:**
```
Which arm? (left/right) [left]:
→ Enter left or right

Enter port [/dev/ttyACM2]:
→ Press Enter for left arm
→ Or enter /dev/ttyACM3 for right arm
```

**Duration:**
```
Test duration in seconds [10]:
→ Press Enter (10 seconds is good)
```

**Task:**
```
Enter task ['pick up the red cube']:
→ Press Enter (use default)
→ Or enter custom task like "pick up the blue block"
```

### Step 4: Watch the Test

The script will run for 10 seconds. **WATCH THE ARM:**

✅ **GOOD signs:**
- Large, purposeful movements
- Reaches toward objects
- Gripper opens/closes at appropriate times
- Looks like it's trying to do the task

❌ **BAD signs:**
- Only tiny movements (like base SmolVLA)
- Random-looking behavior
- Arm stays mostly still
- No gripper activity

### Step 5: Evaluate

After test completes, script will ask:
```
Was the behavior BETTER than base SmolVLA? (y/n):
```

**Answer honestly:**
- `y` if it showed task-relevant movements (even if not perfect)
- `n` if it looked similar to base SmolVLA (tiny movements only)

---

## 📊 Expected Results

### Best Case (20-40% chance)
- ✅ Model shows large, task-relevant movements
- ✅ Approaches objects, attempts grasping
- ✅ 40-70% success rate on pick-place
- **→ Can use this model, proceed to Stage 3**

### Likely Case (60-80% chance)
- ⚠️ Better than base SmolVLA, but not great
- ⚠️ Some large movements, but not task-relevant
- ⚠️ 20-40% success rate
- **→ Try 1-2 more community models, then Path C**

### Worst Case (10-20% chance)
- ❌ No better than base SmolVLA
- ❌ Still tiny movements only
- ❌ <20% success rate
- **→ Skip other models, go straight to Path C**

---

## 🔄 If First Model Doesn't Work Well

### Try These Alternatives (in order)

**Option 1: HuggingFadeUser/my_smolvla**
```bash
python jdocs/testing_basic_components/test_community_smolvla.py
# When prompted for model ID:
# Enter: HuggingFadeUser/my_smolvla
```
- 60 likes (most popular)
- Might have different camera setup

**Option 2: saood65/my_smolvla**
```bash
python jdocs/testing_basic_components/test_community_smolvla.py
# When prompted for model ID:
# Enter: saood65/my_smolvla
```
- Recent (newer LeRobot version)
- 4 likes

**Option 3: leesangoh/smolvla_pickplace**
```bash
python jdocs/testing_basic_components/test_community_smolvla.py
# When prompted for model ID:
# Enter: leesangoh/smolvla_pickplace
```
- Community trained
- Unknown setup

---

## 🎯 Success Criteria

### Model Works Well Enough (40-70% success)
**Indicators:**
- ✅ Approaches object correctly most of the time
- ✅ Grasps object sometimes
- ✅ Moves toward target area
- ⚠️ Doesn't complete perfectly every time

**What to do:**
- Use this model for Stage 3
- Build on it with hierarchical control
- Consider fine-tuning later for better performance

### Model Doesn't Work (<40% success)
**Indicators:**
- ❌ Random-looking movements
- ❌ Doesn't approach objects
- ❌ No better than base SmolVLA

**What to do:**
- Try 1-2 more community models
- If all fail → Proceed to Path C (fine-tuning)
- Accept that fine-tuning is necessary

---

## 🐛 Troubleshooting

### Issue: Model fails to load

**Error:** `cannot import name 'SmolVLAPolicy'`
**Fix:** Check import path (should be `lerobot.policies.smolvla.modeling_smolvla`)

**Error:** `Model not found`
**Fix:**
- Check model ID is correct
- Verify you have internet connection
- Try: `huggingface-cli login` if model is private

### Issue: Camera black screen

**Error:** Camera shows black during test
**Fix:**
1. Check camera indices: `ls /dev/video*`
2. Test camera: `python -c "import cv2; cap=cv2.VideoCapture(4); print(cap.read()[0])"`
3. If false, camera index changed - update in script

### Issue: Arm doesn't move

**Problem:** Arm stays still during test
**Check:**
1. Is arm port correct? (`/dev/ttyACM2` for left, `/dev/ttyACM3` for right)
2. Is arm powered on?
3. Do you see action values in terminal? (should print every second)
4. Are action values very small (< 0.01)? → Model not working

### Issue: Compatibility error

**Error:** Config field errors (like r2owb0/act1)
**Meaning:** Model trained with different LeRobot version
**Fix:** Try a different model (more recent ones more likely compatible)

---

## 📈 Comparison Table

| Aspect | Base SmolVLA | jhou/smolvla_pickplace | Your Fine-tuned (Path C) |
|--------|--------------|------------------------|--------------------------|
| **Training data** | SO-100 only | SO-101 pickplace | YOUR robot, YOUR setup |
| **Expected success** | <20% | 40-70%* | 70-90% |
| **Language conditioning** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Setup match** | ❌ Wrong robot | ⚠️ Similar setup | ✅ Perfect match |
| **Time to test** | 15 min | 15 min | 1-2 weeks |

*If camera/workspace is similar to training

---

## 🎯 Decision Tree After Testing

```
Test jhou/smolvla_pickplace
    ├─ Works well (>40%)
    │   └─ Use it! Proceed to Stage 3
    │
    ├─ Okay but not great (20-40%)
    │   ├─ Try HuggingFadeUser/my_smolvla
    │   ├─ Try saood65/my_smolvla
    │   └─ If all mediocre → Decide: use best one OR fine-tune
    │
    └─ Doesn't work (<20%)
        ├─ Try 1 more model (quick check)
        └─ Then → Path C (fine-tuning)
```

---

## 📝 Notes

### Why jhou/smolvla_pickplace?
- ✅ 42 likes (community validated)
- ✅ Trained on exact dataset: `lerobot/svla_so101_pickplace`
- ✅ Language conditioning (can give it tasks)
- ✅ Recent (likely compatible with current LeRobot)

### Dataset Details (svla_so101_pickplace)
- 50 episodes
- 11,939 frames
- SO-100 follower robot (but SO-101 compatible)
- Two cameras: "up" and "side" (both 480×640)
- Task: Pick and place

### Your Setup vs Training
**Similar:**
- ✅ SO-101 arm (compatible with SO-100)
- ✅ Pick-place task
- ✅ Two cameras

**Different:**
- ⚠️ You have 2 arms, dataset has 1
- ⚠️ Camera positions/angles likely different
- ⚠️ Workspace layout different
- ⚠️ Objects/positions different

**Impact:** 40-70% success if setup is "close enough"

---

## ⏱️ Time Estimates

| Activity | Time |
|----------|------|
| Setup & run first test | 15 min |
| Evaluate results | 5 min |
| Try 2nd model (if needed) | 15 min |
| Try 3rd model (if needed) | 15 min |
| **Total Path B** | **30-60 min** |

After this, you'll know:
- ✅ Whether any community model works for you
- ✅ Whether fine-tuning is necessary
- ✅ What "working" vs "not working" looks like

---

## 🚀 Quick Start Summary

```bash
# 1. Activate environment
conda activate lerobot

# 2. Power off one arm (or hide from cameras)

# 3. Run test
cd ~/project/XLeRobot
python jdocs/testing_basic_components/test_community_smolvla.py

# 4. Watch arm for 10 seconds

# 5. Evaluate: Better than base SmolVLA?
#    - Yes → Use it or try more for better
#    - No → Try 1-2 more models, then Path C

# 6. Make decision:
#    - Good model found → Stage 3
#    - No good model → Path C (fine-tuning)
```

**Good luck!** 🤖
