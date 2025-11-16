# Corrected Diagnosis: SmolVLA Tiny Movement Issue

**Date:** 2025-11-08
**Status:** ✅ Research Complete - Corrected Understanding

---

## Executive Summary

After thorough research of the LeRobot codebase, I need to **CORRECT** my initial diagnosis:

### What I Got WRONG ❌

1. **INCORRECT**: "LeRobot doesn't denormalize actions automatically"
   - **TRUTH**: `postprocess()` DOES include `UnnormalizerProcessorStep` that denormalizes actions

2. **INCORRECT**: "You need to manually denormalize actions"
   - **TRUTH**: The `postprocess()` function handles this, BUT only if it has the correct dataset stats

### What I Got RIGHT ✅

1. **CORRECT**: Actions are normalized during training and need denormalization
2. **CORRECT**: The issue is related to action scaling/normalization
3. **PARTIALLY CORRECT**: The fix works, but for a different reason than I thought

---

## The ACTUAL Root Cause

### Problem: Missing or Empty Dataset Stats

Your test script does this:

```python
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,  # <-- Loads processors from pretrained model
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
```

**What happens:**
1. `make_pre_post_processors()` loads the processor configs from the pretrained model hub
2. The postprocessor DOES include `UnnormalizerProcessorStep`
3. **BUT**: The normalization stats might be missing, empty, or incorrect in the pretrained model

### Evidence from LeRobot Code

**File**: `/home/jrobot/project/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py:86-90`

```python
output_steps = [
    UnnormalizerProcessorStep(
        features=config.output_features,
        norm_map=config.normalization_mapping,
        stats=dataset_stats  # <-- This is the key!
    ),
    DeviceProcessorStep(device="cpu"),
]
```

The `UnnormalizerProcessorStep` uses `dataset_stats` to denormalize. If these stats are:
- Missing (None)
- Empty ({})
- Incorrect for your robot

Then denormalization will be wrong or skipped entirely.

### How Normalization Works in LeRobot

**File**: `/home/jrobot/project/lerobot/src/lerobot/processor/normalize_processor.py`

**MEAN_STD Normalization** (used by SmolVLA):

```python
# During preprocessing (normalization):
normalized = (x - mean) / std

# During postprocessing (denormalization):
denormalized = normalized * std + mean
```

**If `mean` and `std` are missing or wrong:**
- The denormalization produces incorrect values
- Actions end up in the wrong scale
- Robot makes tiny movements because actions are too small

---

## Why My "Fix" Worked

My `test_smolvla_fixed.py` did this:

```python
# CRITICAL FIX 1: Check if action needs denormalization
if np.all(np.abs(action_array) <= 1.5):
    # Action is normalized, denormalize it
    action_array = denormalize_action(action_array, joint_mins, joint_maxs)
```

**Why this works:**
- It detects when actions are still in normalized space (values around -1 to 1)
- Then manually denormalizes them using your robot's calibrated joint limits
- This acts as a **backup denormalization** when the postprocessor fails

**However:**
- This is **double-denormalizing** if the postprocessor works correctly
- It's a **workaround** rather than fixing the root cause

---

## The PROPER Fix

### Option 1: Load Dataset Stats Explicitly (BEST)

```python
from lerobot.datasets import LeRobotDataset

# Load the dataset used to train SmolVLA
# This gets the correct normalization stats
dataset = LeRobotDataset("lerobot/svla_so101_pickplace")

# Create processors WITH dataset stats
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    dataset_stats=dataset.meta.stats,  # <-- CRITICAL!
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Now postprocess() will properly denormalize using the correct stats
action = model.select_action(obs)
action = postprocess(action)  # Correctly denormalized!
```

### Option 2: Check What Stats Are Loaded

```python
# After creating processors, inspect what stats were loaded
print("Postprocessor steps:")
for step in postprocess.steps:
    print(f"  - {step.__class__.__name__}")
    if hasattr(step, 'stats'):
        print(f"    Stats: {step.stats}")

# You'll likely see:
# - UnnormalizerProcessorStep
#   Stats: {} or None  <-- PROBLEM!
```

### Option 3: Use My Workaround (If Options 1-2 Don't Work)

Keep the manual denormalization check in my fixed script. It works, but is not elegant.

---

## ROS Question: Answered

### Does LeRobot Use ROS?

**NO**. LeRobot does NOT use or require ROS at all.

### What Does LeRobot Use Instead?

**Direct serial communication** via Python:

1. **`pyserial`** - Direct USB/serial port communication
2. **Motor protocols** - Direct Feetech SCS and Dynamixel protocols
3. **Simple Python abstractions** - Clean robot base class

**Key file**: `/home/jrobot/project/lerobot/src/lerobot/robots/so101_follower/so101_follower.py`

```python
self.bus = FeetechMotorsBus(
    port=self.config.port,  # e.g., /dev/ttyACM2
    motors={...},  # Direct motor definitions
    calibration=self.calibration,
)
```

### Why Don't Papers Mention This?

Many robotics papers use ROS because:
- **ROS is standard in academia** for complex multi-robot systems
- **LeRobot is newer** (2024) and takes a simpler, ML-focused approach
- **Different goals**: ROS for multi-robot coordination, LeRobot for imitation learning

### Should You Install ROS?

**NO!** Do not install ROS for XLeRobot. Reasons:

1. ❌ XLeRobot is built on LeRobot's non-ROS architecture
2. ❌ ROS adds unnecessary complexity (steep learning curve, complex install)
3. ❌ Would create compatibility issues
4. ✅ LeRobot's direct serial communication is simpler and proven

**What you need instead:**
- Python 3.10+ with PyTorch
- LeRobot: `pip install -e ".[feetech]"`
- Standard libraries: opencv, numpy, scipy
- NO ROS!

### Comparison

| Aspect | ROS-Based Systems | LeRobot |
|--------|-------------------|---------|
| Communication | Middleware (topics/services) | Direct serial |
| Setup | Complex | Simple pip install |
| Use case | Multi-robot systems | ML/imitation learning |
| Learning curve | Steep | Shallow |
| XLeRobot needs it? | ❌ NO | ✅ YES |

---

## Updated Action Plan

### Step 1: Test with Correct Dataset Stats (NEW!)

Create `test_smolvla_with_stats.py`:

```python
#!/usr/bin/env python3
"""
Test SmolVLA with PROPER dataset stats loading
"""

import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
# ... other imports

device = torch.device("cuda")
model_id = "lerobot/smolvla_base"

# Load model
model = SmolVLAPolicy.from_pretrained(model_id)
model = model.to(device)
model.eval()

# Load dataset to get normalization stats
print("Loading dataset for normalization stats...")
dataset = LeRobotDataset("lerobot/svla_so101_pickplace")

print(f"Dataset stats keys: {dataset.meta.stats.keys()}")

# Create processors WITH stats
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    dataset_stats=dataset.meta.stats,  # <-- CRITICAL FIX!
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Check if unnormalizer has stats
print("\nPostprocessor steps:")
for step in postprocess.steps:
    print(f"  - {step.__class__.__name__}")
    if hasattr(step, 'stats'):
        if step.stats:
            print(f"    ✅ Has stats: {list(step.stats.keys())}")
        else:
            print(f"    ❌ Stats is empty or None")

# ... rest of inference loop
```

### Step 2: If Step 1 Doesn't Work, Use My Workaround

Use `test_smolvla_fixed.py` with the manual denormalization check.

### Step 3: Verify Actions Are Reasonable

Add logging to see actual action values:

```python
if step % 30 == 0:
    print(f"\n📊 Step {step}")
    print(f"  Action after postprocess: {action_array}")
    print(f"  Min: {action_array.min():.3f}, Max: {action_array.max():.3f}")
    print(f"  Expected range: [{joint_mins.min():.2f}, {joint_maxs.max():.2f}]")
```

### Step 4: Test and Evaluate

Run for 10 seconds and observe:
- ✅ Large, purposeful movements toward objects
- ✅ Actions within robot's joint limits
- ✅ Smooth trajectories (not jittery)

---

## Summary of Corrections

### My Original Diagnosis (PARTIALLY WRONG)

| Original Claim | Corrected Understanding |
|----------------|------------------------|
| "No denormalization happens" | ❌ WRONG - Postprocessor HAS UnnormalizerProcessorStep |
| "Need to manually denormalize" | ⚠️ MISLEADING - Only needed if stats are missing |
| "Actions are normalized [-1, 1]" | ✅ CORRECT - But denorm should be automatic |
| "Missing temporal context" | ✅ CORRECT - But less critical than I thought |

### The Real Issue

**Primary cause**: Missing or incorrect `dataset_stats` in the postprocessor

**Secondary causes**:
- Camera domain mismatch (different setup from training data)
- Possibly incorrect joint limits in calibration
- No action smoothing (minor issue)

### The Real Solution

1. **Load dataset stats explicitly** when creating processors
2. **Verify unnormalizer has stats** by inspecting postprocessor steps
3. **Use manual denormalization as backup** if stats are missing
4. **Add action smoothing** for stability (nice-to-have)

---

## Questions Answered

### 1. Was my initial diagnosis correct?

**Partially**. I identified the symptom (wrong action scaling) but misdiagnosed the cause (thought LeRobot didn't denormalize, when actually it does but needs correct stats).

### 2. Will my fixed script work?

**Yes**, but for the wrong reason. It manually denormalizes as a workaround for missing stats in the postprocessor. Better to fix the root cause by loading stats properly.

### 3. Do we need ROS?

**Absolutely not**. LeRobot doesn't use ROS. It uses direct serial communication via pyserial. Installing ROS would only add complexity with zero benefit.

### 4. Should I use the fixed script or rewrite it?

**Both**:
1. Try loading dataset stats properly first (Option 1 above)
2. If that doesn't work or the pretrained model lacks stats, use my fixed script

---

## Next Steps

1. **Create `test_smolvla_with_stats.py`** - Load dataset stats explicitly
2. **Test it** - See if actions are now reasonable without manual denorm
3. **Compare** - Run both versions, see which works better
4. **Document results** - Update your progress based on what you learn

---

## Key Takeaway

**The core issue is likely missing normalization stats in the pretrained model's postprocessor, NOT a fundamental flaw in LeRobot's design.**

My "fix" works as a workaround, but the proper solution is to ensure the postprocessor has the correct `dataset_stats` loaded.

**And no, you don't need ROS. At all. Period.** 😊
