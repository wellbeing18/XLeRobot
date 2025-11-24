# Pi0.5 Config Override Guide - CRITICAL for SO-101 Integration

## The Core Problem

Pi0.5 is a **multi-embodiment model** trained on multiple robots with different action dimensions. When loading pretrained pi0.5 for a NEW robot (like SO-101) that wasn't in the training data, you MUST override the config to specify your robot's dimensions.

### Without Config Override (BROKEN):
```python
model = PI05Policy.from_pretrained("lerobot/pi05_base")
# ❌ Loads with default config (unknown action dimensions)
# ❌ Results in dimension mismatch: 32-dim model vs 7-dim robot
# ❌ Error: "The size of tensor a (32) must match the size of tensor b (7)"
```

### With Config Override (CORRECT):
```python
config = PI05Config.from_pretrained("lerobot/pi05_base")
config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
config.input_features = {
    "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    "observation.images.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
}
config.validate_features()
model = PI05Policy.from_pretrained("lerobot/pi05_base", config=config, strict=False)
# ✅ Model knows to expect/output 7-dim actions
# ✅ Padding system handles 7 → 32 → 7 conversion internally
```

---

## How Pi0.5's Padding System Works

Pi0.5 uses a **universal 32-dimensional action space** to support any robot:

### Internal Model Flow:

```
Your Robot (7-dim) → Padding (7→32) → Model Processing (32-dim) → Unpadding (32→7) → Your Robot (7-dim)
```

### In Code:

**During Preprocessing (Input):**
```python
# From modeling_pi05.py line 1106
def prepare_action(self, batch):
    """Pad action to max_action_dim"""
    actions = pad_vector(batch[ACTION], self.config.max_action_dim)
    # Your 7-dim action becomes 32-dim (padded with zeros)
    return actions
```

**During Postprocessing (Output):**
```python
# From modeling_pi05.py lines 1134-1136
def predict_action_chunk(self, batch):
    actions = self.model.sample_actions(...)  # Model outputs 32-dim

    # Unpad actions to actual action dimension from config
    original_action_dim = self.config.output_features[ACTION].shape[0]  # 7 for SO-101
    actions = actions[:, :, :original_action_dim]  # Slice: 32 → 7
    return actions
```

**Key Insight:** The padding/unpadding is AUTOMATIC, but the model needs to know YOUR robot's dimensions via `config.output_features`.

---

## SO-101 Robot Specifications

### Action Space (7 dimensions):
1. `shoulder_pan.pos` (degrees)
2. `shoulder_lift.pos` (degrees)
3. `elbow_flex.pos` (degrees)
4. `wrist_flex.pos` (degrees)
5. `wrist_roll.pos` (degrees)
6. `gripper.pos` (degrees)

**Total: 6 DOF joints + 1 gripper = 7-dim action**

### Observation Space:
- **Visual:** 2 cameras (base_0_rgb, left_wrist_0_rgb) at 224x224 (pi0.5 resizes)
- **State:** 7-dim (6 joint positions + gripper state)

---

## Step-by-Step Integration for SO-101

### 1. Load Base Config
```python
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import PolicyFeature, FeatureType

config = PI05Config.from_pretrained("lerobot/pi05_base")
```

### 2. Override for SO-101
```python
# Define SO-101 action space
config.output_features = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(7,),  # 6 joints + gripper
    )
}

# Define SO-101 observation space
config.input_features = {
    "observation.images.base_0_rgb": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),  # Pi0.5 standard size
    ),
    "observation.images.left_wrist_0_rgb": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),
    ),
    "observation.state": PolicyFeature(
        type=FeatureType.STATE,
        shape=(7,),  # Joint positions + gripper
    ),
}
```

### 3. Validate Config
```python
config.validate_features()
# Creates placeholders for any missing features
# Ensures config is valid before loading model
```

### 4. Load Model with Config
```python
model = PI05Policy.from_pretrained(
    "lerobot/pi05_base",
    config=config,
    strict=False  # Allow dimension mismatch (action head will be adapted)
)
model = model.to(device)
model.eval()
```

### 5. Create Preprocessors (No Dataset Stats)
```python
from lerobot.policies.factory import make_pre_post_processors

preprocess, postprocess = make_pre_post_processors(
    model.config,
    "lerobot/pi05_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
    # No dataset_stats for pretrained baseline
)
```

---

## Why This Works

### Pretrained Baseline Testing:
- **Vision/Language weights:** Transferable from pretraining (general visual understanding)
- **Action head:** May output suboptimal actions (not trained on SO-101 kinematics)
- **Expected success:** 0-15% (normal for new robot + egocentric cameras)
- **Purpose:** Establish baseline before finetuning

### After Finetuning:
- **Vision/Language weights:** Adapted to egocentric cameras
- **Action head:** Learned SO-101 kinematics from your data
- **Expected success:** 40-60% with 50-100 good episodes
- **Config:** Automatically set from dataset during training

---

## Common Mistakes & Fixes

### ❌ Mistake 1: Loading Without Config Override
```python
model = PI05Policy.from_pretrained("lerobot/pi05_base")
```
**Error:** Dimension mismatch (32 vs 7)
**Fix:** Use config override as shown above

### ❌ Mistake 2: Wrong Action Dimensions
```python
config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
```
**Error:** Feature count mismatch (forgot gripper)
**Fix:** Use shape=(7,) for SO-101 (6 joints + gripper)

### ❌ Mistake 3: Trying to Unnormalize with Dataset Stats
```python
preprocess, postprocess = make_pre_post_processors(
    model.config,
    "lerobot/pi05_base",
    dataset_stats=your_dataset_stats,  # ❌ Dimension mismatch!
)
```
**Error:** Can't map 7-dim stats to pretrained model
**Fix:** Don't use dataset_stats for pretrained baseline

### ❌ Mistake 4: Manual Slicing
```python
action = postprocess(action)
action = action[..., :7]  # ❌ Unnecessary! Config handles this
```
**Fix:** Remove manual slicing - config + padding system handles it

---

## Verification Checklist

After implementing config override, verify:

✅ Model loads without errors
✅ Config shows `output_features["action"].shape = (7,)`
✅ Config shows correct input_features (2 cameras + state)
✅ Preprocessors create successfully
✅ Actions are 7-dimensional after postprocessing
✅ Actions are in reasonable value ranges (degrees)

---

## Comparison: Pretrained vs Finetuned Workflow

### Pretrained Inference (Your Current Goal):
```python
# Manual config override required
config = PI05Config.from_pretrained("lerobot/pi05_base")
config.output_features = {"action": PolicyFeature(..., shape=(7,))}
config.input_features = {...}  # Manually define
model = PI05Policy.from_pretrained("lerobot/pi05_base", config=config)
```

### Finetuned Inference (After Training):
```python
# Config automatically saved with checkpoint
model = PI05Policy.from_pretrained("./outputs/pi05_so101/checkpoint-6000")
# ✅ No config override needed!
# ✅ Config with action_dim=7 already baked in
```

---

## Next Steps

### Phase 1: Pretrained Baseline (Now)
1. ✅ Use config override in inference script
2. ✅ Test pretrained pi0.5 on SO-101
3. ✅ Expect 0-15% success (establish baseline)

### Phase 2: Finetuning (Week 2)
1. Collect 50-100 SO-101 episodes
2. Run finetuning: `lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base`
3. Lerobot automatically extracts action_dim=7 from dataset
4. Model learns SO-101 kinematics + egocentric camera views

### Phase 3: Finetuned Inference
1. Load finetuned checkpoint (no config override needed!)
2. Test on same task
3. Expect 40-60% success rate

---

## Key Takeaways

1. **Config override is MANDATORY** for pretrained inference on new robots
2. **Padding system is automatic** - just specify your dimensions
3. **7-dim vs 32-dim is NOT a bug** - it's the multi-embodiment design
4. **Pretrained baseline will be poor** (0-15%) - that's expected
5. **Finetuning handles everything automatically** - no manual config needed

---

## Summary

The dimension mismatch issue was NOT a bug - it was a missing configuration step. Pi0.5's padding system is designed to support ANY robot by:

1. Padding your actions to 32-dim (universal space)
2. Processing in 32-dim (model's internal representation)
3. Unpadding back to your dimensions (7-dim for SO-101)

You just need to tell the model YOUR dimensions via config override, and the padding system handles the rest automatically!
