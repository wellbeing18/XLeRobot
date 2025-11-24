# Postprocessor Override Fix - Critical!

## The Problem You Encountered

Even after loading your dataset stats, actions were STILL normalized:
```
[STEP 0] ⚡ Action to send: {
    'shoulder_pan.pos': -0.17376123368740082,  # ← Still normalized!
    'shoulder_lift.pos': -1.1349276304244995,
    ...
}
```

And the arm didn't move because these values are so small (less than 1 degree).

---

## Root Cause: Empty Postprocessor Features

The pretrained pi0.5 model ships with a `policy_postprocessor.json` file:

```json
{
  "steps": [
    {
      "registry_name": "unnormalizer_processor",
      "config": {
        "eps": 1e-08,
        "features": {},  // ← EMPTY!
        "norm_map": {
          "VISUAL": "IDENTITY",
          "STATE": "QUANTILES",
          "ACTION": "QUANTILES"
        }
      }
    }
  ]
}
```

When you call `make_pre_post_processors()` with a `pretrained_path`, it:
1. Loads the saved postprocessor config from HuggingFace
2. Creates a postprocessor with EMPTY features
3. **Ignores** the `dataset_stats` parameter!

### Why Does This Happen?

Looking at `lerobot/policies/factory.py` line 230-249:

```python
if pretrained_path:
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename="policy_postprocessor.json",
            overrides=kwargs.get("postprocessor_overrides", {}),  # ← Only uses overrides!
            ...
        ),
    )
```

The `dataset_stats` parameter is passed in `kwargs`, but `from_pretrained()` only uses `postprocessor_overrides`, NOT `dataset_stats` directly.

---

## The Fix: Explicit Overrides

We need to explicitly override the empty features, similar to how Groot does it (lines 210-228 in factory.py).

### Before (BROKEN):
```python
preprocess, postprocess = make_pre_post_processors(
    model.config,
    MODEL_ID,
    dataset_stats=dataset_stats,  # ← Passed but IGNORED!
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
```

### After (FIXED):
```python
preprocess, postprocess = make_pre_post_processors(
    model.config,
    MODEL_ID,
    dataset_stats=dataset_stats,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "normalizer_processor": {"features": dataset_stats},  # ← Explicit override!
    },
    postprocessor_overrides={
        "unnormalizer_processor": {"features": dataset_stats},  # ← Explicit override!
    },
)
```

Now the postprocessor will have YOUR robot's action stats and can properly unnormalize!

---

## Expected Behavior After Fix

### Before:
```
[STEP 0] ⚡ Action to send: {
    'shoulder_pan.pos': -0.17,  # ← Normalized (barely moves)
    'shoulder_lift.pos': -1.13,
    'elbow_flex.pos': 1.03,
    ...
}
```

### After:
```
[STEP 0] ⚡ Action to send: {
    'shoulder_pan.pos': -5.42,  # ← Degrees (actual movement!)
    'shoulder_lift.pos': -45.23,
    'elbow_flex.pos': 67.89,
    ...
}
```

The arm should now move with realistic degree values!

---

## About the Frequency

You noticed the frequency was 18.6Hz avg, but this is misleading:
- **Step 0**: 1787ms (first step includes warmup)
- **Steps 10-40**: ~200ms each = **5Hz** ✅

After the first step, the control loop runs at ~5Hz, which matches your dataset's FPS. The 15fps cameras are fine - they're faster than the control loop so you always get fresh frames.

---

## About the Frozen Arm State

You noticed the arm state was identical at steps 10, 20, 30, 40:
```
shoulder_pan.pos: -0.17582417582417584
shoulder_lift.pos: -0.7912087912087912
```

This happened because:
1. Actions were normalized (< 1 degree)
2. SO-ARM101 servos have a dead zone for tiny movements
3. The arm physically didn't move

Once unnormalization is fixed, the arm should actually move and the state should change between steps.

---

## Summary

✅ **Fixed calibration file**: Changed ID from `follower_left_so101` → `xlerobot_left_arm`
✅ **Added dataset loading**: Load stats from your recorded dataset
✅ **Fixed postprocessor override**: Explicitly inject stats into unnormalizer
⏳ **Ready to test**: Actions should now be in degrees and arm should move!

The pretrained pi0.5 model still won't complete the task (expected 0-15% baseline), but at least it will MOVE properly now instead of staying frozen!
