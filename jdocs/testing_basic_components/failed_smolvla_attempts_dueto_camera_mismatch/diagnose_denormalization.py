#!/usr/bin/env python3
"""
Denormalization Diagnostic Tool
================================

Purpose: Diagnose denormalization issue WITHOUT running the robot
This script verifies each component in isolation to identify the root cause

Run this FIRST before testing on the robot!

Usage:
    python diagnose_denormalization.py

Expected output:
    - Verification of model loading
    - Verification of dataset stats loading
    - Verification of postprocessor configuration
    - Test denormalization with dummy data
    - Clear diagnosis of what's broken

Author: System Diagnostic Tool
Date: 2025-11-15
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("="*70)
print(" Denormalization Diagnostic Tool ".center(70, "="))
print("="*70)
print("\nThis tool will verify each component without running the robot.")
print("It will identify exactly what's broken in the denormalization pipeline.\n")

# Track results for final summary
results = {
    "model_loaded": False,
    "dataset_loaded": False,
    "stats_loaded": False,
    "postprocessor_created": False,
    "postprocessor_has_stats": False,
    "denormalization_works": False,
}

issues_found = []


# ==============================================================================
# Test 1: Load SmolVLA Model
# ==============================================================================
print("="*70)
print("Test 1: Loading SmolVLA Model")
print("="*70)

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print("⏳ Loading model from Hugging Face Hub...")
    print("   (First run: downloads ~2-3GB, may take 3-5 minutes)")
    print("   (Subsequent runs: loads from cache, ~30 seconds)")

    model_id = "lerobot/smolvla_base"
    model = SmolVLAPolicy.from_pretrained(model_id)

    print(f"✅ Model loaded successfully: {model_id}")

    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    print(f"   Model config keys: {list(model.config.__dict__.keys())[:5]}...")

    # Check if model has normalization settings
    if hasattr(model.config, 'normalize_actions'):
        print(f"   normalize_actions: {model.config.normalize_actions}")
    if hasattr(model.config, 'action_dim'):
        print(f"   action_dim: {model.config.action_dim}")

    results["model_loaded"] = True

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n💡 Your LeRobot installation might be missing SmolVLA support.")
    print("   Try: cd /home/jrobot/project/lerobot && git pull && pip install -e .")
    issues_found.append("Model import failed - LeRobot may need update")
    sys.exit(1)

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"Model loading failed: {e}")
    sys.exit(1)


# ==============================================================================
# Test 2: Load Dataset for Stats
# ==============================================================================
print("\n" + "="*70)
print("Test 2: Loading Dataset for Normalization Stats")
print("="*70)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("⏳ Loading dataset from Hugging Face Hub...")
    print("   (First run: downloads dataset, may take 2-5 minutes)")
    print("   (Subsequent runs: loads from cache, fast)")

    dataset_id = "lerobot/svla_so101_pickplace"
    dataset = LeRobotDataset(dataset_id)

    print(f"✅ Dataset loaded successfully: {dataset_id}")

    # Try to get episode count (attribute name varies by version)
    try:
        if hasattr(dataset, 'episode_data_index'):
            print(f"   Total episodes: {len(dataset.episode_data_index)}")
        elif hasattr(dataset, 'episodes'):
            print(f"   Total episodes: {len(dataset.episodes)}")
        elif hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
            print(f"   Total episodes: {len(dataset.meta.episodes)}")
    except:
        print(f"   Total episodes: (unable to determine)")

    try:
        print(f"   Total frames: {len(dataset)}")
    except:
        print(f"   Total frames: (unable to determine)")

    results["dataset_loaded"] = True

    # Check if stats exist
    print("\n🔍 Checking dataset stats...")
    if dataset.meta.stats is None:
        print("❌ dataset.meta.stats is None!")
        issues_found.append("Dataset stats is None - cannot denormalize!")
        results["stats_loaded"] = False
    else:
        print("✅ dataset.meta.stats exists")
        print(f"   Stats keys: {list(dataset.meta.stats.keys())}")

        # Check action stats specifically
        if 'action' in dataset.meta.stats:
            print("\n📊 Action statistics:")
            action_stats = dataset.meta.stats['action']
            print(f"   Keys: {list(action_stats.keys())}")

            if 'mean' in action_stats:
                mean = action_stats['mean']
                print(f"   mean: {mean}")
                print(f"   mean type: {type(mean)}")
                print(f"   mean length: {len(mean) if hasattr(mean, '__len__') else 'N/A'}")

            if 'std' in action_stats:
                std = action_stats['std']
                print(f"   std: {std}")
                print(f"   std type: {type(std)}")
                print(f"   std length: {len(std) if hasattr(std, '__len__') else 'N/A'}")

            if 'min' in action_stats:
                print(f"   min: {action_stats['min']}")

            if 'max' in action_stats:
                print(f"   max: {action_stats['max']}")

            results["stats_loaded"] = True
        else:
            print("❌ 'action' key not found in stats!")
            issues_found.append("Action stats missing from dataset")
            results["stats_loaded"] = False

except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"Dataset loading failed: {e}")
    sys.exit(1)


# ==============================================================================
# Test 3: Create Postprocessor with Stats
# ==============================================================================
print("\n" + "="*70)
print("Test 3: Creating Postprocessor with Stats")
print("="*70)

try:
    from lerobot.policies.factory import make_pre_post_processors

    print("⏳ Creating preprocessor and postprocessor...")
    print("   Using dataset_stats parameter...")

    # Method 1: Using dataset_stats parameter only
    print("\n🔬 Method 1: dataset_stats parameter only")
    try:
        preprocess_1, postprocess_1 = make_pre_post_processors(
            model.config,
            model_id,
            dataset_stats=dataset.meta.stats,
        )
        print("   ✅ Created successfully with dataset_stats")
        results["postprocessor_created"] = True
    except Exception as e:
        print(f"   ⚠️  Failed: {e}")
        preprocess_1, postprocess_1 = None, None

    # Method 2: Using postprocessor_overrides
    print("\n🔬 Method 2: With postprocessor_overrides")
    try:
        preprocess_2, postprocess_2 = make_pre_post_processors(
            model.config,
            model_id,
            dataset_stats=dataset.meta.stats,
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": dataset.meta.stats
                }
            },
        )
        print("   ✅ Created successfully with overrides")
        results["postprocessor_created"] = True
    except Exception as e:
        print(f"   ⚠️  Failed: {e}")
        preprocess_2, postprocess_2 = None, None

    # Choose which one to use for testing
    if postprocess_2 is not None:
        postprocess = postprocess_2
        print("\n   Using Method 2 (with overrides) for further testing")
    elif postprocess_1 is not None:
        postprocess = postprocess_1
        print("\n   Using Method 1 (dataset_stats only) for further testing")
    else:
        print("\n❌ Both methods failed to create postprocessor!")
        issues_found.append("Postprocessor creation failed")
        sys.exit(1)

except Exception as e:
    print(f"❌ Postprocessor creation failed: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"Postprocessor creation failed: {e}")
    sys.exit(1)


# ==============================================================================
# Test 4: Inspect Postprocessor Internals
# ==============================================================================
print("\n" + "="*70)
print("Test 4: Inspecting Postprocessor Internals")
print("="*70)

print("🔍 Postprocessor pipeline steps:")
print(f"   Type: {type(postprocess)}")

if hasattr(postprocess, 'steps'):
    print(f"   Number of steps: {len(postprocess.steps)}")

    unnormalizer_found = False

    for i, step in enumerate(postprocess.steps):
        step_name = step.__class__.__name__
        print(f"\n   Step {i}: {step_name}")

        # Check for stats attribute
        if hasattr(step, 'stats'):
            if step.stats is not None:
                print(f"     ✅ Has stats attribute")
                print(f"     Stats keys: {list(step.stats.keys())}")

                if 'action' in step.stats:
                    print(f"     ✅ Has 'action' in stats")
                    print(f"        action.mean: {step.stats['action']['mean']}")
                    print(f"        action.std: {step.stats['action']['std']}")

                    if 'Unnormalizer' in step_name:
                        unnormalizer_found = True
                        results["postprocessor_has_stats"] = True
                else:
                    print(f"     ⚠️  'action' key missing from stats")
            else:
                print(f"     ❌ stats attribute is None!")
                if 'Unnormalizer' in step_name:
                    issues_found.append("UnnormalizerProcessor has stats=None")
        else:
            print(f"     ⚠️  No stats attribute")

        # Check for other relevant attributes
        if hasattr(step, 'mode'):
            print(f"     mode: {step.mode}")

        if hasattr(step, 'features'):
            print(f"     features: {step.features}")

    if unnormalizer_found:
        print("\n✅ UnnormalizerProcessor found with stats loaded!")
    else:
        print("\n⚠️  UnnormalizerProcessor not found or missing stats")
        issues_found.append("UnnormalizerProcessor missing or has no stats")

else:
    print("⚠️  Postprocessor has no 'steps' attribute")
    print(f"   Postprocessor type: {type(postprocess)}")
    issues_found.append("Postprocessor structure unexpected")


# ==============================================================================
# Test 5: Test Denormalization with Dummy Data
# ==============================================================================
print("\n" + "="*70)
print("Test 5: Testing Denormalization with Dummy Data")
print("="*70)

print("🧪 Creating dummy normalized action...")
# Typical normalized action values (mean~0, std~1, range -3 to +3)
dummy_normalized = torch.tensor([[0.5, -1.2, 0.8, 0.0, -0.5, 1.0]], dtype=torch.float32)
print(f"   Input (normalized): {dummy_normalized.numpy().tolist()[0]}")
print(f"   Shape: {dummy_normalized.shape}")
print(f"   Range: [{dummy_normalized.min():.2f}, {dummy_normalized.max():.2f}]")

try:
    print("\n⏳ Running postprocessor...")

    # Postprocessor expects dict input
    dummy_input = {"action": dummy_normalized}
    dummy_output = postprocess(dummy_input)

    print(f"   Output type: {type(dummy_output)}")

    if isinstance(dummy_output, dict) and 'action' in dummy_output:
        dummy_denormalized = dummy_output['action']

        if torch.is_tensor(dummy_denormalized):
            print(f"   Output (denormalized): {dummy_denormalized.numpy().tolist()[0]}")
            print(f"   Shape: {dummy_denormalized.shape}")
            print(f"   Range: [{dummy_denormalized.min():.2f}, {dummy_denormalized.max():.2f}]")

            # Check if values changed
            if torch.allclose(dummy_normalized, dummy_denormalized, rtol=1e-3):
                print("\n❌ FAILED: Values unchanged!")
                print("   Denormalization is NOT working!")
                print("   Input and output are essentially identical.")
                issues_found.append("Denormalization failed - values unchanged")
                results["denormalization_works"] = False
            else:
                print("\n✅ SUCCESS: Values changed!")
                print("   Denormalization is working!")

                # Calculate scale ratios
                scale_ratios = dummy_denormalized / (dummy_normalized + 1e-8)
                print(f"   Scale ratios: {scale_ratios.numpy().tolist()[0]}")

                # Check if ratios match expected std values
                if results["stats_loaded"]:
                    expected_std = dataset.meta.stats['action']['std']
                    print(f"   Expected std: {expected_std}")

                results["denormalization_works"] = True
        else:
            print(f"   ⚠️  Output is not a tensor: {type(dummy_denormalized)}")
    else:
        print(f"   ⚠️  Unexpected output format: {dummy_output}")
        issues_found.append("Postprocessor output format unexpected")

except Exception as e:
    print(f"\n❌ Denormalization test failed: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"Denormalization test error: {e}")


# ==============================================================================
# Test 6: Manual Denormalization for Comparison
# ==============================================================================
print("\n" + "="*70)
print("Test 6: Manual Denormalization (Ground Truth)")
print("="*70)

if results["stats_loaded"]:
    print("🧮 Computing manual denormalization using stats...")

    mean_list = dataset.meta.stats['action']['mean']
    std_list = dataset.meta.stats['action']['std']

    mean = torch.tensor(mean_list, dtype=torch.float32)
    std = torch.tensor(std_list, dtype=torch.float32)

    print(f"   Using mean: {mean.numpy().tolist()}")
    print(f"   Using std: {std.numpy().tolist()}")

    manual_denorm = dummy_normalized * std.unsqueeze(0) + mean.unsqueeze(0)
    print(f"\n   Manual denormalized: {manual_denorm.numpy().tolist()[0]}")
    print(f"   Range: [{manual_denorm.min():.2f}, {manual_denorm.max():.2f}]")

    print("\n📊 This is what the robot SHOULD receive!")
    print("   These values should be in radians (or degrees if use_degrees=True)")
    print("   Typical range for radians: -π to +π (-3.14 to +3.14)")
    print("   Typical range for degrees: -180° to +180°")

    # Check if values are reasonable for joint angles
    if manual_denorm.abs().max() > 10:
        print("\n⚠️  WARNING: Values exceed ±10, might be degrees")
        print("   Check if robot config uses degrees vs radians")
    elif manual_denorm.abs().max() > 3.14:
        print("\n⚠️  WARNING: Values exceed ±π, might exceed joint limits")
else:
    print("⚠️  Skipping manual denormalization (stats not loaded)")


# ==============================================================================
# Test 7: Check Units (Degrees vs Radians)
# ==============================================================================
print("\n" + "="*70)
print("Test 7: Checking Units (Degrees vs Radians)")
print("="*70)

print("🔍 Checking dataset metadata for units...")

if hasattr(dataset.meta, 'info'):
    info = dataset.meta.info
    print(f"   Dataset info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")

    # Look for units information
    units_found = False
    if isinstance(info, dict):
        for key in ['units', 'action_units', 'use_degrees']:
            if key in info:
                print(f"   ✅ {key}: {info[key]}")
                units_found = True

    if not units_found:
        print("   ⚠️  No explicit units information found in dataset metadata")
        print("   You'll need to check dataset documentation or infer from value ranges")
else:
    print("   ⚠️  dataset.meta.info not available")

print("\n💡 Remember to check your robot config:")
print("   In test_smolvla_official.py line 165:")
print("   SO101FollowerConfig(..., use_degrees=True/False)")
print("   This MUST match the dataset units!")


# ==============================================================================
# Final Summary
# ==============================================================================
print("\n" + "="*70)
print(" Diagnostic Summary ".center(70, "="))
print("="*70)

print("\n✅ Tests Passed:")
for test_name, passed in results.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {test_name.replace('_', ' ').title()}")

print("\n" + "="*70)

if len(issues_found) == 0 and all(results.values()):
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\n✅ Your system appears to be configured correctly!")
    print("\nNext steps:")
    print("1. The denormalization should work on the robot")
    print("2. Run test_smolvla_official.py to test on real hardware")
    print("3. If robot still makes tiny movements, check:")
    print("   - Units mismatch (degrees vs radians)")
    print("   - Robot calibration")
    print("   - Camera domain mismatch")

elif not results["denormalization_works"]:
    print("🔴 CRITICAL ISSUE FOUND!")
    print("="*70)
    print("\n❌ Denormalization is NOT working!")
    print("\nRoot cause: Stats are not being used by the postprocessor")
    print("\nPossible fixes:")
    print("1. Verify the postprocessor_overrides syntax in your script")
    print("2. Check if your LeRobot version supports stats overrides")
    print("3. Try manual denormalization as a workaround:")
    print("\n   # After model.select_action:")
    print("   mean = torch.tensor(dataset.meta.stats['action']['mean'])")
    print("   std = torch.tensor(dataset.meta.stats['action']['std'])")
    print("   action = action * std + mean")

else:
    print("⚠️  ISSUES FOUND")
    print("="*70)
    print("\nIssues detected:")
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue}")

print("\n" + "="*70)
print("Diagnostic complete!")
print("="*70)

# Save results to file
output_file = Path(__file__).parent / "diagnostic_results.txt"
with open(output_file, 'w') as f:
    f.write("Denormalization Diagnostic Results\n")
    f.write("="*70 + "\n\n")

    f.write("Test Results:\n")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        f.write(f"  {status}: {test_name}\n")

    f.write("\nIssues Found:\n")
    if issues_found:
        for issue in issues_found:
            f.write(f"  - {issue}\n")
    else:
        f.write("  None\n")

    if results["stats_loaded"]:
        f.write("\nDataset Stats:\n")
        f.write(f"  mean: {dataset.meta.stats['action']['mean']}\n")
        f.write(f"  std: {dataset.meta.stats['action']['std']}\n")

print(f"\n📄 Results saved to: {output_file}")
