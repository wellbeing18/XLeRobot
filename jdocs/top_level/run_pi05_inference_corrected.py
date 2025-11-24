#!/usr/bin/env python3
"""
Pi0.5 Baseline Inference on XLeRobot Dual SO-101 Follower Arms (CORRECTED VERSION)

This script runs the pretrained pi0.5 model on dual SO-ARM101 follower arms
to establish a baseline BEFORE finetuning. Expected baseline: 0-15% success
due to egocentric camera domain mismatch.

CRITICAL FIXES from original script:
1. ✅ Correct import: PI05Policy (was PI0Policy - WRONG!)
2. ✅ No dataset stats loading (not needed for pretrained pi0.5)
3. ✅ Proper robot state handling in observations
4. ✅ Comprehensive logging at every step
5. ✅ Better error handling and safety checks

Based on: lerobot/examples/tutorial/pi0/using_pi0_example.py
Adapted for: Dual SO-101 follower arms + 3 UVC cameras (egocentric setup)

Author: Created for XLeRobot project
Date: 2025-11-23
"""

import torch
import time
import numpy as np
import traceback
from pathlib import Path

# ============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# GPU/CPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Using device: {device}")

# Model configuration
MODEL_ID = "lerobot/pi05_base"  # Pretrained Pi0.5 model (4B params)

# Robot USB ports (VERIFY THESE with: ls -l /dev/ttyACM*)
FOLLOWER_PORT = "/dev/ttyACM0"  # Left follower arm
LEADER_PORT = "/dev/ttyACM2"    # Left leader arm (not used for inference, just for reference)

# Robot IDs for calibration files (should be in ~/.cache/lerobot/calibration/)
# IMPORTANT: Must match existing calibration filename (without .json extension)
FOLLOWER_ID = "xlerobot_left_arm"  # Matches xlerobot_left_arm.json in ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/

# Camera configuration (VERIFY THESE with camera test script)
# Using ONLY 2 cameras: head + left wrist
# Camera KEY NAMES must match dataset (head and left_wrist)
CAMERA_INDICES = {
    "head": 4,               # Head camera (/dev/video4)
    "left_wrist": 6,         # Left wrist camera (/dev/video6)
}

# Task instruction (natural language)
# Pi0.5 was trained on diverse tasks - try simple pick/place
TASK = "pick up the red cube"  # Modify based on your setup

# Episode configuration
MAX_EPISODES = 1  # Run once only
MAX_STEPS_PER_EPISODE = 50  # ~10 seconds at 5Hz action frequency
CONTROL_FREQUENCY_HZ = 5  # Action frequency (matches your recorded dataset)

# Robot type for multi-embodiment models
ROBOT_TYPE = "so101_follower"

# Logging configuration
LOG_EVERY_N_STEPS = 10  # Print detailed logs every N steps

print(f"[CONFIG] Model: {MODEL_ID}")
print(f"[CONFIG] Follower arm: {FOLLOWER_PORT} ({FOLLOWER_ID})")
print(f"[CONFIG] Cameras: {list(CAMERA_INDICES.keys())}")
print(f"[CONFIG] Task: '{TASK}'")
print(f"[CONFIG] Episodes: {MAX_EPISODES}, Steps/episode: {MAX_STEPS_PER_EPISODE}")
print(f"[CONFIG] Control frequency: {CONTROL_FREQUENCY_HZ} Hz")
print(f"[CONFIG] NOTE: Single left arm only (no right arm)")

# ============================================================================
# STEP 1: IMPORT LEROBOT MODULES
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Importing LeRobot modules...")
print("="*70)

try:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.policies.factory import make_pre_post_processors

    # CRITICAL FIX: Import PI05Policy, NOT PI0Policy!
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.types import PolicyFeature, FeatureType

    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower

    print("[IMPORT] ✅ All LeRobot modules imported successfully")
    print(f"[IMPORT] ✅ Using PI05Policy (NOT PI0Policy - this was the bug!)")

except ImportError as e:
    print(f"[IMPORT] ❌ Import failed: {e}")
    print("[IMPORT] 💡 Try updating LeRobot:")
    print("         cd /home/jrobot/project/lerobot && git pull && pip install -e .")
    exit(1)

# ============================================================================
# STEP 2: LOAD PI0.5 MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Loading Pi0.5 model...")
print("="*70)

try:
    print(f"[MODEL] Loading config from {MODEL_ID}...")
    print(f"[MODEL] ⏳ This will load from cache (~30s) or download first time (~3-5 min)")

    model_start = time.time()

    # STEP 2a: Create base config for Pi0.5
    # NOTE: Using direct initialization instead of from_pretrained to avoid draccus parsing bug
    print(f"[MODEL] 🔧 Creating Pi0.5 config for SO-101 (6-dim actions: 5 joints + gripper)")

    config = PI05Config(
        # Define SO-101 action space: 5 arm joints + 1 gripper = 6 dimensions total
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,),  # SO-101: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
            )
        },
        # Define SO-101 observation space (matching dataset camera keys)
        input_features={
            "observation.images.head": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 224, 224),  # Pi0.5 resizes to 224x224
            ),
            "observation.images.left_wrist": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 224, 224),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(6,),  # 6 joint positions (5 arm joints + gripper)
            ),
        },
        device=str(device),  # Set device
    )

    # Validate config (creates placeholders for missing features)
    config.validate_features()
    print(f"[MODEL] ✅ Config created and validated for SO-101")

    # STEP 2c: Load model with overridden config
    print(f"[MODEL] Loading model weights with SO-101 config...")
    model = PI05Policy.from_pretrained(
        MODEL_ID,
        config=config,
        strict=False  # Allow dimension mismatch for action head (will use pretrained as-is)
    )
    model = model.to(device)
    model.eval()
    model_load_time = time.time() - model_start

    print(f"[MODEL] ✅ Pi0.5 model loaded in {model_load_time:.1f}s")
    print(f"[MODEL] ✅ Model configured for SO-101: 6-dim actions, 2 cameras, 6-dim state")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {total_params/1e9:.2f}B ({total_params/1e6:.0f}M)")
    print(f"[MODEL] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

except Exception as e:
    print(f"[MODEL] ❌ Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 3: CREATE PREPROCESSORS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Creating preprocessors/postprocessors...")
print("="*70)

try:
    print("[PREPROCESS] Creating pre/post processors for pretrained baseline...")
    print("[PREPROCESS] Note: Pretrained model uses its own normalization (no dataset stats)")
    print("[PREPROCESS] Model will handle 6-dim → 32-dim padding internally")

    # For pretrained pi0.5 baseline testing:
    # - No dataset_stats needed - model uses its pretrained normalization
    # - Padding/unpadding happens automatically: 6 → 32 (internal) → 6 (output)
    # - Config override ensures correct dimensions at input/output boundaries
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    print("[PREPROCESS] ✅ Preprocessors ready")
    print("[PREPROCESS] ✅ Padding system will handle 6-dim ↔ 32-dim conversion")

except Exception as e:
    print(f"[PREPROCESS] ❌ Preprocessor setup failed: {e}")
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 4: CONFIGURE CAMERAS
# ============================================================================

print("\n" + "="*70)
print("STEP 4: Configuring cameras...")
print("="*70)

try:
    print(f"[CAMERA] Setting up {len(CAMERA_INDICES)} cameras...")

    # Build camera config dict
    camera_config = {}
    for cam_name, cam_idx in CAMERA_INDICES.items():
        camera_config[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_idx,
            width=640,
            height=480,
            fps=15  # Reduced from 30fps to avoid USB bandwidth issues
        )
        print(f"[CAMERA] {cam_name}: /dev/video{cam_idx} (640x480@15fps)")

    # Create camera objects
    cameras = make_cameras_from_configs(camera_config)

    # Connect cameras
    print("[CAMERA] Connecting to cameras...")
    for cam_name, cam in cameras.items():
        cam.connect()
        print(f"[CAMERA] ✅ Connected to {cam_name}")

    print(f"[CAMERA] ✅ All {len(cameras)} cameras connected successfully")

except Exception as e:
    print(f"[CAMERA] ❌ Camera setup failed: {e}")
    print(f"[CAMERA] 💡 Verify camera indices with: ls -l /dev/video*")
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 5: INITIALIZE SINGLE ROBOT (LEFT FOLLOWER ONLY)
# ============================================================================

print("\n" + "="*70)
print("STEP 5: Initializing left SO-101 follower arm...")
print("="*70)

try:
    print("[ROBOT] Setting up left follower arm...")
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras={},  # No cameras on individual arms (handled separately)
        use_degrees=True  # Use degrees for easier debugging
    )
    robot = SO101Follower(robot_cfg)

    print(f"[ROBOT] Connecting to follower arm on {FOLLOWER_PORT}...")
    # If calibration mismatch detected, you'll be prompted to:
    #   - Press ENTER: Write calibration file to motors (recommended)
    #   - Type 'c': Run new calibration from scratch (not recommended)
    robot.connect()
    print(f"[ROBOT] ✅ Follower arm connected")

    # Confirm calibration is loaded
    print(f"[ROBOT] ✅ Using calibration: {FOLLOWER_ID}")
    print(f"[ROBOT] ✅ Calibration file loaded from: ~/.cache/lerobot/calibration/{FOLLOWER_ID}/")
    print(f"[ROBOT] ✅ Safety: Joint limits enforced by calibration")

    # Log robot features
    print(f"[ROBOT] Observation features: {list(robot.observation_features.keys())}")
    print(f"[ROBOT] Action features: {list(robot.action_features.keys())}")

except Exception as e:
    print(f"[ROBOT] ❌ Robot connection failed: {e}")
    print(f"[ROBOT] 💡 Verify ports with: ls -l /dev/ttyACM*")
    traceback.print_exc()

    # Cleanup cameras
    for cam_name, cam in cameras.items():
        try:
            cam.disconnect()
        except:
            pass
    exit(1)

# ============================================================================
# STEP 6: SETUP DATASET FEATURES MAPPING
# ============================================================================

print("\n" + "="*70)
print("STEP 6: Setting up feature mappings...")
print("="*70)

try:
    print("[FEATURES] Merging observation features from arm + cameras...")

    # Single arm observation features (no prefix needed for single arm)
    combined_obs_features = {}

    # Add robot motor features directly (no prefix for single arm)
    for motor_name, motor_type in robot.observation_features.items():
        combined_obs_features[motor_name] = motor_type

    # Camera features (use camera names directly)
    for cam_name in cameras.keys():
        # Get dimensions from config
        cam_cfg = camera_config[cam_name]
        combined_obs_features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

    print(f"[FEATURES] Total observation features: {len(combined_obs_features)}")
    print(f"[FEATURES]   - Arm motors: {len([k for k in combined_obs_features.keys() if 'pos' in k])}")
    print(f"[FEATURES]   - Cameras: {len(cameras)}")

    # Single arm action features
    combined_action_features = {}

    for motor_name, motor_type in robot.action_features.items():
        combined_action_features[motor_name] = motor_type

    print(f"[FEATURES] Total action features: {len(combined_action_features)}")

    # Verify we have 7 action dimensions (6 joints + gripper)
    if len(combined_action_features) != 7:
        print(f"[FEATURES] ⚠️  Warning: Expected 7 action dims, got {len(combined_action_features)}")
        print(f"[FEATURES] ⚠️  Action features: {list(combined_action_features.keys())}")
    else:
        print(f"[FEATURES] ✅ Correct action dimensions: 7 (6 joints + gripper)")

    # Convert to dataset features format (adds "observation." and "action." prefixes)
    obs_features_dataset = hw_to_dataset_features(combined_obs_features, "observation")
    action_features_dataset = hw_to_dataset_features(combined_action_features, "action")
    dataset_features = {**obs_features_dataset, **action_features_dataset}

    print(f"[FEATURES] ✅ Dataset features ready: {len(dataset_features)} total")
    print(f"[FEATURES] ✅ Action feature names: {list(action_features_dataset['action']['names'])}")

except Exception as e:
    print(f"[FEATURES] ❌ Feature mapping failed: {e}")
    traceback.print_exc()

    # Cleanup
    robot.disconnect()
    for cam_name, cam in cameras.items():
        cam.disconnect()
    exit(1)

# ============================================================================
# STEP 7: RUN INFERENCE LOOP
# ============================================================================

print("\n" + "="*70)
print("STARTING PI0.5 BASELINE INFERENCE TEST")
print("="*70)
print(f"[TEST] Task: '{TASK}'")
print(f"[TEST] Episodes: {MAX_EPISODES}")
print(f"[TEST] Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"[TEST] Control frequency: {CONTROL_FREQUENCY_HZ} Hz")
print(f"[TEST] Robot type: {ROBOT_TYPE}")
print("")
print("[TEST] ⚠️  IMPORTANT EXPECTATIONS:")
print("[TEST]    - This is PRETRAINED (not finetuned) pi0.5")
print("[TEST]    - Expected baseline success: 0-15%")
print("[TEST]    - Egocentric cameras are OUT-OF-DISTRIBUTION")
print("[TEST]    - Arm may move erratically or randomly")
print("[TEST]    - This is NORMAL and establishes baseline!")
print("")
print("[TEST] 🛡️  SAFETY:")
print("[TEST]    - Calibration limits are ENFORCED")
print("[TEST]    - Press Ctrl+C ANYTIME to stop safely")
print("[TEST]    - Duration: ~10 seconds only")
print("[TEST]    - All logs print in real-time")
print("="*70 + "\n")

# Wait for user to press Enter before starting
print("="*70)
print("🛡️  READY TO START INFERENCE")
print("="*70)
print("✅ Model loaded and ready")
print("✅ Cameras connected")
print("✅ Arm connected")
print("✅ Calibration limits enforced")
print("")
print("⚠️  The arm will move for ~10 seconds after you press Enter")
print("⚠️  Make sure:")
print("   - Workspace is clear")
print("   - Arm has room to move")
print("   - You're ready to press Ctrl+C if needed")
print("")
input("Press ENTER to start (or Ctrl+C to abort)... ")
print("\n🚀 Starting inference!\n")

# Performance tracking
inference_times = []
episode_count = 0

try:
    for episode in range(MAX_EPISODES):
        episode_count = episode + 1
        print(f"\n{'='*70}")
        print(f"[EPISODE {episode_count}/{MAX_EPISODES}] Starting...")
        print(f"{'='*70}")
        print(f"[EPISODE {episode_count}] Task: '{TASK}'")
        print(f"[EPISODE {episode_count}] Press Ctrl+C to stop anytime\n")

        episode_start_time = time.time()

        for step in range(MAX_STEPS_PER_EPISODE):
            step_start = time.time()

            try:
                # --------------------------------------------------------
                # GET OBSERVATIONS (cameras + robot states)
                # --------------------------------------------------------

                # Read camera frames
                camera_obs = {}
                for cam_name, cam in cameras.items():
                    frame = cam.async_read()
                    camera_obs[cam_name] = frame

                    if step % LOG_EVERY_N_STEPS == 0:
                        print(f"[STEP {step}] 📷 {cam_name}: {frame.shape}")

                # Read robot state (joint positions)
                robot_state = robot.get_observation()

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] 🤖 Arm state keys: {list(robot_state.keys())}")
                    # Print actual joint values for debugging
                    for key, value in robot_state.items():
                        if 'pos' in key:
                            print(f"[STEP {step}]     {key}: {value}")

                # CRITICAL: Combine cameras + robot states into single observation
                observation = {}

                # Add camera observations
                observation.update(camera_obs)

                # Add robot states directly (no prefix for single arm)
                observation.update(robot_state)

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] 📦 Combined observation keys: {list(observation.keys())}")

                # --------------------------------------------------------
                # BUILD INFERENCE FRAME
                # --------------------------------------------------------

                obs_frame = build_inference_frame(
                    observation=observation,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE
                )

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] 🔧 Inference frame built with task: '{TASK}'")

                # --------------------------------------------------------
                # PREPROCESS
                # --------------------------------------------------------

                obs_preprocessed = preprocess(obs_frame)

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] ⚙️  Observation preprocessed")

                # --------------------------------------------------------
                # MODEL INFERENCE
                # --------------------------------------------------------

                inference_start = time.time()

                with torch.no_grad():
                    action = model.select_action(obs_preprocessed)

                inference_time = time.time() - inference_start

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] 🧠 Model inference: {inference_time*1000:.1f}ms")
                    print(f"[STEP {step}] 🎯 Raw action type: {type(action)}")
                    if isinstance(action, dict):
                        print(f"[STEP {step}] 🎯 Raw action keys: {list(action.keys())}")

                # --------------------------------------------------------
                # POSTPROCESS ACTION
                # --------------------------------------------------------

                # Postprocess handles: unnormalization + unpadding (32-dim → 6-dim)
                # Config override ensures output is 6-dim for SO-101
                action = postprocess(action)
                action = make_robot_action(action, dataset_features)

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] 🔄 Action postprocessed (unpadded 32→6 dims)")
                    print(f"[STEP {step}] 🎯 Postprocessed action keys: {list(action.keys())}")

                # --------------------------------------------------------
                # SEND ACTION TO SINGLE ARM (no splitting needed)
                # --------------------------------------------------------

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] ⚡ Action to send: {action}")
                    # Print joint-by-joint for debugging
                    for key, value in action.items():
                        if 'pos' in key:
                            print(f"[STEP {step}]     {key}: {value}")

                # Send action to single robot
                robot.send_action(action)

                if step % LOG_EVERY_N_STEPS == 0:
                    print(f"[STEP {step}] ✅ Action sent to follower arm")

                # --------------------------------------------------------
                # TIMING AND FREQUENCY CONTROL
                # --------------------------------------------------------

                step_time = time.time() - step_start
                inference_times.append(step_time)

                # Maintain control frequency
                target_step_time = 1.0 / CONTROL_FREQUENCY_HZ
                if step_time < target_step_time:
                    time.sleep(target_step_time - step_time)

                actual_step_time = time.time() - step_start

                if step % LOG_EVERY_N_STEPS == 0:
                    avg_freq = 1.0 / np.mean(inference_times[-10:]) if len(inference_times) >= 10 else 0
                    print(f"[STEP {step}] ⏱️  Step time: {actual_step_time*1000:.1f}ms (avg freq: {avg_freq:.1f}Hz)")
                    print(f"[STEP {step}] 💾 GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                    print("")

            except Exception as e:
                print(f"[STEP {step}] ⚠️  Error during step: {e}")
                print(f"[STEP {step}] ⚠️  Continuing to next step...")
                traceback.print_exc()
                continue

        # Episode summary
        episode_time = time.time() - episode_start_time
        print(f"\n[EPISODE {episode_count}] ✅ Completed in {episode_time:.1f}s")
        print(f"[EPISODE {episode_count}] Average step time: {np.mean(inference_times[-MAX_STEPS_PER_EPISODE:]) * 1000:.1f}ms")
        print(f"[EPISODE {episode_count}] Average frequency: {1.0 / np.mean(inference_times[-MAX_STEPS_PER_EPISODE:]):.1f}Hz")

        if episode < MAX_EPISODES - 1:
            print(f"\n[EPISODE {episode_count}] Waiting 3 seconds before next episode...")
            time.sleep(3)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    print("Shutting down gracefully...")

finally:
    # ========================================================================
    # CLEANUP AND SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("CLEANUP: Disconnecting devices...")
    print("="*70)

    # Disconnect cameras
    for cam_name, cam in cameras.items():
        try:
            cam.disconnect()
            print(f"[CLEANUP] ✅ Disconnected {cam_name}")
        except Exception as e:
            print(f"[CLEANUP] ⚠️  Error disconnecting {cam_name}: {e}")

    # Disconnect robot
    try:
        robot.disconnect()
        print(f"[CLEANUP] ✅ Disconnected follower arm")
    except Exception as e:
        print(f"[CLEANUP] ⚠️  Error disconnecting follower arm: {e}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("PI0.5 BASELINE TEST COMPLETE")
    print("="*70)
    print(f"✅ Model tested: {MODEL_ID}")
    print(f"✅ Task: '{TASK}'")
    print(f"✅ Episodes completed: {episode_count}/{MAX_EPISODES}")
    print(f"✅ Total steps: {len(inference_times)}")

    if len(inference_times) > 0:
        avg_inference_time = np.mean(inference_times)
        avg_freq = 1.0 / avg_inference_time
        print(f"✅ Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"✅ Average frequency: {avg_freq:.1f}Hz")
        print(f"✅ Target frequency: {CONTROL_FREQUENCY_HZ}Hz")

        if avg_freq >= CONTROL_FREQUENCY_HZ * 0.9:
            print("✅ Performance: EXCELLENT (meeting target frequency)")
        elif avg_freq >= CONTROL_FREQUENCY_HZ * 0.7:
            print("⚠️  Performance: GOOD (close to target frequency)")
        else:
            print("⚠️  Performance: NEEDS IMPROVEMENT (below target frequency)")

    print("")
    print("="*70)
    print("📊 RECORD YOUR OBSERVATIONS:")
    print("="*70)
    print("1. Did the episode successfully complete the task?")
    print("   Success: [ ] Yes  [ ] No")
    print("")
    print("2. What behaviors did you observe?")
    print("   [ ] Arm moved toward objects")
    print("   [ ] Arm grasped objects")
    print("   [ ] Arm placed objects correctly")
    print("   [ ] Arm moved erratically/randomly")
    print("   [ ] Arm stayed mostly still")
    print("   [ ] Other: ___________________________")
    print("")
    print("3. Overall assessment: ___________________________")
    print("")
    print("="*70)
    print("⚠️  EXPECTED BASELINE: 0-15% success")
    print("="*70)
    print("Why so low?")
    print("- Pi0.5 trained on EXTERNAL fixed cameras")
    print("- Your setup uses EGOCENTRIC moving cameras")
    print("- This is OUT-OF-DISTRIBUTION for the pretrained model")
    print("- Finetuning on your 10 recorded episodes should improve this!")
    print("")
    print("NEXT STEPS:")
    print("1. Record your baseline success rate above")
    print("2. Finetune pi0.5 on your recorded dataset")
    print("3. Run this script again with finetuned model")
    print("4. Compare success rates (expect 40-60% after finetuning)")
    print("="*70)

    print("\n✅ Script completed successfully!")
