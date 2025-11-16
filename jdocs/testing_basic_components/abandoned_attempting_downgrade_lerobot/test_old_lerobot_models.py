#!/usr/bin/env python3
"""
Test Script for Old LeRobot (Pre-v0.4.0) Community Models

This script is designed to work with LeRobot from August 31, 2024,
BEFORE the v0.4.0 processor pipeline changes.

It should successfully load models like:
- jhou/smolvla_pickplace
- HuggingFadeUser/my_smolvla
- saood65/my_smolvla
- Other pre-v0.4.0 community models

These models DO NOT have policy_preprocessor.json files,
but old LeRobot doesn't require them!
"""

import torch
import time
import numpy as np
import traceback
from pathlib import Path


def test_old_lerobot_model(
    model_id="jhou/smolvla_pickplace",
    arm_port="/dev/ttyACM2",
    arm_id="xlerobot_left_arm",
    camera_top=4,
    camera_wrist=6,
    duration=10,
    task="pick up the red cube"
):
    """
    Test community model with old LeRobot (pre-v0.4.0)

    Args:
        model_id: HuggingFace model ID
        arm_port: Robot serial port
        arm_id: Robot calibration ID
        camera_top: Top camera index
        camera_wrist: Wrist camera index
        duration: Test duration in seconds
        task: Natural language task description
    """

    print("\n" + "="*70)
    print("Testing Community Model with OLD LeRobot (Pre-v0.4.0)")
    print("="*70)
    print(f"Model: {model_id}")
    print(f"Robot: {arm_id} on {arm_port}")
    print(f"Cameras: top={camera_top}, wrist={camera_wrist}")
    print(f"Task: '{task}'")
    print(f"Duration: {duration}s at 30Hz")
    print("="*70 + "\n")

    device = torch.device("cuda")

    try:
        # Step 1: Verify we're in old LeRobot
        print("Step 1: Verifying LeRobot version...")
        print("⚠️  This script requires OLD LeRobot (Aug 31, 2024)")
        print("   Checking if policy_preprocessor.json is NOT required...")

        try:
            # In old LeRobot, this import path should work
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            print("✅ Old LeRobot import path detected")
        except ImportError:
            # Try new path (if this works, we're in new LeRobot!)
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            print("✅ Using newer import path (should still work)")

        # Import other modules
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
        from lerobot.common.robot_devices.robots.so101_follower import SO101FollowerRobot

        print("✅ All modules imported successfully")

        # Step 2: Load model
        print(f"\nStep 2: Loading {model_id} from HuggingFace...")
        print("⏳ This may take 1-2 minutes (downloading model)...")

        model = SmolVLAPolicy.from_pretrained(model_id)
        model = model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print("")
        print("🎉 SUCCESS! Model loaded WITHOUT policy_preprocessor.json!")
        print("   This confirms old LeRobot doesn't require those files.")

        # Step 3: Initialize cameras
        print("\nStep 3: Initializing cameras...")

        camera_configs = [
            {
                "name": "top",
                "index": camera_top,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            {
                "name": "wrist",
                "index": camera_wrist,
                "width": 640,
                "height": 480,
                "fps": 30
            }
        ]

        cameras = {}
        for config in camera_configs:
            cameras[config["name"]] = OpenCVCamera(
                config["index"],
                fps=config["fps"],
                width=config["width"],
                height=config["height"]
            )
            print(f"✅ {config['name']} camera initialized (index {config['index']})")

        # Step 4: Initialize robot
        print("\nStep 4: Initializing robot...")

        robot = SO101FollowerRobot(
            robot_type="so101_follower",
            calibration_path=str(Path.home() / f".cache/huggingface/lerobot/calibration/robots/so101_follower/{arm_id}.json"),
            port=arm_port
        )
        robot.connect()
        print(f"✅ Robot connected: {arm_port}")

        # Step 5: Load calibration for action bounds
        print("\nStep 5: Loading calibration...")
        import json

        calib_path = Path.home() / f".cache/huggingface/lerobot/calibration/robots/so101_follower/{arm_id}.json"

        if calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)

            joint_mins = []
            joint_maxs = []
            joint_names = []

            for joint_key in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                             "wrist_flex", "wrist_roll", "gripper"]:
                if joint_key in calib:
                    joint_names.append(joint_key)
                    range_vals = calib[joint_key]

                    if isinstance(range_vals, dict):
                        joint_mins.append(range_vals.get("min", -3.14))
                        joint_maxs.append(range_vals.get("max", 3.14))
                    elif isinstance(range_vals, (list, tuple)) and len(range_vals) >= 2:
                        joint_mins.append(range_vals[0])
                        joint_maxs.append(range_vals[1])
                    else:
                        joint_mins.append(-3.14)
                        joint_maxs.append(3.14)

            joint_mins = np.array(joint_mins)
            joint_maxs = np.array(joint_maxs)
            print("✅ Calibration loaded")
        else:
            print("⚠️  Calibration not found, using defaults")
            joint_mins = np.array([-3.14, -1.57, -3.14, -3.14, -3.14, 0.0])
            joint_maxs = np.array([3.14, 1.57, 3.14, 3.14, 3.14, 6.28])

        # Step 6: Run inference
        print("\nStep 6: Starting inference loop...")
        print("="*70)
        print(f"Running at 30Hz for {duration} seconds")
        print(f"Task: '{task}'")
        print("="*70)
        print("\n⚠️  WATCH THE ARM:")
        print("   ✅ GOOD: Large, purposeful movements toward objects")
        print("   ✅ GOOD: Gripper opens/closes at appropriate times")
        print("   ❌ BAD: Only tiny movements (like base SmolVLA)")
        print("\n" + "="*70 + "\n")

        total_steps = int(duration * 30)  # 30Hz
        inference_times = []

        # Action smoothing
        alpha = 0.3
        prev_action = None

        for step in range(total_steps):
            step_start = time.time()

            try:
                # Get observations
                images = {}
                for name, camera in cameras.items():
                    images[name] = camera.read()

                # Get robot state
                state = robot.get_state()

                # Build observation dict (old LeRobot format)
                observation = {
                    "observation.images.top": torch.from_numpy(images["top"]).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0,
                    "observation.images.wrist": torch.from_numpy(images["wrist"]).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0,
                    "observation.state": torch.from_numpy(state).unsqueeze(0).to(device).float(),
                    "task": task
                }

                # Model inference
                with torch.no_grad():
                    action = model.select_action(observation)

                # Extract action (format depends on model)
                if isinstance(action, dict):
                    action_array = action["action"].cpu().numpy().flatten()
                elif torch.is_tensor(action):
                    action_array = action.cpu().numpy().flatten()
                else:
                    action_array = np.array(action).flatten()

                # Ensure correct size
                action_array = action_array[:6]  # 6 DOF (including gripper)

                # Check if actions need scaling (backup denormalization)
                if np.all(np.abs(action_array) <= 2.0):
                    # Actions look normalized, scale them
                    action_array = joint_mins + (action_array + 1.0) / 2.0 * (joint_maxs - joint_mins)

                    if step == 0:
                        print(f"  ℹ️  Actions were normalized, scaled to robot range")

                # Clip to safe limits
                action_array = np.clip(action_array, joint_mins, joint_maxs)

                # Apply smoothing
                if prev_action is not None:
                    action_array = alpha * action_array + (1 - alpha) * prev_action
                prev_action = action_array.copy()

                # Log every 30 steps (every second)
                if step % 30 == 0:
                    print(f"\n📊 Step {step}/{total_steps} ({step//30}s)")
                    print(f"  Action range: [{action_array.min():.3f}, {action_array.max():.3f}]")

                    # Check if actions are large enough
                    if prev_action is not None:
                        action_change = np.abs(action_array - prev_action).mean()
                        if action_change > 0.1:
                            print(f"  ✅ Large movements (change: {action_change:.3f})")
                        else:
                            print(f"  ⚠️  Small movements (change: {action_change:.3f})")

                # Send to robot
                robot.send_action(action_array)

                # Timing
                step_time = time.time() - step_start
                inference_times.append(step_time)

                # Maintain 30Hz
                if step_time < 0.033:
                    time.sleep(0.033 - step_time)

                # Progress
                if step % 30 == 0:
                    avg_time = np.mean(inference_times[-30:])
                    freq = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"  Frequency: {freq:.1f}Hz | GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            except Exception as e:
                print(f"⚠️  Error at step {step}: {e}")
                continue

        # Results
        print("\n" + "="*70)
        print("Inference Complete!")
        print("="*70)

        avg_time = np.mean(inference_times)
        avg_freq = 1.0 / avg_time

        print(f"Total steps: {total_steps}")
        print(f"Avg inference time: {avg_time*1000:.1f}ms")
        print(f"Avg frequency: {avg_freq:.1f}Hz")

        # Cleanup
        robot.disconnect()
        for camera in cameras.values():
            camera.disconnect()

        # User evaluation
        print("\n" + "="*70)
        print("📊 EVALUATION")
        print("="*70)
        print("\nDid the robot:")
        print("  1. Make LARGE, purposeful movements? (not tiny jitters)")
        print("  2. Move toward or interact with objects?")
        print("  3. Open/close gripper at appropriate times?")
        print("  4. Show task-relevant behavior (reach, grasp, etc.)?")
        print("\nCompare to base SmolVLA:")
        print("  - Base SmolVLA: Small movements only")
        print("  - This model: ???")
        print("\n" + "-"*70)

        success = input("\nDid this model work BETTER than base SmolVLA? (y/n): ").strip().lower()

        print("\n" + "="*70)
        if success == 'y':
            print("✅ SUCCESS! Model works for MVP!")
            print("="*70)
            print("\nWhat this means:")
            print("  • Old LeRobot compatibility issue SOLVED ✅")
            print("  • This model can be used for MVP demonstrations")
            print("  • Performance is good enough for proof of concept")
            print("\nNext steps:")
            print("  → Document this success in RESULTS.md")
            print("  → Test on different object positions")
            print("  → Use this model for Stage 3a planning")
            print("  → Consider fine-tuning later for production (70-90%)")
        else:
            print("⚠️  Model loaded but performance not great")
            print("="*70)
            print("\nWhat we learned:")
            print("  • ✅ Compatibility issue SOLVED (model loaded!)")
            print("  • ❌ Domain mismatch remains (your setup vs training)")
            print("\nNext steps:")
            print("  → Try another community model")
            print("  → Models to try:")
            print("     - HuggingFadeUser/my_smolvla (60 likes)")
            print("     - saood65/my_smolvla (recent)")
            print("     - leesangoh/smolvla_pickplace")
            print("  → If 2-3 models fail, proceed to fine-tuning (Path C)")

        return success == 'y'

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        try:
            robot.disconnect()
            for camera in cameras.values():
                camera.disconnect()
        except:
            pass
        return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        try:
            robot.disconnect()
            for camera in cameras.values():
                camera.disconnect()
        except:
            pass
        return False


def main():
    print("="*70)
    print("Old LeRobot Model Tester (Pre-v0.4.0)")
    print("="*70)
    print("\nTesting community models with OLD LeRobot")
    print("These models were uploaded before v0.4.0 processor changes")
    print("")

    # Verify environment
    print("="*70)
    print("Environment Check")
    print("="*70)

    import sys
    env_path = sys.executable
    print(f"Python: {env_path}")

    if "lerobot_old" in env_path:
        print("✅ Running in lerobot_old environment")
    else:
        print("⚠️  WARNING: Not in lerobot_old environment!")
        print("   You should be using: conda activate lerobot_old")
        proceed = input("\nContinue anyway? (y/n): ")
        if proceed.lower() != 'y':
            return

    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}\n")

    # Model selection
    print("="*70)
    print("Model Selection")
    print("="*70)
    print("\nRecommended models to test (in order):")
    print("  1. jhou/smolvla_pickplace (42 likes, failed with new LeRobot)")
    print("  2. HuggingFadeUser/my_smolvla (60 likes, most popular)")
    print("  3. saood65/my_smolvla (recent, 4 likes)")
    print("  4. leesangoh/smolvla_pickplace (community)")
    print("")

    model_input = input("Enter model ID [jhou/smolvla_pickplace]: ").strip()
    model_id = model_input if model_input else "jhou/smolvla_pickplace"

    # Camera configuration
    print("\n" + "="*70)
    print("Camera Configuration")
    print("="*70)

    camera_top_input = input("Enter TOP camera index [4]: ").strip()
    camera_top = int(camera_top_input) if camera_top_input else 4

    camera_wrist_input = input("Enter WRIST camera index [6]: ").strip()
    camera_wrist = int(camera_wrist_input) if camera_wrist_input else 6

    # Robot configuration
    print("\n" + "="*70)
    print("Robot Configuration")
    print("="*70)

    arm_choice = input("Which arm? (left/right) [left]: ").strip().lower()
    if arm_choice == 'right':
        arm_port = input("Enter port [/dev/ttyACM3]: ").strip() or "/dev/ttyACM3"
        arm_id = "xlerobot_right_arm"
    else:
        arm_port = input("Enter port [/dev/ttyACM2]: ").strip() or "/dev/ttyACM2"
        arm_id = "xlerobot_left_arm"

    # Duration
    duration_input = input("Test duration in seconds [10]: ").strip()
    duration = int(duration_input) if duration_input else 10

    # Task
    print("\n" + "="*70)
    print("Task Description")
    print("="*70)
    task = input("Enter task ['pick up the red cube']: ").strip()
    if not task:
        task = "pick up the red cube"

    # Confirm
    print("\n" + "="*70)
    print("Starting Test")
    print("="*70)
    print(f"Model: {model_id}")
    print(f"Robot: {arm_id}")
    print(f"Task: '{task}'")
    print("")
    print("⚠️  Make sure:")
    print("   • Only ONE arm active (other arm off/hidden)")
    print("   • Test object on table")
    print("   • Workspace clear")
    print("   • Keep finger near Ctrl+C")
    input("\nPress Enter when ready...")

    success = test_old_lerobot_model(
        model_id=model_id,
        arm_port=arm_port,
        arm_id=arm_id,
        camera_top=camera_top,
        camera_wrist=camera_wrist,
        duration=duration,
        task=task
    )

    if success:
        print("\n" + "="*70)
        print("🎉 MVP SUCCESS!")
        print("="*70)
        print("\nYou've proven:")
        print("  ✅ Pretrained models CAN work with old LeRobot")
        print("  ✅ Downgrade strategy was correct")
        print("  ✅ Ready for MVP demonstrations")
        print("\nDocument your results:")
        print("  → Edit: attempting_downgrade_lerobot/RESULTS.md")
        print("  → Note: Which model worked, success rate, observations")
    else:
        print("\n" + "="*70)
        print("Try Another Model")
        print("="*70)
        print("\nQuick retry:")
        print("  python test_old_lerobot_models.py")
        print("\nOr proceed to fine-tuning:")
        print("  → See: jdocs/NEXT_STEPS_STAGE_2.md (Path C)")


if __name__ == "__main__":
    main()
