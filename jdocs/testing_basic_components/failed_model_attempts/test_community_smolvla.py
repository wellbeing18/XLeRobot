#!/usr/bin/env python3
"""
Test Community Fine-tuned SmolVLA Models (Path B)

This script tests community models fine-tuned on SO-101 pickplace task.
Default: jhou/smolvla_pickplace (42 likes, trained on svla_so101_pickplace)

Expected: 40-70% success if camera setup is similar to training
"""

import torch
import time
import numpy as np
import traceback
from pathlib import Path


def test_community_smolvla(
    model_id="jhou/smolvla_pickplace",
    arm_port="/dev/ttyACM2",
    arm_id="xlerobot_left_arm",
    camera_top=4,
    camera_wrist=6,
    duration=10,
    task="pick up the red cube"
):
    """
    Test community fine-tuned SmolVLA model

    Args:
        model_id: HuggingFace model ID (e.g., "jhou/smolvla_pickplace")
        arm_port: Robot serial port
        arm_id: Robot calibration ID
        camera_top: Top camera index
        camera_wrist: Wrist camera index
        duration: Test duration in seconds
        task: Natural language task description
    """

    print("\n" + "="*60)
    print(f"Testing Community SmolVLA Model")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Robot: {arm_id} on {arm_port}")
    print(f"Cameras: top={camera_top}, wrist={camera_wrist}")
    print(f"Task: '{task}'")
    print(f"Duration: {duration}s at 30Hz")
    print("="*60 + "\n")

    device = torch.device("cuda")

    try:
        # Step 1: Import LeRobot modules
        print("Step 1: Importing LeRobot modules...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        print("✅ All modules imported")

        # Step 2: Load community model
        print(f"\nStep 2: Loading {model_id} from HuggingFace...")
        print("⏳ This may take a minute (downloading model)...")

        try:
            model = SmolVLAPolicy.from_pretrained(model_id)
            model = model.to(device)
            model.eval()
            print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("\n💡 Common issues:")
            print("   - Model might not exist or be private")
            print("   - Network connection issues")
            print("   - Compatibility issues with LeRobot version")
            raise

        # Step 3: Setup preprocessors
        print("\nStep 3: Setting up preprocessors...")
        try:
            preprocess, postprocess = make_pre_post_processors(
                model.config,
                model_id,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print("✅ Preprocessors ready")
        except Exception as e:
            print(f"⚠️  Warning: Preprocessor setup issue: {e}")
            print("   Trying without dataset_stats...")
            preprocess, postprocess = make_pre_post_processors(
                model.config,
                model_id,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print("✅ Preprocessors ready (without stats)")

        # Step 4: Configure cameras
        print("\nStep 4: Configuring cameras...")
        print("⚠️  Note: svla_so101_pickplace uses:")
        print("   - Camera 'up': 480×640")
        print("   - Camera 'side': 480×640")

        camera_config = {
            "camera1": OpenCVCameraConfig(
                index_or_path=camera_top,
                width=640, height=480, fps=30
            ),
            "camera2": OpenCVCameraConfig(
                index_or_path=camera_wrist,
                width=640, height=480, fps=30  # Note: 640×480 for both
            ),
        }
        print("✅ Cameras configured")
        print(f"   Top: 640×480 (camera {camera_top})")
        print(f"   Wrist: 640×480 (camera {camera_wrist})")

        # Step 5: Initialize robot
        print("\nStep 5: Initializing robot...")
        robot_cfg = SO101FollowerConfig(port=arm_port, id=arm_id, cameras=camera_config)
        robot = SO101Follower(robot_cfg)
        robot.connect()
        print(f"✅ Robot connected: {arm_port}")

        # Step 6: Setup features
        print("\nStep 6: Setting up feature mappings...")
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        print("✅ Feature mappings ready")

        # Step 7: Load calibration for action scaling
        print("\nStep 7: Loading robot calibration...")
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

        # Step 8: Run inference
        print("\nStep 8: Starting inference loop...")
        print("="*60)
        print(f"Running at 30Hz for {duration} seconds")
        print(f"Task: '{task}'")
        print("="*60)
        print("\n⚠️  WATCH THE ARM:")
        print("   ✅ GOOD: Large, purposeful movements toward objects")
        print("   ✅ GOOD: Gripper opens/closes at appropriate times")
        print("   ❌ BAD: Only tiny movements (like base SmolVLA)")
        print("\n" + "="*60 + "\n")

        total_steps = int(duration * 30)  # 30Hz
        inference_times = []
        robot_type = ""

        # Action smoothing
        alpha = 0.3
        prev_action = None

        for step in range(total_steps):
            step_start = time.time()

            try:
                # Get observation
                obs = robot.get_observation()

                # Build inference frame
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=task,
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

                # Extract action array
                if isinstance(action, dict):
                    action_array = []
                    for joint_key in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                                     "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]:
                        if joint_key in action:
                            val = action[joint_key]
                            if torch.is_tensor(val):
                                val = val.cpu().numpy()
                            if isinstance(val, np.ndarray):
                                val = float(val.flatten()[0])
                            action_array.append(val)
                        else:
                            action_array.append(0.0)
                    action_array = np.array(action_array)
                else:
                    action_array = action
                    if torch.is_tensor(action_array):
                        action_array = action_array.cpu().numpy()

                # Check if actions need scaling (backup denormalization)
                if np.all(np.abs(action_array) <= 2.0):
                    # Actions look normalized, scale them
                    action_array = joint_mins + (action_array + 1.0) / 2.0 * (joint_maxs - joint_mins)

                    if step % 30 == 0 and step < 60:
                        print(f"  ℹ️  [Backup scaling applied - actions were small]")

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
                    action_magnitude = np.abs(action_array - prev_action if prev_action is not None else action_array).mean()
                    if action_magnitude > 0.1:
                        print(f"  ✅ Large movements detected (magnitude: {action_magnitude:.3f})")
                    else:
                        print(f"  ⚠️  Small movements (magnitude: {action_magnitude:.3f})")

                # Reconstruct action dict
                action_dict = {}
                for i, joint_key in enumerate(["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                                               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]):
                    action_dict[joint_key] = action_array[i]

                # Send to robot
                robot.send_action(action_dict)

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
        print("\n" + "="*60)
        print("Inference Complete!")
        print("="*60)

        avg_time = np.mean(inference_times)
        avg_freq = 1.0 / avg_time

        print(f"Total steps: {total_steps}")
        print(f"Avg inference time: {avg_time*1000:.1f}ms")
        print(f"Avg frequency: {avg_freq:.1f}Hz")

        # Cleanup
        robot.disconnect()

        # User evaluation
        print("\n" + "="*60)
        print("📊 EVALUATION")
        print("="*60)
        print("\nDid the robot:")
        print("  1. Make LARGE, purposeful movements? (not tiny jitters)")
        print("  2. Move toward or interact with objects?")
        print("  3. Open/close gripper at appropriate times?")
        print("  4. Show task-relevant behavior (reach, grasp, etc.)?")
        print("\nCompare to your SmolVLA base test:")
        print("  - SmolVLA base: Small movements only")
        print("  - This model: ???")
        print("\n" + "-"*60)

        success = input("\nWas the behavior BETTER than base SmolVLA? (y/n): ").strip().lower()

        print("\n" + "="*60)
        if success == 'y':
            print("✅ SUCCESS! This community model works better!")
            print("="*60)
            print("\nWhat this means:")
            print("  • Someone fine-tuned on similar SO-101 setup")
            print("  • Your camera/workspace is close enough to theirs")
            print("  • You can use this model for your tasks!")
            print("\nLimitations:")
            print("  ⚠️  Performance depends on setup similarity")
            print("  ⚠️  May not work for all object positions")
            print("  ⚠️  Fine-tuning on YOUR data would be better")
            print("\n📍 NEXT STEPS:")
            print("  → Test on different object positions")
            print("  → Try different tasks with language conditioning")
            print("  → If not good enough, proceed to Path C (fine-tuning)")
        else:
            print("❌ Model doesn't work well on your setup")
            print("="*60)
            print("\nThis is expected - common reasons:")
            print("  • Camera positions differ from their training")
            print("  • Lighting conditions different")
            print("  • Workspace layout different")
            print("  • You have 2-arm setup, they might have 1-arm")
            print("\n📍 NEXT STEPS:")
            print("  → Try another community model")
            print("  → Or proceed to Path C (fine-tuning)")
            print("  → See: jdocs/NEXT_STEPS_STAGE_2.md")

        return success == 'y'

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        try:
            robot.disconnect()
        except:
            pass
        return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        try:
            robot.disconnect()
        except:
            pass
        return False


def main():
    print("="*60)
    print("XLeRobot - Test Community Fine-tuned SmolVLA")
    print("="*60)
    print("\nPath B: Testing community models")
    print("This model was fine-tuned by the community on SO-101 data")
    print("")

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}\n")

    # Model selection
    print("="*60)
    print("Model Selection")
    print("="*60)
    print("Default: jhou/smolvla_pickplace")
    print("  - 42 likes")
    print("  - Trained on lerobot/svla_so101_pickplace")
    print("  - Language conditioning: YES")
    print("")

    model_input = input("Enter model ID [jhou/smolvla_pickplace]: ").strip()
    model_id = model_input if model_input else "jhou/smolvla_pickplace"

    # Camera configuration
    print("\n" + "="*60)
    print("Camera Configuration")
    print("="*60)
    print("⚠️  svla_so101_pickplace dataset uses:")
    print("   - Camera 'up': 480×640")
    print("   - Camera 'side': 480×640")
    print("   Both cameras same resolution!")
    print("")

    camera_top_input = input("Enter TOP camera index [4]: ").strip()
    camera_top = int(camera_top_input) if camera_top_input else 4

    camera_wrist_input = input("Enter WRIST camera index [6]: ").strip()
    camera_wrist = int(camera_wrist_input) if camera_wrist_input else 6

    # Robot configuration
    print("\n" + "="*60)
    print("Robot Configuration")
    print("="*60)

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
    print("\n" + "="*60)
    print("Task Description")
    print("="*60)
    print("This model was trained on pick-and-place tasks")
    task = input("Enter task ['pick up the red cube']: ").strip()
    if not task:
        task = "pick up the red cube"

    # Confirm
    print("\n" + "="*60)
    print("Starting Test")
    print("="*60)
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

    success = test_community_smolvla(
        model_id=model_id,
        arm_port=arm_port,
        arm_id=arm_id,
        camera_top=camera_top,
        camera_wrist=camera_wrist,
        duration=duration,
        task=task
    )

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Test more variations:")
        print("   - Different object positions")
        print("   - Different tasks")
        print("   - Different lighting")
        print("\n2. If good enough (60-70%):")
        print("   - Use this model for Stage 3")
        print("   - Build hierarchical control on top")
        print("\n3. If not quite good enough:")
        print("   - Proceed to Path C (fine-tuning)")
        print("   - Fine-tuning will get you to 70-90%")
    else:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Try another community model:")
        print("   - HuggingFadeUser/my_smolvla (60 likes)")
        print("   - saood65/my_smolvla (recent)")
        print("\n2. Or proceed to Path C:")
        print("   - Fine-tune on YOUR setup")
        print("   - 80 episodes → 70-90% success")
        print("   - See: jdocs/NEXT_STEPS_STAGE_2.md")


if __name__ == "__main__":
    main()
