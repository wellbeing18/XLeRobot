#!/usr/bin/env python3
"""
Test r2owb0/act1 - The only true SO-101 pretrained model

This model was trained on 10 episodes of SO-101 pick-place task.
It has NO language conditioning - will repeat the trained behavior.

Expected behavior:
- IF camera/lighting matches training → 60-80% success
- IF setup differs → Poor performance (<30%)
"""

import torch
import time
import numpy as np
import traceback
from pathlib import Path


def test_act_r2owb0(arm_port="/dev/ttyACM2", arm_id="xlerobot_left_arm",
                    camera_top=0, camera_wrist=6, duration=10):
    """
    Test ACT model r2owb0/act1 trained on SO-101

    Args:
        arm_port: Robot serial port
        arm_id: Robot calibration ID
        camera_top: Top camera index (needs 640×480)
        camera_wrist: Wrist camera index (needs 320×240)
        duration: Test duration in seconds
    """

    print("\n" + "="*60)
    print("Testing r2owb0/act1 - SO-101 ACT Model")
    print("="*60)
    print(f"Robot: {arm_id} on {arm_port}")
    print(f"Cameras: top={camera_top} (640×480), wrist={camera_wrist} (320×240)")
    print(f"Duration: {duration}s at 50Hz")
    print("="*60 + "\n")

    device = torch.device("cuda")

    try:
        # Step 1: Import LeRobot modules
        print("Step 1: Importing LeRobot modules...")
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        print("✅ All modules imported")

        # Step 2: Load ACT model
        print("\nStep 2: Loading r2owb0/act1 from HuggingFace...")
        model_id = "r2owb0/act1"
        model = ACTPolicy.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

        # Step 3: Setup preprocessors
        print("\nStep 3: Setting up preprocessors...")
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        print("✅ Preprocessors ready")

        # Step 4: Configure cameras (MATCH TRAINING RESOLUTION!)
        print("\nStep 4: Configuring cameras...")
        camera_config = {
            "camera1": OpenCVCameraConfig(
                index_or_path=camera_top,
                width=640, height=480, fps=30  # Top camera: larger
            ),
            "camera2": OpenCVCameraConfig(
                index_or_path=camera_wrist,
                width=320, height=240, fps=30  # Wrist camera: smaller!
            ),
        }
        print("✅ Cameras configured")
        print("   Top: 640×480 (matches training)")
        print("   Wrist: 320×240 (matches training)")

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

        # Step 7: Run inference
        print("\nStep 7: Starting inference loop...")
        print("="*60)
        print(f"Running at 50Hz for {duration} seconds")
        print("⚠️  ACT has NO language conditioning")
        print("⚠️  Will repeat behavior from 10 training episodes")
        print("="*60 + "\n")

        total_steps = int(duration * 50)  # ACT trained at 50Hz
        inference_times = []
        robot_type = ""

        for step in range(total_steps):
            step_start = time.time()

            try:
                # Get observation
                obs = robot.get_observation()

                # Build inference frame (no task - ACT doesn't use language)
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task="",  # ACT has no language conditioning!
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

                # Timing
                step_time = time.time() - step_start
                inference_times.append(step_time)

                # Maintain 50Hz
                if step_time < 0.02:
                    time.sleep(0.02 - step_time)

                # Progress
                if step % 50 == 0:
                    avg_time = np.mean(inference_times[-50:])
                    freq = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"Step {step}/{total_steps} | {freq:.1f}Hz | GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

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
        print("📊 EVALUATION QUESTIONS")
        print("="*60)
        print("\nDid the robot:")
        print("  1. Make LARGE, purposeful movements? (not tiny jitters)")
        print("  2. Approach or interact with objects on the table?")
        print("  3. Open/close gripper at appropriate times?")
        print("  4. Move toward a target position/area?")
        print("  5. Show task-relevant behavior (pick, place, etc.)?")
        print("\n" + "-"*60)

        success = input("\nWas the behavior task-relevant? (y/n): ").strip().lower()

        print("\n" + "="*60)
        if success == 'y':
            print("✅ SUCCESS! r2owb0/act1 works on your setup!")
            print("="*60)
            print("\nWhat this means:")
            print("  • Your camera/lighting setup matches training data")
            print("  • You can use this model for simple pick-place tasks")
            print("  • Model only knows ONE task (from 10 training episodes)")
            print("\nLimitations:")
            print("  ⚠️  No language conditioning (can't give it new tasks)")
            print("  ⚠️  Will only repeat trained pick-place behavior")
            print("  ⚠️  May fail if objects in different positions")
            print("\n📍 NEXT STEP:")
            print("  → Proceed to Stage 3a with this model")
            print("  → Or fine-tune SmolVLA for language conditioning")
        else:
            print("❌ Model doesn't work on your setup")
            print("="*60)
            print("\nThis is EXPECTED - common reasons:")
            print("  • Camera positions differ from training")
            print("  • Lighting conditions different")
            print("  • Workspace layout different")
            print("  • Object positions don't match training data")
            print("\nThis is NOT a failure - it's normal!")
            print("\n📍 NEXT STEP:")
            print("  → Try Path B: Test community fine-tuned models")
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
    print("XLeRobot - Test r2owb0/act1 (SO-101 ACT Model)")
    print("="*60)
    print("\nThis is the ONLY pretrained model specifically for SO-101")
    print("Trained on 10 episodes of pick-place task")
    print("")

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}\n")

    # Camera configuration
    print("="*60)
    print("Camera Configuration")
    print("="*60)
    print("⚠️  IMPORTANT: Camera resolutions must match training:")
    print("   Top camera: 640×480")
    print("   Wrist camera: 320×240")
    print("")

    camera_top_input = input("Enter TOP camera index [0]: ").strip()
    camera_top = int(camera_top_input) if camera_top_input else 0

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

    # Confirm
    print("\n" + "="*60)
    print("Starting Test")
    print("="*60)
    print("⚠️  Make sure:")
    print("   • Workspace is clear of obstacles")
    print("   • Test object (cube/block) is on table")
    print("   • You can reach Ctrl+C to stop if needed")
    print("\n⚠️  Remember: This model will try to repeat its trained task")
    print("   It CANNOT follow new language instructions")
    input("\nPress Enter when ready...")

    success = test_act_r2owb0(arm_port, arm_id, camera_top, camera_wrist, duration)

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\nOption 1: Use this model for Stage 3")
        print("  Pros: Works immediately, no training needed")
        print("  Cons: One task only, no language conditioning")
        print("\nOption 2: Fine-tune SmolVLA")
        print("  Pros: Language conditioning, flexible tasks")
        print("  Cons: Requires 80+ demos, 1-2 weeks")
        print("\nSee: jdocs/NEXT_STEPS_STAGE_2.md for details")
    else:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Try community models (Path B)")
        print("   See: jdocs/NEXT_STEPS_STAGE_2.md")
        print("\n2. Or proceed directly to fine-tuning (Path C)")
        print("   • Collect 80 episodes")
        print("   • Fine-tune SmolVLA")
        print("   • Achieve 70-90% success")


if __name__ == "__main__":
    main()
