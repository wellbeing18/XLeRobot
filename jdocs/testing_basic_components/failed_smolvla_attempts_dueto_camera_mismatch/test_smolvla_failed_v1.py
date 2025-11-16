#!/usr/bin/env python3
"""
Test SmolVLA with Official LeRobot API (ORIGINAL VERSION - NO STATS FIX)

Based on: /home/jrobot/project/lerobot/examples/tutorial/smolvla/using_smolvla_example.py
Adapted for: XLeRobot SO-101 + UVC cameras

This is the version BEFORE adding dataset stats loading.
"""

import torch
import time
import numpy as np
import traceback
from pathlib import Path


def check_gpu():
    """Check GPU availability"""
    print("="*60)
    print("GPU Check")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU: {gpu_name}")
    print(f"   Total Memory: {gpu_memory:.1f} GB")
    print(f"   Free Memory: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    return True


def test_smolvla_official(arm_port="/dev/ttyACM0", arm_id="xlerobot_left_arm",
                          camera_top=8, camera_wrist=4, duration=10, task="pick up the red cube"):
    """
    Test SmolVLA using official LeRobot API

    Args:
        arm_port: Robot USB port
        arm_id: Calibration ID
        camera_top: Top camera index
        camera_wrist: Wrist camera index
        duration: Test duration (seconds)
        task: Language task description
    """

    print("\n" + "="*60)
    print("SmolVLA Official API Test (ORIGINAL - NO STATS)")
    print("="*60)
    print(f"Robot: {arm_id} on {arm_port}")
    print(f"Cameras: top={camera_top}, wrist={camera_wrist}")
    print(f"Task: {task}")
    print(f"Duration: {duration}s")
    print("="*60 + "\n")

    device = torch.device("cuda")

    try:
        # Step 1: Import LeRobot modules (correct paths)
        print("Step 1: Importing LeRobot modules...")
        try:
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            from lerobot.datasets.utils import hw_to_dataset_features
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.utils import build_inference_frame, make_robot_action
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            from lerobot.robots.so101_follower.so101_follower import SO101Follower
            print("✅ All modules imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("\n💡 Your LeRobot might be too old. Try:")
            print("   cd /home/jrobot/project/lerobot")
            print("   git pull")
            print("   pip install -e .")
            return False

        # Step 2: Load SmolVLA model
        print("\nStep 2: Loading SmolVLA model...")
        print("⏳ First run: downloading ~2-3GB (3-5 minutes)")
        print("   Subsequent runs: loads from cache (~30s)")

        try:
            model_id = "lerobot/smolvla_base"
            model = SmolVLAPolicy.from_pretrained(model_id)
            model = model.to(device)
            model.eval()

            print("✅ SmolVLA model loaded")

            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {total_params/1e6:.1f}M")
            print(f"   GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False

        # Step 3: Setup preprocessors/postprocessors (WITHOUT dataset stats)
        print("\nStep 3: Setting up preprocessors...")
        try:
            preprocess, postprocess = make_pre_post_processors(
                model.config,
                model_id,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print("✅ Preprocessors ready")
            print("   ⚠️  NOTE: No dataset stats loaded (may cause tiny movements)")
        except Exception as e:
            print(f"❌ Preprocessor setup failed: {e}")
            traceback.print_exc()
            return False

        # Step 4: Configure cameras
        print("\nStep 4: Configuring cameras...")
        try:
            # IMPORTANT: SmolVLA expects camera keys: camera1, camera2, camera3
            # NOT "top" and "wrist"! This is a critical fix.
            # Using camera1 for top view, camera2 for wrist view
            camera_config = {
                "camera1": OpenCVCameraConfig(index_or_path=camera_top, width=640, height=480, fps=30),
                "camera2": OpenCVCameraConfig(index_or_path=camera_wrist, width=640, height=480, fps=30),
            }
            print(f"✅ Camera config: camera1={camera_top} (top, 640x480), camera2={camera_wrist} (wrist, 640x480)")
            print(f"   Note: Using camera1/camera2 keys to match SmolVLA training")
        except Exception as e:
            print(f"❌ Camera config failed: {e}")
            return False

        # Step 5: Initialize robot
        print("\nStep 5: Initializing robot...")
        try:
            robot_cfg = SO101FollowerConfig(port=arm_port, id=arm_id, cameras=camera_config)
            robot = SO101Follower(robot_cfg)
            robot.connect()
            print(f"✅ Robot connected: {arm_port}")
        except Exception as e:
            print(f"❌ Robot connection failed: {e}")
            traceback.print_exc()
            return False

        # Step 6: Setup dataset features mapping
        print("\nStep 6: Setting up feature mappings...")
        try:
            action_features = hw_to_dataset_features(robot.action_features, "action")
            obs_features = hw_to_dataset_features(robot.observation_features, "observation")
            dataset_features = {**action_features, **obs_features}
            print("✅ Feature mappings ready")
        except Exception as e:
            print(f"❌ Feature mapping failed: {e}")
            traceback.print_exc()
            robot.disconnect()
            return False

        # Step 7: Run inference loop
        print("\nStep 7: Starting inference loop...")
        print("="*60)
        print(f"Running at 30Hz for {duration} seconds")
        print(f"Task: {task}")
        print("="*60 + "\n")

        total_steps = int(duration * 30)  # 30Hz
        inference_times = []

        # Robot type for multi-embodiment (leave empty for single robot)
        robot_type = ""

        for step in range(total_steps):
            step_start = time.time()

            try:
                # Get observation from robot
                obs = robot.get_observation()

                # Build inference frame (combines obs + task + robot_type)
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=task,
                    robot_type=robot_type
                )

                # Preprocess observation
                obs_processed = preprocess(obs_frame)

                # Model inference
                with torch.no_grad():
                    action = model.select_action(obs_processed)

                # Postprocess action
                action = postprocess(action)

                # Convert to robot action format
                action = make_robot_action(action, dataset_features)

                # Log actions every 30 steps to see what's being sent
                if step % 30 == 0:
                    try:
                        print(f"\n🔍 Action type: {type(action)}")
                        print(f"🔍 Action keys: {action.keys() if isinstance(action, dict) else 'Not a dict'}")

                        # Print all action values
                        if isinstance(action, dict):
                            for key, value in action.items():
                                if torch.is_tensor(value):
                                    val_np = value.cpu().numpy()
                                elif isinstance(value, (list, np.ndarray)):
                                    val_np = np.array(value)
                                else:
                                    val_np = value
                                print(f"🎯 {key}: {val_np}")
                        else:
                            print(f"🎯 Action value: {action}")
                    except Exception as e:
                        print(f"\n⚠️  Could not log action: {e}")
                        import traceback
                        traceback.print_exc()

                # Send to robot
                robot.send_action(action)

                # Timing
                step_time = time.time() - step_start
                inference_times.append(step_time)

                # Maintain 30Hz
                if step_time < 0.033:
                    time.sleep(0.033 - step_time)

                # Progress updates
                if step % 30 == 0:  # Every second
                    avg_time = np.mean(inference_times[-30:])
                    freq = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"Step {step}/{total_steps} | "
                          f"Freq: {freq:.1f}Hz | "
                          f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            except Exception as e:
                print(f"⚠️  Error at step {step}: {e}")
                continue

        # Step 8: Results
        print("\n" + "="*60)
        print("Inference Complete!")
        print("="*60)

        avg_time = np.mean(inference_times)
        avg_freq = 1.0 / avg_time

        print(f"Total steps: {total_steps}")
        print(f"Avg inference time: {avg_time*1000:.1f}ms")
        print(f"Avg frequency: {avg_freq:.1f}Hz")

        if avg_freq >= 25:
            print("✅ Performance: GOOD (≥25Hz)")
        elif avg_freq >= 20:
            print("⚠️  Performance: OK (20-25Hz)")
        else:
            print("❌ Performance: POOR (<20Hz)")

        # Cleanup
        robot.disconnect()
        print("\n✅ Test completed successfully!")
        return True

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
    """Main function with interactive setup"""

    print("="*60)
    print("XLeRobot SmolVLA Test (ORIGINAL - NO STATS FIX)")
    print("="*60)

    # Check GPU
    if not check_gpu():
        print("\n❌ GPU check failed")
        return

    # Camera selection
    print("\n" + "="*60)
    print("Camera Configuration")
    print("="*60)

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

    # Task description
    print("\n" + "="*60)
    print("Task Description")
    print("="*60)
    task = input("Enter task ['pick up the red cube']: ").strip()
    if not task:
        task = "pick up the red cube"

    # Confirm and run
    print("\n" + "="*60)
    print("Starting Test")
    print("="*60)
    print("⚠️  Make sure workspace is clear!")
    input("Press Enter when ready...")

    success = test_smolvla_official(arm_port, arm_id, camera_top, camera_wrist, duration, task)

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Observe robot behavior:")
        print("   ✅ Smooth movements = working")
        print("   ✅ Approaches objects = excellent")
        print("   ⚠️  Random movement = needs tuning")
        print("   ⚠️  Tiny movements = missing stats (need stats fix!)")
        print("\n2. If movements are tiny, use test_smolvla_official.py (with stats)")
    else:
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("1. Check error messages above")
        print("2. Verify cameras: ls /dev/video*")
        print("3. Verify robot: ls /dev/ttyACM*")
        print("4. Check LeRobot installation")


if __name__ == "__main__":
    main()
