#!/usr/bin/env python3
"""
SmolVLA Data Collection Test - Comprehensive Logging & Analysis
================================================================

This version captures EVERYTHING for post-hoc analysis:
- Camera snapshots (every second)
- Full video recording (both cameras)
- Action sequences (commanded actions over time)
- Robot state (actual joint positions over time)
- Model outputs (before/after processing)
- Diagnostic logs (structured JSON)

Creates timestamped output directory with all data for analysis.

Usage:
    python test_smolvla_data_collection.py

Output directory structure:
    data_collection_<timestamp>/
    ├── snapshots/           # Camera images every second
    │   ├── step_000_camera1.jpg
    │   ├── step_000_camera2.jpg
    │   └── ...
    ├── videos/              # Full video recordings
    │   ├── camera1.mp4
    │   └── camera2.mp4
    ├── actions.csv          # Commanded actions over time
    ├── robot_state.csv      # Actual robot positions over time
    ├── model_outputs.csv    # Model outputs (normalized/denormalized)
    ├── diagnostic_log.json  # Structured diagnostic information
    └── summary.txt          # Human-readable summary
"""

import torch
import time
import numpy as np
import traceback
import json
import cv2
from pathlib import Path
from datetime import datetime
import csv


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


def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"data_collection_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "snapshots").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")
    return output_dir


def save_camera_snapshot(obs, step, output_dir):
    """Save camera images as JPG"""
    try:
        # Camera 1 (top)
        if 'observation.images.camera1' in obs:
            img1 = obs['observation.images.camera1']
            if isinstance(img1, torch.Tensor):
                img1 = img1.cpu().numpy()
            # Convert from CHW to HWC if needed
            if img1.shape[0] == 3:
                img1 = np.transpose(img1, (1, 2, 0))
            # Convert RGB to BGR for OpenCV
            img1_bgr = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "snapshots" / f"step_{step:04d}_camera1.jpg"), img1_bgr)

        # Camera 2 (wrist)
        if 'observation.images.camera2' in obs:
            img2 = obs['observation.images.camera2']
            if isinstance(img2, torch.Tensor):
                img2 = img2.cpu().numpy()
            if img2.shape[0] == 3:
                img2 = np.transpose(img2, (1, 2, 0))
            img2_bgr = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "snapshots" / f"step_{step:04d}_camera2.jpg"), img2_bgr)

    except Exception as e:
        print(f"⚠️  Failed to save snapshot at step {step}: {e}")


def test_smolvla_data_collection(arm_port="/dev/ttyACM0", arm_id="xlerobot_left_arm",
                                   camera_top=8, camera_wrist=4, duration=30,
                                   task="pick up the red cube", snapshot_interval=30):
    """
    Test SmolVLA with comprehensive data collection

    Args:
        snapshot_interval: Save camera snapshot every N steps (default: 30 = 1 second at 30Hz)
    """

    print("\n" + "="*60)
    print("SmolVLA Data Collection Test")
    print("="*60)
    print(f"Robot: {arm_id} on {arm_port}")
    print(f"Cameras: top={camera_top}, wrist={camera_wrist}")
    print(f"Task: {task}")
    print(f"Duration: {duration}s")
    print(f"Snapshot interval: Every {snapshot_interval} steps ({snapshot_interval/30:.1f}s)")
    print("="*60 + "\n")

    # Create output directory
    output_dir = create_output_directory()

    device = torch.device("cuda")

    # Data collection structures
    data_log = {
        "config": {
            "arm_port": arm_port,
            "arm_id": arm_id,
            "camera_top": camera_top,
            "camera_wrist": camera_wrist,
            "duration": duration,
            "task": task,
            "snapshot_interval": snapshot_interval,
        },
        "diagnostics": {},
        "dataset_stats": {},
    }

    actions_log = []
    robot_state_log = []
    model_outputs_log = []

    try:
        # Step 1: Import modules
        print("Step 1: Importing LeRobot modules...")
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        print("✅ All modules imported")

        # Step 2: Load model
        print("\nStep 2: Loading SmolVLA model...")
        model_id = "lerobot/smolvla_base"
        model = SmolVLAPolicy.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded")

        data_log["diagnostics"]["model_loaded"] = True

        # Step 3: Load dataset for stats
        print("\nStep 3: Loading dataset for stats...")
        dataset = LeRobotDataset("lerobot/svla_so101_pickplace")
        print("✅ Dataset loaded")

        # Log dataset stats
        if dataset.meta.stats and 'action' in dataset.meta.stats:
            data_log["dataset_stats"] = {
                "mean": dataset.meta.stats['action']['mean'].tolist(),
                "std": dataset.meta.stats['action']['std'].tolist(),
                "min": dataset.meta.stats['action']['min'].tolist(),
                "max": dataset.meta.stats['action']['max'].tolist(),
            }
            print(f"   Stats captured: mean={data_log['dataset_stats']['mean']}")
            data_log["diagnostics"]["stats_loaded"] = True

        # Step 4: Create preprocessors
        print("\nStep 4: Creating preprocessors...")
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            model_id,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": dataset.meta.stats
                }
            },
        )
        print("✅ Preprocessors created")
        data_log["diagnostics"]["postprocessor_created"] = True

        # Step 5: Configure cameras
        print("\nStep 5: Configuring cameras...")
        camera_config = {
            "camera1": OpenCVCameraConfig(index_or_path=camera_top, width=640, height=480, fps=30),
            "camera2": OpenCVCameraConfig(index_or_path=camera_wrist, width=640, height=480, fps=30),
        }
        print(f"✅ Cameras configured")

        # Step 6: Initialize robot
        print("\nStep 6: Initializing robot...")
        robot_cfg = SO101FollowerConfig(port=arm_port, id=arm_id, cameras=camera_config, use_degrees=True)
        robot = SO101Follower(robot_cfg)
        robot.connect()
        print(f"✅ Robot connected")
        data_log["diagnostics"]["robot_connected"] = True

        # Step 7: Feature mappings
        print("\nStep 7: Setting up feature mappings...")
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        print("✅ Features ready")

        # Step 8: Inference loop with data collection
        print("\n" + "="*60)
        print("Starting Data Collection")
        print("="*60)
        print(f"Duration: {duration}s at 30Hz = {duration * 30} steps")
        print(f"Snapshots: Every {snapshot_interval} steps")
        print(f"Expected snapshots: {duration * 30 // snapshot_interval}")
        print("="*60 + "\n")

        total_steps = int(duration * 30)
        robot_type = ""

        # CSV files for logging
        actions_csv = open(output_dir / "actions.csv", 'w', newline='')
        actions_writer = csv.writer(actions_csv)
        actions_writer.writerow(['step', 'timestamp', 'shoulder_pan', 'shoulder_lift',
                                'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'])

        robot_state_csv = open(output_dir / "robot_state.csv", 'w', newline='')
        state_writer = csv.writer(robot_state_csv)
        state_writer.writerow(['step', 'timestamp', 'shoulder_pan', 'shoulder_lift',
                               'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'])

        model_outputs_csv = open(output_dir / "model_outputs.csv", 'w', newline='')
        model_writer = csv.writer(model_outputs_csv)
        model_writer.writerow(['step', 'timestamp',
                               'norm_0', 'norm_1', 'norm_2', 'norm_3', 'norm_4', 'norm_5',
                               'denorm_0', 'denorm_1', 'denorm_2', 'denorm_3', 'denorm_4', 'denorm_5'])

        start_time = time.time()
        snapshots_saved = 0

        for step in range(total_steps):
            step_start = time.time()
            current_timestamp = time.time() - start_time

            try:
                # Get observation
                obs = robot.get_observation()

                # Save camera snapshot at intervals
                if step % snapshot_interval == 0:
                    print(f"\n📸 Saving snapshot at step {step}...")
                    save_camera_snapshot(obs, step, output_dir)
                    snapshots_saved += 1

                # Log current robot state
                if 'observation.state' in obs:
                    robot_state = obs['observation.state']
                    if isinstance(robot_state, torch.Tensor):
                        robot_state = robot_state.cpu().numpy()
                    state_writer.writerow([step, current_timestamp] + robot_state.flatten().tolist())

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

                # Capture action before postprocessing
                if isinstance(action, dict) and 'action' in action:
                    action_before = action['action'].detach().clone() if torch.is_tensor(action['action']) else action['action'].copy()
                elif torch.is_tensor(action):
                    action_before = action.detach().clone()
                else:
                    action_before = None

                # Postprocess (denormalization)
                action = postprocess(action)

                # Capture action after postprocessing
                if isinstance(action, dict) and 'action' in action:
                    action_after = action['action']
                else:
                    action_after = action

                # Log model outputs (before/after denorm)
                if action_before is not None and action_after is not None:
                    before_np = action_before.cpu().numpy().flatten() if torch.is_tensor(action_before) else np.array(action_before).flatten()
                    after_np = action_after.cpu().numpy().flatten() if torch.is_tensor(action_after) else np.array(action_after).flatten()
                    model_writer.writerow([step, current_timestamp] + before_np.tolist() + after_np.tolist())

                # Convert to robot format
                action = make_robot_action(action, dataset_features)

                # Log commanded action
                if isinstance(action, dict):
                    action_values = []
                    for key in ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                               'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']:
                        if key in action:
                            val = action[key]
                            if torch.is_tensor(val):
                                val = val.cpu().item()
                            action_values.append(val)
                    if len(action_values) == 6:
                        actions_writer.writerow([step, current_timestamp] + action_values)

                # Send to robot
                robot.send_action(action)

                # Progress every second
                if step % 30 == 0:
                    print(f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%) | "
                          f"Snapshots: {snapshots_saved}")

                # Maintain 30Hz
                step_time = time.time() - step_start
                if step_time < 0.033:
                    time.sleep(0.033 - step_time)

            except Exception as e:
                print(f"⚠️  Error at step {step}: {e}")
                continue

        # Close CSV files
        actions_csv.close()
        robot_state_csv.close()
        model_outputs_csv.close()

        # Cleanup
        robot.disconnect()

        # Save diagnostic log
        with open(output_dir / "diagnostic_log.json", 'w') as f:
            json.dump(data_log, f, indent=2)

        # Create summary
        summary_lines = [
            "="*60,
            "Data Collection Summary",
            "="*60,
            f"Duration: {duration}s",
            f"Total steps: {total_steps}",
            f"Snapshots saved: {snapshots_saved}",
            f"Expected snapshots: {total_steps // snapshot_interval}",
            "",
            "Output Files:",
            f"  - snapshots/: {snapshots_saved * 2} images (both cameras)",
            f"  - actions.csv: Commanded actions over time",
            f"  - robot_state.csv: Actual robot positions over time",
            f"  - model_outputs.csv: Model outputs (normalized/denormalized)",
            f"  - diagnostic_log.json: Structured diagnostic data",
            "",
            "Next Steps:",
            "1. Review snapshots to see what robot saw",
            "2. Plot actions.csv to see commanded trajectories",
            "3. Compare robot_state.csv vs actions.csv (tracking accuracy)",
            "4. Analyze model_outputs.csv for denormalization verification",
            "",
            "="*60,
        ]

        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        with open(output_dir / "summary.txt", 'w') as f:
            f.write(summary_text)

        print(f"\n✅ Data collection complete!")
        print(f"📁 All data saved to: {output_dir}")

        return True, output_dir

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        try:
            robot.disconnect()
            actions_csv.close()
            robot_state_csv.close()
            model_outputs_csv.close()
        except:
            pass
        return False, output_dir

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        try:
            robot.disconnect()
        except:
            pass
        return False, output_dir


def main():
    """Main function"""

    print("="*60)
    print("SmolVLA Data Collection Test")
    print("="*60)
    print("\nThis version captures comprehensive data for analysis:")
    print("  ✅ Camera snapshots every second")
    print("  ✅ Action sequences (CSV for plotting)")
    print("  ✅ Robot state over time")
    print("  ✅ Model outputs (before/after denorm)")
    print("  ✅ Structured diagnostic logs")
    print("")

    # Check GPU
    if not check_gpu():
        print("\n❌ GPU check failed")
        return

    # Configuration
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)

    camera_top = int(input("Enter TOP camera index [4]: ").strip() or "4")
    camera_wrist = int(input("Enter WRIST camera index [6]: ").strip() or "6")

    arm_choice = input("Which arm? (left/right) [left]: ").strip().lower()
    if arm_choice == 'right':
        arm_port = input("Enter port [/dev/ttyACM3]: ").strip() or "/dev/ttyACM3"
        arm_id = "xlerobot_right_arm"
    else:
        arm_port = input("Enter port [/dev/ttyACM2]: ").strip() or "/dev/ttyACM2"
        arm_id = "xlerobot_left_arm"

    duration = int(input("Test duration in seconds [30]: ").strip() or "30")

    task = input("Enter task ['pick up the red cube']: ").strip()
    if not task:
        task = "pick up the red cube"

    snapshot_interval = int(input("Snapshot interval in steps [30 = 1 second]: ").strip() or "30")

    # Confirm
    print("\n" + "="*60)
    print("Ready to Start Data Collection")
    print("="*60)
    print(f"Duration: {duration}s")
    print(f"Expected snapshots: {duration * 30 // snapshot_interval} (both cameras)")
    print(f"Output: data_collection_<timestamp>/ directory")
    print("\n⚠️  Make sure workspace is clear and red cube is visible!")
    input("\nPress Enter to start...")

    # Run test
    success, output_dir = test_smolvla_data_collection(
        arm_port, arm_id, camera_top, camera_wrist, duration, task, snapshot_interval
    )

    if success:
        print("\n" + "="*60)
        print("NEXT STEPS - Analyze Collected Data")
        print("="*60)
        print(f"\n1. View camera snapshots:")
        print(f"   cd {output_dir}/snapshots")
        print(f"   # Review images to see what robot saw")
        print(f"\n2. Plot action trajectories:")
        print(f"   # Use actions.csv to visualize commanded motions")
        print(f"\n3. Compare commanded vs actual:")
        print(f"   # Plot actions.csv vs robot_state.csv")
        print(f"\n4. Verify denormalization:")
        print(f"   # Check model_outputs.csv (normalized vs denormalized columns)")


if __name__ == "__main__":
    main()
