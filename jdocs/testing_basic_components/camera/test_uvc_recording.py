#!/usr/bin/env python3
"""
Test script for recording episodes with UVC cameras
Replaces section 1.7 from xlerobot_mvp_plan_v1.md for UVC cameras

This script tests the full recording pipeline with UVC cameras and SO-101 follower arms
"""

import argparse
import sys

def test_basic_recording():
    """
    Test basic video recording with UVC camera (no robot)
    """
    print("="*60)
    print("Basic UVC Camera Recording Test (No Robot)")
    print("="*60)

    import cv2
    import time

    camera_index = int(input("Enter camera index (usually 0): "))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False

    # Set to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Recording at {actual_width}x{actual_height}")

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_recording.mp4', fourcc, 30.0, (actual_width, actual_height))

    print("\nRecording for 5 seconds...")
    print("Press 'q' to stop early")

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 5.0:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1

            # Show preview
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    duration = time.time() - start_time
    fps = frame_count / duration

    print(f"\n✅ Recording complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   FPS: {fps:.1f}")
    print(f"   Saved to: test_recording.mp4")

    return True


def test_lerobot_recording(robot_port=None, camera_index=0):
    """
    Test LeRobot recording with UVC camera and SO-101 follower
    This is the corrected version of section 1.7
    """
    print("\n" + "="*60)
    print("LeRobot Recording Test (UVC Camera + SO-101)")
    print("="*60)

    try:
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
        import numpy as np
        import time

        print("\n1. Initializing SO-101 robot...")
        if robot_port is None:
            robot_port = input("Enter robot port (e.g., /dev/ttyACM0): ")

        # Initialize robot
        robot = ManipulatorRobot(
            robot_type="so101",
            port=robot_port,
            calibration_dir="~/.cache/lerobot/calibration"
        )
        print("   ✅ Robot connected")

        print("\n2. Initializing UVC camera...")
        camera = OpenCVCamera(
            camera_index=camera_index,
            fps=30,
            width=1920,
            height=1080
        )
        print(f"   ✅ Camera {camera_index} initialized")

        print("\n3. Testing synchronized data collection...")
        print("   Recording 3 seconds of data (robot state + camera)")

        data = {
            'timestamps': [],
            'joint_positions': [],
            'images': []
        }

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 3.0:
            # Get robot state
            joint_pos = robot.get_joint_positions()

            # Get camera frame
            image = camera.async_read()

            # Record
            data['timestamps'].append(time.time() - start_time)
            data['joint_positions'].append(joint_pos)
            data['images'].append(image)

            frame_count += 1
            time.sleep(1/30)  # 30Hz

        print(f"\n✅ Collected {frame_count} synchronized frames")
        print(f"   Joint positions shape: {np.array(data['joint_positions']).shape}")
        print(f"   Images: {len(data['images'])} frames")
        print(f"   Image shape: {data['images'][0].shape}")

        # Cleanup
        robot.disconnect()
        camera.disconnect()

        return True

    except ImportError as e:
        print(f"\n⚠️  LeRobot not installed: {e}")
        print("\nTo install LeRobot, run:")
        print("  conda activate lerobot")
        print("  pip install -e '.[feetech]'")
        return False

    except Exception as e:
        print(f"\n❌ Error during LeRobot recording test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode_recording():
    """
    Test full episode recording using LeRobot's record script
    This shows the CORRECTED command for UVC cameras
    """
    print("\n" + "="*60)
    print("Full Episode Recording (LeRobot Script)")
    print("="*60)

    print("\nTo record full episodes with your UVC camera, use:")
    print("\n" + "-"*60)
    print("python -m lerobot.scripts.control_robot record \\")
    print("  --robot-path=lerobot/configs/robot/so101.yaml \\")
    print("  --robot-overrides='~cameras' \\")  # Disable default cameras
    print("  --fps=30 \\")
    print("  --repo-id=YOUR_HF_USERNAME/xlerobot_test \\")
    print("  --num-episodes=5 \\")
    print("  --warmup-time-s=3 \\")
    print("  --episode-time-s=30 \\")
    print("  --reset-time-s=5")
    print("-"*60)

    print("\nNote: This requires custom camera configuration")
    print("You'll need to create a config file for your UVC cameras")

    choice = input("\nWould you like me to generate a camera config file? (y/n): ")
    if choice.lower() == 'y':
        generate_camera_config()


def generate_camera_config():
    """
    Generate a LeRobot camera configuration for UVC cameras
    """
    print("\n" + "="*60)
    print("Generating UVC Camera Configuration")
    print("="*60)

    num_cameras = int(input("\nHow many cameras do you have? "))

    config = {
        'cameras': {}
    }

    for i in range(num_cameras):
        print(f"\nCamera {i+1}:")
        name = input(f"  Name (e.g., 'top', 'wrist', 'hand'): ")
        index = int(input(f"  Video device index (e.g., 0 for /dev/video0): "))

        config['cameras'][name] = {
            'type': 'opencv',
            'index': index,
            'fps': 30,
            'width': 1920,
            'height': 1080,
            'use_rgb': True
        }

    # Save config
    import yaml
    config_path = '/home/jrobot/project/XLeRobot/lerobot/configs/robot/xlerobot_cameras.yaml'

    print(f"\n{'='*60}")
    print("Generated configuration:")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False))

    choice = input(f"\nSave to {config_path}? (y/n): ")
    if choice.lower() == 'y':
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UVC camera recording for XLeRobot")
    parser.add_argument('--mode', choices=['basic', 'lerobot', 'full', 'config'],
                       default='basic',
                       help='Test mode: basic (OpenCV only), lerobot (with robot), full (episode recording), config (generate camera config)')
    parser.add_argument('--robot-port', type=str, help='Robot port (e.g., /dev/ttyACM0)')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index')

    args = parser.parse_args()

    if args.mode == 'basic':
        test_basic_recording()
    elif args.mode == 'lerobot':
        test_lerobot_recording(robot_port=args.robot_port, camera_index=args.camera_index)
    elif args.mode == 'full':
        test_full_episode_recording()
    elif args.mode == 'config':
        generate_camera_config()
