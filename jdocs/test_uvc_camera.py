#!/usr/bin/env python3
"""
Test script for USB UVC cameras (1080P USB2.0 UVC Camera)
Replaces section 1.6 from xlerobot_mvp_plan_v1.md for UVC cameras
"""

import cv2
import numpy as np

def list_cameras():
    """Find all available video devices"""
    print("Scanning for cameras...")
    available_cameras = []

    # Check first 10 video device indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                print(f"  Camera {i}: {width}x{height} @ {fps}fps")
            cap.release()

    return available_cameras

def test_camera(camera_index=0, width=1920, height=1080, fps=30):
    """
    Test a single UVC camera with live preview

    Args:
        camera_index: Video device index (e.g., 0 for /dev/video0)
        width: Desired width (1920 for 1080p)
        height: Desired height (1080 for 1080p)
        fps: Desired frame rate
    """
    print(f"\n{'='*60}")
    print(f"Testing Camera {camera_index}")
    print(f"Resolution: {width}x{height} @ {fps}fps")
    print(f"{'='*60}\n")

    # Open camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open camera {camera_index}")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera opened successfully!")
    print(f"Actual resolution: {actual_width}x{actual_height} @ {actual_fps}fps")

    if actual_width != width or actual_height != height:
        print(f"⚠️  Warning: Requested {width}x{height} but got {actual_width}x{actual_height}")

    print("\nPress 'q' to quit, 's' to save a test image")

    frame_count = 0
    import time
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ ERROR: Failed to read frame")
            break

        frame_count += 1

        # Calculate actual FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            measured_fps = frame_count / elapsed

            # Add FPS overlay
            cv2.putText(frame, f"FPS: {measured_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {actual_width}x{actual_height}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Camera {camera_index}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow(f'Camera {camera_index} Test', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f'camera_{camera_index}_test.jpg'
            cv2.imwrite(filename, frame)
            print(f"\n✅ Saved image to {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    print(f"\n{'='*60}")
    print(f"Camera {camera_index} Test Results:")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"{'='*60}\n")

    if avg_fps >= fps * 0.8:  # Within 80% of target
        print(f"✅ Camera {camera_index} working well!")
        return True
    else:
        print(f"⚠️  Camera {camera_index} FPS lower than expected")
        return True  # Still functional, just slower


def test_lerobot_opencv_camera():
    """
    Test using LeRobot's OpenCVCamera class (if available)
    This is what you'll use for actual robot control
    """
    print(f"\n{'='*60}")
    print(f"Testing LeRobot OpenCVCamera Integration")
    print(f"{'='*60}\n")

    try:
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

        # Initialize camera with LeRobot
        camera = OpenCVCamera(
            camera_index=0,
            fps=30,
            width=1920,
            height=1080
        )

        print("✅ LeRobot OpenCVCamera initialized successfully!")

        # Read a frame
        image = camera.async_read()
        print(f"✅ Frame captured! Shape: {image.shape}")
        print(f"   Expected: (1080, 1920, 3) for 1080p RGB")

        # Save test image
        cv2.imwrite('lerobot_camera_test.jpg', image)
        print(f"✅ Saved test image to lerobot_camera_test.jpg")

        # Cleanup
        camera.disconnect()

        return True

    except ImportError as e:
        print(f"⚠️  LeRobot OpenCVCamera not available: {e}")
        print("   This is okay for basic testing. LeRobot will be installed in Stage 1.1")
        return False
    except Exception as e:
        print(f"❌ Error testing LeRobot camera: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("XLeRobot UVC Camera Test")
    print("USB2.0 1080P UVC Camera (130° wide angle)")
    print("="*60)

    # Step 1: List all cameras
    cameras = list_cameras()

    if not cameras:
        print("\n❌ No cameras found!")
        print("\nTroubleshooting:")
        print("1. Check USB connection")
        print("2. Run: ls -l /dev/video*")
        print("3. Check permissions: sudo usermod -a -G video $USER")
        print("   (then log out and log back in)")
        exit(1)

    print(f"\n✅ Found {len(cameras)} camera(s)")

    # Step 2: Test each camera with live preview
    print("\n" + "="*60)
    print("Live Camera Tests")
    print("="*60)

    for cam in cameras:
        choice = input(f"\nTest camera {cam['index']}? (y/n/q to quit): ").lower()
        if choice == 'q':
            break
        elif choice == 'y':
            # Test at full 1080p resolution
            test_camera(cam['index'], width=1920, height=1080, fps=30)

    # Step 3: Test LeRobot integration (if available)
    print("\n" + "="*60)
    print("Optional: LeRobot Integration Test")
    print("="*60)

    choice = input("\nTest LeRobot OpenCVCamera integration? (y/n): ").lower()
    if choice == 'y':
        test_lerobot_opencv_camera()

    print("\n" + "="*60)
    print("Camera testing complete!")
    print("="*60)
