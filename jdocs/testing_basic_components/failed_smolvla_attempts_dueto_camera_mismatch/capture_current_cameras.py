#!/usr/bin/env python3
"""
Capture Current Camera Views
=============================

Captures a few frames from your current camera setup
so you can compare with training data.

Usage:
    python capture_current_cameras.py
"""

import cv2
from pathlib import Path
import time

print("="*60)
print("Capturing Current Camera Views")
print("="*60)

# Configuration (from test_smolvla_official.py)
import sys
camera_top = int(sys.argv[1]) if len(sys.argv) > 1 else 4
camera_wrist = int(sys.argv[2]) if len(sys.argv) > 2 else 6

# Create output directory
output_dir = Path(__file__).parent / "current_camera_views"
output_dir.mkdir(exist_ok=True)

print(f"\nOutput directory: {output_dir}")
print(f"Cameras: {camera_top} (top), {camera_wrist} (wrist)")

try:
    # Open cameras
    print(f"\nOpening camera {camera_top}...")
    cap1 = cv2.VideoCapture(camera_top)
    if not cap1.isOpened():
        print(f"❌ Failed to open camera {camera_top}")
        exit(1)

    print(f"Opening camera {camera_wrist}...")
    cap2 = cv2.VideoCapture(camera_wrist)
    if not cap2.isOpened():
        print(f"❌ Failed to open camera {camera_wrist}")
        cap1.release()
        exit(1)

    print("✅ Both cameras opened")

    # Set resolution
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up cameras
    print("\nWarming up cameras (2 seconds)...")
    time.sleep(2)
    for _ in range(10):
        cap1.read()
        cap2.read()

    # Capture multiple samples
    print("\nCapturing samples...")
    for i in range(5):
        print(f"  Sample {i+1}/5...")

        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1:
            filename1 = output_dir / f"camera_top_{i:02d}.jpg"
            cv2.imwrite(str(filename1), frame1)
            print(f"    ✅ Saved: {filename1.name}")

        if ret2:
            filename2 = output_dir / f"camera_wrist_{i:02d}.jpg"
            cv2.imwrite(str(filename2), frame2)
            print(f"    ✅ Saved: {filename2.name}")

        time.sleep(0.5)

    # Release cameras
    cap1.release()
    cap2.release()

    print("\n" + "="*60)
    print("✅ CAPTURE COMPLETE!")
    print("="*60)
    print(f"\nCurrent camera views saved to: {output_dir}")
    print("\nTo view:")
    print(f"  eog {output_dir}/*.jpg")
    print("\nCompare with training data camera angles!")
    print("\nWhat to check:")
    print("  ✅ Top camera: Can you see entire workspace?")
    print("  ✅ Wrist camera: Can you see gripper and object?")
    print("  ✅ Angles similar to training data?")
    print("  ✅ Cube visible in at least one camera?")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
