#!/usr/bin/env python3
"""
Extract Frames from Training Videos
====================================

Extracts sample frames from downloaded training videos
to see camera angles used during training.

Usage:
    python extract_video_frames.py
"""

import cv2
from pathlib import Path
import numpy as np

print("="*60)
print("Extracting Frames from Training Videos")
print("="*60)

# Paths
videos_dir = Path(__file__).parent / "training_videos" / "videos"
output_dir = Path(__file__).parent / "training_data_samples"
output_dir.mkdir(exist_ok=True)

# Video files
video_up = videos_dir / "observation.images.up" / "chunk-000" / "file-000.mp4"
video_side = videos_dir / "observation.images.side" / "chunk-000" / "file-000.mp4"

print(f"\nOutput directory: {output_dir}")
print(f"\nVideo files:")
print(f"  Top camera: {video_up}")
print(f"    Exists: {video_up.exists()}")
print(f"  Side camera: {video_side}")
print(f"    Exists: {video_side.exists()}")

def extract_frames(video_path, camera_name, output_dir, num_frames=5):
    """Extract evenly spaced frames from a video"""
    print(f"\nProcessing {camera_name}...")

    if not video_path.exists():
        print(f"  ❌ Video not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Failed to open video: {video_path}")
        return 0

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.1f}s")

    # Sample frames evenly
    frame_indices = [
        0,                          # First frame
        total_frames // 4,          # 25%
        total_frames // 2,          # 50%
        total_frames * 3 // 4,      # 75%
        total_frames - 1            # Last frame
    ]

    saved_count = 0
    for idx in frame_indices[:num_frames]:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read frame
        ret, frame = cap.read()
        if ret:
            # Save frame
            filename = output_dir / f"{camera_name}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"    ✅ Saved frame {idx}: {filename.name}")
            saved_count += 1
        else:
            print(f"    ⚠️  Failed to read frame {idx}")

    cap.release()
    return saved_count

# Extract frames from both videos
total_saved = 0
total_saved += extract_frames(video_up, "camera_top", output_dir)
total_saved += extract_frames(video_side, "camera_side", output_dir)

print("\n" + "="*60)
print(f"✅ EXTRACTION COMPLETE! ({total_saved} frames saved)")
print("="*60)
print(f"\nTraining frames saved to: {output_dir}")
print("\nTo view:")
print(f"  eog {output_dir}/*.jpg")
print("\nNow compare these with your current camera views:")
print(f"  eog {Path(__file__).parent / 'current_camera_views'}/*.jpg")
print("\nLook for:")
print("  ✅ Camera height/angle")
print("  ✅ Distance from robot")
print("  ✅ Field of view coverage")
print("  ✅ Workspace visibility")
print("  ✅ Gripper visibility (for side/wrist camera)")
