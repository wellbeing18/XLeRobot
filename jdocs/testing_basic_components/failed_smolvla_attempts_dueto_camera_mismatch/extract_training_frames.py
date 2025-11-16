#!/usr/bin/env python3
"""
Extract Training Data Frames Using LeRobotDataset
==================================================

Uses LeRobotDataset to load videos and extract sample frames
for camera angle comparison.

Usage:
    python extract_training_frames.py
"""

import torch
import cv2
import numpy as np
from pathlib import Path

print("="*60)
print("Extracting Training Frames via LeRobotDataset")
print("="*60)

# Create output directory
output_dir = Path(__file__).parent / "training_data_samples"
output_dir.mkdir(exist_ok=True)

print(f"\nOutput directory: {output_dir}")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("\nLoading dataset with video decoding enabled...")
    print("Dataset: lerobot/svla_so101_pickplace")
    print("(This may take a moment to download/cache videos...)")

    # Load dataset - this should handle video loading
    dataset = LeRobotDataset("lerobot/svla_so101_pickplace")

    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # Sample indices from different parts of dataset
    indices = [
        0,            # First frame
        50,           # Early
        len(dataset)//4,  # 25%
        len(dataset)//2,  # 50%
        len(dataset)*3//4 # 75%
    ]

    print(f"\nExtracting {len(indices)} sample frames...")
    print("Looking for observation keys in dataset...")

    # Get first sample to inspect keys
    sample_0 = dataset[0]
    obs_keys = [k for k in sample_0.keys() if 'observation' in k]
    image_keys = [k for k in obs_keys if 'image' in k]

    print(f"Found observation keys: {obs_keys[:5]}...")
    print(f"Found image keys: {image_keys}")

    if not image_keys:
        print("\n⚠️  No image keys found in sample!")
        print("Available keys:", list(sample_0.keys()))
        raise ValueError("No image observations found")

    saved_count = 0
    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        sample = dataset[idx]

        # Save each image
        for img_key in image_keys:
            if img_key in sample:
                img_tensor = sample[img_key]

                # Convert tensor to numpy array
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.cpu().numpy()

                    # Handle channel-first format (C, H, W) -> (H, W, C)
                    if img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))

                    # Convert to uint8 if needed
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)

                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # Save
                    clean_key = img_key.replace('.', '_').replace('/', '_')
                    filename = output_dir / f"sample_{idx:06d}_{clean_key}.jpg"
                    cv2.imwrite(str(filename), img_bgr)

                    print(f"  ✅ Saved: {filename.name}")
                    saved_count += 1

    print("\n" + "="*60)
    print(f"✅ EXTRACTION COMPLETE! ({saved_count} images saved)")
    print("="*60)
    print(f"\nSample images saved to: {output_dir}")
    print("\nTo view them:")
    print(f"  eog {output_dir}/*.jpg")
    print("\nLook for:")
    print("  - *_observation_images_up_*.jpg  (top/upper camera)")
    print("  - *_observation_images_side_*.jpg (side camera)")
    print("  - OR similar naming pattern")
    print("\nCompare these with your current camera views!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "="*60)
    print("Alternative: Download videos manually")
    print("="*60)
    print("\n1. Go to: https://huggingface.co/datasets/lerobot/svla_so101_pickplace")
    print("2. Click 'Files and versions' tab")
    print("3. Download these videos:")
    print("   - videos/chunk-000/observation.images.up/episode_000000.mp4")
    print("   - videos/chunk-000/observation.images.side/episode_000000.mp4")
    print("\nOr try these commands:")
    print("  huggingface-cli download lerobot/svla_so101_pickplace \\")
    print("    videos/chunk-000/observation.images.up/episode_000000.mp4 \\")
    print("    --repo-type dataset --local-dir training_videos/")
    print("  huggingface-cli download lerobot/svla_so101_pickplace \\")
    print("    videos/chunk-000/observation.images.side/episode_000000.mp4 \\")
    print("    --repo-type dataset --local-dir training_videos/")
