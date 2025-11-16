#!/usr/bin/env python3
"""
Extract Training Data Sample Images
====================================

Downloads and saves sample frames from the training dataset
so you can see what camera angles were used during training.

Usage:
    python extract_training_samples.py
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np

print("="*60)
print("Extracting Training Data Samples")
print("="*60)

# Create output directory
output_dir = Path(__file__).parent / "training_data_samples"
output_dir.mkdir(exist_ok=True)

print(f"\nOutput directory: {output_dir}")

try:
    # Import after creating output dir so we see message first
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("\nLoading dataset (this will download if first time)...")
    print("Dataset: lerobot/svla_so101_pickplace")

    # Load dataset metadata without loading videos
    from datasets import load_dataset

    # Load just metadata first
    print("\nLoading dataset info...")
    ds = load_dataset("lerobot/svla_so101_pickplace", split="train")

    print(f"✅ Dataset loaded: {len(ds)} samples")
    print(f"   Columns: {ds.column_names}")

    # Find image columns
    image_cols = [col for col in ds.column_names if 'image' in col.lower()]
    print(f"   Image columns: {image_cols}")

    # Get a few samples from different parts of dataset
    indices = [
        0,           # First frame
        100,         # Early frame
        len(ds)//4,  # 25% through
        len(ds)//2,  # Middle
        len(ds)*3//4 # 75% through
    ]

    print(f"\nExtracting {len(indices)} sample frames...")

    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        sample = ds[idx]

        # Save each image column
        for col in image_cols:
            if col in sample and sample[col] is not None:
                # Get image - it should be a PIL Image
                img = sample[col]

                # Save it
                clean_col = col.replace('.', '_').replace('/', '_')
                filename = output_dir / f"sample_{idx:06d}_{clean_col}.jpg"

                if isinstance(img, Image.Image):
                    img.save(filename)
                    print(f"  ✅ Saved: {filename.name}")
                elif isinstance(img, dict) and 'bytes' in img:
                    # It's encoded, try to decode
                    from io import BytesIO
                    img_pil = Image.open(BytesIO(img['bytes']))
                    img_pil.save(filename)
                    print(f"  ✅ Saved: {filename.name}")
                else:
                    print(f"  ⚠️  Unknown format for {col}: {type(img)}")

    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nSample images saved to: {output_dir}")
    print("\nTo view them:")
    print(f"  eog {output_dir}/*.jpg")
    print(f"  # OR")
    print(f"  ls {output_dir}/")
    print("\nLook for:")
    print("  - observation_images_up_*.jpg  (top/upper camera)")
    print("  - observation_images_side_*.jpg (side camera)")
    print("\nCompare these angles with your camera setup!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying alternative method...")

    import traceback
    traceback.print_exc()

    print("\n" + "="*60)
    print("Alternative: Download from HuggingFace directly")
    print("="*60)
    print("\n1. Go to: https://huggingface.co/datasets/lerobot/svla_so101_pickplace")
    print("2. Click 'Files and versions' tab")
    print("3. Navigate to: videos/chunk-000/observation.images.up/")
    print("4. Download: episode_000000.mp4")
    print("5. Navigate to: videos/chunk-000/observation.images.side/")
    print("6. Download: episode_000000.mp4")
    print("\nThen open the videos to see camera angles!")
