#!/bin/bash
################################################################################
# Pi0.5 Full Training Script for SO-ARM101
# Purpose: Finetune pretrained Pi0.5 on your SO-101 dataset
# Duration: 6-10 hours for 6000 steps (RTX 5090)
# VRAM Usage: 18-22GB / 24GB
# Prerequisites: 50-100 episodes collected
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 Full Training - SO-ARM101 Left Arm"
echo "========================================================================"
echo ""
echo "Duration: ~6-10 hours (6000 steps)"
echo "Prerequisites: 50-100 episodes in dataset"
echo "Expected Results: 40-60% success (50 eps) to 65-80% (100 eps)"
echo ""
echo "IMPORTANT: This will use significant GPU resources."
echo "  - VRAM: 18-22GB / 24GB"
echo "  - Training time: 6-10 hours"
echo "  - Disk space: ~5GB for checkpoints"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
LEROBOT_ROOT="/home/jrobot/project/lerobot"
DATASET_ROOT="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/pi05_so101_v1"
DATASET_REPO_ID="lerobot/xlerobot_mvp_pick_15fps"

# Training hyperparameters (optimized for SO-101 finetuning)
STEPS=6000
BATCH_SIZE=8
NUM_WORKERS=4
LEARNING_RATE=3e-6          # Lower than default (5e-6) to prevent gradient explosion
GRAD_CLIP_NORM=1.0          # CRITICAL: Prevents NaN losses
WARMUP_STEPS=500            # Longer warmup for stability
DECAY_STEPS=6000
LORA_RANK=32                # Balance of capacity vs VRAM
LORA_ALPHA=64

# Logging and checkpointing
LOG_FREQ=50                 # Log every 50 steps
SAVE_FREQ=1000              # Save checkpoint every 1000 steps
EVAL_FREQ=0                 # Disable evaluation (no separate eval set)

# Activate conda environment
echo "[1/5] Activating lerobot conda environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "[2/5] Verifying environment..."
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify dataset
echo "[3/5] Verifying dataset..."
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    echo "Expected structure:"
    echo "  $DATASET_ROOT/meta/info.json"
    echo "  $DATASET_ROOT/meta/stats.json"
    echo "  $DATASET_ROOT/data/"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['total_episodes'])")
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['total_frames'])")

echo "Dataset found: $DATASET_ROOT"
echo "Episodes: $TOTAL_EPISODES"
echo "Frames: $TOTAL_FRAMES"

# Validate episode count
if [ "$TOTAL_EPISODES" -lt 50 ]; then
    echo ""
    echo "WARNING: Only $TOTAL_EPISODES episodes found!"
    echo "Recommended: 50-100 episodes for reliable performance"
    echo ""
    echo "You can:"
    echo "  1. Continue anyway (may underfit)"
    echo "  2. Collect more episodes first (recommended)"
    echo ""
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read
fi

# Create output directory
echo "[4/5] Creating output directory..."
mkdir -p $OUTPUT_DIR
echo "Checkpoints will be saved to: $OUTPUT_DIR"

# Display training configuration
echo "[5/5] Training Configuration:"
echo "  Model: Pi0.5 (4B params) with LoRA"
echo "  LoRA Rank: $LORA_RANK"
echo "  Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Gradient Clipping: $GRAD_CLIP_NORM"
echo "  Warmup Steps: $WARMUP_STEPS"
echo ""
echo "Starting training in 5 seconds..."
echo "Monitor GPU with: watch -n 5 nvidia-smi"
echo "Monitor CPU RAM with: watch -n 10 'free -h'"
echo ""
sleep 5

# Run training
python $LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/pi05_base \
  --policy.use_lora=true \
  --policy.lora_rank=$LORA_RANK \
  --policy.lora_alpha=$LORA_ALPHA \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.root=$DATASET_ROOT \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --optimizer.lr=$LEARNING_RATE \
  --optimizer.grad_clip_norm=$GRAD_CLIP_NORM \
  --optimizer.weight_decay=0.01 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=$WARMUP_STEPS \
  --scheduler.num_decay_steps=$DECAY_STEPS \
  --scheduler.peak_lr=$LEARNING_RATE \
  --scheduler.decay_lr=2.5e-7 \
  --eval_freq=$EVAL_FREQ \
  --log_freq=$LOG_FREQ \
  --save_checkpoint=true \
  --save_freq=$SAVE_FREQ \
  --output_dir=$OUTPUT_DIR \
  --wandb.enable=false \
  --seed=1000

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Evaluate finetuned model on robot"
echo "  2. Compare to pretrained baseline (expected improvement: 40-80%)"
echo "  3. If performance is poor, see troubleshooting guide"
echo ""
echo "To run inference with your finetuned model:"
echo "  python jdocs/top_level/run_pi05_inference_corrected.py \\"
echo "    --checkpoint $OUTPUT_DIR/checkpoint-$STEPS"
echo ""
echo "See jdocs/top_level/PI05_TRAINING_GUIDE.md for evaluation instructions."
echo ""
