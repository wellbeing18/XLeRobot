#!/bin/bash
################################################################################
# Pi0.5 LoRA MVP Training Script for SO-101
# Purpose: Training with 50+ episodes for initial validation
# Duration: ~1-2 hours for 500 steps
# VRAM Usage: ~18-22GB on RTX 5090
#
# STATUS: READY FOR USE (LoRA implementation validated 2024-11-24)
# Requires: Collected dataset with 50+ episodes
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 LoRA MVP Training - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Purpose: MVP training with 50+ episodes (500 steps)"
echo "Duration: ~1-2 hours"
echo "Expected VRAM: 18-22GB"
echo ""
echo "Prerequisites:"
echo "  - Run mini-MVP validation first: bash scripts/train_pi05_mini_mvp.sh"
echo "  - Collect 50+ episodes of demonstration data"
echo ""

# Configuration - ADJUST THESE FOR YOUR SETUP
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/pi05_mvp_lora"
LEROBOT_ROOT="/home/jrobot/project/lerobot"

# Training hyperparameters
MAX_STEPS=500
BATCH_SIZE=4
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
LOG_FREQ=25
SAVE_FREQ=100

echo "========================================================================"
echo "Step 1/4: Environment Setup"
echo "========================================================================"

# Check LeRobot
if [ ! -d "$LEROBOT_ROOT" ]; then
    echo "ERROR: LeRobot not found at $LEROBOT_ROOT"
    exit 1
fi

# Activate lerobot environment
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "========================================================================"
echo "Step 2/4: Dataset Validation"
echo "========================================================================"

if [ ! -d "$DATASET_PATH/meta" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])")
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_frames'])")

echo "Dataset: $DATASET_PATH"
echo "  - Episodes: $TOTAL_EPISODES"
echo "  - Frames: $TOTAL_FRAMES"
echo "  - Format: LeRobot v3"

if [ "$TOTAL_EPISODES" -lt 50 ]; then
    echo ""
    echo "WARNING: Only $TOTAL_EPISODES episodes found."
    echo "For MVP training, recommend 50+ episodes."
    echo "Continuing anyway..."
fi

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Step 3/4: Run Pi0.5 LoRA MVP Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Model: Pi0.5 (lerobot/pi05_base)"
echo "  - LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  - Steps: $MAX_STEPS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gradient Checkpointing: Enabled"
echo ""
echo "Monitor GPU: watch -n 5 nvidia-smi"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Change to LeRobot directory
cd $LEROBOT_ROOT

# Run MVP training
echo "Starting training at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.use_lora=true \
    --policy.lora_rank=$LORA_RANK \
    --policy.lora_alpha=$LORA_ALPHA \
    --policy.lora_dropout=$LORA_DROPOUT \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=xlerobot_mvp \
    --dataset.root=$DATASET_PATH \
    --dataset.video_backend=pyav \
    --steps=$MAX_STEPS \
    --batch_size=$BATCH_SIZE \
    --log_freq=$LOG_FREQ \
    --save_freq=$SAVE_FREQ \
    --save_checkpoint=true \
    --wandb.enable=false \
    --output_dir=$OUTPUT_DIR \
    --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb"}' \
    2>&1 | tee $OUTPUT_DIR/training.log

TRAINING_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training finished at $(date)"

echo ""
echo "========================================================================"
echo "Step 4/4: Results Summary"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "MVP Training completed successfully!"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    ls -lh $OUTPUT_DIR/
    echo ""
    echo "Loss progression:"
    grep "loss:" $OUTPUT_DIR/training.log | tail -20
    echo ""
    echo "Success Criteria Check:"
    echo "  [ ] Training completed 500 steps without crashes"
    echo "  [ ] Loss decreased from initial value"
    echo "  [ ] VRAM stayed <22GB"
    echo ""
    echo "Next steps:"
    echo "  1. If loss is decreasing: Run inference test"
    echo "  2. Collect 100 episodes for full training"
    echo "  3. Run: bash scripts/train_pi05_full_lora.sh"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $OUTPUT_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi
