#!/bin/bash
################################################################################
# Pi0.5 LoRA Mini-MVP Training Script for SO-101
# Purpose: Quick pipeline validation with existing 10 episodes
# Duration: ~5-10 minutes for 100 steps
# Goal: Verify LoRA implementation works before committing to longer training
#
# STATUS: VALIDATED (2024-11-24)
# Pi0.5 LoRA working with LeRobot training pipeline
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 LoRA Mini-MVP Pipeline Test - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Purpose: Quick validation of Pi0.5 LoRA training pipeline (100 steps)"
echo "Dataset: Existing 10 episodes"
echo "Duration: ~5-10 minutes"
echo ""
echo "What this tests:"
echo "  - Pi0.5 model loading from HuggingFace (lerobot/pi05_base)"
echo "  - LoRA adapter application (PEFT)"
echo "  - Dataset loading (LeRobot v3 format)"
echo "  - Training loop execution"
echo "  - Memory usage verification"
echo ""

# Configuration
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_BASE="/home/jrobot/project/XLeRobot/outputs"
# Use nanoseconds for unique folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S%N)
OUTPUT_DIR="${OUTPUT_BASE}/pi05_mini_mvp_${TIMESTAMP}"
LEROBOT_ROOT="/home/jrobot/project/lerobot"

# Training hyperparameters
MAX_STEPS=100
BATCH_SIZE=2
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

echo "========================================================================"
echo "Step 1/4: Environment Setup"
echo "========================================================================"

# Activate lerobot environment
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "from lerobot.policies.pi05 import PI05Policy; print('Pi0.5 policy available')"

echo ""
echo "========================================================================"
echo "Step 2/4: Dataset Validation"
echo "========================================================================"

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])")
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_frames'])")

echo "Dataset: $DATASET_PATH"
echo "  - Episodes: $TOTAL_EPISODES"
echo "  - Frames: $TOTAL_FRAMES"
echo "  - Format: LeRobot v3 (no conversion needed)"

# NOTE: Do NOT create output directory - let LeRobot create it
# LeRobot will fail if the directory already exists
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Step 3/4: Run Pi0.5 LoRA Mini-MVP Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Model: Pi0.5 (lerobot/pi05_base)"
echo "  - LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  - Steps: $MAX_STEPS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gradient Checkpointing: Enabled"
echo ""

# Change to LeRobot directory
cd $LEROBOT_ROOT

# Run mini-MVP training
echo "Starting training at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create temp log file (output dir doesn't exist yet)
TEMP_LOG="/tmp/pi05_training_${TIMESTAMP}.log"

lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.use_lora=true \
    --policy.lora_rank=$LORA_RANK \
    --policy.lora_alpha=$LORA_ALPHA \
    --policy.lora_dropout=$LORA_DROPOUT \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=xlerobot_mini_mvp \
    --dataset.root=$DATASET_PATH \
    --dataset.video_backend=pyav \
    --steps=$MAX_STEPS \
    --batch_size=$BATCH_SIZE \
    --log_freq=10 \
    --save_freq=$MAX_STEPS \
    --save_checkpoint=true \
    --wandb.enable=false \
    --output_dir=$OUTPUT_DIR \
    --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb"}' \
    2>&1 | tee $TEMP_LOG

TRAINING_EXIT_CODE=$?

# Copy log to output directory (now exists after training)
if [ -d "$OUTPUT_DIR" ]; then
    cp $TEMP_LOG $OUTPUT_DIR/training.log
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training finished at $(date)"

echo ""
echo "========================================================================"
echo "Step 4/4: Results Summary"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    ls -lh $OUTPUT_DIR/ 2>/dev/null || echo "  (directory listing not available)"
    echo ""
    echo "Loss progression:"
    grep "loss:" $TEMP_LOG | tail -10 || echo "  (no loss entries found)"
    echo ""
    echo "Next steps:"
    echo "  1. Review training.log for any warnings"
    echo "  2. If successful, proceed to MVP training:"
    echo "     bash scripts/train_pi05_mvp_lora.sh"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $TEMP_LOG"
    exit $TRAINING_EXIT_CODE
fi
