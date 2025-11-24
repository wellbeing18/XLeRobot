#!/bin/bash
################################################################################
# Pi0.5 LoRA Full Training Script for SO-101
# Purpose: Full LoRA finetuning for production model
# Duration: ~6-8 hours for 6,000 steps
# VRAM Usage: Target 18-22GB (experimental - Pi0.5 LoRA)
# Prerequisites: 75-100 episodes collected
#
# ⚠️  STATUS: NOT IMPLEMENTED YET - DESIGN STUB ONLY
# Pi0.5 LoRA is not yet wired into LeRobot. This script will fail until:
#   1. LoRA fields added to PI05Config (use_lora, lora_rank, lora_alpha, lora_dropout)
#   2. PEFT wrapping implemented in modeling_pi05.py
#   3. CLI flags fixed to match lerobot_train.py expectations
#
# For now, use GR00T scripts instead (train_groot_so101_full.sh)
# See: jdocs/top_level/lora/gpt5_review_comments.md for details
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 LoRA Full Training - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "❌ ERROR: Pi0.5 LoRA is NOT IMPLEMENTED in LeRobot yet"
echo ""
echo "This script is a design stub for future implementation."
echo "It will fail because:"
echo "  - PI05Config does not have LoRA fields (use_lora, lora_rank, etc.)"
echo "  - modeling_pi05.py does not wrap model with PEFT"
echo "  - CLI flags do not match lerobot_train.py expectations"
echo ""
echo "To proceed:"
echo "  1. Use GR00T scripts instead (proven implementation):"
echo "     bash scripts/train_groot_so101_full.sh"
echo ""
echo "  2. OR implement Pi0.5 LoRA first (see implementation guide):"
echo "     jdocs/top_level/lora/gpt5_lora_suggestion.md"
echo ""
echo "========================================================================"
exit 1

# Below is the design spec for when LoRA is implemented
# DO NOT RUN until implementation is complete

echo "Duration: ~6-8 hours (6,000 steps)"
echo "Prerequisites: 75-100 episodes in dataset"
echo "Expected Results: TBD (experimental implementation)"
echo ""
echo "IMPORTANT: This will use significant GPU resources."
echo "  - VRAM: Target 18-22GB / 24GB (experimental)"
echo "  - Training time: 6-8 hours"
echo "  - Disk space: ~5-10GB for checkpoints"
echo ""
echo "NOTE: This is experimental - Pi0.5 LoRA was just implemented."
echo "      If issues arise, fallback to GR00T (proven to work)."
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/pi05_full_lora_v1"
LEROBOT_ROOT="/home/jrobot/project/lerobot"

# Training hyperparameters (adapted from Pi0.5 config + LoRA best practices)
MAX_STEPS=6000
BATCH_SIZE=16
LEARNING_RATE=2.5e-5      # Pi0.5 default optimizer_lr
LORA_RANK=16              # NVIDIA's recommended rank for 24GB VRAM
SAVE_STEPS=1000           # Save checkpoint every 1000 steps
LOGGING_STEPS=100         # Log every 100 steps

# Check if LeRobot is installed
if [ ! -d "$LEROBOT_ROOT" ]; then
    echo "❌ ERROR: LeRobot not found at $LEROBOT_ROOT"
    echo ""
    echo "Please ensure LeRobot is installed at this location."
    exit 1
fi

# Activate conda environment
echo "[1/5] Activating lerobot conda environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "[2/5] Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check PEFT is installed
echo "Checking PEFT installation..."
if ! python -c "import peft" 2>/dev/null; then
    echo "❌ ERROR: PEFT not installed"
    echo "Installing PEFT..."
    pip install peft
fi

# Verify dataset
echo "[3/5] Verifying dataset..."
if [ ! -d "$DATASET_PATH/meta" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")

echo "Dataset found: $DATASET_PATH"
echo "Episodes: $TOTAL_EPISODES"

# Validate episode count
if [ "$TOTAL_EPISODES" -lt 75 ]; then
    echo ""
    echo "⚠️  WARNING: Only $TOTAL_EPISODES episodes found!"
    echo "Recommended: 75-100 episodes for reliable performance"
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
echo "  LoRA Alpha: 32"
echo "  Steps: $MAX_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Save Frequency: Every $SAVE_STEPS steps"
echo "  Memory Optimization: Gradient checkpointing enabled"
echo ""
echo "Starting training in 5 seconds..."
echo "Monitor GPU with: watch -n 5 nvidia-smi"
echo "Monitor CPU RAM with: watch -n 10 'free -h'"
echo ""
sleep 5

# Change to LeRobot directory
cd $LEROBOT_ROOT

# Run full training with LoRA
python src/lerobot/scripts/lerobot_train.py \
    --policy.path=lerobot/pi05_base \
    --policy.use_lora=true \
    --policy.lora_rank=$LORA_RANK \
    --policy.lora_alpha=32 \
    --policy.lora_dropout=0.1 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=lerobot/xlerobot_mvp_pick_15fps \
    --dataset.root=$DATASET_PATH \
    --steps=$MAX_STEPS \
    --batch_size=$BATCH_SIZE \
    --optimizer.lr=$LEARNING_RATE \
    --optimizer.grad_clip_norm=1.0 \
    --optimizer.weight_decay=0.01 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=$MAX_STEPS \
    --scheduler.peak_lr=$LEARNING_RATE \
    --scheduler.decay_lr=2.5e-6 \
    --output_dir=$OUTPUT_DIR \
    --save_checkpoint=true \
    --save_freq=$SAVE_STEPS \
    --log_freq=$LOGGING_STEPS \
    --seed=1000

echo ""
echo "========================================================================"
echo "Pi0.5 LoRA Full Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Available checkpoints:"
ls -lh $OUTPUT_DIR/checkpoint-*/
echo ""
echo "Next Steps:"
echo "  1. Evaluate finetuned model on robot"
echo "  2. Compare to GR00T model performance"
echo "  3. If performance good, deploy to production"
echo "  4. If performance poor, use GR00T instead (proven implementation)"
echo ""
echo "To run inference:"
echo "  cd $LEROBOT_ROOT"
echo "  python src/lerobot/scripts/lerobot_eval.py \\\\"
echo "    --policy.path=$OUTPUT_DIR/checkpoint-$MAX_STEPS \\\\"
echo "    --robot.type=so101_follower \\\\"
echo "    --robot.cameras=[head,left_wrist]"
echo ""
echo "See jdocs/top_level/finetuning_pi0_5_issue.md for evaluation guide."
echo ""
