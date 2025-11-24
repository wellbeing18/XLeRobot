#!/bin/bash
################################################################################
# Pi0.5 LoRA MVP Training Script for SO-101
# Purpose: Quick validation that Pi0.5 LoRA training works with your dataset
# Duration: ~1-2 hours for 500 steps
# VRAM Usage: Target 18-22GB (experimental - first test of Pi0.5 LoRA)
#
# ⚠️  STATUS: NOT IMPLEMENTED YET - DESIGN STUB ONLY
# Pi0.5 LoRA is not yet wired into LeRobot. This script will fail until:
#   1. LoRA fields added to PI05Config (use_lora, lora_rank, lora_alpha, lora_dropout)
#   2. PEFT wrapping implemented in modeling_pi05.py
#   3. CLI flags fixed to match lerobot_train.py expectations
#
# For now, use GR00T scripts instead (train_groot_so101_mvp.sh)
# See: jdocs/top_level/lora/gpt5_review_comments.md for details
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 LoRA MVP Training Test - SO-101 Left Arm"
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
echo "     bash scripts/train_groot_so101_mvp.sh"
echo ""
echo "  2. OR implement Pi0.5 LoRA first (see implementation guide):"
echo "     jdocs/top_level/lora/gpt5_lora_suggestion.md"
echo ""
echo "========================================================================"
exit 1

# Below is the design spec for when LoRA is implemented
# DO NOT RUN until implementation is complete

echo "Purpose: Validate Pi0.5 LoRA training pipeline (EXPERIMENTAL)"
echo "Duration: ~1-2 hours (500 steps)"
echo "Dataset: Your existing 10 episodes"
echo "LoRA Rank: 16"
echo ""
echo "What this tests:"
echo "  ✓ Dataset compatible with Pi0.5"
echo "  ✓ LoRA implementation works correctly"
echo "  ✓ Training fits in 24GB VRAM"
echo "  ✓ Loss decreases consistently"
echo ""
echo "NOTE: This is experimental - Pi0.5 LoRA was just implemented."
echo "      If issues arise, fallback to GR00T (proven to work)."
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/pi05_mvp_lora"
LEROBOT_ROOT="/home/jrobot/project/lerobot"

# Check if LeRobot is installed
if [ ! -d "$LEROBOT_ROOT" ]; then
    echo "❌ ERROR: LeRobot not found at $LEROBOT_ROOT"
    echo ""
    echo "Please ensure LeRobot is installed at this location."
    exit 1
fi

# Activate conda environment
echo "[1/4] Activating lerobot conda environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "[2/4] Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check PEFT is installed
echo "Checking PEFT installation..."
if ! python -c "import peft" 2>/dev/null; then
    echo "❌ ERROR: PEFT not installed"
    echo "Installing PEFT..."
    pip install peft
fi

# Check dataset
echo "[3/4] Verifying dataset..."
if [ ! -d "$DATASET_PATH/meta" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    echo "Expected structure:"
    echo "  $DATASET_PATH/meta/info.json"
    echo "  $DATASET_PATH/data/"
    echo "  $DATASET_PATH/videos/"
    exit 1
fi

echo "Dataset found: $DATASET_PATH"
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    echo "Episodes: $(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "unknown")"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run MVP training test
echo "[4/4] Starting Pi0.5 LoRA MVP training (500 steps)..."
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor GPU usage in another terminal: watch -n 5 nvidia-smi"
echo "Monitor CPU RAM: watch -n 10 'free -h'"
echo ""
echo "Training Configuration:"
echo "  - Model: Pi0.5 (4B params)"
echo "  - LoRA: Enabled (rank 16, alpha 32)"
echo "  - Batch Size: 8"
echo "  - Steps: 500 (quick test)"
echo "  - Learning Rate: 3e-6 (conservative for MVP)"
echo "  - Gradient Checkpointing: Enabled (saves VRAM)"
echo ""

cd $LEROBOT_ROOT

python src/lerobot/scripts/lerobot_train.py \
    --policy.path=lerobot/pi05_base \
    --policy.use_lora=true \
    --policy.lora_rank=16 \
    --policy.lora_alpha=32 \
    --policy.lora_dropout=0.1 \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=lerobot/xlerobot_mvp_pick_15fps \
    --dataset.root=$DATASET_PATH \
    --steps=500 \
    --batch_size=8 \
    --optimizer.lr=3e-6 \
    --optimizer.grad_clip_norm=1.0 \
    --optimizer.weight_decay=0.01 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=50 \
    --scheduler.num_decay_steps=500 \
    --scheduler.peak_lr=3e-6 \
    --scheduler.decay_lr=3e-7 \
    --output_dir=$OUTPUT_DIR \
    --save_checkpoint=true \
    --save_freq=500 \
    --log_freq=10 \
    --seed=1000

echo ""
echo "========================================================================"
echo "Pi0.5 LoRA MVP Training Test Complete!"
echo "========================================================================"
echo ""
echo "✅ Pass Criteria:"
echo "   - Training completed 500 steps without crashes"
echo "   - VRAM stayed <22GB throughout"
echo "   - Loss decreased from initial value"
echo "   - Checkpoint saved successfully"
echo "   - LoRA adapters loaded correctly"
echo ""
echo "Check output at: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. If test passed: Collect 65-90 more episodes (reach 75-100 total)"
echo "  2. Then run full training: bash scripts/train_pi05_full_lora.sh"
echo "  3. Compare with GR00T results"
echo ""
echo "If test failed, fallback to GR00T (proven implementation)."
echo "See: jdocs/top_level/finetuning_pi0_5_issue.md"
echo ""
