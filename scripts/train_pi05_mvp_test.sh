#!/bin/bash
################################################################################
# Pi0.5 MVP Training Test Script
# Purpose: Quick validation that training pipeline works with your 10 episodes
# Duration: ~30-45 minutes for 100 steps
# VRAM Usage: 18-22GB / 24GB
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "Pi0.5 MVP Training Test - SO-ARM101 Left Arm"
echo "========================================================================"
echo ""
echo "Purpose: Validate training pipeline works before collecting more data"
echo "Duration: ~30-45 minutes (100 steps only)"
echo "Dataset: Your existing 10 episodes"
echo ""
echo "What this tests:"
echo "  ✓ Dataset loads without errors"
echo "  ✓ Pi0.5 model fits in VRAM (24GB)"
echo "  ✓ Training runs without crashes"
echo "  ✓ No NaN losses or gradient explosions"
echo "  ✓ Checkpoint saves correctly"
echo ""
echo "What this DOES NOT test:"
echo "  ✗ Model quality (can't learn from 10 episodes)"
echo "  ✗ Performance improvement (need 50-100 episodes)"
echo "  ✗ Generalization (too little data)"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
LEROBOT_ROOT="/home/jrobot/project/lerobot"
DATASET_ROOT="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/pi05_mvp_test"
DATASET_REPO_ID="lerobot/xlerobot_mvp_pick_15fps"

# Activate conda environment
echo "[1/4] Activating lerobot conda environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Verify environment
echo "[2/4] Verifying environment..."
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify dataset exists
echo "[3/4] Verifying dataset..."
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    echo "Expected structure:"
    echo "  $DATASET_ROOT/meta/info.json"
    echo "  $DATASET_ROOT/meta/stats.json"
    echo "  $DATASET_ROOT/data/"
    exit 1
fi

echo "Dataset found: $DATASET_ROOT"
echo "Episodes: $(python -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['total_episodes'])")"
echo "Frames: $(python -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['total_frames'])")"

# Run MVP training test
echo "[4/4] Starting MVP training test (100 steps)..."
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor GPU usage in another terminal with: watch -n 5 nvidia-smi"
echo ""

python $LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/pi05_base \
  --gradient_checkpointing=true \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.root=$DATASET_ROOT \
  --steps=100 \
  --batch_size=2 \
  --num_workers=4 \
  --optimizer_lr=3e-6 \
  --optimizer_grad_clip_norm=1.0 \
  --optimizer_weight_decay=0.01 \
  --scheduler_warmup_steps=10 \
  --scheduler_decay_steps=100 \
  --scheduler_decay_lr=2.5e-7 \
  --eval_freq=0 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=100 \
  --output_dir=$OUTPUT_DIR \
  --wandb.enable=false \
  --seed=1000

echo ""
echo "========================================================================"
echo "MVP Training Test Complete!"
echo "========================================================================"
echo ""
echo "✅ Pass Criteria:"
echo "   - Training completed 100 steps without crashes"
echo "   - No OOM (out of memory) errors"
echo "   - No NaN losses"
echo "   - Checkpoint saved successfully"
echo ""
echo "Check output at: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. If test passed: Collect 40-90 more episodes (reach 50-100 total)"
echo "  2. Then run full training: bash scripts/train_pi05_so101.sh"
echo ""
echo "If test failed, see troubleshooting in jdocs/top_level/PI05_TRAINING_GUIDE.md"
echo ""
