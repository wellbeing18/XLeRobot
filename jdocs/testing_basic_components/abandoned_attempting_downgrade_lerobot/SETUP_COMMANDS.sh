#!/bin/bash
#
# LeRobot Downgrade Setup Script
# Complete automation of environment setup
#
# Usage: bash SETUP_COMMANDS.sh
#

set -e  # Exit on error

echo "======================================================================"
echo "LeRobot Downgrade Setup - Automated Installation"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Create lerobot_old conda environment"
echo "  2. Checkout LeRobot from August 31, 2024"
echo "  3. Install old LeRobot with dependencies"
echo "  4. Verify GPU support"
echo ""
echo "Estimated time: 10-15 minutes"
echo "======================================================================"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================================
# STEP 1: Create Conda Environment
# ============================================================================

echo ""
echo "======================================================================"
echo "STEP 1: Creating lerobot_old conda environment"
echo "======================================================================"
echo ""

# Check if environment already exists
if conda env list | grep -q "lerobot_old"; then
    echo "⚠️  lerobot_old environment already exists"
    read -p "Remove and recreate? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        echo "Removing existing environment..."
        conda env remove -n lerobot_old -y
    else
        echo "Skipping environment creation..."
        SKIP_ENV=true
    fi
fi

if [ "$SKIP_ENV" != "true" ]; then
    echo "Creating new environment..."
    conda create -n lerobot_old python=3.10 -y
    echo "✅ Environment created"
fi

# ============================================================================
# STEP 2: Checkout Old LeRobot
# ============================================================================

echo ""
echo "======================================================================"
echo "STEP 2: Checking out LeRobot from August 31, 2024"
echo "======================================================================"
echo ""

cd /home/jrobot/project/lerobot

# Save current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check if branch already exists
if git show-ref --verify --quiet refs/heads/lerobot_v03_august31; then
    echo "⚠️  lerobot_v03_august31 branch already exists"
    read -p "Checkout existing branch? (y/n): " checkout
    if [ "$checkout" = "y" ]; then
        git checkout lerobot_v03_august31
    fi
else
    echo "Creating new branch from August 31 commit..."
    git checkout -b lerobot_v03_august31 c0da806
fi

# Verify checkout
CURRENT_COMMIT=$(git log --oneline -1)
echo "Current commit: $CURRENT_COMMIT"
echo "✅ Checked out pre-v0.4.0 LeRobot"

# ============================================================================
# STEP 3: Install LeRobot
# ============================================================================

echo ""
echo "======================================================================"
echo "STEP 3: Installing LeRobot with dependencies"
echo "======================================================================"
echo ""
echo "⏳ This will take 5-10 minutes..."
echo ""

# Activate environment and install
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot_old

# Install PyAV via conda (required for video processing)
echo "Installing PyAV via conda..."
conda install -c conda-forge av -y

# Install LeRobot
echo "Installing LeRobot (old version)..."
pip install -e ".[feetech]"

echo "✅ LeRobot installed"

# ============================================================================
# STEP 4: Verify Installation
# ============================================================================

echo ""
echo "======================================================================"
echo "STEP 4: Verifying installation"
echo "======================================================================"
echo ""

# Test Python version
echo "Python version:"
python --version

# Test imports
echo ""
echo "Testing imports..."
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('✅ SmolVLA import works')"

# Test GPU
echo ""
echo "Testing GPU support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print('✅ GPU support working')
else:
    print('❌ GPU not available - check CUDA installation')
"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "======================================================================"
echo "✅ SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Environment: lerobot_old"
echo "LeRobot version: August 31, 2024 (pre-v0.4.0)"
echo "Branch: lerobot_v03_august31"
echo ""
echo "======================================================================"
echo "NEXT STEPS:"
echo "======================================================================"
echo ""
echo "1. The environment is already activated (lerobot_old)"
echo ""
echo "2. Navigate to test directory:"
echo "   cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot"
echo ""
echo "3. Run the test script:"
echo "   python test_old_lerobot_models.py"
echo ""
echo "4. When prompted, test these models in order:"
echo "   - jhou/smolvla_pickplace"
echo "   - HuggingFadeUser/my_smolvla"
echo "   - saood65/my_smolvla"
echo ""
echo "======================================================================"
echo "ENVIRONMENT SWITCHING:"
echo "======================================================================"
echo ""
echo "To use OLD LeRobot (for testing pretrained):"
echo "   conda activate lerobot_old"
echo "   cd /home/jrobot/project/lerobot"
echo "   git checkout lerobot_v03_august31"
echo ""
echo "To use NEW LeRobot (for fine-tuning later):"
echo "   conda activate lerobot"
echo "   cd /home/jrobot/project/lerobot"
echo "   git checkout $CURRENT_BRANCH"
echo ""
echo "======================================================================"
echo ""
echo "Good luck with testing! 🤖"
echo ""
