#!/bin/bash
# Fix RTX 5090 PyTorch CUDA Compatibility
# This script reinstalls PyTorch with CUDA 12.8 support for sm_120 architecture

set -e

echo "============================================================"
echo "RTX 5090 PyTorch CUDA 12.8 Reinstallation Script"
echo "============================================================"
echo ""

# Check if we're in the lerobot environment
if [[ "$CONDA_DEFAULT_ENV" != "lerobot" ]]; then
    echo "❌ Error: Not in 'lerobot' conda environment"
    echo "   Please run: conda activate lerobot"
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Show current PyTorch version
echo "Current PyTorch installation:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch not found or error"
echo ""

# Confirm with user
read -p "Uninstall and reinstall PyTorch with CUDA 12.8? (y/n) [y]: " confirm
confirm=${confirm:-y}

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Step 1: Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio pytorch-triton -y || true

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.8 support..."
echo "  This will download ~2-3 GB and may take a few minutes..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "  Using uv for faster installation..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    echo "  Using pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

echo ""
echo "Step 3: Verifying installation..."
python << 'EOF'
import torch
import sys

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA version: {torch.version.cuda}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU compute capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")

    # Test CUDA operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.T)
        print("✅ CUDA operations working!")
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")
        sys.exit(1)

    # Check if cu128
    if 'cu128' in torch.__version__:
        print("✅ Correct CUDA 12.8 build detected")
    else:
        print(f"⚠️  Warning: Expected cu128, got {torch.__version__}")
else:
    print("❌ CUDA not available!")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ SUCCESS! PyTorch with CUDA 12.8 is installed and working"
    echo "============================================================"
    echo ""
    echo "You can now run your SmolVLA test:"
    echo "  cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components"
    echo "  python test_smolvla_official.py"
else
    echo ""
    echo "============================================================"
    echo "❌ Installation completed but verification failed"
    echo "============================================================"
    echo "Please check the errors above and try manual installation."
    exit 1
fi
