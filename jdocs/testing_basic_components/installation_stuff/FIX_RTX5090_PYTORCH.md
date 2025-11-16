# Fix RTX 5090 PyTorch CUDA Compatibility Issue

## Problem
Your RTX 5090 has CUDA compute capability `sm_120`, but PyTorch 2.7.1+cu126 only supports up to `sm_90`.

Error:
```
CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
```

## Root Cause
You installed PyTorch compiled for CUDA 12.6, which was built before the RTX 5090 (Blackwell/sm_120) existed.

## Solution
Reinstall PyTorch with CUDA 12.8 support, which includes sm_120 architecture.

## Step-by-Step Fix

### 1. Activate your conda environment
```bash
conda activate lerobot
```

### 2. Uninstall current PyTorch
```bash
pip uninstall torch torchvision torchaudio -y
```

### 3. Install PyTorch with CUDA 12.8 support
```bash
# Using pip (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# OR using uv (faster)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.7.1+cu128  (or similar with cu128)
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090 Laptop GPU
```

### 5. Test with a simple CUDA operation
```bash
python -c "import torch; x = torch.randn(100, 100).cuda(); y = torch.matmul(x, x.T); print('✅ CUDA operations working!')"
```

### 6. Re-run your SmolVLA test
```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components
python test_smolvla_official.py
```

## Important Notes

1. **CUDA Toolkit Version**: You have `cuda-toolkit=12.6` from conda, but this is fine. PyTorch comes with its own CUDA runtime (12.8), which will be used for PyTorch operations.

2. **No Need to Change conda CUDA**: The conda `cuda-toolkit` is for compilation tools (nvcc, etc.), not for PyTorch runtime. PyTorch bundles its own CUDA libraries.

3. **Driver Version**: Your nvidia-driver-580-open is sufficient for CUDA 12.8.

4. **Why cu128 not cu126**: RTX 5090 (sm_120) support was added in CUDA 12.8, so you must use PyTorch compiled with CUDA 12.8+.

## Verification Checklist

- [ ] PyTorch version shows `cu128` (not `cu126`)
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU name shows correctly
- [ ] Simple CUDA operations work without errors
- [ ] SmolVLA test runs without "no kernel image" errors

## Troubleshooting

### If you still see compatibility warnings after reinstall:

1. Check PyTorch was fully uninstalled:
```bash
pip list | grep torch
```

2. Force reinstall:
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### If CUDA operations still fail:

1. Restart your shell/terminal
2. Reactivate conda environment
3. Try a fresh Python session

## Quick One-Liner Fix

```bash
conda activate lerobot && pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && python -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}')"
```
