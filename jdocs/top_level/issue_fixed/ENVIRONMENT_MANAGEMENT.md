# Environment Management Guide

**Last Updated:** 2025-11-24

---

## Overview: Two Separate Environments

This project uses **two separate conda environments** for different models:

| Environment | Purpose | Models | Key Packages |
|-------------|---------|--------|--------------|
| **`lerobot`** | LeRobot policies (Pi0.5, etc.) | Pi0.5 | PyTorch 2.9, Transformers 4.53, LeRobot 0.4.1 |
| **`groot`** | NVIDIA GR00T training | GR00T N1.5 | PyTorch 2.5, flash-attn 2.7, Isaac-GR00T |

---

## Why Separate Environments?

### Technical Reasons

1. **Version Conflicts:**
   - `lerobot`: PyTorch 2.9.0+cu128 (cutting edge)
   - `groot`: PyTorch 2.5.1 (NVIDIA tested)
   - These can have different APIs/behaviors

2. **Different Dependencies:**
   - `groot` requires flash-attn (NVIDIA-optimized attention)
   - `lerobot` uses standard PyTorch attention
   - flash-attn can conflict with other packages

3. **Isolation:**
   - If one environment breaks, the other still works
   - Can test/debug independently

### Practical Benefits

- ✅ Official support (NVIDIA tests GR00T with specific versions)
- ✅ Clean rollback (can recreate one without affecting the other)
- ✅ Clear separation (no confusion about which packages for which model)

### Cost

- ~5GB extra disk space per environment
- Need to remember to switch environments (but scripts handle this)

---

## Current Status

### ✅ `lerobot` Environment (Already Set Up)

**Location:** `/home/jrobot/anaconda3/envs/lerobot`

**Key Packages:**
- Python 3.10.18
- PyTorch 2.9.0+cu128
- Transformers 4.53.3
- Diffusers 0.35.2
- Accelerate 1.10.1
- LeRobot 0.4.1 (editable install from `/home/jrobot/project/lerobot`)

**Used For:**
- Pi0.5 model (when LoRA implemented)
- LeRobot data collection/processing
- General robotics development

**Activate:**
```bash
conda activate lerobot
```

### ❌ `groot` Environment (Not Yet Set Up)

**Will Be Located:** `/home/jrobot/anaconda3/envs/groot`

**Will Include:**
- Python 3.10
- PyTorch 2.5.1
- Transformers 4.51.3
- flash-attn 2.7.1.post4
- Isaac-GR00T (editable install from `~/project/Isaac-GR00T`)

**Used For:**
- GR00T N1.5 LoRA training
- GR00T model inference

**Setup:**
```bash
cd /home/jrobot/project/XLeRobot
bash scripts/setup_groot_env.sh
```

---

## Quick Setup Guide

### Set Up GR00T Environment (One-Time, 10 minutes)

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/setup_groot_env.sh
```

This will:
1. Clone Isaac-GR00T to `~/project/Isaac-GR00T`
2. Create `groot` conda environment
3. Install all dependencies
4. Install flash-attn (takes 5-10 min to compile)

**After Setup:**
```bash
# Add to ~/.bashrc for convenience
echo 'export ISAAC_GROOT_ROOT=$HOME/project/Isaac-GR00T' >> ~/.bashrc
source ~/.bashrc
```

---

## Daily Usage

### Working with GR00T

```bash
# Activate groot environment
conda activate groot

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should show: PyTorch: 2.5.1

# Run training
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

### Working with LeRobot/Pi0.5

```bash
# Activate lerobot environment
conda activate lerobot

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should show: PyTorch: 2.9.0+cu128

# Run LeRobot commands
cd /home/jrobot/project/lerobot
python src/lerobot/scripts/lerobot_train.py ...
```

---

## Environment Switching

### Method 1: Let Scripts Handle It (Recommended)

Our training scripts automatically activate the correct environment:

```bash
# GR00T scripts automatically activate 'groot'
bash scripts/train_groot_mini_mvp.sh

# Pi0.5 scripts automatically activate 'lerobot'
bash scripts/train_pi05_mvp_lora.sh
```

### Method 2: Manual Switching

```bash
# Check current environment
conda env list
# Active environment has '*' marker

# Switch to groot
conda activate groot

# Switch to lerobot
conda activate lerobot

# Deactivate (return to base)
conda deactivate
```

---

## Troubleshooting

### "conda: command not found"

```bash
# Initialize conda for your shell
conda init bash
# Restart shell
exec bash
```

### "Environment already exists"

```bash
# Remove and recreate
conda env remove -n groot
bash scripts/setup_groot_env.sh
```

### "Wrong PyTorch version"

Make sure you're in the right environment:

```bash
# Check active environment
which python
# Should show: .../envs/groot/bin/python or .../envs/lerobot/bin/python

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

### "ImportError: No module named X"

You're probably in the wrong environment:

```bash
# For GR00T work
conda activate groot

# For LeRobot work
conda activate lerobot
```

---

## Managing Disk Space

### Check Environment Sizes

```bash
du -sh ~/anaconda3/envs/*
```

Expected sizes:
- `lerobot`: ~5GB
- `groot`: ~5GB
- Total: ~10GB

### Clean Up Old Environments

```bash
# List all environments
conda env list

# Remove unused environments
conda env remove -n lerobot_old
conda env remove -n old_env_name

# Clean conda cache
conda clean --all
```

---

## Advanced: Environment Comparison

### View Differences

```bash
# Export package lists
conda activate lerobot
pip list > /tmp/lerobot_packages.txt

conda activate groot
pip list > /tmp/groot_packages.txt

# Compare
diff /tmp/lerobot_packages.txt /tmp/groot_packages.txt
```

### Key Differences

| Package | lerobot | groot | Notes |
|---------|---------|-------|-------|
| **PyTorch** | 2.9.0+cu128 | 2.5.1 | Major version difference |
| **Transformers** | 4.53.3 | 4.51.3 | Minor difference |
| **flash-attn** | Not installed | 2.7.1.post4 | GR00T only |
| **LeRobot** | 0.4.1 | Not installed | LeRobot only |
| **Isaac-GR00T** | Not installed | Latest | GR00T only |

---

## Best Practices

### 1. Always Verify Environment

Before running any script:
```bash
conda env list
# Check which has '*' marker
```

### 2. Use Environment-Specific Terminals

Keep separate terminal tabs/windows:
- Terminal 1: `conda activate groot` (for GR00T work)
- Terminal 2: `conda activate lerobot` (for LeRobot work)

### 3. Add Aliases (Optional)

Add to `~/.bashrc`:
```bash
alias groot='conda activate groot && cd $HOME/Isaac-GR00T'
alias lerobot='conda activate lerobot && cd $HOME/project/lerobot'
alias xle='cd /home/jrobot/project/XLeRobot'
```

Then just type:
```bash
groot    # Activates groot env + goes to GR00T directory
lerobot  # Activates lerobot env + goes to LeRobot directory
xle      # Goes to XLeRobot project
```

### 4. Check Before Installing

Always verify you're in the right environment before `pip install`:
```bash
which python
# Should show the environment path
```

---

## Quick Reference

### Common Commands

```bash
# List environments
conda env list

# Activate environment
conda activate groot
conda activate lerobot

# Deactivate
conda deactivate

# Check active environment
echo $CONDA_DEFAULT_ENV

# Check Python location
which python

# Check package version
python -c "import PACKAGE; print(PACKAGE.__version__)"

# Export environment
conda env export > environment.yml

# Remove environment
conda env remove -n ENV_NAME
```

---

## Summary

**Two Environments = Better:**
- ✅ No version conflicts
- ✅ Clean separation
- ✅ Easy debugging
- ✅ Official support
- ✅ Can work on both models simultaneously

**Setup Once:**
```bash
bash scripts/setup_groot_env.sh
```

**Daily Use:**
```bash
# For GR00T
conda activate groot

# For LeRobot
conda activate lerobot
```

**Training scripts handle environment switching automatically!**
