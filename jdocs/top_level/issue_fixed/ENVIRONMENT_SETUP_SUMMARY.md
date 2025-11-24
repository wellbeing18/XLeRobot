# Environment Setup Summary

**Date:** 2025-11-24

---

## Quick Answer to Your Questions

### 1. Pi0.5 LoRA Future Environment?

**Answer: No new environment needed! Use existing `lerobot` environment.**

**Why:**
- Pi0.5 is already part of LeRobot (installed in `lerobot` env)
- PyTorch 2.9.0 in `lerobot` is actually better (newer, supports Pi0.5)
- When implementing Pi0.5 LoRA, just add PEFT to `lerobot`:
  ```bash
  conda activate lerobot
  pip install peft
  ```

### 2. GR00T Location Changed to ~/project?

**Answer: Yes! ✅ Updated all scripts.**

**New location:** `~/project/Isaac-GR00T` (keeps all projects organized)

---

## Your Final Environment Setup

### Two Environments, Clear Separation

```
~/project/
├── lerobot/          → LeRobot codebase
├── Isaac-GR00T/      → GR00T codebase (will be cloned here)
└── XLeRobot/         → Your project

Conda Environments:
├── lerobot env       → For Pi0.5, LeRobot models, data collection
└── groot env         → For GR00T N1.5 training
```

### Environment Usage

| Task | Environment | Location |
|------|-------------|----------|
| **GR00T N1.5 training** | `groot` | `~/project/Isaac-GR00T` |
| **Pi0.5 LoRA (future)** | `lerobot` | `~/project/lerobot` |
| **Data collection** | `lerobot` | `~/project/XLeRobot` |
| **Your scripts** | Either (auto-switches) | `~/project/XLeRobot` |

---

## Setup Instructions (10 minutes)

### Step 1: Set Up GR00T Environment

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/setup_groot_env.sh
```

**What this does:**
1. Creates `~/project/` directory
2. Clones Isaac-GR00T to `~/project/Isaac-GR00T`
3. Creates `groot` conda environment (Python 3.10)
4. Installs Isaac-GR00T with PyTorch 2.5.1
5. Installs flash-attn 2.7.1 (~5-10 min compilation)

**Time:** ~10 minutes total

### Step 2: Add to ~/.bashrc (Optional but Recommended)

```bash
echo 'export ISAAC_GROOT_ROOT=$HOME/project/Isaac-GR00T' >> ~/.bashrc
source ~/.bashrc
```

This makes the GR00T path available in all new terminals.

### Step 3: Verify Setup

```bash
# Activate groot environment
conda activate groot

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should show: PyTorch: 2.5.1

python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
# Should show: flash-attn: 2.7.1.post4

python -c "import gr00t; print('GR00T: OK')"
# Should show: GR00T: OK
```

---

## After Setup: Run Mini-MVP Test

```bash
# Already in groot environment
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

This will:
- Use your existing 10 episodes
- Run ultra-short training (100 steps, ~5-10 minutes)
- Generate inspection report for validation
- Create logs for debugging

---

## Why This Design?

### Benefits of Two Separate Environments

1. **✅ Version Safety**
   - `groot`: PyTorch 2.5.1 (NVIDIA tested)
   - `lerobot`: PyTorch 2.9.0 (newer, for Pi0.5)
   - No conflicts!

2. **✅ Clear Separation**
   - NVIDIA models → `groot` env
   - LeRobot models → `lerobot` env
   - Easy to remember!

3. **✅ Risk Isolation**
   - If one breaks, other still works
   - Can debug independently

4. **✅ Official Support**
   - NVIDIA tests GR00T with specific versions
   - We follow their recommendations

### Cost

- **Disk:** ~5GB per environment (10GB total) - negligible
- **Setup time:** 10 minutes (one-time)
- **Mental overhead:** None (scripts handle switching)

---

## Daily Usage

### For GR00T Work

```bash
conda activate groot
cd ~/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

### For Pi0.5 Work (Future)

```bash
conda activate lerobot
cd ~/project/lerobot
# Pi0.5 commands here
```

### Scripts Handle Switching Automatically

All training scripts automatically activate the correct environment:
- `train_groot_*.sh` → activates `groot`
- `train_pi05_*.sh` → activates `lerobot`

You don't need to remember!

---

## Project Structure After Setup

```
/home/jrobot/
├── project/
│   ├── lerobot/              ← LeRobot codebase (already exists)
│   │   ├── src/lerobot/      ← Pi0.5 code
│   │   └── ...
│   ├── Isaac-GR00T/          ← GR00T codebase (will be created)
│   │   ├── scripts/
│   │   │   └── gr00t_finetune.py
│   │   └── ...
│   └── XLeRobot/             ← Your project (already exists)
│       ├── scripts/
│       │   ├── setup_groot_env.sh
│       │   ├── train_groot_mini_mvp.sh
│       │   └── ...
│       └── jdocs/
│
└── anaconda3/envs/
    ├── lerobot/              ← LeRobot environment (exists)
    │   ├── PyTorch 2.9.0
    │   ├── LeRobot 0.4.1
    │   └── (Pi0.5 ready)
    └── groot/                ← GR00T environment (will be created)
        ├── PyTorch 2.5.1
        ├── flash-attn 2.7.1
        └── Isaac-GR00T
```

---

## Troubleshooting

### "conda: command not found"

```bash
conda init bash
exec bash
```

### "Isaac-GR00T not found"

Make sure you ran setup script:
```bash
bash scripts/setup_groot_env.sh
```

### "Wrong PyTorch version"

Check you're in the right environment:
```bash
conda env list
# Active environment has '*' marker

# Switch to correct one
conda activate groot    # For GR00T
conda activate lerobot  # For Pi0.5
```

---

## Next Steps

1. **✅ Run setup script** (creates groot environment):
   ```bash
   bash scripts/setup_groot_env.sh
   ```

2. **✅ Run mini-MVP test** (validates pipeline):
   ```bash
   conda activate groot
   bash scripts/train_groot_mini_mvp.sh
   ```

3. **✅ Review inspection report**:
   ```bash
   cat outputs/groot_mini_mvp_test/mini_mvp_inspection_report.txt
   ```

4. **✅ Share results** for validation

---

## Summary

**Your Questions Answered:**

1. **Pi0.5 LoRA environment?**
   - ✅ Use existing `lerobot` environment
   - No new environment needed!

2. **GR00T location?**
   - ✅ Changed to `~/project/Isaac-GR00T`
   - All scripts updated!

**Setup:**
- Run one command: `bash scripts/setup_groot_env.sh`
- Wait 10 minutes
- Done!

**Daily Use:**
- Scripts handle environment switching
- You just run the script you need
- No mental overhead!

Ready to set up? 🚀
