# Commands Explained - Where Everything Is

**You said:** "I don't see the creating new conda environment or downloading commands anywhere"

**Here's where they are!**

---

## 📍 The Commands ARE in SETUP_COMMANDS.sh

### Line 52: Create Conda Environment
```bash
conda create -n lerobot_old python=3.10 -y
```
**This creates the new environment!**

### Line 107: Install/Download Packages
```bash
pip install -e ".[feetech]"
```
**This downloads:**
- PyTorch + CUDA
- LeRobot dependencies
- OpenCV
- Transformers
- All other packages (~2GB download)

---

## 🔍 Full Command Flow

When you run `bash SETUP_COMMANDS.sh`, here's what happens:

### Step 1: Environment Creation (Lines 37-54)
```bash
# Check if lerobot_old exists
if conda env list | grep -q "lerobot_old"; then
    # Ask if you want to recreate
    ...
else
    # Create it!
    conda create -n lerobot_old python=3.10 -y  # ← HERE!
fi
```

### Step 2: Git Checkout (Lines 66-87)
```bash
cd /home/jrobot/project/lerobot

# Create branch from Aug 31 commit
git checkout -b lerobot_v03_august31 c0da806  # ← Downloads old code
```

### Step 3: Install Packages (Lines 101-108)
```bash
# Activate the new environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot_old

# Install LeRobot (this DOWNLOADS everything)
pip install -e ".[feetech]"  # ← HERE! Downloads 2GB+
```

### Step 4: Verify (Lines 120-143)
```bash
# Test Python
python --version

# Test imports (makes sure download worked)
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; ..."

# Test GPU
python -c "import torch; print(torch.cuda.is_available()); ..."
```

---

## 💡 Why You Might Not See Them Clearly

The script uses **bash logic** (if/else statements) that hides the main commands in the flow.

**Here's the simplified version:**

```bash
# Pseudocode of what SETUP_COMMANDS.sh does:

if lerobot_old exists:
    ask to recreate
else:
    conda create -n lerobot_old python=3.10 -y  # ← CREATE ENVIRONMENT

cd lerobot repo
git checkout -b lerobot_v03_august31 c0da806      # ← GET OLD CODE

activate lerobot_old
pip install -e ".[feetech]"                       # ← DOWNLOAD PACKAGES

test everything works
```

---

## 🚀 To Run It

```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot

# Run the script (it has all the commands inside!)
bash SETUP_COMMANDS.sh
```

**You'll see it execute:**
1. ✅ Creating environment... (line 52 runs)
2. ✅ Checking out code... (line 81 runs)
3. ✅ Installing packages... (line 107 runs, this downloads everything)
4. ✅ Verifying... (lines 122-143 run)

---

## 📊 What Gets Downloaded

When `pip install -e ".[feetech]"` runs (line 107):

| Package | Size | What It's For |
|---------|------|---------------|
| PyTorch | ~800MB | ML framework |
| CUDA libraries | ~500MB | GPU support |
| Transformers | ~300MB | Language models |
| OpenCV | ~100MB | Camera processing |
| Other deps | ~300MB | Various utilities |
| **Total** | **~2GB** | **Everything needed** |

This happens automatically when the script runs!

---

## ✅ Summary

**The commands ARE there:**
- Line 52: `conda create -n lerobot_old python=3.10 -y`
- Line 107: `pip install -e ".[feetech]"`

**They're inside the bash script, wrapped in if/else logic.**

**To execute them, just run:**
```bash
bash SETUP_COMMANDS.sh
```

**The script will:**
1. Create environment ✅
2. Download old LeRobot code ✅
3. Download all packages (~2GB) ✅
4. Verify everything works ✅

**Time: 10-15 minutes (mostly downloading)**

---

**Ready to run it?** 🤖

```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot
bash SETUP_COMMANDS.sh
```
