# START HERE - Simple Instructions

**You moved this folder under `testing_basic_components/` ✅**

---

## 🎯 What To Do (3 Steps)

### Step 1: Run Setup (10-15 min)
```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot

bash SETUP_COMMANDS.sh
```

**This will:**
- ✅ Create `lerobot_old` conda environment (line 52 of script)
- ✅ Download old LeRobot code (August 31, 2024)
- ✅ Download & install all packages (~2GB, line 107 of script)
- ✅ Verify everything works

**Time:** 10-15 minutes (mostly waiting for downloads)

---

### Step 2: Test First Model (15 min)
```bash
# Environment will already be activated from setup
python test_old_lerobot_models.py
```

**When prompted:**
- Model: `jhou/smolvla_pickplace` (just press Enter)
- Top camera: `4`
- Wrist camera: `6`
- Arm: `left`
- Rest: press Enter

**Watch the arm move for 10 seconds!**

---

### Step 3: Evaluate

**Did it work?**
- ✅ Model loaded without errors?
- ✅ Arm made LARGE movements (not tiny)?
- ✅ Reached toward objects?

**If YES:** You're done! MVP success! 🎉
**If NO:** Try another model (HuggingFadeUser/my_smolvla)

---

## 📋 Where Are The Commands?

**You asked:** "I don't see the conda create or downloading commands"

**Answer:** They're IN the bash script!

- **Line 52:** `conda create -n lerobot_old python=3.10 -y`
- **Line 107:** `pip install -e ".[feetech]"` (downloads ~2GB packages)

See `COMMANDS_EXPLAINED.md` for details.

---

## ⚡ Quick Reference

**Your location (you moved it):**
```
/home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot/
```

**Files here:**
- `SETUP_COMMANDS.sh` ← Run this first!
- `test_old_lerobot_models.py` ← Run this second!
- `START_HERE.md` ← You are here
- `COMMANDS_EXPLAINED.md` ← Explains where commands are
- `DOWNGRADE_GUIDE.md` ← Full detailed guide
- `QUICK_START.md` ← Quick overview
- `RESULTS.md` ← Fill in after testing

---

## 🚀 Ready? Run This:

```bash
cd /home/jrobot/project/XLeRobot/jdocs/testing_basic_components/attempting_downgrade_lerobot

# Step 1: Setup (runs conda create, pip install, etc.)
bash SETUP_COMMANDS.sh

# Step 2: Test (after setup finishes)
python test_old_lerobot_models.py
```

**That's it!** 🤖
