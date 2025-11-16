# Quick Start - LeRobot Downgrade Testing

**5-Minute Guide to Get Started**

---

## 🚀 Run Setup (10-15 minutes)

```bash
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot

# Make script executable
chmod +x SETUP_COMMANDS.sh

# Run automated setup
bash SETUP_COMMANDS.sh
```

**What this does:**
1. Creates `lerobot_old` conda environment
2. Checks out LeRobot from August 31, 2024
3. Installs all dependencies
4. Verifies GPU support

**Time:** 10-15 minutes (mostly downloading packages)

---

## 🧪 Run First Test (15 minutes)

After setup completes, the terminal will show:

```
Environment: lerobot_old (already activated)
```

Then run:

```bash
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot

python test_old_lerobot_models.py
```

**Answer prompts:**
- Model: `jhou/smolvla_pickplace` (press Enter)
- Top camera: `4` (or your index)
- Wrist camera: `6` (or your index)
- Arm: `left`
- Port: Press Enter
- Duration: Press Enter
- Task: Press Enter

**Watch the arm for 10 seconds!**

---

## ✅ Evaluate Results

**The model worked if:**
- ✅ Loaded without "policy_preprocessor.json" error
- ✅ Made large, purposeful movements
- ✅ Reached toward objects
- ✅ Gripper opened/closed

**Success = 30-50% task completion for MVP**

---

## 📊 What to Do Next

### If It Worked (>40% success)
```bash
# Document in RESULTS.md
nano RESULTS.md

# You're done! Use this model for MVP
```

### If Mediocre (20-40% success)
```bash
# Try another model
python test_old_lerobot_models.py
# Model: HuggingFadeUser/my_smolvla
```

### If Failed (<20% success)
```bash
# Try 1-2 more models
python test_old_lerobot_models.py

# If all fail → Proceed to fine-tuning (Path C)
# See: jdocs/NEXT_STEPS_STAGE_2.md
```

---

## 🔄 Environment Switching

**For testing pretrained (old LeRobot):**
```bash
conda activate lerobot_old
cd /home/jrobot/project/lerobot
git checkout lerobot_v03_august31
```

**For fine-tuning later (new LeRobot):**
```bash
conda activate lerobot
cd /home/jrobot/project/lerobot
git checkout main
```

**Check which you're in:**
```bash
echo $CONDA_DEFAULT_ENV
# Shows: lerobot_old OR lerobot
```

---

## 📝 Files Created

```
attempting_downgrade_lerobot/
├── QUICK_START.md              ← You are here
├── DOWNGRADE_GUIDE.md          ← Full detailed guide
├── SETUP_COMMANDS.sh           ← Automated setup script
├── test_old_lerobot_models.py  ← Test script
└── RESULTS.md                  ← Document your results
```

---

## ⚡ TL;DR Commands

```bash
# 1. Setup (one time, 10-15 min)
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot
chmod +x SETUP_COMMANDS.sh
bash SETUP_COMMANDS.sh

# 2. Test (15 min per model)
python test_old_lerobot_models.py

# 3. Try different models if needed
python test_old_lerobot_models.py  # Enter different model ID

# 4. Document results
nano RESULTS.md
```

---

## 🎯 Expected Outcome

**Optimistic (50% chance):**
- One model works well (>40% success)
- MVP complete in 1.5 hours
- Ready for demonstrations

**Realistic (30% chance):**
- Models work okay (20-40% success)
- Acceptable for MVP proof of concept
- Consider fine-tuning for production

**Pessimistic (20% chance):**
- Models don't work (<20% success)
- Compatibility solved, but domain mismatch remains
- Proceed to fine-tuning (Path C)

**Regardless:** You'll learn a lot and have clear next steps! 🤖

---

**Ready? Run the setup script!**

```bash
bash SETUP_COMMANDS.sh
```
