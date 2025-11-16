# LeRobot Downgrade Testing Directory

**Purpose:** Test pre-v0.4.0 community models for MVP proof of concept

---

## 📁 Directory Contents

| File | Purpose | Time |
|------|---------|------|
| **QUICK_START.md** | 5-minute guide to get started | Read first! |
| **SETUP_COMMANDS.sh** | Automated setup script | Run this (10-15 min) |
| **test_old_lerobot_models.py** | Test script for old LeRobot | Run after setup |
| **DOWNGRADE_GUIDE.md** | Complete detailed instructions | Reference |
| **RESULTS.md** | Template to document your results | Fill in as you test |
| **README.md** | This file | Overview |

---

## 🎯 Goal

Test community SmolVLA models (jhou, HuggingFadeUser, etc.) that were uploaded before LeRobot v0.4.0. These models:
- ❌ Don't work with current LeRobot (missing `policy_preprocessor.json`)
- ✅ Should work with old LeRobot (pre-v0.4.0)

**Success = 40-50% task completion for MVP** (not 70-90% production)

---

## 🚀 Quick Start (30 minutes total)

### Step 1: Setup (10-15 min)
```bash
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot
bash SETUP_COMMANDS.sh
```

### Step 2: Test (15 min)
```bash
python test_old_lerobot_models.py
```

### Step 3: Evaluate
- Did it work better than base SmolVLA?
- If yes → MVP done!
- If no → Try 1-2 more models

---

## 📊 What We're Testing

### The Hypothesis
**Community models fail with new LeRobot due to:**
1. ✅ **Version incompatibility** (missing files, config changes)
2. ⚠️ **Domain mismatch** (your setup vs training setup)

**Downgrading solves #1, but #2 remains**

### Expected Results

| Outcome | Probability | What It Means |
|---------|-------------|---------------|
| **One model works well** | 50% | ✅ MVP success, compatibility solved |
| **Models work okay** | 30% | ⚠️ Acceptable for MVP |
| **All fail** | 20% | ❌ Domain mismatch too large, need fine-tuning |

---

## 🔬 Technical Details

### Old vs New LeRobot

**Old LeRobot (Aug 31, 2024):**
- Simpler processor system
- No `policy_preprocessor.json` required
- Compatible with June-Sept 2024 models

**New LeRobot (v0.4.0, Oct 2024):**
- New processor pipeline system
- Requires `policy_preprocessor.json`
- Better features (plugins, multi-GPU)
- Official models updated, community models not

### Two-Environment Strategy

```
lerobot_old              lerobot (current)
     ↓                          ↓
Test pretrained          Fine-tune later
     ↓                          ↓
MVP (this week)          Production (weeks 2-3)
```

---

## 📚 Documentation Guide

### If You're New: Start Here
1. **QUICK_START.md** (5 min read)
2. Run `SETUP_COMMANDS.sh` (10-15 min)
3. Run `test_old_lerobot_models.py` (15 min)
4. Fill in **RESULTS.md**

### If You Want Details
1. **DOWNGRADE_GUIDE.md** (complete walkthrough)
2. Understand the analysis and reasoning
3. Follow step-by-step instructions

### If Something Breaks
1. Check **DOWNGRADE_GUIDE.md** → Troubleshooting section
2. Verify environment: `echo $CONDA_DEFAULT_ENV`
3. Verify git branch: `cd ~/project/lerobot && git branch`

---

## ✅ Success Criteria

### Model Loads Successfully
- [ ] Downloads without errors
- [ ] Loads without `policy_preprocessor.json` error
- [ ] Preprocessors initialize
- [ ] Robot connects

### Model Performs Adequately
- [ ] Makes LARGE movements (not tiny)
- [ ] Shows task-relevant behavior
- [ ] Gripper opens/closes appropriately
- [ ] Estimated 30-50% success on pick-place

**You don't need perfection!** This is MVP.

---

## 🎓 Learning Objectives

By completing this exercise, you'll understand:

1. **Version compatibility matters**
   - API changes break older models
   - Managing multiple environments is common

2. **Pretrained models have two hurdles**
   - Compatibility (can it load?)
   - Domain match (does it perform?)

3. **MVP vs Production**
   - Different success criteria
   - Different time investments

4. **When to fine-tune**
   - Not always necessary for proof of concept
   - Essential for production quality

---

## 🔄 Next Steps After Testing

### If Successful (>40% success)
- ✅ Document best model in RESULTS.md
- ✅ Test on variations (different positions)
- ✅ Prepare MVP demonstration
- ✅ Proceed to Stage 3a
- ⏳ Fine-tune later for production

### If Mediocre (20-40% success)
- ⚠️ Decide: Good enough for MVP?
- ⚠️ If yes: Use it, document limitations
- ⚠️ If no: Try 1-2 more models
- ⏳ Plan fine-tuning for better performance

### If Unsuccessful (<20% success)
- ❌ Acknowledge domain mismatch is real
- ❌ Pretrained won't work for your setup
- ✅ Start Path C (fine-tuning) planning
- ✅ Keep old environment for reference

---

## 📞 Getting Help

### Scripts Not Working?
1. Check you're in `lerobot_old` environment
2. Check git branch is `lerobot_v03_august31`
3. Re-run setup: `bash SETUP_COMMANDS.sh`

### Model Not Loading?
1. Check internet connection (downloads from HuggingFace)
2. Try different model
3. Check error message in DOWNGRADE_GUIDE.md troubleshooting

### Performance Issues?
1. This is domain mismatch, not a bug
2. Try 2-3 different models
3. If all fail, fine-tuning is the answer

---

## 📊 Time Investment Summary

| Activity | Time | Cumulative |
|----------|------|------------|
| Read QUICK_START.md | 5 min | 5 min |
| Run SETUP_COMMANDS.sh | 10-15 min | 20 min |
| Test first model | 15 min | 35 min |
| Test second model (if needed) | 15 min | 50 min |
| Test third model (if needed) | 15 min | 65 min |
| Document results | 10 min | 75 min |
| **Total (worst case)** | **~1.5 hours** | **MVP complete or pivot decision** |

---

## 🎯 Bottom Line

**This is about finding the fastest path to MVP:**

- ✅ If pretrained works (50% chance) → MVP in 1.5 hours
- ⚠️ If pretrained mediocre (30% chance) → Decide: use or fine-tune
- ❌ If pretrained fails (20% chance) → Clear decision to fine-tune

**Either way, you make progress!** 🤖

---

**Ready to start?**

```bash
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot
bash SETUP_COMMANDS.sh
```
