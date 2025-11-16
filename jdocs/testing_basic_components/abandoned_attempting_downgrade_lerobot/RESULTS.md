# Downgrade Testing Results

**Date Started:** 2025-11-09
**Purpose:** Test community models with old LeRobot (pre-v0.4.0) for MVP

---

## Setup Summary

**Environment:** `lerobot_old`
**LeRobot Version:** August 31, 2024 (commit c0da806)
**Git Branch:** `lerobot_v03_august31`
**Python:** 3.10
**GPU:** NVIDIA GeForce RTX 5090 Laptop GPU

---

## Models Tested

### Model 1: jhou/smolvla_pickplace

**Date Tested:** _____
**Time:** _____

**Loading:**
- [ ] Model downloaded successfully
- [ ] Loaded without `policy_preprocessor.json` error
- [ ] No compatibility issues

**Performance:**
- Task: _____
- Duration: _____ seconds
- Large movements: Yes / No
- Task-relevant behavior: Yes / No
- Gripper activity: Yes / No
- Estimated success rate: _____%

**Notes:**
```
[Your observations here]
```

**Verdict:** ✅ Works / ⚠️ Mediocre / ❌ Doesn't work

---

### Model 2: HuggingFadeUser/my_smolvla

**Date Tested:** _____
**Time:** _____

**Loading:**
- [ ] Model downloaded successfully
- [ ] Loaded without errors
- [ ] No compatibility issues

**Performance:**
- Task: _____
- Duration: _____ seconds
- Large movements: Yes / No
- Task-relevant behavior: Yes / No
- Gripper activity: Yes / No
- Estimated success rate: _____%

**Notes:**
```
[Your observations here]
```

**Verdict:** ✅ Works / ⚠️ Mediocre / ❌ Doesn't work

---

### Model 3: saood65/my_smolvla

**Date Tested:** _____
**Time:** _____

**Loading:**
- [ ] Model downloaded successfully
- [ ] Loaded without errors
- [ ] No compatibility issues

**Performance:**
- Task: _____
- Duration: _____ seconds
- Large movements: Yes / No
- Task-relevant behavior: Yes / No
- Gripper activity: Yes / No
- Estimated success rate: _____%

**Notes:**
```
[Your observations here]
```

**Verdict:** ✅ Works / ⚠️ Mediocre / ❌ Doesn't work

---

### Model 4: [Other Model]

**Date Tested:** _____
**Model ID:** _____
**Time:** _____

**Loading:**
- [ ] Model downloaded successfully
- [ ] Loaded without errors
- [ ] No compatibility issues

**Performance:**
- Task: _____
- Duration: _____ seconds
- Large movements: Yes / No
- Task-relevant behavior: Yes / No
- Gripper activity: Yes / No
- Estimated success rate: _____%

**Notes:**
```
[Your observations here]
```

**Verdict:** ✅ Works / ⚠️ Mediocre / ❌ Doesn't work

---

## Overall Results

### Best Model

**Model ID:** _____
**Success Rate:** _____%
**Why it worked:** _____

### Key Learnings

1. **Compatibility:**
   - Did downgrading solve loading errors? Yes / No
   - Which models loaded successfully? _____
   - Any unexpected issues? _____

2. **Performance:**
   - Best success rate achieved: _____%
   - Domain mismatch still present? Yes / No
   - Camera/workspace issues? _____

3. **Time Investment:**
   - Setup time: _____ minutes
   - Testing time: _____ minutes
   - Total: _____ minutes

### Decision

- [ ] **Found working model** → Use for MVP, proceed to Stage 3
- [ ] **Models work okay** → Acceptable for demonstrations
- [ ] **No model works well** → Proceed to Path C (fine-tuning)

---

## Next Steps

Based on results:

**If successful (>40% success rate):**
- [ ] Document best model and settings
- [ ] Test on different object positions
- [ ] Prepare demonstrations
- [ ] Plan Stage 3a (hierarchical control)

**If mediocre (20-40% success rate):**
- [ ] Decide: Use for MVP or fine-tune?
- [ ] If using: Document limitations
- [ ] If fine-tuning: Start Path C planning

**If unsuccessful (<20% success rate):**
- [ ] Confirm domain mismatch is the issue
- [ ] Start Path C (fine-tuning) planning
- [ ] Order leader arm or plan keyboard teleoperation
- [ ] Keep old environment for reference

---

## Technical Notes

### Camera Setup
- Top camera index: _____
- Wrist camera index: _____
- Resolution: _____
- Any issues: _____

### Robot Setup
- Arm tested: Left / Right
- Port: _____
- Other arm: Hidden / Powered off
- Calibration: ✅ Loaded / ❌ Using defaults

### Environment Issues
- Any import errors: Yes / No
- GPU working: Yes / No
- Memory issues: Yes / No

---

## Comparison: Old vs New LeRobot

### What Worked with Old LeRobot
- [ ] Models loaded that failed with new LeRobot
- [ ] No `policy_preprocessor.json` required
- [ ] Compatible with community models

### What Was Different
- Import paths: _____
- API differences: _____
- Performance differences: _____

---

## Recommendations

### For Future MVP Testing
```
[Your recommendations based on experience]
```

### For Production (Fine-tuning)
```
[Notes on what you'd do differently for fine-tuning]
```

---

## Appendix: Command Reference

### Activate Old Environment
```bash
conda activate lerobot_old
cd /home/jrobot/project/lerobot
git checkout lerobot_v03_august31
```

### Run Test
```bash
cd /home/jrobot/project/XLeRobot/jdocs/attempting_downgrade_lerobot
python test_old_lerobot_models.py
```

### Switch Back to New
```bash
conda activate lerobot
cd /home/jrobot/project/lerobot
git checkout main
```

---

**Status:** 🚧 In Progress / ✅ Complete
**Last Updated:** _____
