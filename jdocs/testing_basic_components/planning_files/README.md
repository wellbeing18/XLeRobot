# SmolVLA Testing - Quick Reference

## Problem: Tiny Incremental Movements ❌

Your robot made tiny, random movements instead of executing the task.

**CORRECTED Root Cause (after research):**
- **Missing dataset stats** in the postprocessor
- LeRobot DOES denormalize actions, but needs correct mean/std stats
- Without stats, actions aren't properly scaled to robot's joint ranges

**Secondary issues:**
- Camera domain mismatch (different from training data)
- No action smoothing (noisy predictions)

---

## Solution: Two Approaches ✅

### Files Created

1. **`diagnose_smolvla_issue.py`** - Diagnostic tool
2. **`test_smolvla_with_stats.py`** - PROPER FIX (loads dataset stats)
3. **`test_smolvla_fixed.py`** - WORKAROUND (manual denormalization)
4. **`CORRECTED_DIAGNOSIS.md`** - Full research findings
5. **`RESEARCH_SUMMARY.md`** - Complete explanation
6. **`../QUICK_START_STEP_2_FIX.md`** - Original explanation (partially incorrect)

### Quick Start (UPDATED)

```bash
# 1. Run diagnostic (optional but helpful)
python diagnose_smolvla_issue.py

# 2. Try proper fix first (RECOMMENDED)
python test_smolvla_with_stats.py

# 3. If #2 doesn't work, use workaround
python test_smolvla_fixed.py
```

### Approach 1: test_smolvla_with_stats.py (PROPER FIX)

✅ **Loads dataset stats explicitly** - Gets correct mean/std for denormalization
✅ **Verifies stats are loaded** - Checks postprocessor has correct configuration
✅ **Relies on LeRobot's built-in denorm** - Uses official implementation
✅ **Action smoothing** - Exponential moving average for stability
✅ **Detailed logging** - Shows what's working and what's not

### Approach 2: test_smolvla_fixed.py (WORKAROUND)

✅ **Manual denormalization** - Backup when stats aren't available
✅ **Action clipping** - Safety limits
✅ **Action smoothing** - Exponential moving average
✅ **Temporal buffer** - Observation history
✅ **Better logging** - Action values and ranges

---

## Expected Results

### Best Case (50-80% success)
- Robot moves smoothly toward objects
- Actions are in reasonable ranges (e.g., -3 to 3 radians)
- Gripper opens/closes appropriately
- Task executed correctly most of the time
→ **Proceed to Stage 3a**

### Medium Case (20-50% success)
- Movements are smooth but not always task-relevant
- Actions look reasonable but behavior is off
- Needs camera adjustment or fine-tuning
→ **Collect 10-20 demos, fine-tune for 1000 epochs**

### Still Not Working (< 20% success)
- Try ACT model: `r2owb0/act1`
- Check calibration exists: `cat ~/.cache/lerobot/calibration/xlerobot_left_arm.json`
- Verify cameras working: `ls /dev/video*`
→ **See CORRECTED_DIAGNOSIS.md for full troubleshooting**

---

## Research Findings

### About Normalization (CORRECTED)

| Initial Understanding | After Research |
|----------------------|----------------|
| ❌ "LeRobot doesn't denormalize" | ✅ LeRobot DOES denormalize via `UnnormalizerProcessorStep` |
| ❌ "Need manual denormalization" | ✅ Only if dataset stats are missing |
| ✅ "Actions are normalized" | ✅ CORRECT - using MEAN_STD normalization |
| ❌ "No automatic processing" | ✅ Postprocessor handles it automatically (if stats present) |

**Key insight**: The issue is missing stats, not missing functionality.

---

---

## About ROS: Do You Need It?

### Short Answer: **NO**

LeRobot does NOT use or require ROS. Do NOT install ROS for XLeRobot.

### What LeRobot Uses Instead

- **Direct serial communication** via `pyserial`
- **Python-based robot control** (no middleware)
- **Simple, lightweight architecture** for ML/imitation learning

### Why Papers Mention ROS But LeRobot Doesn't

| ROS | LeRobot |
|-----|---------|
| Multi-robot coordination | Single-arm ML focus |
| Complex system setup | Simple pip install |
| C++ centric | Pure Python |
| Academic standard (2007) | Modern ML approach (2024) |

**Conclusion**: Skip ROS entirely. It would only add complexity without any benefit.

See `RESEARCH_SUMMARY.md` for detailed explanation.

---

## Understanding "Plug and Go"

**Your expectation:** Pretrained model should work immediately

**Reality:**
- Pretrained models need proper preprocessing (esp. normalization stats)
- May need fine-tuning for your exact setup
- Camera positioning and lighting matter a lot
- 50-60% success is good for a proof-of-concept!

**This is normal!** Most robotics projects require debugging and adaptation.

---

## Next Steps

1. **Test proper fix**: `python test_smolvla_with_stats.py`
2. **If needed, test workaround**: `python test_smolvla_fixed.py`
3. **Compare results** - see which works better
4. **Evaluate success rate** (try 5-10 times)
5. **Based on results**:
   - **Good (≥50%):** Move to Stage 3a (language control)
   - **Medium (20-50%):** Fine-tune with 10-20 demos
   - **Poor (<20%):** Try ACT model or collect more data

---

## Alternative: ACT Model

If SmolVLA doesn't work well, try ACT:

```bash
# ACT is specifically trained for SO-101
# No language conditioning, but more reliable
# See xlerobot_mvp_plan_v1.md for details
```

---

## Further Reading

- **`CORRECTED_DIAGNOSIS.md`** - Full technical analysis of the issue
- **`RESEARCH_SUMMARY.md`** - Complete research findings (normalization + ROS)
- **`QUICK_START_STEP_2_FIX.md`** - Original explanation (some parts corrected)
- **`../design/xlerobot_mvp_plan_v1.md`** - Overall project plan

**Most important**: `RESEARCH_SUMMARY.md` has everything you need to know.
