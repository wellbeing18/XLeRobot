# Minimal Validation Strategy for Pi0.5 Finetuning

**Purpose**: Test your finetuning pipeline with MINIMAL time/data investment before committing to full-scale data collection

**Time Investment**: 3-5 hours total (MVP + minimal set)
**Data Required**: 7-50 episodes (staged approach)
**Goal**: Catch bugs early, verify egocentric cameras work, see "before/after" finetuning effect

---

## Table of Contents

1. [Why Minimal Validation?](#why-minimal-validation)
2. [Three-Stage Validation Approach](#three-stage-validation-approach)
3. [Stage 0: MVP - Technical Pipeline Validation](#stage-0-mvp---technical-pipeline-validation)
   - [What You're Testing](#what-youre-testing)
   - [Step-by-Step MVP](#step-by-step-mvp)
   - [MVP Pass/Fail Criteria](#mvp-passfail-criteria)
4. [Stage 1: Minimal Set - Learning & Camera Validation](#stage-1-minimal-set---learning--camera-validation)
   - [What You're Testing](#what-youre-testing-1)
   - [Minimal Dataset Composition](#minimal-dataset-composition-50-episodes)
   - [Collection Strategy](#collection-strategy)
   - [Full Training](#full-training-6000-steps)
   - [Evaluation](#evaluation)
   - [Stage 1 Pass/Fail Criteria](#stage-1-passfail-criteria)
5. [Stage 2: Full Set - Maximize Performance](#stage-2-full-set---maximize-performance)
   - [When to Proceed](#when-to-proceed-to-stage-2)
   - [Additional Data to Collect](#additional-data-to-collect)
   - [Expected Performance](#expected-performance-full-set)
6. [Decision Trees](#decision-trees)
   - [After MVP](#after-mvp-stage-0)
   - [After Stage 1](#after-stage-1-50-episodes)
   - [After Stage 2](#after-stage-2-75-100-episodes)
7. [Timeline Summary](#timeline-summary)
8. [Key Research Citations](#key-research-citations)

**🎯 Quick Start:**
- **First time?** → Read sections 1-3 (Why, Approach, MVP)
- **Ready for MVP?** → Jump to [Stage 0: Step-by-Step](#step-by-step-mvp)
- **MVP passed?** → Read [Stage 1](#stage-1-minimal-set---learning--camera-validation)
- **Need decision help?** → See [Decision Trees](#decision-trees)

---

## Why Minimal Validation?

### The Problem

Collecting 75-100 high-quality episodes takes **20-30 hours**. What if:
- Data format is wrong (must re-collect everything)
- Action frequency configured incorrectly (incompatible dataset)
- Training crashes due to VRAM/memory issues
- Egocentric cameras create too much embodiment gap (model can't learn)
- Camera views don't capture objects properly

**These issues waste 20-30 hours of work!**

### The Solution

**Test with minimal data FIRST** to catch technical problems before scaling up.

### Research Support

**OpenVLA Documentation**:
> "Two key verification steps are recommended: (1) replay actions from a demonstration... (2) once you've fine-tuned a model, load it in your inference pipeline and feed images from the fine-tuning dataset to verify you can reproduce the token accuracies"

**LeRobot Community Best Practice**:
> "Best practices recommend first recording just a handful of episodes to confirm data was saved correctly, then running a full small-scale cycle with a small dataset (e.g., 10 episodes)"

**Community Experience**:
- "5-10 episodes sufficient to catch pipeline bugs"
- "Test replay before scaling to 100 episodes"
- "Quick sanity check saves hours of debugging"

---

## Three-Stage Validation Approach

```
Stage 0: MVP (5-10 episodes)          → Catch technical bugs      [2-3 hours]
   ↓
Stage 1: Minimal Set (50 episodes)    → Test learning & cameras   [7-8 hours]
   ↓
Stage 2: Full Set (75-100 episodes)   → Maximize performance      [10-15 hours]
```

**Key Principle**: Only proceed to next stage if previous stage PASSES.

---

## Stage 0: MVP - Technical Pipeline Validation

**Time**: 2-3 hours
**Episodes**: 5-10 (ONE primitive only)
**Goal**: Verify pipeline runs without crashes

### What You're Testing

✅ **Technical correctness**:
- Data collection saves episodes correctly
- Dataset loads without import errors
- Training runs without crashes
- Model fits in GPU VRAM
- Checkpoints save and load
- Inference pipeline works

❌ **What you're NOT testing**:
- Model quality (can't learn from 5-10 episodes!)
- Demonstration quality (not enough data)
- Generalization (need real training)

### Step-by-Step MVP

#### 0. FIRST: Test Pretrained Pi0.5 Baseline (30 minutes) ⭐ **CRITICAL - DO THIS FIRST**

**Why this is essential:**
- You MUST have a baseline to compare against
- Without it, you can't tell if finetuning helped!
- Example: 40% success after finetuning could mean:
  - Pretrained got 10% → +30% improvement = EXCELLENT ✅
  - Pretrained got 35% → +5% improvement = MINIMAL ⚠️

**How to test baseline:**

```bash
# Download pretrained Pi0.5
from lerobot.common.policies.pi05.modeling_pi05 import Pi05ForActionPrediction

model = Pi05ForActionPrediction.from_pretrained("lerobot/pi05_base")
model.eval().cuda()

# Test on your setup (WITHOUT finetuning)
# Try 5-10 pick attempts with pretrained model
# Record: success rate, behavior, failure modes
```

**Expected pretrained performance on YOUR egocentric setup:**
- Best case: 5-15% success (model confused by egocentric cameras)
- Likely: 0-5% success (random flailing, wrong motions)
- This is NORMAL - egocentric cameras are out-of-distribution!

**Record this number!** You'll compare after finetuning.

**If pretrained gets >20% success:** 🎉 Great news! Egocentric cameras not as hard as expected.

**If pretrained gets 0-5% success:** Expected. Your goal: Finetune to 30-40%+.

---

#### 1. Collect 7-10 Episodes (1 hour)

**Updated recommendation: 7-10 episodes (not 5-10)**

```bash
# Only collect ONE primitive: pick_center
python scripts/collect_primitives.py --task pick_center --count 10 \
  --dataset lerobot/xlerobot_mvp_test
```

**Quality Standard for MVP**:
- ✅ Task completes (object picked up)
- ✅ Smooth-ish motion (doesn't need to be perfect)
- ✅ Object visible in cameras
- ❌ Don't stress about perfect consistency yet

**Time per episode**: ~5 minutes (setup + 30s recording + reset)

#### 2. Validate Dataset (5 minutes)

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/xlerobot_mvp_test")

# Critical checks
assert dataset.num_episodes >= 7, f"Need ≥7 episodes, got {dataset.num_episodes}"
assert dataset.meta.fps == 5, f"Action freq must be 5Hz, got {dataset.meta.fps}Hz"
assert len(dataset.meta.camera_keys) == 3, f"Need 3 cameras, got {len(dataset.meta.camera_keys)}"

print(f"✅ Dataset valid!")
print(f"   Episodes: {dataset.num_episodes}")
print(f"   FPS: {dataset.meta.fps}")
print(f"   Cameras: {dataset.meta.camera_keys}")
```

**If this fails** → Fix config before proceeding!

#### 3. Quick Training Test (30-45 minutes)

```bash
# Train for ONLY 100 steps (not 6000!)
python lerobot/scripts/lerobot_train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --dataset.repo_id=lerobot/xlerobot_mvp_test \
  --steps=100 \
  --batch_size=4 \
  --lr=3e-6 \
  --gradient_clip_norm=1.0 \
  --output_directory=outputs/pi05_mvp_test

# Monitor for crashes
watch -n 10 'nvidia-smi && free -h'
```

**What "success" means**:
- ✅ Training completes all 100 steps
- ✅ No crashes, no NaN losses
- ✅ VRAM usage < 24GB (leaves headroom)
- ✅ CPU memory stable (not increasing)

**Loss values DON'T MATTER!**
- Might decrease (overfitting to 10 examples)
- Might stay flat (normal for tiny dataset)
- Might increase slightly (also normal)
- **You're testing for crashes, not learning!**

#### 4. Inference Test (5 minutes)

```python
from lerobot.common.policies.pi05.modeling_pi05 import Pi05ForActionPrediction
import torch

# Load checkpoint
model = Pi05ForActionPrediction.from_pretrained(
    "outputs/pi05_mvp_test/checkpoint-100"
)
model.eval().cuda()

# Dummy inference
dummy_images = torch.randn(1, 3, 3, 480, 640).cuda()
dummy_state = torch.randn(1, 12).cuda()

with torch.no_grad():
    output = model(
        images=dummy_images,
        state=dummy_state,
        task="pick red_cube from center"
    )

print(f"✅ Inference works! Output shape: {output.shape}")
```

**If this fails** → Checkpoint loading or inference broken!

---

#### 5. Compare vs Pretrained Baseline (10 minutes) ⭐ **IMPORTANT**

**Now compare your mini-finetuned model vs pretrained baseline:**

```python
# You tested pretrained in Step 0, recorded: X% success
# Now test finetuned model (same setup, same task):
# - Load checkpoint from outputs/pi05_mvp_test/checkpoint-100
# - Try 5-10 pick attempts
# - Record: success rate, behavior

# Expected with 7-10 episodes + 100 steps training:
# - Finetuned: 0-20% success (not enough data to learn!)
# - This is NORMAL for MVP - you're just testing pipeline

# What you're checking:
# ✅ Finetuned model ATTEMPTS the task (not random flailing)
# ✅ Some indication of learning (even if fails)
# ❌ Don't expect actual success with only 7-10 episodes!
```

**The point of MVP comparison:**
- NOT to see if model learned (too little data!)
- BUT to verify model responds to finetuning at all
- AND to establish baseline for Stage 1 comparison

### MVP Pass/Fail Criteria

#### ✅ PASS - Proceed to Stage 1

All of these must be true:
- [x] Dataset loads without errors
- [x] Action frequency = 5Hz (not 30Hz!)
- [x] All 3 cameras present
- [x] Training completes 100 steps
- [x] No NaN losses
- [x] No memory crashes/OOM
- [x] Checkpoint saves and loads
- [x] Inference runs without errors

**→ Continue to Stage 1 (50 episodes)**

#### ❌ FAIL - Debug Before Continuing

If ANY of these occur:
- [ ] Import errors when loading dataset
- [ ] FPS mismatch (got 30Hz instead of 5Hz)
- [ ] Camera key mismatches
- [ ] Training crashes before 100 steps
- [ ] Gradient explosion (NaN)
- [ ] CUDA OOM (out of memory)
- [ ] Checkpoint won't load
- [ ] Inference errors

**→ FIX ISSUES before collecting more data!**

See troubleshooting: `VLM_VLA_FINETUNING_STRATEGY.md` → "Known Pi0.5 Issues"

---

## Stage 1: Minimal Set - Learning & Camera Validation

**Time**: 7-8 hours collection + 6-10 hours training
**Episodes**: 50 total
**Goal**: Verify egocentric cameras work and model can learn

### What You're Testing

✅ **Actual learning**:
- Model improves from epoch 1 → final epoch
- Success rate >0% on primitives
- Egocentric cameras provide learnable signal
- Spatial generalization (3 positions per primitive)

✅ **Camera viability**:
- Wrist cameras + head camera sufficient for learning
- Embodiment gap not insurmountable
- Visual features stable enough for policy learning

### Minimal Dataset Composition (50 episodes)

**Optimized for fastest path to validation:**

| Primitive | Episodes | Rationale | Variety Type |
|-----------|----------|-----------|--------------|
| **pick_center** | 7 | Core skill baseline | Natural variation only* |
| **pick_left** | 7 | Test spatial generalization | Object 8-10cm left |
| **pick_right** | 6 | Test spatial generalization | Object 8-10cm right |
| **place_box** | 5 | Complement pick (enables sequences) | Natural variation only |
| **place_left** | 5 | Test place spatial transfer | Target left edge |
| **place_right** | 5 | Test place spatial transfer | Target right edge |
| **grasp** | 8 | Sub-component aids pick learning | Natural variation only |
| **release** | 7 | Sub-component aids place learning | Natural variation only |

**Total: 50 episodes (~25 minutes of demos)**

**\*Natural variation** = Slight differences in starting arm pose, exact timing, but SAME strategy

**What to INCLUDE in these 50 episodes:**
- ✅ Spatial diversity (3 positions for pick/place)
- ✅ All successful demonstrations
- ✅ Consistent approach strategy per primitive
- ✅ Natural variation in starting poses

**What to EXCLUDE from these 50 episodes:**
- ❌ NO failure cases (all demos succeed)
- ❌ NO retry/recovery demonstrations
- ❌ NO different approach strategies (pick one, stick to it)
- ❌ NO clutter or distractor objects
- ❌ NO different object types (one cube only)

**Why exclude failures/variety for Stage 1?**
1. **First prove model CAN learn** with clean data
2. **Pi0 paper guidance** applies to production models (100+ episodes)
3. **Stage 1 goal**: Test if egocentric cameras work AT ALL
4. **Add complexity later**: If Stage 1 shows >30% success, add 10-15 recovery demos in Stage 2

**Why this set:**
- ✅ Meets Pi0.5 minimum (≥15 min data requirement)
- ✅ Tests pick+place (most important primitives)
- ✅ Spatial diversity (3 positions each)
- ✅ Enables "pick then place" sequences
- ✅ Fast to collect (7-8 hours vs 20-30 hours)
- ✅ Can iterate if fails (less wasted time)

**What's omitted** (add in Stage 2 if Stage 1 works):
- Push (non-prehensile, more complex)
- Reach (less critical than pick/place)

### Collection Strategy

**Daily breakdown:**

**Day 1**: Pick primitives (20 episodes, ~3 hours)
- Morning: pick_center (7 episodes)
- Afternoon: pick_left (7 episodes), pick_right (6 episodes)

**Day 2**: Place primitives (15 episodes, ~2.5 hours)
- Morning: place_box (5), place_left (5)
- Afternoon: place_right (5)

**Day 3**: Grasp/Release (15 episodes, ~2 hours)
- Morning: grasp (8 episodes)
- Afternoon: release (7 episodes)

**Day 4**: Quality review and re-recording
- Review all 50 episodes
- Re-record any failures
- Validate dataset ready for training

### Full Training (6000 steps)

```bash
# Full training run with optimized config
python lerobot/scripts/lerobot_train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --dataset.repo_id=lerobot/xlerobot_minimal_50 \
  --steps=6000 \
  --batch_size=8 \
  --gradient_accumulation_steps=2 \
  --lr=3e-6 \
  --gradient_clip_norm=1.0 \
  --lr_warmup_steps=500 \
  --weight_decay=0.01 \
  --eval_freq=1000 \
  --save_freq=1000 \
  --early_stopping_patience=5 \
  --output_directory=outputs/pi05_minimal_50

# Expected time: 6-10 hours on RTX 5090
```

**Monitor during training:**
```bash
# Watch for issues
watch -n 30 nvidia-smi
watch -n 30 'free -h'

# Check for NaN
tail -f outputs/pi05_minimal_50/training.log | grep -i nan

# Check val loss trend (should decrease or plateau, not increase)
```

### Evaluation

**FIRST: Re-test pretrained baseline (5-10 trials)**
```python
# Load PRETRAINED Pi0.5 (no finetuning)
pretrained = Pi05ForActionPrediction.from_pretrained("lerobot/pi05_base")

# Test on pick_center (10 trials)
# Record: X% success rate

# This is your BASELINE for comparison
```

**THEN: Test your finetuned model (same trials)**
```bash
python scripts/eval_vla_model.py \
  --checkpoint outputs/pi05_minimal_50/checkpoint-best \
  --trials-per-primitive 10
```

**Qualitative (manual testing)**:
- Test on pick_center (IN distribution)
- Test on pick between left/center (OUT of distribution)
- Test pick→place sequence
- Record: success rate, quality of motion, failure modes

**CRITICAL: Always compare vs baseline!**

| Metric | Pretrained | Finetuned (50 episodes) | Delta |
|--------|------------|-------------------------|-------|
| pick_center success | X% | Y% | +Z% |
| Motion quality | (describe) | (describe) | (better?) |
| Failure modes | (describe) | (describe) | (different?) |

### Stage 1 Pass/Fail Criteria

#### ✅ PASS - Proceed to Stage 2 (Scale Up)

**Minimum bar**: Model shows SOME learning capability
- [x] Training loss decreases (not flat/increasing)
- [x] Val loss decreases or plateaus (not diverging)
- [x] Success rate ≥30% on pick primitives
- [x] Model attempts correct motion direction
- [x] Occasional successful task completions
- [x] Better than random baseline

**Interpretation**: Egocentric cameras work! Pipeline validated! Scale up to 75-100 episodes.

**→ Proceed to Stage 2: Collect remaining 25-50 episodes + add push/reach**

#### ⚠️ MARGINAL - Debug & Retry

**Model learns but poorly**: Success 10-30%
- [ ] Review demonstration quality
- [ ] Check camera views (object visible?)
- [ ] Try collecting 25 MORE episodes (→ 75 total)
- [ ] Adjust training hyperparameters

**→ Iterate before full scale-up**

#### ❌ FAIL - Fundamental Issue

**No learning at all**: Success <10%, random motion
- [ ] Check data quality (demos too inconsistent?)
- [ ] Verify egocentric cameras capture useful info
- [ ] Consider hybrid setup: Add 1-2 fixed external cameras
- [ ] Try SmolVLA (better SO-101 kinematics knowledge)

**→ Don't scale up yet! Fix fundamental issue first.**

---

## Stage 2: Full Set - Maximize Performance

**Time**: 10-15 hours collection (+ Stage 1 already done)
**Episodes**: 75-100 total (add 25-50 to existing 50)
**Goal**: Achieve best possible performance

### When to Proceed to Stage 2

Only if Stage 1 shows:
- ✅ Model learns (>30% success)
- ✅ Egocentric cameras viable
- ✅ No fundamental blocking issues

### Additional Data to Collect

**Add to existing 50 episodes:**

| Primitive | Add Episodes | New Total |
|-----------|--------------|-----------|
| pick_center | +3 | 10 |
| pick_left | +3 | 10 |
| pick_right | +4 | 10 |
| place_box | +5 | 10 |
| place_left | +5 | 10 |
| place_right | +5 | 10 |
| **push** | +10 | 10 (NEW) |
| **reach** | +10 | 10 (NEW) |
| grasp | +2 | 10 |
| release | +1 | 8 |

**Total additions: 48 episodes → Grand total: 98 episodes**

**Why these additions:**
- More examples of existing primitives → better performance
- Push + reach → expand capability repertoire
- 98 episodes ≈ 45min data (3x Pi0.5 minimum)

### Expected Performance (Full Set)

With 75-100 high-quality episodes:
- **Fixed cameras (standard)**: 70-80% success expected
- **Egocentric cameras (yours)**: 50-65% success expected
- **Hybrid cameras**: 60-75% success expected

**Performance gaps explained:**
- Egocentric embodiment gap: -15 to -25% penalty (research-proven)
- Pi0.5's mobile data training: +10 to +15% recovery
- Net effect: -5 to -15% vs standard setup

---

## Decision Trees

### After MVP (Stage 0)

```
MVP Result?
  ├─ PASS (no crashes) → Continue to Stage 1
  └─ FAIL (crashes/errors)
       ├─ FPS wrong → Fix config, re-collect MVP
       ├─ Import errors → Fix dataset schema
       ├─ OOM → Reduce batch size / LoRA rank
       ├─ NaN losses → Lower LR, add gradient clipping
       └─ Other → Debug per "Known Issues" section
```

### After Stage 1 (50 episodes)

```
Success Rate?
  ├─ ≥30% → GOOD! → Proceed to Stage 2 (collect 25-50 more)
  ├─ 10-30% → MARGINAL
  │    ├─ Demos inconsistent? → Improve quality, collect 25 more
  │    ├─ Cameras blocked? → Adjust camera positions
  │    └─ Training issues? → Tune hyperparameters
  └─ <10% → FAIL
       ├─ Egocentric gap too severe? → Add 1-2 fixed external cameras
       ├─ Data quality bad? → Restart with better demos
       └─ Pi0.5 not working? → Try SmolVLA (knows SO-101 better)
```

### After Stage 2 (75-100 episodes)

```
Success Rate?
  ├─ ≥50% → EXCELLENT! → Proceed to VLM integration (Week 3)
  ├─ 30-50% → ACCEPTABLE → Can integrate VLM or collect more data
  ├─ 15-30% → CONCERNING
  │    ├─ Add hybrid cameras (1-2 fixed external)
  │    ├─ Collect 50 MORE episodes (→ 150 total)
  │    └─ Try SmolVLA as alternative
  └─ <15% → CRITICAL ISSUE
       ├─ Reassess camera setup (too much embodiment gap)
       ├─ Check if robot kinematics too different from training
       └─ May need external cameras or different VLA model
```

---

## Timeline Summary

**Conservative estimate** (first-time, careful approach):

```
Stage 0 (MVP):              3 hours
  ├─ Collection:            1 hour
  ├─ Validation:            5 min
  ├─ Training test:         45 min
  └─ Inference test:        5 min
  └─ Contingency/debug:     1 hour

Stage 1 (50 episodes):      16-18 hours
  ├─ Collection:            7-8 hours
  ├─ Quality review:        1-2 hours
  ├─ Training (6000 steps): 6-10 hours
  └─ Evaluation:            1-2 hours

[DECISION: Pass → Stage 2 | Marginal → Iterate | Fail → Pivot]

Stage 2 (add 25-50):        12-15 hours
  ├─ Collection:            5-7 hours
  ├─ Quality review:        1 hour
  ├─ Re-training:           6-10 hours
  └─ Final evaluation:      1-2 hours

TOTAL: 31-36 hours (vs 50+ hours if no validation strategy!)
```

**Optimistic estimate** (experienced, everything works first try):

```
Stage 0:  2 hours
Stage 1:  13 hours (7 collect + 6 train)
Stage 2:  11 hours (5 collect + 6 train)
TOTAL:    26 hours
```

**Pessimistic estimate** (issues encountered, iterations needed):

```
Stage 0:  5 hours (MVP fails twice, need debugging)
Stage 1:  20 hours (collect, train, marginal result, collect 25 more, re-train)
Stage 2:  15 hours (full collection + training)
TOTAL:    40 hours (still better than 50+ hours with no strategy!)
```

---

## Key Research Citations

**Pi0.5 Minimum Data**:
- Official guidance: "Use a dataset with at least 15 minutes of data"
- Research: "50 trajectories can effectively adapt a pre-trained foundational model"

**Egocentric Camera Challenges**:
- EgoMI (arXiv 2511.00153): "Dynamic egocentric views create distribution shifts"
- EMMA (arXiv 2509.04443): "Embodiment gap creates severe distribution shifts"

**Validation Best Practices**:
- LeRobot docs: "Record a handful of episodes first, run small-scale cycle"
- OpenVLA docs: "Verify you can reproduce token accuracies from training"
- Community: "5-10 episodes sufficient to catch pipeline bugs"

**Training Challenges**:
- Pi0 paper: "Training only on high-quality data doesn't teach recovery from mistakes"
- GitHub issues: "Gradient explosion, memory leaks, loss of generalization"

---

## Conclusion

**This strategy saves time by:**
1. Catching bugs with 5-10 episodes (Stage 0) before wasting 20-30 hours
2. Validating egocentric cameras work with 50 episodes (Stage 1) before full commitment
3. Only scaling to 75-100 episodes (Stage 2) if pipeline proven

**Expected outcomes:**
- **Best case**: Pipeline works, egocentric viable, 50 episodes → >40% success → scale up
- **Likely case**: MVP passes, 50 episodes shows learning, iterate to 75-100 for better performance
- **Worst case**: MVP/Stage 1 reveals issues early, pivot before wasting 20+ hours

**Key principle**: **Validate early, iterate fast, scale confidently.**

Good luck with your minimal validation! 🤖
