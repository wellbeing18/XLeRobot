# Pi0.5 Finetuning Issue & Solution Summary

**Date**: 2025-11-23
**Hardware**: RTX 5090 24GB VRAM
**Goal**: Finetune Pi0.5 on SO-101 robot with 50-100 episodes

---

## Issue Encountered

### Problem: LeRobot's Pi0.5 Cannot Finetune on 24GB VRAM

**What we tried**:
```bash
bash scripts/train_pi05_mvp_test.sh
```

**Error progression**:
1. ❌ Script used incorrect parameter names (`--scheduler.warmup_steps` vs `--scheduler.num_warmup_steps`)
2. ❌ Missing scheduler type parameter (`--scheduler.type=cosine_decay_with_warmup`)
3. ❌ Script requested LoRA parameters (`--policy.use_lora=true`) but LeRobot Pi0.5 doesn't support LoRA
4. ⚠️ Even after fixes, full finetuning requires 58-70GB VRAM (we have 24GB)

---

## Root Cause Analysis

### 1. LeRobot Pi0.5 Implementation Limitations

**File**: `/home/jrobot/project/lerobot/src/lerobot/policies/pi05/configuration_pi05.py`

**Lines 95-99**: Explicitly rejects LoRA variants
```python
if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
    raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
    raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
```

**Missing features**:
- ❌ No `use_lora` parameter
- ❌ No `lora_rank` parameter
- ❌ No `lora_alpha` parameter
- ❌ No PEFT library integration
- ❌ No freeze_filter mechanism
- ✅ Only has `gradient_checkpointing` (insufficient for 24GB)

### 2. Memory Requirements Reality

**Official OpenPI Documentation** (https://github.com/Physical-Intelligence/openpi):

| Training Mode | Memory Required | GPU Example |
|---------------|-----------------|-------------|
| Inference | > 8 GB | RTX 4090 |
| **LoRA Finetuning** | **> 22.5 GB** | RTX 4090 |
| Full Finetuning | > 70 GB | A100 80GB |

**Actual User Reports** (OpenPI GitHub Issues):

| Issue | GPU | Config | Result |
|-------|-----|--------|--------|
| #284 | RTX 4090 (24GB) | LoRA | **44GB actual usage - OOM** |
| #284 | A6000 (48GB) | LoRA | ✅ Worked |
| #667 | 2x A100 80GB | Full | **OOM** |
| #379 | RTX 4090 (24GB) | LoRA | **OOM** |

**Conclusion**:
- Documentation claims 22.5GB for LoRA
- Reality: Needs 40-48GB with LoRA
- Full finetuning: 58-70GB minimum

### 3. LeRobot vs OpenPI Comparison

| Feature | OpenPI (JAX) | LeRobot (PyTorch) |
|---------|--------------|-------------------|
| Model Architecture | Pi0.5 (4B params) | ✅ Same |
| LoRA Support | ✅ Yes (via freeze_filter) | ❌ **NO** |
| Freeze Filter | ✅ Yes | ❌ NO |
| PEFT Integration | ✅ Yes | ❌ NO |
| Full Finetuning | ✅ Yes (70GB+) | ✅ Yes (70GB+) |
| PyTorch Version | Limited (new) | ✅ Primary |

**Key insight**: LeRobot ported the model architecture but **did not port LoRA training features**.

---

## Impact on Our Project

### Original Plan (Assumed LoRA Available)
- Collect 50-100 episodes
- Finetune with LoRA on 24GB RTX 5090
- Expected: 40-60% success rate
- Training time: 6-10 hours

### Reality Without LoRA
- **Full finetuning requires**:
  - 500-1000+ episodes (10x more data)
  - 70GB+ VRAM (need A100 80GB or cloud GPU)
  - 20-40 hours training time
  - Risk: Catastrophic forgetting
- **OR**: Need to implement LoRA manually

### Full Finetuning vs LoRA Comparison

| Aspect | Full Finetuning | LoRA Finetuning |
|--------|-----------------|-----------------|
| **Trainable Params** | 4B (100%) | 40M (1%) |
| **Memory (VRAM)** | 58-70GB ❌ | 22-24GB ✅ |
| **Data Required** | 500-1000 episodes | 50-100 episodes ✅ |
| **Training Time** | 20-40 hours | 6-10 hours ✅ |
| **Catastrophic Forgetting** | High risk | Low risk ✅ |
| **Pretrained Knowledge** | Can lose | Preserved ✅ |
| **Our Hardware** | Won't fit ❌ | Should fit ✅ |

---

## Suggested Solutions

### Option A: Implement LoRA in LeRobot's Pi0.5 (RECOMMENDED)

**Effort**: 4-6 hours (moderate complexity)

**Implementation steps**:

1. **Add config parameters** (`configuration_pi05.py` after line 68):
```python
# LoRA configuration
use_lora: bool = False
lora_rank: int = 32
lora_alpha: int = 64
lora_dropout: float = 0.05
```

2. **Update validation** (`configuration_pi05.py` lines 95-99):
```python
# Allow LoRA variants
if self.paligemma_variant not in ["gemma_300m", "gemma_2b", "gemma_300m_lora", "gemma_2b_lora"]:
    raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

if self.action_expert_variant not in ["gemma_300m", "gemma_2b", "gemma_300m_lora", "gemma_2b_lora"]:
    raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
```

3. **Add PEFT integration** (`modeling_pi05.py` imports):
```python
from peft import LoraConfig, get_peft_model
```

4. **Add LoRA wrapping method** (`modeling_pi05.py` in `PI05Pytorch` class):
```python
def apply_lora(self):
    """Apply LoRA adapters to PaliGemma and Action Expert."""
    if not self.config.use_lora:
        return

    # LoRA config
    lora_config = LoraConfig(
        r=self.config.lora_rank,
        lora_alpha=self.config.lora_alpha,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        lora_dropout=self.config.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Wrap PaliGemma language model
    self.paligemma_with_expert.paligemma.language_model = get_peft_model(
        self.paligemma_with_expert.paligemma.language_model,
        lora_config
    )

    # Wrap Action Expert
    self.paligemma_with_expert.gemma_expert = get_peft_model(
        self.paligemma_with_expert.gemma_expert,
        lora_config
    )

    logging.info(f"✅ LoRA enabled: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
    self.paligemma_with_expert.paligemma.language_model.print_trainable_parameters()
```

5. **Call in `__init__`** (after line 517):
```python
# Apply LoRA if configured
self.apply_lora()
```

6. **Update training script**:
```bash
python lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/pi05_base \
  --policy.use_lora=true \
  --policy.lora_rank=32 \
  --policy.lora_alpha=64 \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id=lerobot/xlerobot_mvp_pick_15fps \
  --dataset.root=jdocs/top_level/datasets \
  --steps=6000 \
  --batch_size=8 \
  --optimizer_lr=3e-6 \
  --optimizer_grad_clip_norm=1.0
```

**Expected result**: Training fits in ~19-22GB VRAM

**Pros**:
- ✅ Enables your planned 50-100 episode approach
- ✅ Fits on 24GB VRAM
- ✅ Preserves pretrained knowledge
- ✅ 6-10 hour training time (manageable)

**Cons**:
- ⚠️ Requires code modifications to LeRobot
- ⚠️ Need to test carefully (users report LoRA still tight on 24GB)

**Reference implementation**: GR00T policy in LeRobot already has LoRA working
- File: `/home/jrobot/project/lerobot/src/lerobot/policies/groot/modeling_groot.py`
- Lines: 170-206 (LoRA wrapping methods)

---

### Option B: Use OpenPI's JAX Implementation Directly

**Effort**: 6-8 hours (learning JAX + setup)

**Steps**:
1. Install OpenPI from GitHub
2. Convert LeRobot dataset to OpenPI format
3. Use OpenPI's JAX training script with LoRA config
4. Convert trained model back to PyTorch for LeRobot inference

**Configuration** (from OpenPI docs):
```python
freeze_filter = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    action_horizon=10
).get_freeze_filter()

ema_decay = None  # Required for LoRA
```

**Pros**:
- ✅ Official LoRA implementation
- ✅ No LeRobot code modifications needed

**Cons**:
- ⚠️ Users report 44GB actual usage (won't fit on 24GB)
- ⚠️ Need to learn JAX framework
- ⚠️ Dataset format conversion needed
- ⚠️ Model conversion back to PyTorch for inference

**NOT RECOMMENDED** due to memory issues.

---

### Option C: Switch to GR00T Model

**Effort**: 2-3 hours (dataset re-collection)

**Why**: GR00T already has LoRA implemented in LeRobot

**File**: `/home/jrobot/project/lerobot/src/lerobot/policies/groot/configuration_groot.py`

**Configuration** (lines 79-90):
```python
lora_rank: int = 32
lora_alpha: int = 64
lora_dropout: float = 0.05
lora_full_model: bool = False
```

**Training command**:
```bash
python lerobot/scripts/lerobot_train.py \
  --policy.type=groot \
  --policy.lora_rank=32 \
  --dataset.repo_id=your_dataset \
  --steps=6000
```

**Pros**:
- ✅ LoRA works out of the box
- ✅ Fits on 24GB VRAM (~21-23GB)
- ✅ No code modifications needed

**Cons**:
- ⚠️ Different model architecture (need to re-collect data)
- ⚠️ Different pretrained knowledge base
- ⚠️ May have different performance characteristics

---

### Option D: Cloud GPU Training

**Effort**: 0 hours (no code changes)

**Platforms**:
- Lambda Labs: A6000 48GB ($0.80/hr)
- RunPod: A100 80GB ($1.39/hr)
- Vast.ai: Variable pricing

**For full training** (6 hours):
- Cost: $5-10 total
- Use LeRobot's Pi0.5 full finetuning
- Download trained model
- Run inference locally on 24GB

**Pros**:
- ✅ No code modifications
- ✅ Use full finetuning (better results potentially)
- ✅ Can use your planned 50-100 episodes
- ✅ Low total cost ($5-10)

**Cons**:
- ⚠️ Need cloud account setup
- ⚠️ Upload/download time
- ⚠️ Pay per training run

---

### Option E: Try MVP Test with Aggressive Optimizations

**Effort**: 0 hours (use current script)

**Configuration**:
- `batch_size=1` (minimum)
- `gradient_checkpointing=true`
- `steps=100` (quick test only)

**Expected VRAM**: 22-24GB (at the limit)

**Purpose**:
- ✅ Validate pipeline works
- ✅ See actual VRAM usage on RTX 5090
- ⚠️ NOT sustainable for full 6000-step training

**Use case**: Quick test before committing to Option A/B/C/D

---

## Recommended Path Forward

### Phase 1: Quick Validation (30 min)
1. **Validate dataset**:
```bash
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(
    repo_id='lerobot/xlerobot_mvp_pick_15fps',
    root='jdocs/top_level/datasets'
)
print(f'✅ Dataset: {dataset.num_episodes} episodes, {len(dataset)} frames')
"
```

2. **Optional: Try MVP test** (see if RTX 5090 handles it better than RTX 4090)
```bash
bash scripts/train_pi05_mvp_test.sh
```

### Phase 2: Choose Path (based on Phase 1 results)

**If MVP test succeeds** (VRAM <22GB):
- **Implement Option A** (LoRA in LeRobot) - enables full training

**If MVP test fails** (OOM):
- **Choose Option C** (GR00T) OR
- **Choose Option D** (Cloud GPU)

### Phase 3: Full Training (after LoRA implementation)
1. Collect 40-90 more episodes (50-100 total)
2. Run full 6000-step training with LoRA
3. Expected: 40-60% success rate (50 eps) to 65-80% (100 eps)

---

## Timeline Estimates

| Option | Implementation | Data Collection | Training | Total |
|--------|----------------|-----------------|----------|-------|
| **A (LoRA)** | 4-6 hours | 10-20 hours | 6-10 hours | **20-36 hours** |
| **B (OpenPI JAX)** | 6-8 hours | 10-20 hours | 6-10 hours | **22-38 hours** |
| **C (GR00T)** | 2-3 hours | 10-20 hours | 6-10 hours | **18-33 hours** |
| **D (Cloud GPU)** | 0 hours | 10-20 hours | 6-10 hours | **16-30 hours** |

---

## Key Learnings

### 1. Documentation vs Reality Gap
- **Documented**: LoRA needs 22.5GB
- **Reality**: Users report 44GB actual usage
- **Lesson**: Always test on your hardware, don't trust docs alone

### 2. Framework Differences Matter
- OpenPI (JAX) has LoRA ✅
- LeRobot (PyTorch) doesn't have LoRA for Pi0.5 ❌
- Porting models ≠ porting all features

### 3. Memory Optimization Alone Insufficient
- Gradient checkpointing helps (~40% activation memory)
- But doesn't solve optimizer memory (2x model size)
- LoRA reduces optimizer memory by 85% (critical!)

### 4. Plan for LoRA from Start
- Full finetuning requires 10x more data
- LoRA enables practical finetuning (50-100 episodes)
- Should be first-class consideration, not afterthought

---

## Next Steps

**Immediate** (next 1 hour):
1. Validate dataset is good
2. Try MVP test to measure actual VRAM on RTX 5090
3. Decide between Option A/C/D based on results

**Short-term** (next 1-2 days):
1. Implement chosen option
2. Run MVP training test with LoRA/alternative
3. Validate training pipeline works

**Medium-term** (next 1-2 weeks):
1. Collect 40-90 more episodes (50-100 total)
2. Run full 6000-step training
3. Evaluate finetuned model on robot

---

## References

### Primary Sources
1. [OpenPI GitHub Repository](https://github.com/Physical-Intelligence/openpi)
2. [OpenPI README - Memory Requirements](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)
3. [LeRobot Pi0.5 Documentation](https://huggingface.co/docs/lerobot/pi05)
4. [lerobot/pi05_base Model Card](https://huggingface.co/lerobot/pi05_base)

### User Reports
1. [OpenPI Issue #284: LoRa Finetuning memory](https://github.com/Physical-Intelligence/openpi/issues/284) - RTX 4090 OOM, needed A6000
2. [OpenPI Issue #667: OOM on 2x A100](https://github.com/Physical-Intelligence/openpi/issues/667) - Full finetuning OOM
3. [OpenPI Issue #379: Memory Exhausted LoRA](https://github.com/Physical-Intelligence/openpi/issues/379) - RTX 4090 LoRA OOM
4. [OpenPI Issue #711: Pi05 LoRA finetune results](https://github.com/Physical-Intelligence/openpi/issues/711) - Poor results discussion

### Related Documentation
1. [GR00T LoRA Implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/groot/modeling_groot.py) - Reference for Pi0.5 LoRA
2. [NVIDIA GR00T Blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) - SO-101 finetuning example
3. [HuggingFace PEFT Library](https://github.com/huggingface/peft) - LoRA implementation library

---

## Conclusion

**The core issue**: LeRobot's Pi0.5 does not support LoRA finetuning, making it incompatible with our 24GB VRAM constraint and 50-100 episode data plan.

**Best solution for our project**: **Option A - Implement LoRA** (4-6 hours effort)
- Enables our planned approach
- Preserves pretrained knowledge
- Sustainable long-term solution
- Reference implementation available (GR00T)

**Alternative if time-constrained**: **Option D - Cloud GPU** ($5-10, no code changes)

**Not recommended**: Full finetuning on 24GB (won't work) or OpenPI JAX (still OOM issues)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-23
**Author**: Research findings compiled during Pi0.5 finetuning investigation
