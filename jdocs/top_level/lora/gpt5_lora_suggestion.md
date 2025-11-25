# GPT‑5 LoRA Strategy Suggestions for XLeRobot

_Last updated: 2025‑11‑24_

## 1. Context & Constraints

- **Robot**: XLeRobot SO‑101 (single/dual arm), egocentric cameras.
- **GPU**: RTX 5090, **24 GB VRAM**.
- **Goal**: VLA fine‑tuning for SO‑101 using **LoRA**, both:
  - Week‑0/Week‑1 **MVP** (20–50 demos) to validate datasets + training/inference pipeline.
  - Later **full fine‑tuning** (50–100+ demos) for usable performance.
- **Hard requirement**: MVP must exercise the **same LoRA‑based training path** we intend to use long‑term (no “full‑finetune‑only” shortcuts).

## 2. Models Considered

### 2.1 Pi0.5 (`lerobot/pi05_base`)

**Pros**
- 4B VLA with **very strong cross‑embodiment + viewpoint generalization**, trained on 10k+ hours + 400 h mobile manipulation (good fit for egocentric cameras).
- Strong vision‑language backbone (Paligemma), good for out‑of‑distribution scenes.
- Already integrated into LeRobot (`PI05Config`, `PI05Policy`) and we have a working **baseline inference script** on XLeRobot.

**Cons / Blockers (current LeRobot checkout)**
- **No LoRA support implemented**:
  - `PI05Config` has no `use_lora`, `lora_rank`, `lora_alpha`, etc.
  - `modeling_pi05.py` has no PEFT/LoRA plumbing.
  - CLI flags like `--policy.use_lora` / `--use_lora=true` cause `unrecognized arguments` errors.
- **Full fine‑tune VRAM too high**:
  - OpenPI + community reports show full Pi0.5 training needing **≫24 GB**, often 40–70 GB.
  - Our 24 GB card is realistically limited to **LoRA‑style** fine‑tuning.
- No official, end‑to‑end “Pi0.5 + LoRA + SO‑101 + single‑GPU 24 GB” recipe yet; we would be first‑movers here.

### 2.2 GR00T N1.5 (`nvidia/GR00T-N1.5-3B`)

**Key external references**
- **SO‑101 finetuning tutorial**: [Post‑Training Isaac GR00T N1.5 for LeRobot SO‑101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning?ncid=so-yout-577961-vt48).
- **Isaac‑GR00T getting started** (scripts & examples): <https://github.com/NVIDIA/Isaac-GR00T/tree/main/getting_started?ncid=so-yout-106562-vt48>.
- **LeRobot × NVIDIA Healthcare blog** (GR00T + LeRobot integration): <https://huggingface.co/blog/lerobotxnvidia-healthcare>.

**Pros**
- **Exact target match**: Official tutorial demonstrates **post‑training GR00T N1.5 on a single SO‑101 arm** using teleop data, deployed back onto SO‑101 [HF GR00T SO‑101 blog].
- Designed for **new embodiments** via `embodiment_tag="new_embodiment"`; SO‑101 is handled explicitly as a new embodiment in the tutorial.
- **LoRA finetuning is first‑class**:
  - Isaac‑GR00T scripts (`gr00t_finetune.py`) expose `--lora-rank` etc., with default settings needing ~25 GB; there are flags like `--no-tune_diffusion_model` to lower VRAM if needed.
  - In our local LeRobot tree:
    - `GrootConfig` (in `policies/groot/configuration_groot.py`) defines `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model`.
    - `GrootPolicy` is wired into `get_policy_class("groot")` / `make_policy_config("groot")`.
    - `make_pre_post_processors` has a **special case for `GrootConfig`** that:
      - Uses **LeRobot dataset stats** (`dataset_stats`) and
      - Configures `groot_pack_inputs_v3` + `groot_action_unpack_unnormalize_v1` to map **LeRobot dataset features** → GR00T inputs/outputs.
  - This means we can use **LeRobot datasets + LoRA + GrootPolicy + lerobot_train.py** with only modest dataset metadata tweaks (e.g., `modality.json`).
- **Documented pipeline** from dataset → finetune → open‑loop eval → real‑robot deployment using SO‑101, including:
  - Using a LeRobot dataset (`youliangtan/so101-table-cleanup`).
  - Adding `<DATASET_PATH>/meta/modality.json` (e.g., copying `so100_dualcam__modality.json` for dual‑camera setups).
  - Running `gr00t_finetune.py` with `--dataset-path`, `--data-config so100_dualcam`, `--video-backend torchvision_av`.
  - Running `inference_service.py` as a GR00T policy server and LeRobot’s `eval_lerobot.py` client on the SO‑101 robot [HF GR00T SO‑101 blog + Isaac‑GR00T getting_started].
- VRAM + compute profile matches our hardware:
  - **~25 GB VRAM** default (fits 24 GB with minor compromises).
  - Single‑GPU flow is explicitly supported (`--num-gpus 1`).

**Cons / Trade‑offs**
- Different architecture & backbone (Eagle‑based, ~3B) vs Pi0.5’s 4B Paligemma‑based stack.
- Input modality is **video‑centric** (`video` tensor + state) rather than Pi‑style multi‑camera dict, so:
  - We need to ensure our dataset has (or can be converted to) appropriate **video streams** and **state/action vectors**.
  - Might need to convert LeRobot **dataset v3 → v2** for now, as some GR00T tooling assumes v2 (see comments + PR link in HF blog).
- We would be leaning more on **NVIDIA’s and LeRobot’s GR00T path** than on the Pi0.5 ecosystem we already studied; some Pi0.5‑specific research/notes become less central for the first MVP.

## 3. High‑Level Recommendation

Given:
- We **must** exercise a **LoRA‑based finetuning pipeline** for the MVP.
- We have only **24 GB VRAM**.
- There is a **fully documented, SO‑101‑specific GR00T N1.5 finetuning & deployment path** (HF GR00T SO‑101 blog + Isaac‑GR00T getting_started + LeRobot×NVIDIA healthcare).
- Our current LeRobot checkout has **no Pi0.5 LoRA implementation**, and full Pi0.5 finetune is not feasible on 24 GB.

> **Recommendation**  
> **Use GR00T N1.5 as the primary VLA** for both MVP and full LoRA finetuning on SO‑101, and treat Pi0.5 LoRA support as a **secondary, longer‑term engineering task** once the GR00T pipeline is validated.

Rationale:
- GR00T N1.5 already has:
  - Official **SO‑101 examples** and strong integration with LeRobot datasets.
  - **LoRA finetuning** and LoRA‑friendly VRAM usage.
  - A known path from LeRobot dataset → GR00T → SO‑101 deployment.
- Pi0.5 remains attractive theoretically (diverse pretraining, egocentric robustness), but requires **non‑trivial framework work** to even start LoRA training on our hardware.

## 4. Suggested GR00T‑First Plan

### 4.1 Align Dataset for GR00T

1. **Check dataset version**
   - GR00T examples currently assume **LeRobot dataset v2**.
   - If our dataset is in **v3**, use the conversion script referenced in the HF blog comments / LeRobot PR 2109 to produce a v2‑compatible copy for GR00T.

2. **Add `meta/modality.json`**
   - Start from GR00T examples:
     - `getting_started/examples/so100_dualcam__modality.json` for dual‑camera setups.
     - `getting_started/examples/so100__modality.json` for single‑camera setups.
   - Copy and adapt to our dataset:
     - Map **video streams** (e.g., egocentric head/wrist cameras).
     - Map **state vector** (joint positions for SO‑101).
     - Map **action vector** (same joint set).
   - Place it at: `<DATASET_ROOT>/meta/modality.json`.

3. **Quick dataset sanity checks**
   - Use either:
     - Isaac‑GR00T’s `scripts/load_dataset.py --dataset-path <DATASET_ROOT> --plot-state-action --video-backend torchvision_av`, or
     - LeRobot’s dataset loading utilities, to confirm:
       - Episode count, FPS, and cameras match expectations.
       - Actions and state shapes line up with SO‑101 (6 or 12 DOF, etc.).

### 4.2 MVP GR00T LoRA Finetuning

Two main options (we can revisit which one we choose when we implement):

#### Option A: Use Isaac‑GR00T scripts directly

- **Finetune** with `gr00t_finetune.py` as in the HF SO‑101 tutorial:
  - Example (to be adapted to our dataset paths):
    ```bash
    python scripts/gr00t_finetune.py \
       --dataset-path <DATASET_ROOT> \
       --num-gpus 1 \
       --output-dir <OUTPUT_CHECKPOINT_DIR> \
       --max-steps 6000 \
       --data-config so100_dualcam \
       --video-backend torchvision_av \
       --lora-rank 16
       # Optionally: --no-tune_diffusion_model to reduce VRAM
    ```
- **Pros**
  - Follows the **official SO‑101 recipe** almost exactly.
  - Good documentation + community Q&A around these scripts.
- **Cons**
  - Requires keeping an additional `Isaac-GR00T` environment / repo in sync.
  - Need to bridge the resulting GR00T checkpoint back into our XLeRobot control stack (via server/client or custom script).

#### Option B: Use LeRobot’s `GrootPolicy` + `lerobot_train.py`

- Use **LeRobot as the single training interface**, by:
  - Setting `policy.type=groot` and providing a `GrootConfig` with:
    - `base_model_path="nvidia/GR00T-N1.5-3B"`.
    - `embodiment_tag="new_embodiment"`.
    - `lora_rank > 0` (e.g., 16 or 32), `lora_alpha`, etc.
  - Passing our LeRobot dataset (`--dataset.repo_id`, `--dataset.root`) and `dataset_stats`.
  - Relying on `make_pre_post_processors`’ special `GrootConfig` handling to wire dataset stats into GR00T’s pack/unpack steps.
- **Pros**
  - Single toolchain: **same `lerobot_train.py`** used for GR00T and (later) Pi0.5 experiments.
  - Easier integration with existing LeRobot‑based control/inference scripts.
- **Cons**
  - LeRobot’s GR00T path is newer than Isaac‑GR00T; examples are a bit less “tutorial‑like”.
  - We may need to cross‑reference Isaac‑GR00T configs to choose good defaults for GR00T hyperparameters inside LeRobot.

**MVP suggestion**: Start with **Option A (Isaac‑GR00T)** for the first successful SO‑101 finetune (minimum friction), then migrate to **Option B** once we’re confident in the dataset and GR00T hyperparameters.

### 4.3 GR00T Inference & Deployment

1. **Open‑loop evaluation** (no robot, visualize trajectories)
   - Use `scripts/eval_policy.py` from Isaac‑GR00T with:
     - `--model_path <OUTPUT_CHECKPOINT_DIR>`
     - `--embodiment_tag new_embodiment`
     - `--data_config so100_dualcam`
     - `--dataset_path <DATASET_ROOT>`
     - `--video_backend torchvision_av`
   - Check that predicted trajectories look reasonable for our SO‑101 task.

2. **On‑robot deployment** (SO‑101)
   - Follow the HF blog + Isaac‑GR00T getting_started:
     - Run `inference_service.py --server` with our finetuned checkpoint and data config.
     - Use `getting_started/examples/eval_lerobot.py` (updated with our robot ports, IDs, and camera indices) as a client:
       - This client uses LeRobot robot drivers (SO‑101 followers, cameras) and sends observations to the GR00T policy server.
   - This gives a complete **GR00T policy server ↔ LeRobot robot client** flow on our hardware.

3. **Later improvement**: Once stable, we can optionally:
   - Implement a **pure‑LeRobot inference script** that loads `GrootPolicy.from_pretrained()` directly and uses `make_pre_post_processors` to avoid the separate server/client, similar to how `run_pi05_inference_corrected.py` works for Pi0.5.

## 5. Pi0.5 LoRA as a Secondary, Longer‑Term Task

Pi0.5 is still worth revisiting once the GR00T pipeline is working, but it should not block the MVP.

### 5.1 Minimal Pi0.5 LoRA Implementation Plan

1. **Extend `PI05Config`**
   - Add LoRA fields:
     - `use_lora: bool = False`
     - `lora_rank: int = 32`
     - `lora_alpha: int = 64`
     - `lora_dropout: float = 0.05`
   - If there is variant validation (e.g., Paligemma/action expert variants), consider allowing LoRA variants if following OpenPI naming, or keep variants the same and just inject adapters.

2. **Wire PEFT into `modeling_pi05.py`**
   - Import PEFT:
     - `from peft import LoraConfig, get_peft_model`
   - Add an `apply_lora()` method in the Pi0.5 model/policy class:
     - Build a `LoraConfig` based on `config.lora_rank`, `config.lora_alpha`, `config.lora_dropout`.
     - Wrap:
       - The Paligemma language model.
       - The action expert / diffusion head.
     - Use an appropriate `target_modules` list (similar to what we sketched for Pi0.5 and what GR00T uses for its Eagle backbone).
     - Freeze base model weights as needed so only adapters train.
   - Call `self.apply_lora()` in `__init__` when `config.use_lora` is `True`.

3. **Expose LoRA via CLI / `lerobot_train.py`**
   - Ensure the training script passes:
     - `--policy.use_lora=true`
     - `--policy.lora_rank=32`
     - `--policy.lora_alpha=64`
     - `--policy.lora_dropout=0.05`
   - Align optimizer/scheduler defaults with our Pi0.5 fine‑tuning strategy doc (LR ~3e‑6, gradient clipping, cosine decay with warmup).

4. **Validation sequence**
   - Start with **tiny MVP**:
     - 5–10 episodes, 100 training steps, batch size 1–2.
     - Confirm:
       - Training runs without `NaN` or OOM.
       - VRAM stays below ~22 GB.
   - Once stable, compare:
     - **Pi0.5‑LoRA** vs **GR00T‑LoRA** on the **same SO‑101 dataset**.
     - Metrics: success rate on our primitive task(s), behavior smoothness, robustness to camera variation.

### 5.2 When to Prioritize Pi0.5 LoRA

We should only invest in Pi0.5 LoRA implementation if:
- **GR00T‑LoRA pipeline is already working** and we want to:
  - Squeeze extra performance in challenging egocentric setups.
  - Run a fair comparison across models.
- We’re prepared to take on:
  - **4–6 hours** of low‑level framework work + debugging.
  - Ongoing maintenance to keep the Pi0.5 LoRA patch in sync with upstream LeRobot.

## 6. Open Questions / Future Revisions

Things to revisit as we gain experience:
- **Camera setup**:
  - GR00T examples tend to use external/front or dual‑cam setups.
  - We need empirical data on how GR00T N1.5 behaves with our fully egocentric cameras vs. Pi0.5.
- **Dataset versioning**:
  - If/when GR00T fully supports LeRobot dataset v3, we can simplify the pipeline (no v2→v3 conversion).
- **Model comparison**:
  - After we have both GR00T‑LoRA and Pi0.5‑LoRA runs, we should add a small results table here comparing:
    - Success rate.
    - Data efficiency (episodes to reach X% success).
    - Behavior smoothness.
    - Robustness to viewpoint / object changes.

This document should be updated as we implement the GR00T pipeline and, later, Pi0.5 LoRA. For now, it captures the current best recommendation: **GR00T N1.5 as the primary LoRA‑finetuned VLA on SO‑101, with Pi0.5 LoRA as a follow‑up experiment once the GR00T path is solid.**


## 7. Post‑Implementation Review of Pi0.5 LoRA Integration

This section captures a focused review of the **actual Pi0.5 LoRA implementation** we now have in the LeRobot fork, so future readers can quickly verify that the code matches the design intent from this document.

### 7.1 Overall Verdict

- **Status**: The Pi0.5 LoRA implementation in `configuration_pi05.py` and `modeling_pi05.py` is **technically sound and consistent** with:
  - LeRobot’s config / training pipeline (`TrainPipelineConfig`, `make_policy`, `lerobot_train.py`).
  - PEFT best practices for base transformer models (no `task_type`, apply LoRA after weight loading, handle GC carefully).
- **Consequence**: It is reasonable to proceed to longer runs (MVP 500 steps, then ~6000 steps), watching VRAM and normalization as usual but not expecting structural issues from the LoRA plumbing itself.

### 7.2 Config & CLI Wiring (`PI05Config` + `lerobot-train`)

- **LoRA fields** (`configuration_pi05.py`):
  - `use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout` are correctly added to `PI05Config` with sensible defaults (rank 16, alpha 32, dropout 0.1).
  - These fields are exactly what your training script uses:
    - `--policy.use_lora=true`
    - `--policy.lora_rank=16`
    - `--policy.lora_alpha=32`
    - `--policy.lora_dropout=0.1`
- **Config loading path**:
  - `TrainPipelineConfig.validate()` sees `--policy.path=lerobot/pi05_base`, calls `PreTrainedConfig.from_pretrained(...)` with CLI overrides, and sets `policy.pretrained_path`.
  - `make_policy` then:
    - Infers input/output features from dataset metadata.
    - Calls `policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, config=cfg)` for `PI05Policy`.
- **Normalization stats**:
  - When `cfg.policy.pretrained_path` is set and we’re not resuming, `lerobot_train` injects `dataset.meta.stats` into the normalizer processor.
  - This is aligned with the mitigation for OpenPI issue #711 (fresh quantile stats on the finetuning dataset, not reusing pretraining stats).

### 7.3 Where LoRA Is Actually Applied (`PI05Pytorch`)

- **Core logic**:
  - `PI05Pytorch` defines `_apply_lora()` and `apply_lora_if_enabled()`.
  - `_apply_lora()`:
    - Builds a `LoraConfig` with:
      - `r=config.lora_rank`, `lora_alpha=config.lora_alpha`, `lora_dropout=config.lora_dropout`.
      - `target_modules`:
        - `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`
        - `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
    - Wraps:
      - `paligemma.model.language_model` (text transformer inside `PaliGemmaForConditionalGeneration`).
      - `gemma_expert.model` (Gemma action expert transformer).
  - `apply_lora_if_enabled()` is called **after** the pretrained state dict is loaded (see §7.4).
- **Precision & gradient checkpointing handling**:
  - Before wrapping, the code snapshots each submodel’s `gradient_checkpointing` flag, sets it to `False`, then recreates it on the `PeftModel` and restores the flag.
  - This is important because:
    - PEFT’s `enable_input_require_grads()` logic is fragile for models with missing `embed_tokens` (like the expert).
    - You want to control checkpointing at the Pi0.5 wrapper level (`PI05Pytorch.gradient_checkpointing_enable()`), not from inside PEFT.
- **Non‑LoRA’d components** (remain standard PyTorch modules as intended):
  - Vision tower (SigLIP).
  - `action_in_proj`, `action_out_proj` (small, task‑specific projections).
  - `time_mlp_in`, `time_mlp_out` (flow‑matching conditioning MLP).
  - This matches the design in the validation report and keeps the LoRA scope focused on the two big text backbones.

### 7.4 Load‑Order & State‑Dict Remapping (`PI05Policy.from_pretrained`)

- **Correct load order**:
  - In `PI05Policy.from_pretrained`:
    1. If needed, load a `PI05Config` via `PreTrainedConfig.from_pretrained(...)`.
    2. Instantiate `PI05Policy(config)`, which builds `PI05Pytorch(config)` (Paligemma + Gemma expert).
    3. Use `transformers.utils.cached_file` + `safetensors.torch.load_file` to get the original OpenPI‑style `model.safetensors`.
    4. Run `_fix_pytorch_state_dict_keys(...)` to:
       - Handle adaRMS vs standard layernorm mismatches.
       - Rename `action_time_mlp_*` → `time_mlp_*`.
       - Drop `state_proj.*` keys that don’t exist in Pi0.5.
       - Emit warnings for vision `patch_embedding` keys (informational only).
    5. Prefix any keys that don’t start with `"model."` with `"model."`.
    6. Call `model.load_state_dict(remapped_state_dict, strict=strict)`.
    7. **Only then** call `model.model.apply_lora_if_enabled()`.
- **Effect**:
  - The base weights loaded from `lerobot/pi05_base` are fully applied to the un‑wrapped transformer modules.
  - PEFT wraps the final, correct transformers, so the LoRA adapter state is initialized on top of the pretrained weights (no key mismatch).
- **Behavior vs base `PreTrainedPolicy`**:
  - The override does **all** of the above remapping, which the generic version cannot do; this is required for OpenPI → LeRobot compatibility.
  - The only divergence is that `PI05Policy.from_pretrained` does **not** call `.eval()` or `.to(config.device)` at the end. This is harmless for `lerobot-train` (which calls `policy.to(cfg.device)` and then `.train()`), but is good to remember if you ever use `PI05Policy.from_pretrained` directly for inference.

### 7.5 Training Script Review (`train_pi05_mini_mvp.sh`)

- **Script wiring**:
  - Uses `lerobot-train` with:
    - `--policy.path=lerobot/pi05_base`
    - `--policy.use_lora=true`, `--policy.lora_rank=16`, `--policy.lora_alpha=32`, `--policy.lora_dropout=0.1`
    - `--policy.gradient_checkpointing=true`
    - Dataset settings pointing at a LeRobot v3 dataset and `pyav` as the video backend.
  - Respects LeRobot’s requirement that `--output_dir` **must not exist** before the run (lets the framework create it).
  - Saves a temp log to `/tmp/...` and copies it into the output directory after training, which is useful for inspection and matches the validation report.
- **Environment checks**:
  - Explicitly verifies PyTorch, CUDA, PEFT, and `PI05Policy` imports in the `lerobot` conda env before training.
  - This is consistent with how we expect the Pi0.5 LoRA path to be used in practice and supports the “mini‑MVP” validation goal.

### 7.6 Alignment with the Strategy in This Document

- The **implemented Pi0.5 LoRA path** matches the long‑term plan sketched in §5:
  - Config has LoRA toggles and hyperparameters.
  - `modeling_pi05.py` wires PEFT into both PaliGemma and the action expert.
  - LoRA is exposed and controlled via the standard `lerobot-train` CLI.
  - A tiny 100‑step MVP run exists and behaves healthily (loss and gradient trends as reported).
- The only notable differences from the earlier “wish list” are:
  - LoRA is applied **after** loading pretrained weights instead of at `__init__` time (which is actually better than the original sketch).
  - Rank/alpha defaults are tuned to 16/32 instead of the earlier 32/64 suggestion, to keep memory more comfortable on 24 GB.

### 7.7 Suggested Small Follow‑Ups (Non‑Blocking)

- **Config hygiene**:
  - `PI05Config` currently defines `tokenizer_max_length` twice with the same value; keeping just one definition would reduce confusion.
- **Optional behavior parity with `PreTrainedPolicy`**:
  - If you expect frequent direct use of `PI05Policy.from_pretrained` outside `lerobot-train`, consider adding `model.to(config.device)` and `model.eval()` at the end for consistency with other policies.
- **Future flexibility** (if/when needed):
  - Add an optional flag like `lora_target_mlp: bool = True` to support an “attention‑only LoRA” mode (GR00T‑style) when VRAM is extremely tight.
  - Optionally disable EMA when `use_lora` is `True` (mirroring OpenPI’s JAX LoRA setup), if you later adopt EMA in your training recipes.

In summary, the **current Pi0.5 LoRA implementation is in good shape**: it respects the architecture and training stack described in this document, follows PEFT best practices, and has already passed a mini‑MVP validation run. It is now a viable secondary path alongside the GR00T‑first strategy described above.


