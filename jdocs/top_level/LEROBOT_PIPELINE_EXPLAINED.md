# LeRobot Pipeline Explained: A Complete Guide for Beginners

**Created:** 2025-11-12
**Author:** Research and Analysis
**Status:** Comprehensive Reference

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What is LeRobot?](#what-is-lerobot)
3. [The Complete Pipeline: High-Level View](#the-complete-pipeline-high-level-view)
4. [Stage 1: Data Collection](#stage-1-data-collection)
5. [Stage 2: Dataset Management](#stage-2-dataset-management)
6. [Stage 3: Configuration System](#stage-3-configuration-system)
7. [Stage 4: Preprocessing & Normalization](#stage-4-preprocessing--normalization)
8. [Stage 5: Policy Training](#stage-5-policy-training)
9. [Stage 6: Evaluation](#stage-6-evaluation)
10. [Stage 7: Real Robot Deployment](#stage-7-real-robot-deployment)
11. [Recent Updates and Breaking Changes](#recent-updates-and-breaking-changes)
12. [Common Issues and Solutions](#common-issues-and-solutions)
13. [Summary: Putting It All Together](#summary-putting-it-all-together)

---

## Executive Summary

**LeRobot** is an end-to-end robotics framework from Hugging Face that enables:
1. Collecting robot demonstration data via teleoperation
2. Training imitation learning policies on that data
3. Deploying trained policies on real robots

**Key Insight:** LeRobot standardizes the entire robotics ML pipeline so you can focus on innovation rather than infrastructure.

**For Your Situation:** Understanding this pipeline explains:
- Why pretrained models may not work out-of-the-box (domain mismatch)
- What configuration files do (define robot setup, cameras, normalization)
- What preprocessors are (normalize inputs, process images)
- How recent updates broke compatibility (processor architecture refactor)

---

## What is LeRobot?

### Design Philosophy

LeRobot is designed to be the "Hugging Face Transformers for Robotics":
- **Standardized datasets** (like ImageNet for vision)
- **Pretrained models** (like BERT for NLP)
- **Simple APIs** (like `model.from_pretrained()`)
- **Community-driven** (share models and datasets on the Hub)

### What Makes It Different?

Most robotics research uses custom, incompatible codebases. LeRobot provides:
- **Universal dataset format** (LeRobotDataset)
- **Multiple policy architectures** (ACT, Diffusion, VQ-BeT, SmolVLA)
- **Direct robot control** (no ROS required!)
- **Cross-robot compatibility** (same code for different arms)

---

## The Complete Pipeline: High-Level View

```
┌─────────────────────────────────────────────────────────────┐
│                    LEROBOT PIPELINE                          │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Stage 1    │  Data Collection
│ TELEOPERATION│  ↓ Record demonstrations using leader arm
└──────────────┘

┌──────────────┐
│   Stage 2    │  Dataset Management
│   DATASET    │  ↓ Store in standardized LeRobotDataset format
└──────────────┘

┌──────────────┐
│   Stage 3    │  Configuration System
│    CONFIG    │  ↓ Define robot, cameras, training settings
└──────────────┘

┌──────────────┐
│   Stage 4    │  Preprocessing & Normalization
│ PREPROCESSOR │  ↓ Normalize actions, process images
└──────────────┘

┌──────────────┐
│   Stage 5    │  Policy Training
│   TRAINING   │  ↓ Learn policy from demonstrations
└──────────────┘

┌──────────────┐
│   Stage 6    │  Evaluation
│   EVALUATION │  ↓ Test policy in simulation or on robot
└──────────────┘

┌──────────────┐
│   Stage 7    │  Real Robot Deployment
│  DEPLOYMENT  │  ↓ Run policy on physical robot at 10-50Hz
└──────────────┘
```

**Important:** Pretrained models skip Stages 1-5 and go directly to Stage 7. But they still need Stages 2-4 (dataset stats, configs, preprocessors) to work correctly.

---

## Stage 1: Data Collection

### How It Works

LeRobot uses **leader-follower teleoperation**:
1. **Leader arm** (you control manually) records your movements
2. **Follower arm** (robot) mirrors the leader movements
3. **Cameras** record visual observations
4. **Both** are synchronized and saved together

### What Gets Recorded

Each **episode** (one demonstration) contains:
- **Camera frames** (RGB images at 10-30 FPS)
- **Joint positions** (robot state at each timestep)
- **Actions** (commanded joint positions)
- **Timestamps** (synchronization info)

### Data Format During Collection

```python
episode = {
    "observation.images.top": [frame_0, frame_1, ..., frame_N],    # Top camera
    "observation.images.wrist": [frame_0, frame_1, ..., frame_N],  # Wrist camera
    "observation.state": [state_0, state_1, ..., state_N],         # Joint positions (6D)
    "action": [action_0, action_1, ..., action_N],                 # Target positions (6D)
    "timestamp": [t_0, t_1, ..., t_N],                             # Time in seconds
}
```

### Example Collection Command

```bash
python lerobot/scripts/control_robot.py \
    --robot-path lerobot.robots.so101_leader \
    --robot-overrides='~cameras' \
    --control-mode teleoperate \
    --display-cameras 0
```

This opens a GUI where you:
1. Start recording
2. Perform the task (e.g., pick up a cube)
3. Stop recording
4. Repeat for 50-100 episodes

---

## Stage 2: Dataset Management

### LeRobotDataset Format

LeRobot saves data in a standardized format:

```
my_dataset/
├── meta/
│   ├── info.json              # Dataset metadata
│   ├── stats.json             # Normalization statistics
│   └── episodes.json          # Episode boundaries
├── videos/
│   ├── observation.images.top/
│   │   ├── episode_000000.mp4
│   │   ├── episode_000001.mp4
│   │   └── ...
│   └── observation.images.wrist/
│       ├── episode_000000.mp4
│       └── ...
└── data/
    ├── chunk-000/
    │   └── episode_000000.parquet  # Joint positions, actions, timestamps
    └── ...
```

### Dataset Statistics (CRITICAL!)

The `stats.json` file contains:

```json
{
  "observation.state": {
    "mean": [0.1, -0.5, 0.2, 0.0, 0.0, 1.5],
    "std": [0.8, 0.6, 0.9, 1.0, 1.2, 2.0],
    "min": [-3.14, -1.57, -3.14, -3.14, -3.14, 0.0],
    "max": [3.14, 1.57, 3.14, 3.14, 3.14, 6.28]
  },
  "action": {
    "mean": [0.1, -0.5, 0.2, 0.0, 0.0, 1.5],
    "std": [0.8, 0.6, 0.9, 1.0, 1.2, 2.0],
    "min": [-3.14, -1.57, -3.14, -3.14, -3.14, 0.0],
    "max": [3.14, 1.57, 3.14, 3.14, 3.14, 6.28]
  }
}
```

**Why This Matters:**
- These stats are used to **normalize** actions during training
- They're used to **denormalize** actions during inference
- **Missing or incorrect stats** cause the "tiny movements" problem you experienced!

### Delta Timestamps

LeRobot has a clever system for temporal data:

```python
dataset = LeRobotDataset("my_dataset", delta_timestamps={
    "observation.images.top": [-0.1, 0.0],       # Load 2 frames: 0.1s ago and now
    "observation.state": [-0.1, -0.05, 0.0],     # Load 3 states: history
    "action": [0.0, 0.033, 0.066],               # Predict 3 future actions
})
```

This enables policies to:
- See **observation history** (helps with motion estimation)
- Predict **action sequences** (smoother control)

---

## Stage 3: Configuration System

### What Are Configuration Files?

LeRobot uses **Hydra** for configuration management. Configs define:
1. **Robot setup** (motors, joints, calibration)
2. **Camera setup** (resolution, FPS, indices)
3. **Policy architecture** (model type, size, hyperparameters)
4. **Training settings** (learning rate, batch size, epochs)
5. **Normalization** (how to scale inputs/outputs)

### Configuration Hierarchy

```
lerobot/configs/
├── default.yaml                   # Base config (rarely modified)
├── robot/
│   ├── so101_follower.yaml       # SO-101 robot definition
│   └── koch.yaml                  # Koch robot definition
├── policy/
│   ├── act.yaml                   # ACT policy settings
│   ├── diffusion.yaml             # Diffusion policy settings
│   └── smolvla.yaml               # SmolVLA settings
├── env/
│   └── pusht.yaml                 # Simulation environment configs
└── training/
    └── default.yaml               # Training hyperparameters
```

### Example: SO-101 Robot Config

```yaml
# lerobot/configs/robot/so101_follower.yaml
name: so101_follower
robot_type: so101_follower

port: /dev/ttyACM2
calibration_path: ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/xlerobot_left_arm.json

motors:
  shoulder_pan:
    index: 1
    drive_mode: 1
  shoulder_lift:
    index: 2
    drive_mode: 1
  # ... more motors

cameras:
  camera1:
    index: 4
    width: 640
    height: 480
    fps: 30
  camera2:
    index: 6
    width: 640
    height: 480
    fps: 30
```

### Where Configs Are Used

**During Training:**
```bash
python lerobot/scripts/train.py \
    policy=smolvla \
    robot=so101_follower \
    dataset_repo_id=lerobot/svla_so101_pickplace
```

This loads:
- `policy/smolvla.yaml` - Model architecture
- `robot/so101_follower.yaml` - Robot definition
- Dataset stats from `lerobot/svla_so101_pickplace`

**During Inference:**
```python
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
# This loads config.json from the model repo
```

---

## Stage 4: Preprocessing & Normalization

### Why Normalize?

Neural networks work best with inputs in the range [-1, 1] or [0, 1]. Raw robot data:
- Joint positions: [-3.14, 3.14] radians
- Actions: [-3.14, 3.14] radians
- Images: [0, 255] pixel values

**Without normalization:**
- Gradients explode or vanish
- Training is unstable
- Model performance suffers

### The Preprocessor Pipeline

As of **LeRobot 0.3.0+**, preprocessing uses a modular pipeline:

```python
preprocessor = ProcessorPipeline([
    NormalizerProcessorStep(
        features=["observation.state", "action"],
        mode="mean_std",  # or "min_max"
        stats=dataset.meta.stats,
    ),
    ImageProcessorStep(
        features=["observation.images.top", "observation.images.wrist"],
        resize=(224, 224),
        normalize=True,
    ),
    DeviceProcessorStep(device="cuda"),
])

# Apply preprocessing
obs_normalized = preprocessor(obs_raw)
```

### Normalization Modes

**1. MEAN_STD (used by SmolVLA, ACT):**

```python
# Normalize (during training)
normalized = (x - mean) / std

# Denormalize (during inference)
denormalized = normalized * std + mean
```

**2. MIN_MAX (used by Diffusion Policy):**

```python
# Normalize
normalized = 2 * (x - min) / (max - min) - 1  # Maps to [-1, 1]

# Denormalize
denormalized = (normalized + 1) / 2 * (max - min) + min
```

### The Postprocessor Pipeline

After the model outputs actions, they need to be **denormalized**:

```python
postprocessor = ProcessorPipeline([
    UnnormalizerProcessorStep(
        features=["action"],
        mode="mean_std",
        stats=dataset.meta.stats,  # CRITICAL: Must match training stats!
    ),
    DeviceProcessorStep(device="cpu"),
])

# Apply postprocessing
action_denormalized = postprocessor(action_normalized)
```

### What The Recent Update Changed

**Before LeRobot 0.3.0:**
- Normalization was **hardcoded inside the policy**
- Stats were loaded automatically during model initialization
- Preprocessor/postprocessor were implicit

**After LeRobot 0.3.0:**
- Normalization is **separate from the policy**
- Stats must be **explicitly passed** to preprocessors
- Models save `preprocessor_config.json` and `postprocessor_config.json`

**Your Problem:**
- Pretrained models may not include these config files
- Without stats, the postprocessor can't denormalize
- Actions stay normalized (values around 0), causing tiny movements!

---

## Stage 5: Policy Training

### What is a Policy?

A **policy** is a neural network that maps observations to actions:

```python
action = policy(observation)
```

Where:
- `observation` = camera images + robot joint positions
- `action` = target joint positions for the robot

### Policy Architectures in LeRobot

#### 1. ACT (Action Chunking Transformer)

**Architecture:**
- **Vision encoder**: ResNet-18 to process camera images
- **Transformer**: Encoder-decoder architecture
- **Action chunking**: Predicts sequences of 100 actions at once

**Advantages:**
- Fast training (2-3 hours)
- Smooth trajectories (predicts action sequences)
- High control frequency (50 Hz)

**Disadvantages:**
- No language conditioning
- Requires 50+ demonstrations

**When to use:** Simple, repetitive tasks (pick-and-place)

#### 2. Diffusion Policy

**Architecture:**
- **Vision encoder**: ResNet-18
- **Diffusion model**: Iteratively denoises action sequences
- **U-Net**: Learns to remove noise from actions

**Advantages:**
- Multimodal behavior (handles multiple solutions)
- Robust to distribution shifts

**Disadvantages:**
- Slower inference (10-20 steps to denoise)
- No language conditioning

**When to use:** Complex tasks with multiple valid solutions

#### 3. SmolVLA (Small Vision-Language-Action Model)

**Architecture:**
- **Vision encoder**: SigLIP (450M parameters)
- **Language encoder**: Qwen2 (450M parameters)
- **Action decoder**: Small MLP (50M parameters)

**Advantages:**
- **Language conditioning** (you can give it natural language commands!)
- Transfer learning from internet data
- 30 Hz control frequency

**Disadvantages:**
- Slower training (requires more data)
- More complex setup

**When to use:** When you need language instructions or want to leverage pretrained vision models

### Training Loop

```python
# Simplified training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get data
        obs = batch["observation"]
        action_true = batch["action"]

        # Preprocess
        obs_normalized = preprocessor(obs)
        action_normalized = preprocessor(action_true)

        # Forward pass
        action_pred = policy(obs_normalized)

        # Loss (mean squared error)
        loss = F.mse_loss(action_pred, action_normalized)

        # Backward pass
        loss.backward()
        optimizer.step()
```

### What Training Produces

After training, you get:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `preprocessor_config.json` - How to normalize inputs (NEW in 0.3.0+)
- `postprocessor_config.json` - How to denormalize outputs (NEW in 0.3.0+)

---

## Stage 6: Evaluation

### Evaluation Modes

**1. Simulation:**
```bash
python lerobot/scripts/eval.py \
    -p outputs/train/smolvla_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10
```

**2. Real Robot:**
```bash
python lerobot/scripts/eval.py \
    -p outputs/train/smolvla_so101 \
    eval.n_episodes=5 \
    eval.use_real_robot=True
```

### Success Metrics

- **Success rate**: % of episodes where task completed
- **Average reward**: Cumulative reward per episode
- **Completion time**: How long to complete task

### When Pretrained Models Fail

Common reasons:
1. **Camera domain mismatch**: Different camera positions, lighting, resolution
2. **Action space mismatch**: Robot calibration different from training
3. **Object differences**: Different objects, positions, or workspace
4. **Missing stats**: Postprocessor can't denormalize correctly

---

## Stage 7: Real Robot Deployment

### Inference Loop

```python
# Initialize
robot = SO101Follower(robot_config)
robot.connect()

# Load model
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
model.eval()

# Create preprocessor/postprocessor
preprocess, postprocess = make_pre_post_processors(
    model.config,
    "lerobot/smolvla_base",
    dataset_stats=dataset.meta.stats,  # CRITICAL!
)

# Control loop
while True:
    # Get observation
    obs = robot.get_observation()  # Images + joint positions

    # Preprocess
    obs_processed = preprocess(obs)

    # Model inference
    action_normalized = model.select_action(obs_processed)

    # Postprocess (denormalize)
    action = postprocess(action_normalized)

    # Send to robot
    robot.send_action(action)

    # Maintain control frequency (e.g., 30 Hz)
    time.sleep(1/30)
```

### Critical Components

**1. Observation:**
- Camera images (must match training resolution!)
- Joint positions (must match training robot)

**2. Preprocessing:**
- Image resizing, normalization
- State normalization using dataset stats

**3. Model Inference:**
- Forward pass through neural network
- Returns normalized actions

**4. Postprocessing:**
- **Denormalization** using dataset stats
- Converts normalized actions back to real joint angles

**5. Robot Control:**
- Sends actions to motors via serial communication
- Safety checks (limits, watchdog)

---

## Recent Updates and Breaking Changes

### Major Update: Preprocessor/Postprocessor Refactor (August 2024, v0.3.0)

**What Changed:**
- Normalization moved out of policies into separate pipelines
- Models now save `preprocessor_config.json` and `postprocessor_config.json`
- Old models (before August 2024) don't have these files

**Impact:**
- **Old pretrained models** may not work with new code without migration
- **New code** expects preprocessor configs alongside model weights
- **Stats handling** changed: must be explicitly passed to processors

### Dataset Format Updates

**Dataset v3.0 (October 2024, v0.4.0):**
- Multiple episodes per file (more efficient)
- Relational metadata for episode boundaries
- Streaming support for large datasets

**Impact:**
- Old datasets (v1.x, v2.x) still supported but deprecated
- New datasets use more efficient storage

### Model Loading Changes

**Fixed in October 2024:**
```python
# OLD (broken):
policy.from_pretrained(path)  # Returns new object, not assigned!

# NEW (fixed):
policy = policy.from_pretrained(path)  # Correctly assigned
```

---

## Common Issues and Solutions

### Issue 1: "Tiny Movements" (Your Problem!)

**Symptom:**
- Robot makes very small movements
- Actions seem to hover around 0
- No purposeful behavior

**Root Cause:**
- Actions are still normalized (values around 0)
- Postprocessor isn't denormalizing correctly
- Missing or incorrect dataset stats

**Solution:**
```python
# Option 1: Load dataset stats explicitly
dataset = LeRobotDataset("lerobot/svla_so101_pickplace")
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    dataset_stats=dataset.meta.stats,  # CRITICAL!
)

# Option 2: Manual denormalization as backup
if np.all(np.abs(action_array) <= 1.5):
    # Action is still normalized, manually denormalize
    action_array = (action_array * std) + mean
```

### Issue 2: Missing Preprocessor Files

**Symptom:**
- Error: "Could not find 'policy_preprocessor.json'"
- Model fails to load

**Root Cause:**
- Old pretrained model (before v0.3.0)
- Doesn't include separate preprocessor files

**Solution:**
- Use model from after August 2024
- Or manually create preprocessor using model config

### Issue 3: Camera Resolution Mismatch

**Symptom:**
- Model inference works but behavior is poor
- Success rate much lower than reported

**Root Cause:**
- Training used different camera resolution
- Model expects specific image sizes

**Solution:**
```python
# Match training resolution exactly
camera_config = {
    "camera1": OpenCVCameraConfig(
        index=4,
        width=640,   # Must match training!
        height=480,  # Must match training!
        fps=30
    )
}
```

### Issue 4: Action Space Mismatch

**Symptom:**
- Actions are out of robot's joint limits
- Motors hit limits frequently
- Jerky movements

**Root Cause:**
- Robot calibration different from training
- Different joint limits

**Solution:**
- Recalibrate robot to match training setup
- Or clip actions to your robot's safe limits:
```python
action_clipped = np.clip(action, joint_mins, joint_maxs)
```

---

## Summary: Putting It All Together

### The Full Picture

LeRobot is a **complete pipeline** from data collection to deployment:

1. **Collect** demonstrations via teleoperation
2. **Store** in standardized LeRobotDataset with stats
3. **Configure** robot, cameras, and training settings
4. **Preprocess** data (normalize, resize images)
5. **Train** policy (ACT, Diffusion, SmolVLA, etc.)
6. **Evaluate** in simulation or on real robot
7. **Deploy** with proper postprocessing (denormalization!)

### Why Pretrained Models May Not Work

Pretrained models are trained on:
- Specific camera setup (position, resolution, lighting)
- Specific robot calibration (joint limits, offsets)
- Specific task setup (objects, workspace layout)

**Your setup differs in:**
- Camera positions (dual arms vs single)
- Robot calibration (different joint ranges?)
- Workspace (different lighting, objects)
- **Missing stats** (postprocessor can't denormalize)

### What The Recent Update Broke

**LeRobot 0.3.0 (August 2024)** introduced:
- Separate preprocessor/postprocessor configs
- Stats must be explicitly passed
- Old models don't have these files

**Your Issues:**
1. **SmolVLA twitchy**: Missing stats in postprocessor
2. **ACT failed to run**: Old model format, missing preprocessor files
3. **Community SmolVLA failed**: Same issue + potential version mismatch

### Your Path Forward

You now understand:
- ✅ How LeRobot works end-to-end
- ✅ What configuration files do
- ✅ What preprocessors are and why they matter
- ✅ What the update changed
- ✅ Why your models aren't working

**Next**: Read `PATH_FORWARD_RECOMMENDATION.md` for detailed advice on what to do next.

---

## Additional Resources

### Key Files in LeRobot Codebase

- `/home/jrobot/project/lerobot/lerobot/scripts/train.py` - Training orchestration (train.py:1-650)
- `/home/jrobot/project/lerobot/lerobot/scripts/eval.py` - Evaluation script
- `/home/jrobot/project/lerobot/lerobot/common/datasets/lerobot_dataset.py` - Dataset loading (lerobot_dataset.py:1-500)
- `/home/jrobot/project/lerobot/lerobot/common/policies/normalize.py` - Normalization system
- `/home/jrobot/project/lerobot/lerobot/common/policies/factory.py` - Policy creation

### Documentation

- **LeRobot Docs**: https://huggingface.co/docs/lerobot
- **SmolVLA Paper**: https://huggingface.co/lerobot/smolvla_base
- **ACT Paper**: https://arxiv.org/abs/2304.13705

### Community

- **Discord**: Hugging Face Discord, #robotics channel
- **GitHub**: https://github.com/huggingface/lerobot

---

**END OF DOCUMENT**
