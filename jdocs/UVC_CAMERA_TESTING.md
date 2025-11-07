# XLeRobot UVC Camera Testing Guide

## Camera Specifications
- **Model**: 1080P USB2.0 UVC Camera
- **FOV**: 130° wide angle
- **Resolution**: 1920x1080 (1080p)
- **Interface**: USB 2.0, Plug and Play
- **Compatibility**: Windows, Linux, Android, macOS

## Important: Why Sections 1.6 and 1.7 Don't Work

### Section 1.6 Issue
The plan assumes **Intel RealSense** cameras:
```python
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
```

Your camera is a **UVC camera**, which requires the `OpenCVCamera` class instead.

### Section 1.7 Issue
The recording script is configured for Intel RealSense cameras. You need custom configuration for UVC cameras.

## Corrected Testing Workflow

### Step 1: Basic Camera Detection (Replaces 1.6)

```bash
# List all video devices
ls -l /dev/video*

# Expected output:
# /dev/video0
# /dev/video1
# etc.
```

### Step 2: Visual Camera Test (Enhanced 1.6)

Run the test script I created for you:

```bash
cd /home/jrobot/project/XLeRobot/jdocs
python test_uvc_camera.py
```

**What this does:**
1. Automatically finds all connected cameras
2. Shows you a **live preview** of each camera (unlike the plan)
3. Measures actual FPS and resolution
4. Tests LeRobot OpenCVCamera integration
5. Saves test images for verification

**Expected output:**
```
Scanning for cameras...
  Camera 0: 1920x1080 @ 30fps
  Camera 1: 1920x1080 @ 30fps  (if you have 2 cameras)

✅ Found 2 camera(s)

Test camera 0? (y/n/q to quit): y
[Live preview window opens]
Press 'q' to quit, 's' to save a test image
```

### Step 3: Recording Test (Replaces 1.7)

```bash
# Basic recording test (no robot required)
python test_uvc_recording.py --mode=basic

# With robot integration
python test_uvc_recording.py --mode=lerobot --robot-port=/dev/ttyACM0

# Generate camera configuration file
python test_uvc_recording.py --mode=config
```

### Step 4: Full Episode Recording (Corrected 1.7)

After generating your camera config:

```bash
python -m lerobot.scripts.control_robot record \
  --robot-path=lerobot/configs/robot/xlerobot_cameras.yaml \
  --fps=30 \
  --repo-id=${HF_USER}/xlerobot_test \
  --num-episodes=5 \
  --warmup-time-s=3 \
  --episode-time-s=30 \
  --reset-time-s=5
```

## Comparison: Plan vs Reality

| Section | Plan (Intel RealSense) | Your Setup (UVC Camera) |
|---------|----------------------|-------------------------|
| 1.6 Camera Test | `IntelRealSenseCamera` | `OpenCVCamera` |
| 1.6 Import | `from ...intelrealsense import IntelRealSenseCamera` | `from ...opencv import OpenCVCamera` |
| 1.7 Recording | Uses RealSense config | Needs custom UVC config |
| Install command | `pip install -e ".[intelrealsense]"` | `pip install -e ".[opencv]"` (included by default) |

## Quick Start Commands

### Just want to test cameras right now?

```bash
# 1. Install opencv-python if not already installed
pip install opencv-python

# 2. Run camera test
cd /home/jrobot/project/XLeRobot/jdocs
python test_uvc_camera.py

# 3. Follow the prompts to test each camera
```

### Testing with LeRobot (after Stage 1.1 installation)

```bash
# 1. Activate LeRobot environment
conda activate lerobot

# 2. Test camera integration
python test_uvc_camera.py

# 3. Test recording
python test_uvc_recording.py --mode=lerobot --robot-port=/dev/ttyACM0 --camera-index=0
```

## Camera Configuration for LeRobot

Your cameras need to be configured in LeRobot's YAML format. Example:

```yaml
# xlerobot_cameras.yaml
cameras:
  top:
    type: opencv
    index: 0  # /dev/video0
    fps: 30
    width: 1920
    height: 1080
    use_rgb: true

  wrist:
    type: opencv
    index: 1  # /dev/video1
    fps: 30
    width: 1920
    height: 1080
    use_rgb: true
```

## Troubleshooting

### Camera not found
```bash
# Check permissions
sudo usermod -a -G video $USER
# Log out and log back in

# Check if camera detected
ls -l /dev/video*
lsusb | grep -i camera
```

### Low FPS (< 20fps)
- USB 2.0 bandwidth limitation with 1080p
- Consider reducing resolution:
  ```python
  camera = OpenCVCamera(width=640, height=480, fps=30)
  ```

### "Camera already in use" error
```bash
# Find process using camera
lsof /dev/video0

# Kill the process
sudo fuser -k /dev/video0
```

### Image quality issues
- Check lighting conditions
- Verify focus (some cameras have manual focus rings)
- Test with: `python test_uvc_camera.py` and save images with 's' key

## Success Criteria

Your cameras are working if:
- ✅ `test_uvc_camera.py` shows live preview
- ✅ Resolution is 1920x1080
- ✅ FPS is ≥ 24 (ideally 30)
- ✅ No frame drops or freezing
- ✅ Image quality is clear and focused
- ✅ LeRobot OpenCVCamera test passes

## Next Steps

After cameras are verified:
1. Proceed to Section 1.7 using `test_uvc_recording.py`
2. Continue with Stage 2 (VLA Inference)
3. Your UVC cameras will work with all pretrained models (SmolVLA, ACT, Pi0)

## Notes

- Your 130° wide angle is **excellent** for robot manipulation (better than standard 60-90° cameras)
- 1080p is higher resolution than the plan assumes (640x480) - you may want to downsample to 640x480 for faster inference
- UVC cameras have lower latency than RealSense depth cameras (good for real-time control)
