# SO-101 Leader-Follower Teleoperation Setup Guide

**Date:** 2025-11-01
**Status:** Tested and Working

## Overview

This guide walks through setting up and using SO-101 leader arms to control SO-101 follower arms in real-time. You can control individual arms or both arms simultaneously.

---

## Hardware Setup

### Required Hardware
- 2x SO-101 Follower Arms (with micro USB cables)
- 2x SO-101 Leader Arms (with USB-C cables)
- Power supplies for all 4 arms
- Computer with at least 4 USB ports (USB-A or USB-C)

### Physical Connections

1. **Connect Follower Arms:**
   - Left follower → Micro USB → Computer
   - Right follower → Micro USB → Computer

2. **Connect Leader Arms:**
   - Left leader → USB-C cable → Computer (may need USB-C to USB-A adapter)
   - Right leader → USB-C cable → Computer

3. **Power on all 4 arms** (important - arms won't be detected without power)

### Verify USB Connections

After connecting and powering on all arms:

```bash
ls -l /dev/ttyACM*
```

**Expected output:** 4 USB ports
```
/dev/ttyACM0  (Left leader)
/dev/ttyACM1  (Right leader)
/dev/ttyACM2  (Left follower)
/dev/ttyACM3  (Right follower)
```

**Note:** Port assignments depend on USB plug-in order and can change between reboots!

**To identify current port mapping:**
```bash
python -m lerobot.scripts.lerobot_find_port
```
Then unplug one arm at a time and run again to see which port disappears.

---

## Calibration

### Step 1: Calibrate Follower Arms

If not already calibrated:

```bash
# Calibrate left follower
python -m lerobot.scripts.lerobot_calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm

# Calibrate right follower
python -m lerobot.scripts.lerobot_calibrate \
  --robot.port=/dev/ttyACM3 \
  --robot.id=xlerobot_right_arm
```

**Calibration process:**
1. Move robot to middle of range, press Enter
2. Slowly move each joint through full range of motion
3. Calibration saved to `~/.cache/huggingface/lerobot/calibration/`

### Step 2: Calibrate Leader Arms

```bash
# Calibrate left leader
python -m lerobot.scripts.lerobot_calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=xlerobot_left_leader

# Calibrate right leader
python -m lerobot.scripts.lerobot_calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=xlerobot_right_leader
```

**Note:** Leader arms use `--teleop.type` not `--robot.type` in the calibration command.

---

## Teleoperation

### Single Arm Control

#### Control Left Arm Only

```bash
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=xlerobot_left_leader
```

**What happens:** Move the left leader arm → left follower arm mirrors the movements at ~60Hz

#### Control Right Arm Only

```bash
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --robot.id=xlerobot_right_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=xlerobot_right_leader
```

---

### Bimanual Control (Both Arms Simultaneously)

**Note:** SO-101 doesn't have native bimanual support in the teleoperate script yet. Use this workaround:

**Open two terminal windows and run each command in parallel:**

**Terminal 1 (Left arm):**
```bash
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=xlerobot_left_leader
```

**Terminal 2 (Right arm):**
```bash
python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --robot.id=xlerobot_right_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=xlerobot_right_leader
```

**What happens:** Both arms run independently - you can control both leader arms simultaneously and both follower arms will mirror them.

---

## Usage Tips

### Starting Teleoperation

**Important:** Before starting the teleoperation script, manually move the follower arm to roughly match the leader arm's current position. This ensures smooth tracking from the start.

**Why?** When you stop teleoperation (Ctrl+C), the follower stays in its last position. If you restart without syncing, there may be a large positional error.

### Stopping Teleoperation

Press **`Ctrl+C`** to stop the program.

**Note:** The follower arm will remain in its current position when stopped (motors hold position by design).

### Expected Behavior

When running correctly, you should see continuous output like:
```
time: 16.84ms (59Hz)
time: 17.12ms (58Hz)
...
```

This shows the control loop frequency (~60Hz is normal).

---

## Troubleshooting

### Motors Not Found

**Error:** `Missing motor IDs: 1, 2, 3, 4, 5, 6`

**Solutions:**
1. Check that power is turned ON for all arms
2. Verify USB connections are secure
3. Run `ls -l /dev/ttyACM*` to verify ports exist
4. Try `python -m lerobot.scripts.lerobot_find_port` to auto-detect motors

### Wrong Port Mapping

If arms don't respond or respond incorrectly:

1. **Identify ports by elimination:**
   - Disconnect one arm at a time
   - Run `ls -l /dev/ttyACM*` to see which port disappears
   - Update your commands with correct port mappings

### Follower Doesn't Track Leader Smoothly

**Possible causes:**
1. Initial position mismatch - manually sync positions before starting
2. Calibration issues - recalibrate both leader and follower
3. USB communication issues - try different USB ports

### Permission Denied Error

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Log out and log back in
```

---

## Port Reference

Your current port mapping (verified 2025-11-08):

| Device | USB Port | Calibration ID |
|--------|----------|----------------|
| Left Leader | `/dev/ttyACM0` | `xlerobot_left_leader` |
| Right Leader | `/dev/ttyACM1` | `xlerobot_right_leader` |
| Left Follower | `/dev/ttyACM2` | `xlerobot_left_arm` |
| Right Follower | `/dev/ttyACM3` | `xlerobot_right_arm` |

**⚠️ Important:** These port assignments can change! Always verify with `python -m lerobot.scripts.lerobot_find_port` before running commands.

---

## Next Steps

Now that leader-follower control works, you can:

1. **Collect demonstration data** for training policies (see MVP plan Stage "Fallback: Custom Data Collection")
2. **Record episodes** with cameras for VLA training
3. **Test bimanual manipulation tasks** (pick and place, handovers, etc.)

See `/home/jrobot/project/XLeRobot/jdocs/design/xlerobot_mvp_plan_v1.md` for the full roadmap.
