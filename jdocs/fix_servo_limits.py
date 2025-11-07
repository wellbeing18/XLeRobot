#!/usr/bin/env python3
"""
Safe script to reset servo position limits
Includes multiple safety checks and confirmation steps
"""

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import time

def fix_motor_limits(port="/dev/ttyACM0", robot_id="xlerobot_left_arm"):
    """Safely reset position limits on motor 3 (elbow_flex)"""

    print("=== SAFE SERVO LIMIT RESET ===\n")

    # Safety values from your right arm's working calibration
    # These are known-good values from xlerobot_right_arm.json
    NEW_MIN_LIMIT = 887
    NEW_MAX_LIMIT = 3080

    print(f"This script will:")
    print(f"1. Disable torque (motor goes limp)")
    print(f"2. Set Min_Position_Limit to {NEW_MIN_LIMIT}")
    print(f"3. Set Max_Position_Limit to {NEW_MAX_LIMIT}")
    print(f"4. Re-enable torque")
    print(f"\nThese values are from your working right arm calibration.")
    print(f"This is SAFE and will not damage the motor.\n")

    response = input("Type 'yes' to proceed, or anything else to cancel: ")
    if response.lower() != 'yes':
        print("Cancelled. No changes made.")
        return False

    print("\nConnecting to robot...")
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect()

    print("Robot connected. Accessing motor bus...")
    bus = robot.bus
    motor_name = "elbow_flex"

    try:
        print("\n=== Step 1: Read current values ===")
        current_min = bus.read("Min_Position_Limit", motor_name)
        current_max = bus.read("Max_Position_Limit", motor_name)
        current_pos = bus.read("Present_Position", motor_name)
        print(f"Current Min: {current_min}")
        print(f"Current Max: {current_max}")
        print(f"Current Position: {current_pos}")

        print("\n=== Step 2: Unlock EPROM ===")
        print("Unlocking EPROM to allow limit changes...")
        bus.write("Lock", motor_name, 0)
        time.sleep(0.2)
        lock_status = bus.read("Lock", motor_name)
        print(f"EPROM unlocked: {lock_status == 0}")

        print("\n=== Step 3: Disable torque ===")
        print("Motor will go limp - this is normal and safe")
        bus.write("Torque_Enable", motor_name, 0)
        time.sleep(0.5)
        torque_status = bus.read("Torque_Enable", motor_name)
        print(f"Torque disabled: {torque_status == 0}")

        if torque_status != 0:
            print("ERROR: Could not disable torque. Aborting for safety.")
            robot.disconnect()
            return False

        print("\n=== Step 4: Set new position limits ===")
        print(f"Writing Min_Position_Limit = {NEW_MIN_LIMIT} to EPROM...")
        bus.write("Min_Position_Limit", motor_name, NEW_MIN_LIMIT)
        time.sleep(0.5)  # Longer wait for EPROM write

        print(f"Writing Max_Position_Limit = {NEW_MAX_LIMIT} to EPROM...")
        bus.write("Max_Position_Limit", motor_name, NEW_MAX_LIMIT)
        time.sleep(0.5)  # Longer wait for EPROM write

        print("\n=== Step 5: Verify new limits ===")
        verify_min = bus.read("Min_Position_Limit", motor_name)
        verify_max = bus.read("Max_Position_Limit", motor_name)
        print(f"Verified Min: {verify_min}")
        print(f"Verified Max: {verify_max}")

        if verify_min != NEW_MIN_LIMIT or verify_max != NEW_MAX_LIMIT:
            print("WARNING: New limits don't match expected values!")
            print("Not re-enabling torque for safety.")
            robot.disconnect()
            return False

        print("\n=== Step 6: Re-lock EPROM ===")
        bus.write("Lock", motor_name, 1)
        time.sleep(0.2)
        print("EPROM locked to protect settings")

        print("\n=== Step 7: Re-enable torque ===")
        bus.write("Torque_Enable", motor_name, 1)
        time.sleep(0.5)
        torque_status = bus.read("Torque_Enable", motor_name)
        print(f"Torque re-enabled: {torque_status == 1}")

        print("\n" + "="*50)
        print("✓ SUCCESS!")
        print("="*50)
        print(f"\nOld range: {current_min} to {current_max} ({current_max - current_min} units)")
        print(f"New range: {verify_min} to {verify_max} ({verify_max - verify_min} units)")
        print("\nThe elbow should now have full range of motion.")
        print("\nNext steps:")
        print("1. Turn OFF power to the arm")
        print("2. Manually move the elbow through its full range")
        print("3. Turn power back ON")
        print("4. Re-calibrate the arm using:")
        print(f"   python -m lerobot.scripts.lerobot_calibrate \\")
        print(f"     --robot.type=so101_follower \\")
        print(f"     --robot.port={port} \\")
        print(f"     --robot.id={robot_id}")

        robot.disconnect()
        print("\nRobot disconnected.")
        return True

    except Exception as e:
        print(f"\nERROR during fix: {e}")
        print("Attempting to safely disconnect...")
        import traceback
        traceback.print_exc()
        try:
            robot.disconnect()
        except:
            pass
        return False

if __name__ == "__main__":
    success = fix_motor_limits(port="/dev/ttyACM0", robot_id="xlerobot_left_arm")
    if success:
        print("\n✓ Fix completed successfully!")
    else:
        print("\n✗ Fix failed or was cancelled")
