#!/usr/bin/env python3
"""
Force reset servo position by adjusting homing offset
This is a last-resort nuclear option
"""

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import time

def force_reset_position(port="/dev/ttyACM0", robot_id="xlerobot_left_arm"):
    """
    Last resort: Force the servo to think it's at a valid position
    by adjusting the homing offset
    """

    print("=== NUCLEAR OPTION: FORCE POSITION RESET ===\n")
    print("WARNING: This is a last-resort fix that will:")
    print("1. Unlock EPROM")
    print("2. Disable torque")
    print("3. Reset homing offset to force position into valid range")
    print("4. This WILL change the zero position of the elbow")
    print("\nYou will need to recalibrate the entire arm after this.\n")

    response = input("Type 'FORCE' to proceed (all caps), or anything else to cancel: ")
    if response != 'FORCE':
        print("Cancelled.")
        return False

    print("\nConnecting to robot...")
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect()

    bus = robot.bus
    motor_name = "elbow_flex"

    try:
        # Read current state
        print("\n=== Current State ===")
        current_pos = bus.read("Present_Position", motor_name)
        current_min = bus.read("Min_Position_Limit", motor_name)
        current_max = bus.read("Max_Position_Limit", motor_name)
        current_offset = bus.read("Homing_Offset", motor_name)

        print(f"Present Position: {current_pos}")
        print(f"Min Limit: {current_min}")
        print(f"Max Limit: {current_max}")
        print(f"Current Homing Offset: {current_offset}")

        # The servo is at raw position ~99, but with offset 1210,
        # it reports as -1111 degrees
        # We need to adjust offset so reported position is within 1811-2777

        # Target: Make present position = 2300 (middle of 1811-2777 range)
        # Present_Position = Raw_Position - Homing_Offset
        # So: Homing_Offset = Raw_Position - Target_Position

        # Assuming raw position is around 2048 (middle of 4096 range)
        # Let's just set offset to 0 and see where we are
        new_offset = -1700  # This should put us around position 1800-1900

        print(f"\n=== Proposed Fix ===")
        print(f"Set Homing_Offset to: {new_offset}")
        print(f"This should move reported position into valid range")

        confirm = input("\nProceed? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            robot.disconnect()
            return False

        # Unlock EPROM
        print("\nUnlocking EPROM...")
        bus.write("Lock", motor_name, 0)
        time.sleep(0.2)

        # Disable torque
        print("Disabling torque...")
        bus.write("Torque_Enable", motor_name, 0)
        time.sleep(0.5)

        # Write new homing offset
        print(f"Writing Homing_Offset = {new_offset}...")
        bus.write("Homing_Offset", motor_name, new_offset)
        time.sleep(0.5)

        # Verify
        verify_offset = bus.read("Homing_Offset", motor_name)
        print(f"Verified Homing_Offset: {verify_offset}")

        # Check new position
        time.sleep(0.5)
        new_pos = bus.read("Present_Position", motor_name)
        print(f"New Present Position: {new_pos}")

        if new_pos >= current_min and new_pos <= current_max:
            print("\n✓ SUCCESS! Position is now within valid range!")
        else:
            print(f"\n⚠ Position still outside range ({current_min} to {current_max})")
            print("May need to adjust offset further")

        # Re-lock EPROM
        print("\nRe-locking EPROM...")
        bus.write("Lock", motor_name, 1)
        time.sleep(0.2)

        # Re-enable torque
        print("Re-enabling torque...")
        bus.write("Torque_Enable", motor_name, 1)
        time.sleep(0.5)

        print("\n" + "="*50)
        print("DONE")
        print("="*50)
        print("\nNext steps:")
        print("1. Power cycle the arm")
        print("2. Run diagnostic: python jdocs/diagnose_servo.py")
        print("3. If position is in range, recalibrate the entire arm")

        robot.disconnect()
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            robot.disconnect()
        except:
            pass
        return False

if __name__ == "__main__":
    force_reset_position()
