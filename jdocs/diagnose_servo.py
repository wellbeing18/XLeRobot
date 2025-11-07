#!/usr/bin/env python3
"""
Read-only diagnostic script for servo motor 3 (elbow_flex)
This script ONLY reads parameters, it does NOT write anything
"""

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import time

def diagnose_motor(port="/dev/ttyACM0", robot_id="xlerobot_left_arm"):
    """Read and display all relevant parameters from motor 3"""

    print("=== READ-ONLY SERVO DIAGNOSTIC ===")
    print("This script will NOT modify any settings\n")

    print("Connecting to robot...")
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect()

    print("Robot connected. Accessing motor bus...")
    bus = robot.bus

    motor_name = "elbow_flex"
    motor_id = 3

    print(f"\n{'='*50}")
    print(f"Motor {motor_id} ({motor_name}) Diagnostic Report")
    print(f"{'='*50}\n")

    # List of parameters to read
    params_to_check = [
        "Present_Position",
        "Min_Position_Limit",
        "Max_Position_Limit",
        "Torque_Enable",
        "Goal_Position",
        "Present_Load",
        "Present_Voltage",
        "Present_Temperature",
    ]

    results = {}

    for param in params_to_check:
        try:
            value = bus.read(param, motor_name)
            results[param] = value
            print(f"✓ {param:25s}: {value}")
        except Exception as e:
            results[param] = f"ERROR: {e}"
            print(f"✗ {param:25s}: Could not read - {e}")

    print(f"\n{'='*50}")
    print("Analysis:")
    print(f"{'='*50}\n")

    # Analyze the results
    if "Min_Position_Limit" in results and "Max_Position_Limit" in results:
        min_limit = results["Min_Position_Limit"]
        max_limit = results["Max_Position_Limit"]

        if isinstance(min_limit, (int, float)) and isinstance(max_limit, (int, float)):
            range_size = max_limit - min_limit
            print(f"Position Range: {min_limit} to {max_limit}")
            print(f"Range Size: {range_size} units")

            # STS3215 typically has 0-4095 range (12-bit, ~360 degrees)
            # Normal elbow should have ~180-270 degree range = ~2000-3000 units
            if range_size < 1000:
                print("\n⚠️  WARNING: Range is very restricted!")
                print(f"   Normal range is typically 2000-3000 units")
                print(f"   Current range is only {range_size} units")
                print("\n   This confirms the servo is in a locked/restricted state.")
            elif range_size < 2000:
                print("\n⚠️  Range is somewhat restricted")
                print(f"   Consider resetting to wider limits")
            else:
                print("\n✓ Range looks normal")
        else:
            print("Could not read position limits properly")

    if "Present_Position" in results:
        pos = results["Present_Position"]
        if isinstance(pos, (int, float)):
            print(f"\nCurrent position: {pos}")
            if "Min_Position_Limit" in results and "Max_Position_Limit" in results:
                if isinstance(results["Min_Position_Limit"], (int, float)):
                    if pos < results["Min_Position_Limit"] or pos > results["Max_Position_Limit"]:
                        print("⚠️  Current position is OUTSIDE the set limits!")
                        print("   This could cause the servo to not move")

    print(f"\n{'='*50}\n")

    robot.disconnect()
    print("Diagnostic complete. Robot disconnected.")

    return results

if __name__ == "__main__":
    try:
        results = diagnose_motor(port="/dev/ttyACM0", robot_id="xlerobot_left_arm")
    except Exception as e:
        print(f"\nError running diagnostic: {e}")
        import traceback
        traceback.print_exc()
