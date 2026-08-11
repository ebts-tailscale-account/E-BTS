#!/usr/bin/env python3

# Franka Emika Panda:
#   1. Go to a recorded JOINT pose (exact, joint-space move -- same idea as test.py).
#   2. Then give the user interactive controls to indent the END EFFECTOR
#      straight DOWN along the GLOBAL Z axis, one step at a time, at a
#      resolution (step size in mm) the user dictates.
#
# The downward motion is a CARTESIAN (end-effector) move in the robot base
# frame (panda_link0), whose +Z points up along global Z. We use MoveIt's
# compute_cartesian_path so the tip travels in a straight vertical line.

import sys
import rospy
import moveit_commander

# ------- recorded target pose, as 7 joint angles -------
target_joints = [-0.099759, 0.453835, 0.003536, -2.286833, -0.01887, 2.715436, 0.714523]
# -------------------------------------------------------

# Home, as 7 joint angles (same as move_to_start / test.py).
home_joints = [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397]


def go_to_joints(arm, joints, name):
    print("Going to", name, "(joint-space) ...")
    ok = arm.go(joints, wait=True)
    arm.stop()
    if ok:
        print("Reached", name, ".")
    else:
        print("FAILED to reach", name, "- stopping here.")
    return ok


def move_z(arm, dz):
    """Move the end effector by dz metres along GLOBAL Z (up = +, down = -),
    in a straight Cartesian line. Returns True on (near-)full success."""
    waypoint = arm.get_current_pose().pose
    waypoint.position.z += dz  # base frame Z == global Z

    # eef_step 1mm; jump_threshold 0.0 (disabled).
    (plan, fraction) = arm.compute_cartesian_path([waypoint], 0.001, 0.0)

    if fraction < 0.99:
        print("  WARNING: only planned %.0f%% of the straight-line path -- "
              "not executing (would deviate or hit a limit)." % (fraction * 100.0))
        return False

    ok = arm.execute(plan, wait=True)
    arm.stop()
    return ok


def print_help(step_mm):
    print("")
    print("Controls (step = %.2f mm):" % step_mm)
    print("   d  or  Enter  -> indent DOWN one step  (global -Z)")
    print("   u             -> lift  UP   one step  (global +Z)")
    print("   s <mm>        -> set step size, e.g.  s 0.5")
    print("   z             -> show current EEF position")
    print("   h             -> show this help")
    print("   q             -> quit")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("indent_down")

    arm = moveit_commander.MoveGroupCommander("panda_arm", wait_for_servers=30.0)

    # Move slowly so it is safe (10% speed).
    arm.set_max_velocity_scaling_factor(0.1)
    arm.set_max_acceleration_scaling_factor(0.1)

    print("Planning frame:", arm.get_planning_frame(),
          "| end-effector link:", arm.get_end_effector_link())

    # Step 1: go to the exact recorded joint pose.
    if not go_to_joints(arm, target_joints, "recorded target pose"):
        print("Could not reach the target pose. Aborting.")
        return

    # Step 2: interactive indent controls.
    step_mm = 1.0  # default step resolution: 1 mm
    print_help(step_mm)

    while not rospy.is_shutdown():
        try:
            raw = raw_input("indent> ") if sys.version_info[0] < 3 else input("indent> ")
        except (EOFError, KeyboardInterrupt):
            print("")
            break

        cmd = raw.strip().lower()

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("", "d", "down"):
            print("Indenting DOWN %.2f mm ..." % step_mm)
            if move_z(arm, -step_mm / 1000.0):
                print("  done.")
        elif cmd in ("u", "up"):
            print("Lifting UP %.2f mm ..." % step_mm)
            if move_z(arm, +step_mm / 1000.0):
                print("  done.")
        elif cmd.startswith("s"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    step_mm = float(parts[1])
                    print("Step size set to %.3f mm." % step_mm)
                except ValueError:
                    print("Could not parse step size. Use e.g.  s 0.5")
            else:
                print("Usage:  s <mm>   e.g.  s 0.5")
        elif cmd in ("z", "pos"):
            p = arm.get_current_pose().pose.position
            print("  EEF position (base frame): x=%.4f, y=%.4f, z=%.4f" %
                  (p.x, p.y, p.z))
        elif cmd in ("h", "help", "?"):
            print_help(step_mm)
        else:
            print("Unknown command '%s'. Press h for help." % cmd)

    arm.stop()
    print("All done!")


if __name__ == "__main__":
    main()
