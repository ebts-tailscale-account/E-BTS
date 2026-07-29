#!/usr/bin/env python3
"""Runs the snake grid sweep on the Franka AND logs franka_states as a
UNIX-timestamped time series to a CSV. Invoked ON TACTILE by master.py over ssh.

Reuses the validated snake_indent motion (home -> 16-point boustrophedon, 2 mm
dip at each point) and, throughout, a franka_states subscriber logs pose /
endpoint effort / joints / collision at the controller's publish rate. rospy
dispatches the subscriber callback on its own thread, so logging runs
concurrently with the (blocking) MoveIt motion on the main thread.

Prereq: the clean MoveIt bringup must already be running so move_group and
/franka_state_controller/franka_states exist:
    roslaunch panda_moveit_config franka_control.launch robot_ip:=10.1.196.5 load_gripper:=false

Usage (normally via master.py):  python3 franka_grid_logger.py [out_csv]
"""

import sys
import threading
import time

import rospy
import moveit_commander
from franka_msgs.msg import FrankaState

# --- level + just-touching reference (bottom-left corner), from level_jog_capture ---
start_joints = [-0.001577, 0.456974, 0.018536, -1.883728, -0.006198, 2.377692, 0.750261]
home_joints  = [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397]

# --- grid (full 30 x 36 mm surface) ---
N_ROWS, N_COLS       = 4, 4
V_SPACING, H_SPACING = 0.012, 0.010     # 12 mm vertical, 10 mm horizontal
VERT, RIGHT          = (1.0, 0.0), (0.0, -1.0)  # base-frame directions (verified)
INDENT, HOVER, DWELL = 0.002, 0.005, 2.0        # 2 mm dip, 5 mm hover, 2 s hold


class FrankaLogger:
    """Streams franka_states rows to a CSV with a UNIX timestamp per sample."""

    def __init__(self, path):
        self.file = open(path, "w")
        self.file.write("unix_time_s,robot_time_s,"
                        "q0,q1,q2,q3,q4,q5,q6,"
                        "ee_x,ee_y,ee_z,"
                        "Fx_ext_N,Fy_ext_N,Fz_ext_N,Tx_ext_Nm,Ty_ext_Nm,Tz_ext_Nm,"
                        "collided,robot_mode\n")
        self.lock = threading.Lock()
        self.count = 0
        self.sub = rospy.Subscriber("/franka_state_controller/franka_states", FrankaState,
                                    self._on_state, queue_size=400)

    def _on_state(self, msg):
        unix_s = time.time()
        robot_s = msg.header.stamp.to_sec()
        # O_T_EE is column-major 4x4; the EE position is the last column (12,13,14).
        ee = (msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14])
        # O_F_ext_hat_K = estimated external wrench at the EE: [Fx Fy Fz Tx Ty Tz] (N, Nm).
        w = msg.O_F_ext_hat_K
        collided = int(any(msg.cartesian_collision) or any(msg.joint_collision))

        row = "%.6f,%.6f,%s,%s,%s,%d,%d\n" % (
            unix_s, robot_s,
            ",".join("%.6f" % v for v in msg.q),
            ",".join("%.6f" % v for v in ee),
            ",".join("%.4f" % v for v in w),
            collided, msg.robot_mode,
        )
        with self.lock:
            self.file.write(row)
            self.count += 1

    def close(self):
        self.sub.unregister()
        with self.lock:
            self.file.flush()
            self.file.close()


def go_to(arm, joints, name):
    print("Going to", name, "...")
    ok = arm.go(joints, wait=True)
    arm.stop()
    print("Reached", name, "." if ok else "- FAILED.")
    return ok


def goto_cart(arm, robot, x, y, z, speed):
    wpose = arm.get_current_pose().pose
    wpose.position.x, wpose.position.y, wpose.position.z = x, y, z
    wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w = 1.0, 0.0, 0.0, 0.0
    plan, frac = arm.compute_cartesian_path([wpose], 0.001, True)  # eef_step, avoid_collisions
    if frac < 0.99:
        print("  WARN: only %.0f%% reachable - skipped." % (frac * 100))
        return False
    plan = arm.retime_trajectory(robot.get_current_state(), plan, speed, speed)
    ok = arm.execute(plan, wait=True)
    arm.stop()
    return ok


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ebts_franka_run.csv"

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("franka_grid_logger")

    robot = moveit_commander.RobotCommander()
    arm = moveit_commander.MoveGroupCommander("panda_arm", wait_for_servers=30.0)
    arm.set_max_velocity_scaling_factor(0.1)
    arm.set_max_acceleration_scaling_factor(0.1)

    logger = FrankaLogger(out_csv)
    print("Logging franka_states to", out_csv)

    try:
        if not go_to(arm, home_joints, "home"):
            return
        if not go_to(arm, start_joints, "start (bottom-left touch)"):
            return

        p = arm.get_current_pose().pose.position
        start_x, start_y, touch_z = p.x, p.y, p.z
        hover_z = touch_z + HOVER

        order = []
        for c in range(N_COLS):
            rows = list(range(N_ROWS))
            if c % 2 == 1:
                rows = rows[::-1]
            for r in rows:
                order.append((c, r))

        goto_cart(arm, robot, start_x, start_y, hover_z, 0.1)  # lift off the corner first
        for i, (c, r) in enumerate(order, 1):
            x = start_x + VERT[0] * (r * V_SPACING) + RIGHT[0] * (c * H_SPACING)
            y = start_y + VERT[1] * (r * V_SPACING) + RIGHT[1] * (c * H_SPACING)
            print("Point %d/%d  (col %d, row %d)" % (i, len(order), c, r))
            if not goto_cart(arm, robot, x, y, hover_z, 0.1):
                continue
            if goto_cart(arm, robot, x, y, touch_z - INDENT, 0.05):  # dip 2 mm, slow
                rospy.sleep(DWELL)
                goto_cart(arm, robot, x, y, hover_z, 0.05)           # retract

        go_to(arm, home_joints, "home")
    finally:
        logger.close()
        print("Logged %d franka_states samples to %s" % (logger.count, out_csv))


if __name__ == "__main__":
    main()
