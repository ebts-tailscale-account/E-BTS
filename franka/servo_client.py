#!/usr/bin/env python3
"""
Python client for the CartesianPoseServoController (incremental position servo).

Bring up the servo controller FIRST (this replaces the MoveIt bringup -- it
claims the same Cartesian interface, so do not run both):

    source ~/ws_franka/devel/setup.bash
    roslaunch franka_example_controllers cartesian_pose_servo.launch \\
        robot_ip:=10.1.196.5 load_gripper:=false

Then stream target poses (base frame, panda_link0) to
    /cartesian_pose_servo_controller/target_pose
The controller integrates toward the target with hard velocity/acceleration
limits, so you can send incremental targets and it moves smoothly (no lunge).

This client just sets targets and waits until the measured EE pose (from
/franka_state_controller/franka_states, O_T_EE) reaches them. Orientation is
held at the current value unless you pass one.
"""

import argparse
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState

TARGET_TOPIC = "/cartesian_pose_servo_controller/target_pose"
BASE_FRAME = "panda_link0"


def _R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2
        w, x, y, z = 0.25 * S, (R[2, 1]-R[1, 2])/S, (R[0, 2]-R[2, 0])/S, (R[1, 0]-R[0, 1])/S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        w, x, y, z = (R[2, 1]-R[1, 2])/S, 0.25*S, (R[0, 1]+R[1, 0])/S, (R[0, 2]+R[2, 0])/S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        w, x, y, z = (R[0, 2]-R[2, 0])/S, (R[0, 1]+R[1, 0])/S, 0.25*S, (R[1, 2]+R[2, 1])/S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        w, x, y, z = (R[1, 0]-R[0, 1])/S, (R[0, 2]+R[2, 0])/S, (R[1, 2]+R[2, 1])/S, 0.25*S
    return [x, y, z, w]


class CartesianServo:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node("cartesian_servo_client", anonymous=True, disable_signals=True)
        self.pub = rospy.Publisher(TARGET_TOPIC, PoseStamped, queue_size=1)
        # wait for the servo controller's subscriber to connect
        t0 = time.time()
        while self.pub.get_num_connections() == 0 and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if self.pub.get_num_connections() == 0:
            rospy.logwarn("No subscriber on %s -- is cartesian_pose_servo_controller running?",
                          TARGET_TOPIC)

    def current_pose(self):
        """(pos[3], quat[xyzw]) of the EE in the base frame, from franka_states."""
        msg = rospy.wait_for_message("/franka_state_controller/franka_states",
                                     FrankaState, timeout=5.0)
        T = np.array(msg.O_T_EE, dtype=float).reshape(4, 4, order="F")  # column-major
        pos = T[:3, 3].tolist()
        quat = _R_to_quat(T[:3, :3])
        return pos, quat

    def send_target(self, x, y, z, quat):
        m = PoseStamped()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = rospy.Time.now()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = [float(v) for v in quat]
        self.pub.publish(m)

    def move_to(self, x, y, z, quat=None, tol=0.001, timeout=30.0, poll=0.05):
        """Set a target and block until the measured EE reaches it (or timeout).
        Returns the final xyz error (m)."""
        if quat is None:
            _, quat = self.current_pose()
        deadline = time.time() + timeout
        err = float("inf")
        while time.time() < deadline:
            self.send_target(x, y, z, quat)      # re-publish so the target stays latched
            pos, _ = self.current_pose()
            err = max(abs(pos[0]-x), abs(pos[1]-y), abs(pos[2]-z))
            if err <= tol:
                return err
            time.sleep(poll)
        rospy.logwarn("move_to timeout: residual %.4f m", err)
        return err

    def step(self, dx=0.0, dy=0.0, dz=0.0, **kw):
        pos, quat = self.current_pose()
        return self.move_to(pos[0]+dx, pos[1]+dy, pos[2]+dz, quat=quat, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nudge-z-mm", type=float, default=None,
                    help="SUPERVISED test: move Z by this many mm (e.g. -1 = down 1 mm).")
    args = ap.parse_args()

    servo = CartesianServo()
    pos, quat = servo.current_pose()
    print(f"current EE pos = [{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}] m")
    print(f"current EE quat= {['%.4f' % q for q in quat]}")

    if args.nudge_z_mm is not None:
        dz = args.nudge_z_mm / 1000.0
        print(f"nudging Z by {args.nudge_z_mm:+.1f} mm ...")
        err = servo.step(dz=dz, tol=0.0005, timeout=15.0)
        p2, _ = servo.current_pose()
        print(f"done. new Z = {p2[2]:.5f} m (residual {err*1000:.2f} mm)")


if __name__ == "__main__":
    main()
