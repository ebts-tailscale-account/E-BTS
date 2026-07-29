#!/usr/bin/env python3
"""Record the Franka's estimated external force/torque for a few seconds.

Reads O_F_ext_hat_K from franka_states (state only -- does NOT move the robot),
prints it live, and saves a CSV. Run on tactile (sourced) after setting the
payload load to check the offset is gone.

Usage:  python3 record_franka_force.py [out.csv]  [seconds]
"""

import sys
import time

import rospy
from franka_msgs.msg import FrankaState

out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/franka_force_check.csv"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

rows = []
last_print = [0.0]


def cb(msg):
    t = time.time()
    w = msg.O_F_ext_hat_K  # [Fx Fy Fz Tx Ty Tz], N and Nm
    rows.append((t,) + tuple(w))
    if t - last_print[0] > 0.25:  # print ~4x/s so you can watch it live
        last_print[0] = t
        print("Fx %+6.2f  Fy %+6.2f  Fz %+6.2f N    Tx %+6.3f  Ty %+6.3f  Tz %+6.3f Nm"
              % (w[0], w[1], w[2], w[3], w[4], w[5]))


rospy.init_node("record_franka_force")
sub = rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, cb)
print("Recording external wrench for %.0f s (keep the arm still)..." % duration)
t0 = time.time()
while not rospy.is_shutdown() and time.time() - t0 < duration:
    time.sleep(0.05)
sub.unregister()

with open(out_path, "w") as f:
    f.write("unix_time_s,Fx_N,Fy_N,Fz_N,Tx_Nm,Ty_Nm,Tz_Nm\n")
    for r in rows:
        f.write("%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % r)

print("")
if rows:
    for i, name in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], start=1):
        vals = [r[i] for r in rows]
        print("mean %s = %+.3f    (min %+.3f, max %+.3f)" % (name, sum(vals) / len(vals), min(vals), max(vals)))
else:
    print("No samples -- is the MoveIt/franka_control bringup running?")
print("Saved %d samples to %s" % (len(rows), out_path))
