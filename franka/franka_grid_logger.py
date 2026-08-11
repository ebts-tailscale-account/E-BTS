#!/usr/bin/env python3
"""Experiment sweep: servo the Franka to every surface-mapped point, indent 2 mm
below the MEASURED surface height, dwell, retract -- while logging franka_states
(93 cols) at the controller's publish rate. Invoked ON TACTILE by master.py.

Built on the surface mapper (map_surface.py):
  * reads surface_map.csv, so each indent lands INDENT_MM below that point's true
    z_touch (accounts for tilt/sag -- no single flat touch-Z),
  * reuses map_surface's reach-aware homing (home -> elastomer center),
    _gross_move settling, and reflex-avoidance,
  * does NOT force-probe and does NOT read the F/T -- it servos to the mapped
    heights. During the EXPERIMENT the camera (.raw) and Wittenstein F/T are
    recorded by the E_BTS_GUI on the WORKSTATION, so the HEX21 is on the
    WORKSTATION now (NOT tactile, unlike the surface map).

The franka_states logger runs on a subscriber thread from the START (so its
timeline covers the same window as the camera/F/T recording), tagging every row
with phase / point / commanded target / that point's mapped surface_z.

Prereq (on tactile), start from HOME with the REACH launch:
    source /opt/ros/noetic/setup.bash && source ~/ws_franka/devel/setup.bash
    roslaunch franka_example_controllers cartesian_pose_servo_reach.launch \
        robot_ip:=10.1.196.5 load_gripper:=false

Usage (via master.py):  python3 franka_grid_logger.py [out_csv] [--max-points N]
"""

import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path

import rospy
from franka_msgs.msg import FrankaState

from servo_client import CartesianServo
import map_surface as m          # reuse motion helpers (_gross_move, approach_start, preflight_reach)

# --- indentation experiment params ---
INDENT_MM = 2.0                   # dip this far BELOW the mapped surface at each point
HOVER_MM  = 5.0                   # travel/retract this far ABOVE the surface
DWELL_S   = 2.0                   # hold the 2 mm indent this long (measurement window)
# TARE: the Wittenstein Fz drifts over a long run, so a single baseline taken at the
# start of the recording goes stale. Instead we hold still, OUT OF CONTACT, at hover
# height for TARE_S immediately before every dip. That gives each indentation its own
# fresh no-load window, and postprocess.py subtracts THAT as the zero -- so the drift
# cancels per point. Tagged phase="tare" in the log.
TARE_S    = 1.0                   # settle + no-load window before each dip

# DEPTH CORRECTION -- why it is needed
# franka_control runs internal_controller=joint_impedance, i.e. a COMPLIANT
# controller: it holds position by generating torques proportional to error, so a
# steady contact force necessarily leaves a steady position error. The servo being
# "force-blind" means it never gives up or aborts on contact -- NOT that it is
# infinitely stiff. Measured on the 99-point run: achieved depth fell short of the
# command by 0.39 mm on average (0.07-0.71 mm), correlating with contact force at
# r=+0.79, i.e. shortfall ~= 0.10 mm (free-space tracking) + F / 3.8 N/mm.
# Since we MEASURE ee_z, we can close the loop: push deeper by whatever is missing
# and re-check. That converges in a few iterations and yields a true depth.
DEPTH_TOL_MM   = 0.05             # accept when within this of the target depth
DEPTH_ITERS    = 6                # max correction iterations per point
DEPTH_EXTRA_MAX_MM = 1.5          # SAFETY: never command more than this extra depth
DEPTH_SETTLE_S = 0.5              # let the impedance settle before re-measuring

MAP_CSV = m.OUT_CSV               # ~/E-BTS/surface_map.csv (same dir as map_surface.py)


class Ctx:
    """Thread-safe hand-off of the current motion context to the logger thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "init"
        self.idx = self.col = self.row = -1
        self.tx = self.ty = self.tz = float("nan")
        self.surface_z = float("nan")

    def set(self, phase, idx=None, col=None, row=None, target=None, surface_z=None):
        with self.lock:
            self.phase = phase
            if idx is not None:
                self.idx = idx
            if col is not None:
                self.col = col
            if row is not None:
                self.row = row
            if target is not None:
                self.tx, self.ty, self.tz = target
            if surface_z is not None:
                self.surface_z = surface_z

    def snapshot(self):
        with self.lock:
            return (self.phase, self.idx, self.col, self.row,
                    self.tx, self.ty, self.tz, self.surface_z)


class FrankaLogger:
    """Streams a comprehensive franka_states row per sample, UNIX-timestamped.

    log_hz throttles how many samples we KEEP (0 = every message). This matters:
    deserializing FrankaState (30+ arrays) in Python at the full 1 kHz publish rate
    is a real CPU load on the control PC, and starving the 1 kHz FCI loop causes
    'communication_constraints_violation' reflexes. It costs no information either --
    franka_control already low-pass filters state at cutoff_frequency=100 Hz, so
    ~200 Hz sampling is above Nyquist for the data actually in the signal.
    """

    def __init__(self, path, ctx, log_hz=200.0):
        self.ctx = ctx
        self.file = open(path, "w")
        self.file.write(",".join(self._header()) + "\n")
        self.lock = threading.Lock()
        self.count = 0
        self.seen = 0
        self.min_dt = (1.0 / log_hz) if log_hz and log_hz > 0 else 0.0
        self.last_kept = 0.0
        self.sub = rospy.Subscriber("/franka_state_controller/franka_states", FrankaState,
                                    self._on_state, queue_size=100)

    @staticmethod
    def _header():
        cols = ["unix_time_s", "robot_time_s", "franka_time_s", "seq",
                "phase", "point_index", "col", "row",
                "target_x", "target_y", "target_z", "surface_z",
                "ee_x", "ee_y", "ee_z"]
        cols += ["q%d" % i for i in range(7)]
        cols += ["q_d%d" % i for i in range(7)]
        cols += ["dq%d" % i for i in range(7)]
        cols += ["tau_J%d" % i for i in range(7)]
        cols += ["tau_ext%d" % i for i in range(7)]
        # external wrench, base frame (kept names for postprocess compatibility)
        cols += ["Fx_ext_N", "Fy_ext_N", "Fz_ext_N", "Tx_ext_Nm", "Ty_ext_Nm", "Tz_ext_Nm"]
        # external wrench, stiffness/EE (K) frame
        cols += ["KFx_N", "KFy_N", "KFz_N", "KTx_Nm", "KTy_Nm", "KTz_Nm"]
        cols += ["O_T_EE_%d" % i for i in range(16)]   # full pose, column-major
        cols += ["elbow0", "elbow1", "m_total", "control_success_rate"]
        cols += ["collided", "cart_contact", "joint_contact", "robot_mode"]
        cols += ["err_joint_reflex", "err_cartesian_reflex", "err_cart_pos_lim",
                 "err_joint_pos_lim", "err_self_collision", "err_communication",
                 "err_instability"]
        return cols

    def _on_state(self, msg):
        unix_s = time.time()
        self.seen += 1
        # throttle: bail out BEFORE any formatting work (the expensive part)
        if self.min_dt and (unix_s - self.last_kept) < self.min_dt:
            return
        self.last_kept = unix_s
        phase, idx, col, row, tx, ty, tz, surface_z = self.ctx.snapshot()
        # O_T_EE is column-major 4x4; the EE position is the last column (12,13,14).
        ee = (msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14])
        collided = int(any(msg.cartesian_collision) or any(msg.joint_collision))
        cart_contact = int(any(msg.cartesian_contact))
        joint_contact = int(any(msg.joint_contact))
        e = msg.current_errors

        vals = ["%.6f" % unix_s, "%.6f" % msg.header.stamp.to_sec(),
                "%.6f" % msg.time, str(msg.header.seq),
                phase, str(idx), str(col), str(row),
                "%.6f" % tx, "%.6f" % ty, "%.6f" % tz, "%.6f" % surface_z,
                "%.6f" % ee[0], "%.6f" % ee[1], "%.6f" % ee[2]]
        vals += ["%.6f" % v for v in msg.q]
        vals += ["%.6f" % v for v in msg.q_d]
        vals += ["%.6f" % v for v in msg.dq]
        vals += ["%.4f" % v for v in msg.tau_J]
        vals += ["%.4f" % v for v in msg.tau_ext_hat_filtered]
        vals += ["%.4f" % v for v in msg.O_F_ext_hat_K]
        vals += ["%.4f" % v for v in msg.K_F_ext_hat_K]
        vals += ["%.6f" % v for v in msg.O_T_EE]
        vals += ["%.4f" % msg.elbow[0], "%.4f" % msg.elbow[1],
                 "%.4f" % msg.m_total, "%.4f" % msg.control_command_success_rate]
        vals += [str(collided), str(cart_contact), str(joint_contact), str(msg.robot_mode)]
        vals += [str(int(e.joint_reflex)), str(int(e.cartesian_reflex)),
                 str(int(e.cartesian_position_limits_violation)),
                 str(int(e.joint_position_limits_violation)),
                 str(int(e.self_collision_avoidance_violation)),
                 str(int(e.communication_constraints_violation)),
                 str(int(e.instability_detected))]

        with self.lock:
            self.file.write(",".join(vals) + "\n")
            self.count += 1

    def close(self):
        self.sub.unregister()
        with self.lock:
            self.file.flush()
            self.file.close()


def read_surface_map(path):
    """Ordered list of the successfully-mapped points: {point_id,row,col,x,y,z_touch}."""
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or r["z_touch"] in ("", None):
                continue
            pts.append({"point_id": int(r["point_id"]),
                        "row": int(float(r["row"])), "col": int(float(r["col"])),
                        "x": float(r["x"]), "y": float(r["y"]),
                        "z_touch": float(r["z_touch"])})
    return pts


def clear_reflex(timeout=3.0):
    """Ask franka_control to clear a latched reflex/error so the arm will move.

    Safe: error_recovery only resets the error state, it does not move the robot.
    Saves having to relaunch the controller after a reflex (e.g. a
    communication_constraints_violation from a momentarily overloaded control PC).
    """
    try:
        from franka_msgs.msg import ErrorRecoveryActionGoal
        pub = rospy.Publisher("/franka_control/error_recovery/goal",
                              ErrorRecoveryActionGoal, queue_size=1)
        t0 = time.time()
        while pub.get_num_connections() == 0 and time.time() - t0 < timeout:
            time.sleep(0.1)
        if pub.get_num_connections() == 0:
            print("[WARN] no error_recovery subscriber; skipping auto-recovery.")
            return
        pub.publish(ErrorRecoveryActionGoal())
        time.sleep(1.0)
        print("Sent error_recovery (clears any latched reflex).")
    except Exception as e:
        print("[WARN] error_recovery failed: %s" % e)


def warn_if_loaded(threshold=4.0):
    """A busy control PC starves the 1 kHz FCI loop -> communication_constraints_violation.

    The usual culprits are a browser / IDE / desktop session left running on the
    control PC. Warn loudly rather than fail, so the operator can quiesce it.
    """
    try:
        import os
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        print("Control PC load average: %.2f (%d cores)" % (load1, ncpu))
        if load1 > threshold:
            print("*" * 72)
            print("WARNING: load average %.2f is high. A loaded control PC drops FCI" % load1)
            print("packets and triggers 'communication_constraints_violation' reflexes.")
            print("Close Chrome / VS Code / heavy desktop apps on this machine first.")
            print("*" * 72)
    except Exception:
        pass


def hold(servo, x, y, z, quat, seconds):
    """Keep the target latched (arm pressed at the indent) for `seconds`."""
    end = time.time() + seconds
    while time.time() < end:
        servo.send_target(x, y, z, quat)
        time.sleep(0.1)


def dip_to_depth(servo, x, y, surface_z, depth_m, quat, correct=True):
    """Indent to a TRUE depth below surface_z, compensating impedance sag.

    Commands the nominal depth, then measures the real EE height and pushes deeper
    by whatever is still missing, repeating until within DEPTH_TOL_MM. Returns
    (achieved_depth_m, commanded_z, iterations).
    """
    target_z = surface_z - depth_m
    m._gross_move(servo, x, y, target_z, quat, name="dip %.1fmm" % (depth_m * 1e3))
    if not correct:
        pos, _ = servo.current_pose()
        return surface_z - pos[2], target_z, 0

    cmd_z = target_z
    tol = DEPTH_TOL_MM / 1000.0
    extra_cap = DEPTH_EXTRA_MAX_MM / 1000.0
    for it in range(1, DEPTH_ITERS + 1):
        time.sleep(DEPTH_SETTLE_S)
        pos, _ = servo.current_pose()
        achieved = surface_z - pos[2]
        missing = depth_m - achieved            # >0 => still too shallow
        if missing <= tol:
            return achieved, cmd_z, it
        # push deeper by the shortfall, clamped by the safety cap
        want_z = cmd_z - missing
        floor_z = target_z - extra_cap
        if want_z < floor_z:
            want_z = floor_z
            print("        depth correction hit the %.1f mm safety cap"
                  % DEPTH_EXTRA_MAX_MM, flush=True)
        if abs(want_z - cmd_z) < 1e-6:
            break
        cmd_z = want_z
        servo.send_target(x, y, cmd_z, quat)
    time.sleep(DEPTH_SETTLE_S)
    pos, _ = servo.current_pose()
    return surface_z - pos[2], cmd_z, DEPTH_ITERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", nargs="?",
                    default=str(Path.home() / "E-BTS/recordings/ebts_franka_run.csv"),
                    help="where to write the franka_states CSV (default: "
                         "~/E-BTS/recordings/ -- NOT /tmp, so runs stay together).")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Indent only the first N mapped points (cautious sync test).")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--log-hz", type=float, default=200.0,
                    help="franka_states sampling rate to KEEP (default 200; 0 = full "
                         "1 kHz). Above ~200 Hz adds CPU load but no information -- "
                         "franka_control already filters state at 100 Hz.")
    ap.add_argument("--no-recovery", action="store_true",
                    help="Skip the automatic error-recovery request at startup.")
    ap.add_argument("--no-depth-correction", action="store_true",
                    help="Command the nominal depth only, without closing the loop on "
                         "the measured EE height (leaves the ~0.4 mm impedance sag).")
    args = ap.parse_args()

    if not MAP_CSV.exists():
        sys.exit("[ERROR] %s not found -- run map_surface.py first." % MAP_CSV)
    grid = read_surface_map(MAP_CSV)
    if args.max_points is not None:
        grid = grid[:args.max_points]
    if not grid:
        sys.exit("[ERROR] no 'ok' points in the surface map.")

    hover = HOVER_MM / 1000.0
    indent = INDENT_MM / 1000.0

    # share ONE node between the logger subscriber and the servo client
    rospy.init_node("franka_grid_logger", disable_signals=True)

    if not args.no_recovery:
        clear_reflex()
    warn_if_loaded()

    servo = CartesianServo()
    points = m.load_points()

    # reach preflight: feed a grid whose z_plane = z_touch so the clamp check
    # covers z_touch + hover (highest) and z_touch - INDENT (deepest) targets.
    pre_grid = [{"x": g["x"], "y": g["y"], "z_plane": g["z_touch"]} for g in grid]
    # temporarily reflect our indent depth in the map's floor-depth constant
    saved_depth = m.MAX_DEPTH_MM
    m.MAX_DEPTH_MM = INDENT_MM
    try:
        m.preflight_reach(servo, points, pre_grid, hover)
    finally:
        m.MAX_DEPTH_MM = saved_depth

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    logger = FrankaLogger(args.out_csv, ctx, log_hz=args.log_hz)  # log BEFORE homing
    print("Logging franka_states to %s | %d points | %s" %
          (args.out_csv, len(grid),
           "full rate" if not args.log_hz else "%.0f Hz" % args.log_hz))

    depths = []
    try:
        ctx.set("home")
        flat_quat, (cx, cy) = m.approach_start(servo, points, level=not args.no_level)

        print("Indenting %d points at %.1f mm below the mapped surface%s ..."
              % (len(grid), INDENT_MM,
                 "" if args.no_depth_correction else " (closed-loop depth)"))
        for i, g in enumerate(grid, 1):
            x, y, zt = g["x"], g["y"], g["z_touch"]
            hover_z, dip_z = zt + hover, zt - indent
            print("Point %d/%d  pid %d (r%d c%d)  surface_z=%.5f  dip_z=%.5f" %
                  (i, len(grid), g["point_id"], g["row"], g["col"], zt, dip_z))

            ctx.set("travel", idx=g["point_id"], col=g["col"], row=g["row"],
                    target=(x, y, hover_z), surface_z=zt)
            m._gross_move(servo, x, y, hover_z, flat_quat, name="travel")

            # TARE: hold still at hover, OUT OF CONTACT, so this point gets its own
            # fresh F/T zero right before the dip (kills Wittenstein Fz drift).
            ctx.set("tare", target=(x, y, hover_z))
            hold(servo, x, y, hover_z, flat_quat, TARE_S)

            ctx.set("dip", target=(x, y, dip_z))
            achieved, cmd_z, iters = dip_to_depth(servo, x, y, zt, indent, flat_quat,
                                                  correct=not args.no_depth_correction)
            print("        depth: %.2f mm achieved (target %.2f, commanded z=%.5f, %d iters)"
                  % (achieved * 1e3, INDENT_MM, cmd_z, iters), flush=True)
            depths.append(achieved * 1e3)

            # dwell at the CORRECTED command so the true depth is held all through
            ctx.set("dwell", target=(x, y, cmd_z))
            hold(servo, x, y, cmd_z, flat_quat, DWELL_S)

            ctx.set("retract", target=(x, y, hover_z))
            m._gross_move(servo, x, y, hover_z, flat_quat, name="retract")

        # park back over the taught center
        ctx.set("park")
        m._gross_move(servo, cx, cy, points["center"]["xyz"][2] + m.CENTER_APPROACH_MM / 1000.0,
                      flat_quat, name="park above center")
        ctx.set("done")
    finally:
        logger.close()
        print("Logged %d franka_states samples to %s" % (logger.count, args.out_csv))
        if depths:
            import statistics as _st
            print("Achieved indentation depth over %d points: mean %.2f mm "
                  "(min %.2f, max %.2f) vs %.2f mm target"
                  % (len(depths), _st.mean(depths), min(depths), max(depths), INDENT_MM))


if __name__ == "__main__":
    main()
