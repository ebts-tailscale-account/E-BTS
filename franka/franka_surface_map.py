#!/usr/bin/env python3
"""
Surface offset mapping for the Franka Emika Panda + Wittenstein Resense HEX21.

This is the Franka/Wittenstein port of the surface-mapping logic from the UR5 +
ATI experiment (detect_first_touch / detect_and_save_z_offsets in experiment.py).

WHAT IT DOES
------------
For every (x, y) probe point it lowers the end effector straight down in small
Z steps, watching the Wittenstein normal force (Fz). When the force first
diverges from the no-contact baseline it records the Z at which that happened.
The result is a surface map:

        {(x, y): z_touch}

i.e. the height of the (uneven) silicone surface under each probe point, in the
robot base frame (panda_link0). This is exactly the z_offset.csv table the old
script produced, and it is what the indentation phase later uses to reference
depths from the true surface instead of a flat plane.

STRUCTURE
---------
The math is hardware-agnostic and lives in ONE function: `map_surface_offsets`.
It takes two callables you inject:

    move_to(x, y, z)          -> block until the EE is at that base-frame pose
    read_window(seconds)      -> (mean_fz, std_fz, n_samples) over that window

Two batteries-included adapters build those callables for your rig:

    WittensteinFT             -> serial reader for the HEX21 (gives read_window)
    make_franka_mover(...)    -> MoveIt-based mover (gives move_to)

so you can run it end-to-end, or swap either side (e.g. read Fz off a ROS topic
instead of local serial) without touching the mapping logic.

NOTE ON YOUR IMPEDANCE / FORCE-SENSITIVITY CONCERN
--------------------------------------------------
For *mapping* we only need first light contact, so the loop stops the instant
Fz rises above baseline. Two Franka-specific things matter here:
  * The Panda's contact reflex can protective-stop on a light touch. Before
    mapping, relax the Cartesian force collision thresholds (see the note in
    make_franka_mover) so a gentle first-touch doesn't trip a reflex mid-map.
  * Keep descent_step tiny (default 0.1 mm) so you stop right at the surface.
The "normal force must not change the indentation rate" requirement is a
separate concern for the indentation phase (position/velocity-controlled dips),
not for this mapping pass.
"""

from __future__ import annotations

import csv
import struct
import threading
import time
from collections import deque
from pathlib import Path


# ============================================================================
#  CORE  --  the one function you asked for. Robot/sensor agnostic.
# ============================================================================

def map_surface_offsets(
    xy_points,
    start_xyz,
    move_to,
    read_window,
    *,
    descent_step=1e-4,        # 0.1 mm per step (matches the UR5 TOUCH_DESCENT_STEP)
    min_z=None,               # lowest Z to try; default = start_z - 20 mm
    safe_z=None,              # Z to travel/retract at; default = start_z
    baseline_seconds=1.0,     # no-contact baseline window
    sample_seconds=0.35,      # force window measured at each step
    settle_seconds=0.25,      # extra dwell after each move before measuring
    force_sigmas=2.0,         # threshold = baseline + max(sigmas*std, min_delta)
    force_min_delta=0.05,     # N; floor above baseline noise (TUNE for HEX21)
    csv_path=None,            # if set, write the z_offset table here
    verbose=True,
):
    """Probe every (x, y) and return the surface map {(x, y): z_touch}.

    Parameters
    ----------
    xy_points : iterable of (x, y) in metres, base frame.
    start_xyz : (x, y, z) safe pose above the surface, base frame.
    move_to(x, y, z) : blocking move callable.
    read_window(seconds) -> (mean_fz, std_fz, n) : force sampler callable.

    Returns
    -------
    (touch_map, results)
        touch_map : {(round(x,6), round(y,6)): z_touch} for detected points.
        results   : list of per-point dicts (also what gets written to CSV).
    """
    start_z = float(start_xyz[2])
    safe_z = start_z if safe_z is None else float(safe_z)
    min_z = (start_z - 0.020) if min_z is None else float(min_z)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    results = []
    touch_map = {}
    xy_points = list(xy_points)
    total = len(xy_points)
    log(f"[MAP] Probing {total} points | step={descent_step*1000:.3f} mm "
        f"| safe_z={safe_z:.5f} | min_z={min_z:.5f}")

    for i, (x, y) in enumerate(xy_points, start=1):
        x = float(x)
        y = float(y)
        log(f"\n[MAP] Point {i}/{total}: XY=({x:.6f}, {y:.6f})")

        # 1) go to the safe height directly above the probe point
        move_to(x, y, safe_z)

        # 2) no-contact baseline
        b_mean, b_std, b_n = read_window(baseline_seconds)
        threshold = max(force_sigmas * b_std, force_min_delta)
        log(f"[MAP]   baseline Fz={b_mean:+.4f} N std={b_std:.4f} N "
            f"n={b_n} -> trip when |dFz| >= {threshold:.4f} N")

        # 3) descend step by step until |Fz - baseline| exceeds threshold
        touch = _descend_one_point(
            x, y, safe_z, min_z, descent_step,
            move_to, read_window,
            b_mean, threshold, sample_seconds, settle_seconds, log,
        )

        # 4) retract before moving on
        move_to(x, y, safe_z)

        row = {
            "point_id": i,
            "x": x,
            "y": y,
            "z_touch": touch["z_touch"],
            "z_offset": (start_z - touch["z_touch"]) if touch["detected"] else 0.0,
            "contact_fz": touch["contact_fz"],
            "baseline_fz": b_mean,
            "baseline_std": b_std,
            "threshold": threshold,
            "status": "success" if touch["detected"] else "not_detected",
        }
        results.append(row)
        if touch["detected"]:
            touch_map[(round(x, 6), round(y, 6))] = touch["z_touch"]

    if csv_path is not None:
        _write_csv(csv_path, results)
        log(f"\n[MAP] Wrote {len(results)} rows -> {csv_path}")

    ok = sum(1 for r in results if r["status"] == "success")
    log(f"[MAP] Detected surface for {ok}/{len(results)} points.")
    return touch_map, results


def _descend_one_point(x, y, safe_z, min_z, step, move_to, read_window,
                       baseline_mean, threshold, sample_seconds,
                       settle_seconds, log):
    """Lower straight down from safe_z until Fz diverges from baseline."""
    z = safe_z
    while z > min_z:
        z = max(z - step, min_z)
        move_to(x, y, z)
        if settle_seconds:
            time.sleep(settle_seconds)

        mean_fz, _std, n = read_window(sample_seconds)
        d = abs(mean_fz - baseline_mean)
        log(f"[MAP]   z={z:.6f}  Fz={mean_fz:+.4f} N  |dFz|={d:.4f} N")

        if n and d >= threshold:
            log(f"[MAP]   >> first touch at z={z:.6f}  (dFz={d:.4f} N)")
            return {"detected": True, "z_touch": z, "contact_fz": mean_fz}

    log(f"[MAP]   !! no contact before min_z={min_z:.6f}")
    return {"detected": False, "z_touch": safe_z, "contact_fz": 0.0}


def _write_csv(csv_path, results):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["point_id", "x", "y", "z_touch", "z_offset", "contact_fz",
              "baseline_fz", "baseline_std", "threshold", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)


# ============================================================================
#  ADAPTER 1  --  Wittenstein Resense HEX21 serial reader (gives read_window)
# ============================================================================

class WittensteinFT:
    """Background serial reader for the HEX21 6-axis F/T sensor.

    Packet: 28 bytes = 7 little-endian float32 -> Fx Fy Fz (N),
    Mx My Mz (mNm), Temp (C). No framing bytes, so we resync by sliding
    until Temp is plausible. Requires B2000000 baud + DTR/RTS asserted, and
    DIP switch 6 = ON on the electronics box (values already calibrated in N;
    do NOT apply the calibration matrix again in software).

    Only the port owner may read it -- close the vendor Resense GUI / read_ft.py
    first.
    """

    PACKET = 28
    FMT = "<7f"
    TEMP_MIN, TEMP_MAX = 0.0, 80.0
    USB_VID = 0x0483   # STMicroelectronics Virtual COM Port
    USB_PID = 0x5740

    def __init__(self, port="auto", baud=2000000, buffer_seconds=5.0):
        # port="auto" (or None) resolves the HEX21 by USB VID:PID at start(),
        # so it survives /dev/ttyACMn renumbering across replugs.
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # rolling (t, fx, fy, fz) buffer
        self._buf = deque(maxlen=int(buffer_seconds * 4000))

    @classmethod
    def find_port(cls):
        """Return the /dev path of the HEX21 by USB VID:PID, or None."""
        from serial.tools import list_ports
        for p in list_ports.comports():
            if p.vid == cls.USB_VID and p.pid == cls.USB_PID:
                return p.device
        return None

    def start(self):
        import serial  # pyserial
        port = self.port
        if port in (None, "auto"):
            port = self.find_port()
            if port is None:
                raise RuntimeError(
                    "Wittenstein HEX21 not found on USB "
                    f"({self.USB_VID:#06x}:{self.USB_PID:#06x}). "
                    "Check the cable / that no other program owns the port.")
            print(f"[HEX21] auto-detected port {port}", flush=True)
        self.port = port
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self._ser.dtr = True
        self._ser.rts = True
        time.sleep(0.2)
        self._ser.reset_input_buffer()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # let the buffer fill so the first read_window has data
        time.sleep(0.3)
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _loop(self):
        raw = bytearray()
        while not self._stop.is_set():
            chunk = self._ser.read(self.PACKET * 8)
            if not chunk:
                continue
            raw.extend(chunk)
            # resync + parse every complete, plausible packet
            while len(raw) >= self.PACKET:
                fields = struct.unpack(self.FMT, bytes(raw[:self.PACKET]))
                temp = fields[6]
                if self.TEMP_MIN <= temp <= self.TEMP_MAX:
                    fx, fy, fz = fields[0], fields[1], fields[2]
                    with self._lock:
                        self._buf.append((time.time(), fx, fy, fz))
                    del raw[:self.PACKET]
                else:
                    # not aligned -> drop one byte and try again
                    del raw[0]
            # keep the leftover bytes bounded
            if len(raw) > self.PACKET * 64:
                del raw[:-self.PACKET]

    def read_window(self, seconds):
        """(mean_fz, std_fz, n) over the most recent `seconds` of samples."""
        time.sleep(seconds)
        now = time.time()
        with self._lock:
            fz = [s[3] for s in self._buf if now - s[0] <= seconds]
        n = len(fz)
        if n == 0:
            return 0.0, 0.0, 0
        mean = sum(fz) / n
        var = sum((v - mean) ** 2 for v in fz) / n
        return float(mean), float(var ** 0.5), n


# ============================================================================
#  ADAPTER 2  --  Franka Panda mover via MoveIt (gives move_to)
# ============================================================================

class FrankaArm:
    """MoveIt-backed Panda motion: joint moves + straight-down Cartesian moves.

    Requires the CLEAN stack up on the Franka PC (see franka-moveit notes):
        source ~/ws_franka/devel/setup.bash   # AFTER bashrc's catkin_ws source
        roslaunch panda_moveit_config franka_control.launch \\
            robot_ip:=10.1.196.5 load_gripper:=false
    and a ROS node initialised in the calling process.

    Typical use: move_to_joints(centre) -> lock_orientation() -> ee_xyz(), then
    move_to(x, y, z) for every probe. Orientation is locked ONCE (after you are
    at the centre pose), so every probe descends straight down.

    IMPORTANT (impedance / reflex): before probing, loosen the contact
    thresholds so a light first-touch doesn't trigger a protective stop, e.g.
        rosservice call /franka_control/set_force_torque_collision_behavior ...
    with generous Cartesian force limits. Otherwise the arm may reflex-stop the
    instant the Wittenstein would have registered contact.
    """

    def __init__(self, group_name="panda_arm", vel_scale=0.1, acc_scale=0.1,
                 planning_time=5.0):
        import rospy
        import moveit_commander

        if not rospy.core.is_initialized():
            rospy.init_node("franka_surface_map", anonymous=True,
                            disable_signals=True)

        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_max_velocity_scaling_factor(vel_scale)
        self.group.set_max_acceleration_scaling_factor(acc_scale)
        self.group.set_planning_time(planning_time)
        self._ori = None
        self._ik = None

    # --- safe straight-line Cartesian motion -------------------------------
    # compute_cartesian_path interpolates the EE in a straight line and solves
    # IK incrementally from the current state, so it stays in ONE arm branch.
    # The newer MoveIt python binding dropped the jump_threshold argument, so we
    # inspect the returned trajectory ourselves and REFUSE it if any consecutive
    # 1 mm waypoint shows a joint jump (a branch flip / lunge) -- same safety.
    EEF_STEP = 0.001            # m, Cartesian interpolation resolution
    MAX_WAYPOINT_JUMP_RAD = 0.5 # reject if any joint jumps more than this per 1 mm

    def _reject_joint_jumps(self, plan):
        pts = plan.joint_trajectory.points
        for i in range(1, len(pts)):
            d = max(abs(a - b) for a, b in zip(pts[i].positions, pts[i - 1].positions))
            if d > self.MAX_WAYPOINT_JUMP_RAD:
                raise RuntimeError(
                    f"cartesian path has a {d:.2f} rad joint jump -- "
                    f"refusing (would flip/lunge)")

    def move_cartesian(self, x, y, z, quat=None, vel=0.05):
        """Move the EE in a STRAIGHT LINE to (x, y, z). Raises (without moving,
        or after a safe partial) if the path isn't fully feasible or would flip
        the arm -- callers should catch and retract."""
        from geometry_msgs.msg import Pose
        cur = self.group.get_current_pose().pose
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        if quat is None:
            target.orientation = cur.orientation
        else:
            (target.orientation.x, target.orientation.y,
             target.orientation.z, target.orientation.w) = [float(v) for v in quat]

        # binding: compute_cartesian_path(waypoints, eef_step, avoid_collisions)
        plan, frac = self.group.compute_cartesian_path(
            [target], self.EEF_STEP, True)
        if frac < 0.99:
            raise RuntimeError(
                f"cartesian path only {int(frac*100)}% feasible to "
                f"({x:.4f},{y:.4f},{z:.4f}) -- refusing (flip/limit/singularity)")
        self._reject_joint_jumps(plan)
        plan = self.group.retime_trajectory(
            self.robot.get_current_state(), plan,
            velocity_scaling_factor=vel, acceleration_scaling_factor=vel)
        ok = self.group.execute(plan, wait=True)
        self.group.stop()
        if not ok:
            reached, err = self._xyz_reached(x, y, z)
            if not reached:
                raise RuntimeError(
                    f"cartesian execute failed to ({x:.4f},{y:.4f},{z:.4f}) "
                    f"(xyz error {err*1000:.1f} mm)")

    def compute_ik(self, x, y, z, quat, seed):
        """IK for a base-frame pose, seeded from `seed` joints (list of 7).

        Returns the 7 arm joint values or None if IK fails. Seeding from a
        nearby configuration keeps the solution in the same arm branch (no
        elbow flips), which is what we want for a smooth raster scan.
        """
        import rospy
        from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        if self._ik is None:
            rospy.wait_for_service("/compute_ik", timeout=5.0)
            self._ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group.get_name()
        req.ik_request.ik_link_name = self.group.get_end_effector_link()
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.1)

        ps = PoseStamped()
        ps.header.frame_id = self.group.get_planning_frame()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = [float(v) for v in quat]
        req.ik_request.pose_stamped = ps

        joints = self.group.get_active_joints()
        js = JointState()
        js.name = joints
        js.position = list(seed)
        req.ik_request.robot_state.joint_state = js

        resp = self._ik(req)
        if resp.error_code.val != 1:      # 1 == SUCCESS
            return None
        name2pos = dict(zip(resp.solution.joint_state.name,
                            resp.solution.joint_state.position))
        return [name2pos[n] for n in joints]

    # Controller may report GOAL_TOLERANCE_VIOLATED (joint still settling at
    # goal_time) even though the arm creeps onto the goal a beat later -- common
    # under an unmodeled payload (m_load = 0). So on an aborted go() we wait and
    # check the ACTUAL joint values before deciding it truly failed.
    JOINT_ARRIVAL_TOL = 0.05   # rad, per joint
    ARRIVAL_SETTLE_S = 2.5     # how long to wait for the slow settle

    def _joints_reached(self, target):
        import time
        deadline = time.time() + self.ARRIVAL_SETTLE_S
        while True:
            actual = self.group.get_current_joint_values()
            err = max(abs(a - b) for a, b in zip(actual, target))
            if err <= self.JOINT_ARRIVAL_TOL:
                return True, err
            if time.time() >= deadline:
                return False, err
            time.sleep(0.1)

    def move_to_joints(self, joints):
        """Go to a 7-DOF joint configuration (blocking).

        Tolerant of the benign GOAL_TOLERANCE_VIOLATED abort: if MoveIt reports
        failure but the arm actually settled onto the goal, accept it.
        """
        target = list(joints)
        ok = self.group.go(target, wait=True)
        if not ok:
            reached, err = self._joints_reached(target)
            if not reached:
                self.group.stop()
                raise RuntimeError(
                    f"MoveIt failed to reach joints {joints} "
                    f"(max joint error {err:.3f} rad after settle)")
            print(f"[FrankaArm] controller aborted but arm settled on goal "
                  f"(max err {err:.3f} rad) -- accepting", flush=True)
        self.group.stop()

    XYZ_ARRIVAL_TOL = 0.005    # m; accept an aborted Cartesian move if this close

    def _xyz_reached(self, x, y, z):
        import time
        target = (float(x), float(y), float(z))
        deadline = time.time() + self.ARRIVAL_SETTLE_S
        while True:
            ax, ay, az = self.ee_xyz()
            err = max(abs(ax - target[0]), abs(ay - target[1]), abs(az - target[2]))
            if err <= self.XYZ_ARRIVAL_TOL:
                return True, err
            if time.time() >= deadline:
                return False, err
            time.sleep(0.1)

    def ee_xyz(self):
        p = self.group.get_current_pose().pose.position
        return p.x, p.y, p.z

    def lock_orientation(self):
        """Freeze the current EE orientation for all subsequent move_to()."""
        self._ori = self.group.get_current_pose().pose.orientation
        return self._ori

    def move_to(self, x, y, z):
        """Move to (x, y, z) in the base frame, keeping the locked orientation."""
        from geometry_msgs.msg import Pose
        if self._ori is None:
            self.lock_orientation()
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation = self._ori
        self.group.set_pose_target(target)
        ok = self.group.go(wait=True)
        if not ok:
            reached, err = self._xyz_reached(x, y, z)
            if not reached:
                self.group.stop()
                self.group.clear_pose_targets()
                raise RuntimeError(
                    f"MoveIt failed to reach ({x:.4f},{y:.4f},{z:.4f}) "
                    f"(xyz error {err*1000:.1f} mm after settle)")
        self.group.stop()
        self.group.clear_pose_targets()

    def current_quat(self):
        """Current EE orientation quaternion as [x, y, z, w]."""
        o = self.group.get_current_pose().pose.orientation
        return [o.x, o.y, o.z, o.w]

    def move_to_pose(self, x, y, z, quat):
        """Move to (x, y, z) with an explicit orientation quaternion [x,y,z,w]."""
        from geometry_msgs.msg import Pose
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        (target.orientation.x, target.orientation.y,
         target.orientation.z, target.orientation.w) = [float(v) for v in quat]
        self.group.set_pose_target(target)
        ok = self.group.go(wait=True)
        if not ok:
            reached, err = self._xyz_reached(x, y, z)
            if not reached:
                self.group.stop()
                self.group.clear_pose_targets()
                raise RuntimeError(
                    f"MoveIt failed to reach pose ({x:.4f},{y:.4f},{z:.4f}) "
                    f"(xyz error {err*1000:.1f} mm after settle)")
        self.group.stop()
        self.group.clear_pose_targets()


def make_franka_mover(group_name="panda_arm", vel_scale=0.1, acc_scale=0.1,
                      planning_time=5.0):
    """Backward-compatible helper: return a blocking move_to(x, y, z) that
    locks orientation to wherever the arm is now. For repeatable centring,
    prefer FrankaArm (move_to_joints -> lock_orientation -> move_to)."""
    arm = FrankaArm(group_name, vel_scale, acc_scale, planning_time)
    arm.lock_orientation()
    return arm.move_to


def read_franka_ee_xyz():
    """Read current EE position from /franka_state_controller/franka_states.

    Uses O_T_EE (column-major 4x4); position is elements [12],[13],[14] in
    panda_link0. Works with just the bare franka_control driver -- no MoveIt.
    """
    import rospy
    from franka_msgs.msg import FrankaState
    msg = rospy.wait_for_message(
        "/franka_state_controller/franka_states", FrankaState, timeout=5.0)
    T = msg.O_T_EE
    return float(T[12]), float(T[13]), float(T[14])


# ============================================================================
#  DEMO
# ============================================================================

if __name__ == "__main__":
    # Example: 3x3 grid of probe points, 8 mm spacing, around a start pose.
    # Fill in your real surface centre in the base frame.
    START_XYZ = (0.55, -0.10, 0.30)   # x, y, safe z above the silicone

    xs = [START_XYZ[0] + dx for dx in (-0.008, 0.0, 0.008)]
    ys = [START_XYZ[1] + dy for dy in (-0.008, 0.0, 0.008)]
    probe_points = [(x, y) for y in ys for x in xs]

    ft = WittensteinFT(port="/dev/ttyACM0").start()
    try:
        move_to = make_franka_mover(vel_scale=0.05)
        touch_map, results = map_surface_offsets(
            probe_points,
            START_XYZ,
            move_to,
            ft.read_window,
            descent_step=1e-4,       # 0.1 mm
            min_z=START_XYZ[2] - 0.05,
            force_min_delta=0.05,    # N -- TUNE to just above HEX21 rest noise
            csv_path="surface_offsets.csv",
        )
    finally:
        ft.stop()

    print("\nSurface map:")
    for (x, y), z in touch_map.items():
        print(f"  ({x:.4f}, {y:.4f}) -> z_touch {z:.5f} m")
