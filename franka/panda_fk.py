#!/usr/bin/env python3
"""Numpy-only forward kinematics for the Franka Emika Panda (no ROS, no MoveIt).

WHY THIS EXISTS
---------------
The Cartesian pose servo controller accepts Cartesian PoseStamped targets only --
there is no joint-target interface. So to work in JOINT space (which is what the
taught points actually record, and what the encoders actually measure) and still
command the servo, we need FK: q[7] -> O_T_EE.

VALIDATION
----------
record_two_points.py / record_corners.py capture q and O_T_EE from the SAME
FrankaState snapshot, so every taught point is an exact FK ground-truth pair.
Validated against the 5 points in corner_joints.json (taught 2026-07-31):

    max position residual 0.001 mm, rms 0.001 mm

and the recovered flange->EE translation was [0, 0, 0], i.e. with
`load_gripper:=false` and no EE transform configured, O_T_EE IS the flange pose.
`self_check()` re-runs that comparison on live data so a changed tool/EE
transform is caught immediately instead of silently biasing every target.

WHY JOINT SPACE MATTERS HERE
----------------------------
Joint-space interpolation makes the EE follow an arc, not the chord. Measured on
the taught corners, the joint-space midpoint differs from the Cartesian midpoint
by 0.14 mm (22 mm apart) up to 0.78 mm (47 mm apart) -- on a 2 mm indent that is
a 7-40% error, so the two are NOT interchangeable.
"""

import numpy as np

# Franka Panda modified-DH (Craig convention). Rows: (a_{i-1}, d_i, alpha_{i-1}).
DH = [
    (0.0,     0.333,  0.0),
    (0.0,     0.0,   -np.pi / 2),
    (0.0,     0.316,  np.pi / 2),
    (0.0825,  0.0,    np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0,     0.0,    np.pi / 2),
    (0.088,   0.0,    np.pi / 2),
]
D_FLANGE = 0.107          # joint7 -> flange (link8)

# Set from a live snapshot by calibrate_tool() when a tool transform is present.
# Default identity: verified correct for load_gripper:=false with no EE transform.
T_FLANGE_EE = np.eye(4)


def _link(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,       -st,      0.0,  a],
        [st * ca,   ct * ca, -sa, -d * sa],
        [st * sa,   ct * sa,  ca,  d * ca],
        [0.0,       0.0,     0.0,  1.0],
    ])


def fk_flange(q):
    """O_T_F (4x4) for joint vector q[7]."""
    q = np.asarray(q, float).ravel()
    if q.size != 7:
        raise ValueError("need 7 joint angles, got %d" % q.size)
    T = np.eye(4)
    for (a, d, alpha), th in zip(DH, q):
        T = T @ _link(a, d, alpha, th)
    return T @ _link(0.0, D_FLANGE, 0.0, 0.0)


def fk(q, T_flange_ee=None):
    """O_T_EE (4x4) for joint vector q[7], including any tool transform."""
    tool = T_FLANGE_EE if T_flange_ee is None else np.asarray(T_flange_ee, float)
    return fk_flange(q) @ tool


def R_to_quat(R):
    """Rotation matrix -> quaternion [x, y, z, w] (the servo_client / ROS order)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    qv = np.array([x, y, z, w], float)
    return list(qv / np.linalg.norm(qv))


def colmajor_to_T(v16):
    """FrankaState O_T_EE (column-major 16) -> 4x4 matrix."""
    return np.asarray(v16, float).reshape(4, 4).T


def pose_of(q, T_flange_ee=None):
    """(xyz[3], quat[xyzw]) for joint vector q."""
    T = fk(q, T_flange_ee)
    return list(T[:3, 3]), R_to_quat(T[:3, :3])


def self_check(q, O_T_EE_colmajor, tol_mm=1.0, label="", verbose=True):
    """Compare fk(q) against a measured O_T_EE from the SAME snapshot.

    Returns the position residual in mm. Raises SystemExit above tol_mm, because a
    mismatch means the DH/tool assumption is wrong and every joint-space target
    computed from it would be silently biased.
    """
    T_meas = colmajor_to_T(O_T_EE_colmajor)
    T_pred = fk(q)
    err_mm = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]) * 1e3)
    # orientation error as the angle of R_pred^T R_meas
    dR = T_pred[:3, :3].T @ T_meas[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
    if verbose:
        print("  [fk check%s] position residual %.4f mm | orientation %.4f deg"
              % ((" " + label) if label else "", err_mm, ang))
    if err_mm > tol_mm:
        raise SystemExit(
            "\n[ABORT] FK does not match the measured O_T_EE (%.3f mm > %.3f mm).\n"
            "        The DH table or the flange->EE tool transform is wrong for this\n"
            "        setup, so any joint-space target would be biased. Check whether a\n"
            "        gripper/EE transform is configured (load_gripper, F_T_NE/NE_T_EE)\n"
            "        and call calibrate_tool() with a live snapshot.\n" % (err_mm, tol_mm))
    return err_mm


def calibrate_tool(q, O_T_EE_colmajor):
    """Recover and install the constant flange->EE transform from one snapshot."""
    global T_FLANGE_EE
    T_FLANGE_EE = np.linalg.inv(fk_flange(q)) @ colmajor_to_T(O_T_EE_colmajor)
    return T_FLANGE_EE


def joint_midpoint(qa, qb):
    """Midway joint configuration between two taught poses."""
    return list(0.5 * (np.asarray(qa, float) + np.asarray(qb, float)))
