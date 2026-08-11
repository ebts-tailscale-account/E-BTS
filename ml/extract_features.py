#!/usr/bin/env python3
"""Run `marker_features.py` over the whole pilot HDF5 -> one feature table per indent.

Step 1 of §14.8: the marker-displacement features that ridge regression is supposed to
beat the robot-only baseline with. One row per indent, every column of `/indents` carried
through unchanged so the table is self-contained.

Displacements are measured from the TARE-MEAN baseline of the tracked roster to the
MIDDLE DWELL frame, in mm (px / (radius_px / 0.75 mm), so the scale is re-measured per
indent rather than assumed).

⚠ The magnitude features alone were measured to carry essentially no force information
(R^2 = -0.012). The SHAPE features -- contact centroid, radial decay length, anisotropy,
radial-vs-tangential split, concentration -- are the point of this table.

  python3 extract_features.py                       # all 648, multiprocess
  python3 extract_features.py --limit 20 --workers 4
"""

import argparse
import os
import sys
import time

import numpy as np
import h5py

import marker_features as mf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml/ -> repo root
sys.path.insert(0, REPO)   # so marker_overlay.py (still in the root) imports
OUTPUT = os.path.join(REPO, "output")
DEFAULT_H5 = os.path.join(OUTPUT, "pilot_20260807_134855_frames.h5")

_W = {"h5": None, "path": None}


def _init(path):
    import cv2
    cv2.setNumThreads(1)                 # workers are already the parallelism
    _W["path"] = path


def _handle():
    if _W["h5"] is None:
        _W["h5"] = h5py.File(_W["path"], "r")
    return _W["h5"]


def _one(args):
    i, offset, n = args
    h5 = _handle()
    frames = h5["/frames"][offset:offset + n]
    phases = h5["/frame_phase"][offset:offset + n]
    try:
        r = mf.process_indent(frames, phases)
    except Exception as e:                                     # never lose the campaign
        return i, None, "%s: %s" % (type(e).__name__, e)
    if r is None:
        return i, None, "process_indent returned None"
    f = mf.shape_features(r["baseline_px"], r["disp_px"], r["px_per_mm"])
    f["n_markers_seeded"] = r["n_seeded"]
    f["n_markers_kept"] = r["n_kept"]
    f["px_per_mm"] = r["px_per_mm"]
    f["marker_radius_px"] = r["radius_px"]
    f["peak_frame"] = r["peak_frame"]
    f["dets_median"] = float(np.median(r["dets_per_frame"]))
    return i, f, None


EXTRA_COLS = ["n_markers_seeded", "n_markers_kept", "px_per_mm", "marker_radius_px",
              "peak_frame", "dets_median"]


def main():
    ap = argparse.ArgumentParser()
    # Accept a RUN FOLDER positionally, like build_dataset.py and evaluate_holdout.py.
    # Every other script in ml/ is invoked as `<script> recordings/<run>`; making this
    # one the exception guarantees somebody types the wrong thing eventually.
    ap.add_argument("run", nargs="?", default=None,
                    help="run folder, e.g. recordings/hard_20260807_223744. Resolved to "
                         "output/<run>_frames.h5. Overrides --h5.")
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--out", default=None, help="stem for the .csv/.npz (default: the h5 "
                                                "stem with _frames -> _features)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.run:
        run = os.path.basename(a.run.rstrip("/"))
        a.h5 = os.path.join(OUTPUT, run + "_frames.h5")
        if not os.path.exists(a.h5):
            raise SystemExit("[ERROR] %s not found.\n"
                             "        Build the frames first:\n"
                             "            python3 ml/build_dataset.py recordings/%s"
                             % (a.h5, run))

    h5 = h5py.File(a.h5, "r")
    ind = h5["/indents"][:]
    h5.close()
    if a.limit:
        ind = ind[:a.limit]
    N = ind.size

    stem = a.out
    if stem is None:
        stem = os.path.splitext(a.h5)[0]
        stem = stem[:-7] + "_features" if stem.endswith("_frames") else stem + "_features"

    print("HDF5    : %s" % a.h5)
    print("indents : %d" % N)
    print("workers : %d" % a.workers)
    print("out     : %s.csv / .npz\n" % stem)

    jobs = [(i, int(ind[i]["frame_offset"]), int(ind[i]["n_frames"])) for i in range(N)]
    cols = mf.FEATURE_COLS + EXTRA_COLS
    res = [None] * N
    errs = []

    t0 = time.time()
    done = 0
    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(a.workers, initializer=_init, initargs=(a.h5,)) as pool:
            for i, f, err in pool.imap_unordered(_one, jobs, chunksize=1):
                res[i] = f
                if err:
                    errs.append((int(ind[i]["point_index"]), err))
                done += 1
                if done % 10 == 0 or done == N:
                    el = time.time() - t0
                    eta = el / done * (N - done)
                    sys.stdout.write("\r  %4d/%d  %5.1f%%  elapsed %5.1f min  "
                                     "ETA %5.1f min  (%.2f s/indent)  errors %d"
                                     % (done, N, 100.0 * done / N, el / 60, eta / 60,
                                        el / done, len(errs)))
                    sys.stdout.flush()
    else:
        _init(a.h5)
        for job in jobs:
            i, f, err = _one(job)
            res[i] = f
            if err:
                errs.append((int(ind[i]["point_index"]), err))
            done += 1
            if done % 10 == 0 or done == N:
                el = time.time() - t0
                eta = el / done * (N - done)
                sys.stdout.write("\r  %4d/%d  %5.1f%%  elapsed %5.1f min  ETA %5.1f min  "
                                 "(%.2f s/indent)  errors %d"
                                 % (done, N, 100.0 * done / N, el / 60, eta / 60,
                                    el / done, len(errs)))
                sys.stdout.flush()
    total = time.time() - t0
    print("\n\nextracted %d/%d in %.1f min (%.2f s/indent)"
          % (N - len(errs), N, total / 60, total / N))
    for pi, e in errs[:20]:
        print("  [FAIL] point_index %d: %s" % (pi, e))

    # ---- assemble: every /indents column unchanged, then the features
    data = {}
    for name in ind.dtype.names:
        data[name] = np.asarray(ind[name])
    for c in cols:
        data[c] = np.array([np.nan if res[i] is None else res[i][c]
                            for i in range(N)], float)
    data["ok"] = np.array([res[i] is not None for i in range(N)], bool)

    order = list(ind.dtype.names) + cols + ["ok"]
    np.savez_compressed(stem + ".npz", **{k: data[k] for k in order})

    import pandas as pd
    df = pd.DataFrame({k: data[k] for k in order}, columns=order)
    df.to_csv(stem + ".csv", index=False)

    print("wrote %s.csv  (%d rows x %d cols, %.1f kB)"
          % (stem, len(df), len(order), os.path.getsize(stem + ".csv") / 1e3))
    print("wrote %s.npz" % stem)
    print("columns: %s" % ", ".join(order))

    ok = df["ok"].values
    print("\nsanity (%d ok):" % ok.sum())
    print("  markers seeded  %s" % np.unique(df["n_markers_seeded"][ok]).astype(int)[:12])
    print("  markers kept    min %d  median %d  max %d"
          % (df["n_markers_kept"][ok].min(), np.median(df["n_markers_kept"][ok]),
             df["n_markers_kept"][ok].max()))
    print("  px_per_mm       %.3f +- %.3f"
          % (df["px_per_mm"][ok].mean(), df["px_per_mm"][ok].std()))
    print("  disp_max_mm     %.3f .. %.3f"
          % (df["disp_max_mm"][ok].min(), df["disp_max_mm"][ok].max()))


if __name__ == "__main__":
    main()
