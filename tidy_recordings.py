#!/usr/bin/env python3
"""Group loose files in recordings/ into one folder per run, with canonical names.

master.py now writes each run straight into recordings/<run>_<stamp>/ , but older
runs left their files loose and flat (<run>_franka.csv, <run>_<stamp>.raw, ...).
This moves each run's files into its own folder and renames them to the canonical
names postprocess.py reads:

    camera.raw   ft.csv   camera.bias   camera.roi   franka.csv   metadata.json

Anything it does not recognise is left exactly where it is. Use --dry-run first.

Usage:  python3 tidy_recordings.py [--dry-run] [--recordings DIR]
"""

import argparse
import re
import shutil
import time
from pathlib import Path

# (suffix, canonical name) -- MOST SPECIFIC FIRST: "_ft.csv" must be tested before
# a bare ".csv", and "_BW.mp4" before ".mp4".
CANON = [("_franka.csv", "franka.csv"), ("_metadata.json", "metadata.json"),
         ("_ft.csv", "ft.csv"), ("_BW.mp4", "video_bw.mp4"),
         (".raw", "camera.raw"), (".bias", "camera.bias"), (".roi", "camera.roi"),
         (".mp4", "video.mp4"), (".csv", "events.csv")]
STAMP_RE = re.compile(r"^(?P<base>.+?)_(?P<stamp>\d{8}_\d{6})(?P<rest>.*)$")


def run_key(p):
    """(run_folder_name, canonical_name) for a loose file, or None to leave it."""
    name = p.name
    for suf, canon in CANON:            # most specific suffix wins (ordered list)
        if not name.endswith(suf):
            continue
        stem = name[: -len(suf)]
        # GUI files carry their own timestamp: <run>_<YYYYmmdd_HHMMSS><suffix>
        m = STAMP_RE.match(stem)
        if m and m.group("stamp") and not m.group("rest"):
            return "%s_%s" % (m.group("base"), m.group("stamp")), canon
        # no stamp in the name (e.g. run1_franka.csv) -> stamp from mtime
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(p.stat().st_mtime))
        return "%s_%s" % (stem, stamp), canon
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings",
                    default=str(Path(__file__).resolve().parent / "recordings"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rec = Path(args.recordings)

    # group loose files by the run folder they belong to
    groups = {}
    for p in sorted(rec.iterdir()):
        if p.is_dir() or p.name == ".gitkeep":
            continue
        key = run_key(p)
        if key is None:
            print("  skip (unrecognised): %s" % p.name)
            continue
        folder, canon = key
        groups.setdefault(folder, []).append((p, canon))

    # a run's files must share ONE folder: prefer the stamp seen in GUI filenames
    for folder in sorted(groups):
        print("\n%s/" % folder)
        dest = rec / folder
        for src, canon in groups[folder]:
            target = dest / canon
            print("   %-34s -> %s" % (src.name, canon), end="")
            if target.exists():
                print("   [SKIP: %s already there]" % canon)
                continue
            print()
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(target))
    if args.dry_run:
        print("\n(dry run -- nothing moved)")
    else:
        print("\nDone. Post-process a run with:  python3 postprocess.py <folder_name>")


if __name__ == "__main__":
    main()
