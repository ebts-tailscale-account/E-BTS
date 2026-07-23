#!/usr/bin/env python3
"""
record_testing.py

Runs natively on the host (NOT inside Docker). Drives E_BTS_record_sequence's
start.cmd/stop.cmd/quit.cmd file protocol (see src/sequence_recording_controller.h)
against a copy of E_BTS_record_sequence that's already running -- either the native
build or, for the docker comparison, one started with:

    ./docker/record_sequence.sh [camera_serial]

Do NOT use "./docker/run.sh record" for this test: run.sh sets the container's
working directory to /data/recordings, so record_sequence's relative "control"
and "recordings" paths resolve to recordings/control/ and recordings/recordings/
instead of the repo-root control/ and recordings/ this script watches --
the two processes would never see each other's files. record_sequence.sh sets
-w /data, which is the layout this script expects.

For each sequence name this script:
  1. drops control/start.cmd (contents = sequence base name)
  2. waits for the corresponding .raw file to appear in recordings/
  3. lets it record for --record-seconds, confirming the file actually grows
     (proves CD events are flowing, not just that the file was created)
  4. drops control/stop.cmd and confirms it gets consumed (proves the watcher
     process is alive and polling, whether native or containerized)

Exit code is 0 only if every sequence passed.
"""

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent
CONTROL_DIR = REPO_ROOT / "control"
RECORDINGS_DIR = REPO_ROOT / "recordings"


def wait_for(predicate, timeout, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def newest_raw(before):
    fresh = set(RECORDINGS_DIR.glob("*.raw")) - before
    return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None


def run_sequence(name, record_seconds):
    print(f"\n=== sequence '{name}' ===")
    before = set(RECORDINGS_DIR.glob("*.raw"))

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    (CONTROL_DIR / "start.cmd").write_text(name)
    print(f"-> wrote control/start.cmd ('{name}')")

    if not wait_for(lambda: newest_raw(before) is not None, timeout=10):
        print("FAIL: no new .raw file appeared in recordings/ within 10s")
        return None
    raw_path = newest_raw(before)
    print(f"-> detected new recording: {raw_path.name}")

    size_at_start = raw_path.stat().st_size
    print(f"   size right after start: {size_at_start} bytes")

    time.sleep(record_seconds)

    size_before_stop = raw_path.stat().st_size
    grew = size_before_stop > size_at_start
    print(f"   size after {record_seconds}s: {size_before_stop} bytes "
          f"({'grew' if grew else 'did NOT grow'})")

    (CONTROL_DIR / "stop.cmd").write_text("")
    print("-> wrote control/stop.cmd")

    if not wait_for(lambda: not (CONTROL_DIR / "stop.cmd").exists(), timeout=5):
        print("FAIL: stop.cmd was not consumed within 5s -- watcher may not be polling")
        return None
    print("-> stop.cmd consumed by watcher")

    time.sleep(0.5)
    final_size = raw_path.stat().st_size
    print(f"   final size: {final_size} bytes")

    if not grew:
        print(f"FAIL: '{name}' created a file but it never grew "
              f"-- camera may not be streaming events")
        return None

    print(f"PASS: '{name}' -> {raw_path.name} ({final_size} bytes)")
    return raw_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", nargs="+",
                         default=["record_test_a", "record_test_b"],
                         help="base names to use for each test recording")
    parser.add_argument("--record-seconds", type=float, default=3.0,
                         help="how long to let each recording run before stopping")
    parser.add_argument("--quit", action="store_true",
                         help="also drop quit.cmd at the end to stop the watcher process")
    args = parser.parse_args()

    print(f"control dir:    {CONTROL_DIR}")
    print(f"recordings dir: {RECORDINGS_DIR}")
    print("(E_BTS_record_sequence must already be running and watching these "
          "same directories)")

    results = [(name, run_sequence(name, args.record_seconds)) for name in args.sequences]

    if args.quit:
        (CONTROL_DIR / "quit.cmd").write_text("")
        print("\n-> wrote control/quit.cmd (watcher should exit)")

    print("\n=== summary ===")
    ok = True
    for name, result in results:
        print(f"  {name}: {'PASS (' + result.name + ')' if result else 'FAIL'}")
        ok = ok and result is not None

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
