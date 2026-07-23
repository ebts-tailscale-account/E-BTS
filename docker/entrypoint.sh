#!/usr/bin/env bash
set -Eeuo pipefail

: "${E_BTS_RECORDINGS_DIR:=/data/recordings}"

mkdir -p "$E_BTS_RECORDINGS_DIR"

if [ ! -w "$E_BTS_RECORDINGS_DIR" ]; then
  echo "ERROR: recordings directory is not writable: $E_BTS_RECORDINGS_DIR" >&2
  echo "Make sure the repo is mounted with: -v \"\$PWD:/data\"" >&2
  exit 73
fi

cmd="${1:-viewer}"

case "$cmd" in
  record|record_sequence|E_BTS_record_sequence)
    # record_sequence.cpp writes both control/ and recordings/ as paths
    # relative to its working directory. Running it from E_BTS_RECORDINGS_DIR
    # (recordings/ itself) would resolve those to recordings/control and
    # recordings/recordings, where nothing driving it from the host would
    # find them -- so it needs to run from the mount root instead.
    cd "$(dirname "$E_BTS_RECORDINGS_DIR")"
    shift || true
    exec /app/build/E_BTS_record_sequence "$@"
    ;;

  viewer|gui|E_BTS_GUI)
    cd "$E_BTS_RECORDINGS_DIR"
    shift || true
    exec /app/build/E_BTS_GUI "$@"
    ;;

  tracker|E_BTS_event_circle_tracker)
    cd "$E_BTS_RECORDINGS_DIR"
    shift || true
    exec /app/build/E_BTS_event_circle_tracker "$@"
    ;;

  csv|raw_to_csv|E_BTS_raw_to_csv)
    cd "$E_BTS_RECORDINGS_DIR"
    shift || true
    exec /app/build/E_BTS_raw_to_csv "$@"
    ;;

  /app/build/*)
    cd "$E_BTS_RECORDINGS_DIR"
    exec "$@"
    ;;

  *)
    cd "$E_BTS_RECORDINGS_DIR"
    exec /app/build/E_BTS_GUI "$@"
    ;;
esac