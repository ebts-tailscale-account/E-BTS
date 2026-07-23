#!/usr/bin/env bash
set -Eeuo pipefail

: "${E_BTS_RECORDINGS_DIR:=/data/recordings}"

mkdir -p "$E_BTS_RECORDINGS_DIR"

if [ ! -w "$E_BTS_RECORDINGS_DIR" ]; then
  echo "ERROR: recordings directory is not writable: $E_BTS_RECORDINGS_DIR" >&2
  echo "Make sure the repo is mounted with: -v \"\$PWD:/data\"" >&2
  exit 73
fi

cd "$E_BTS_RECORDINGS_DIR"

cmd="${1:-viewer}"

case "$cmd" in
  viewer|gui|E_BTS_GUI)
    shift || true
    exec /app/build/E_BTS_GUI "$@"
    ;;

  tracker|E_BTS_event_circle_tracker)
    shift || true
    exec /app/build/E_BTS_event_circle_tracker "$@"
    ;;

  record|record_sequence|E_BTS_record_sequence)
    shift || true
    exec /app/build/E_BTS_record_sequence "$@"
    ;;

  csv|raw_to_csv|E_BTS_raw_to_csv)
    shift || true
    exec /app/build/E_BTS_raw_to_csv "$@"
    ;;

  /app/build/*)
    exec "$@"
    ;;

  *)
    exec /app/build/E_BTS_GUI "$@"
    ;;
esac