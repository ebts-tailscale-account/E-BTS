#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-e-bts:openeb-3.1.2}"

mkdir -p "$ROOT_DIR/recordings"

# record_sequence.cpp reads/writes control/ and recordings/ as paths relative
# to its working directory. For every other command, -w /data/recordings lets
# relative output paths land directly in recordings/. But record_sequence
# needs *both* directories, so it has to run from /data instead -- otherwise
# control/ and recordings/ would resolve inside recordings/ (recordings/control
# and recordings/recordings), where the host-side Python script driving it
# would never find them.
case "${1:-}" in
  record|record_sequence|E_BTS_record_sequence)
    mkdir -p "$ROOT_DIR/control"
    WORKDIR="/data"
    ;;
  *)
    WORKDIR="/data/recordings"
    ;;
esac

DOCKER_ARGS=(
  --rm
  -it
  --privileged
  --net=host
  --ipc=host
  -e "E_BTS_RECORDINGS_DIR=/data/recordings"
  -e "QT_X11_NO_MITSHM=1"
  -v "$ROOT_DIR:/data:rw"
  -v "/dev/bus/usb:/dev/bus/usb"
  -w "$WORKDIR"
)

if [ -n "${DISPLAY:-}" ]; then
  DOCKER_ARGS+=(-e "DISPLAY=$DISPLAY")

  if [ -d /tmp/.X11-unix ]; then
    DOCKER_ARGS+=(-v "/tmp/.X11-unix:/tmp/.X11-unix:rw")
  fi

  if command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:root >/dev/null 2>&1 || true
    trap 'xhost -SI:localuser:root >/dev/null 2>&1 || true' EXIT
  fi
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$@"