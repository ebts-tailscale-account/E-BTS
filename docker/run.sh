#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-e-bts:openeb-3.1.2}"

mkdir -p "$ROOT_DIR/recordings"

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
  -w "/data/recordings"
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