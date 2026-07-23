#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-e-bts:openeb-3.1.2}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or is not in PATH." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

# record_sequence.cpp reads/writes control/ and recordings/ relative to its
# working directory (/data below, i.e. project_dir on the host, which is
# already bind-mounted as a whole). Create them ahead of time as the host
# user so the Python script driving the robotic hand -- which runs on the
# host and writes control/start.cmd, control/stop.cmd, control/quit.cmd --
# can always read/write them, regardless of what user the container runs as.
mkdir -p "${project_dir}/control" "${project_dir}/recordings"

docker run --rm -it \
    --privileged \
    --user "$(id -u):$(id -g)" \
    -v /dev/bus/usb:/dev/bus/usb \
    -v "${project_dir}:/data:rw" \
    -w /data \
    --entrypoint /app/build/E_BTS_record_sequence \
    "${IMAGE_NAME}" "$@"
