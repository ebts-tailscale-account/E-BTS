#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-e-bts:openeb-3.1.2}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or is not in PATH." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

docker run --rm -it \
    -v "${project_dir}:/data:rw" \
    -w /data \
    --entrypoint /app/build/E_BTS_raw_to_csv \
    "${IMAGE_NAME}" "$@"
