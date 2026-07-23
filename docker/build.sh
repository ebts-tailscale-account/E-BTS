#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-e-bts:openeb-3.1.2}"
OPENEB_VERSION="${OPENEB_VERSION:-3.1.2}"

cd "$ROOT_DIR"

docker build \
  --pull \
  --build-arg OPENEB_VERSION="$OPENEB_VERSION" \
  -t "$IMAGE" \
  .

docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
  test -x /app/build/E_BTS_GUI &&
  test -x /app/build/E_BTS_event_circle_tracker &&
  test -x /app/build/E_BTS_record_sequence &&
  test -x /app/build/E_BTS_raw_to_csv &&
  test -x /usr/local/bin/e-bts-entrypoint &&
  echo "Docker image sanity check passed."
'