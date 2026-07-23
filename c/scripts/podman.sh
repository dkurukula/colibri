#!/usr/bin/env bash
# One-command podman path: builds the image once, then runs colibrì inside it —
# no more hand-writing `podman run -v ... -e ... image chat`. This is the
# recommended default way to run colibrì; see README "Run it in podman".
#
# Usage (from repo root or c/):
#   COLI_MODEL=/nvme/glm52_i4 scripts/podman.sh chat
#   COLI_MODEL=/nvme/glm52_i4 scripts/podman.sh serve --host 0.0.0.0
#   COLI_MODEL=/nvme/glm52_i4 scripts/podman.sh plan
#   scripts/podman.sh --build-only     # just (re)build the image, run nothing
#
# Tunables (env):
#   COLI_MODEL   model directory on the HOST (required for chat/serve/run/plan/bench)
#   IMAGE        image tag (default localhost/colibri:latest)
#   ARCH         CPU tier to build for (default native; see bench_cpu_tiers.sh for the list)
#   RAM_GB       RAM budget passed through to the engine (default: engine auto-sizes it)
#   REPIN        override the live re-pin interval (default: engine's own default, 64; 0=off)
#   REPIN_EPS    override the re-pin gate's deviation threshold (default: engine's own, 0.15)
#   PORT         port to publish for `serve` (default 8000)
#   REBUILD=1    force a rebuild even if the image already exists
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-localhost/colibri:latest}"
ARCH="${ARCH:-native}"

build(){
    if [ "${REBUILD:-0}" = 1 ] || ! podman image exists "$IMAGE" 2>/dev/null; then
        echo "[podman] building $IMAGE (ARCH=$ARCH)..." >&2
        podman build --build-arg ARCH="$ARCH" -t "$IMAGE" "$ROOT"
    fi
}

if [ "${1:-}" = "--build-only" ]; then
    build
    exit 0
fi

build

[ -n "${COLI_MODEL:-}" ] || { echo "COLI_MODEL is not set — point it at your downloaded model directory (see 'make quickstart')." >&2; exit 1; }
[ -d "$COLI_MODEL" ] || { echo "COLI_MODEL=$COLI_MODEL does not exist" >&2; exit 1; }

PORT_ARGS=()
[ "${1:-}" = "serve" ] && PORT_ARGS=(-p "${PORT:-8000}:${PORT:-8000}")

# -t only when there's an actual terminal attached (interactive `chat`, run from a shell).
# Forcing it unconditionally hangs non-interactive invocations (scripts, piped input,
# non-tty SSH) since podman then waits on a tty that will never show up. -i (keep stdin
# open) stays on always: chat/serve both read from stdin, tty or not.
TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)

exec podman run --rm "${TTY_ARGS[@]}" \
    --userns=keep-id --security-opt label=disable \
    ${RAM_GB:+-e RAM_GB="$RAM_GB"} \
    ${REPIN:+-e REPIN="$REPIN"} \
    ${REPIN_EPS:+-e REPIN_EPS="$REPIN_EPS"} \
    -v "$COLI_MODEL:/model" \
    "${PORT_ARGS[@]}" \
    "$IMAGE" "$@"
