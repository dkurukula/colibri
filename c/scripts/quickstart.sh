#!/usr/bin/env bash
# One command: download the real GLM-5.2 int4 model and benchmark it.
#
# Wraps everything the README's "Download the model" and "Benchmark the full
# model" sections ask a human to do by hand: pick/confirm a download location,
# check there's enough free disk space FIRST, download the pre-converted
# model, detect and auto-fix the known int4-vs-int8 MTP head gotcha
# (https://github.com/JustVugg/colibri/issues/8 — some snapshots still ship
# an unusable int4 MTP head), build for this CPU, then run the disk/decode/
# quality benchmark. Nothing here is invented: every step matches a command
# already documented in the README, just chained and guarded.
#
# Usage:
#   scripts/quickstart.sh                      # interactive, asks before downloading
#   scripts/quickstart.sh -y                   # non-interactive, accepts every default
#   scripts/quickstart.sh --dir /data/glm52    # different download location
#   scripts/quickstart.sh --skip-download      # model already there; just build+benchmark
#   scripts/quickstart.sh --arch ivybridge     # cross-build for an AVX-only machine
#
# Env var overrides: MODEL_DIR, ARCH, NGEN (decode length, default 32),
# BENCH_LIMIT (quality-benchmark examples/task, default 10 — raise on fast hardware).
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE"

REPO="jlnsrk/GLM-5.2-colibri-int4"
MTP_REPO="mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
DEST="${MODEL_DIR:-$HOME/glm52_i4}"
ARCH="${ARCH:-native}"
NGEN="${NGEN:-32}"
BENCH_LIMIT="${BENCH_LIMIT:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-390}"   # ~370 GB model + ~5% margin (same rule of thumb as tools/download_glm52.py)
YES=0
SKIP_DOWNLOAD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DEST="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --repo) REPO="$2"; shift 2;;
        --skip-download) SKIP_DOWNLOAD=1; shift;;
        -y|--yes) YES=1; shift;;
        -h|--help) sed -n '2,20p' "$0"; exit 0;;
        *) echo "unknown flag: $1 (see --help)" >&2; exit 2;;
    esac
done

ask(){ # ask "prompt" "default" -> prints the answer
    local prompt="$1" default="$2" reply
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then printf '%s\n' "$default"; return; fi
    read -r -p "$prompt [$default]: " reply || true
    printf '%s\n' "${reply:-$default}"
}
confirm(){ # confirm "prompt" -> exit status 0=yes
    local prompt="$1" reply
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then return 0; fi
    read -r -p "$prompt [y/N] " reply || true
    case "$reply" in y|Y|yes|YES) return 0;; *) return 1;; esac
}
free_gb_at(){ # free_gb_at PATH -> free space in GB on the nearest existing ancestor
    local p="$1"
    while [ ! -d "$p" ]; do p="$(dirname "$p")"; done
    df -Pk "$p" | awk 'NR==2{printf "%d", $4/1024/1024}'
}

echo "colibri — download + benchmark GLM-5.2 (744B, int4)"
echo

if [ "$SKIP_DOWNLOAD" = 0 ]; then
    DEST="$(ask "Download destination (local ext4/NTFS only — never /mnt/c or a network/9p mount)" "$DEST")"
    case "$DEST" in
        /mnt/*) echo "ERROR: '$DEST' looks like a 9p/Windows mount. Pick a native ext4/NTFS path." >&2; exit 1;;
    esac

    free=$(free_gb_at "$DEST")
    echo "  model       : $REPO (~370 GB)"
    echo "  destination : $DEST"
    echo "  free space  : ${free} GB available (need >= ${MIN_FREE_GB} GB)"
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
        echo "ERROR: not enough free space at $DEST (have ${free} GB, need ${MIN_FREE_GB} GB)." >&2
        echo "       Free up space or re-run with --dir to pick a different disk." >&2
        exit 1
    fi
    confirm "Download now? This is ~370 GB and can take hours depending on your connection." \
        || { echo "aborted — nothing was downloaded."; exit 1; }

    mkdir -p "$DEST"
    command -v huggingface-cli >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
    huggingface-cli download "$REPO" --local-dir "$DEST"

    # issue #8: some snapshots of this repo still ship the int4 MTP head, which
    # measures 0-4% draft acceptance (unusable) instead of 39-59% at int8.
    # Detect it by the known int4 shard sizes and, if found, pull the int8
    # heads from the community fork automatically instead of asking the user
    # to go check file sizes by hand.
    echo
    echo "[MTP] checking MTP head precision (issue #8)..."
    need_fix=0
    shopt -s nullglob
    for f in "$DEST"/*mtp*.safetensors "$DEST"/*mtp*/*.safetensors; do
        sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        case "$sz" in 1765523544|2686077736|536747200) need_fix=1;; esac
    done
    shopt -u nullglob
    if [ "$need_fix" = 1 ]; then
        echo "[MTP] int4 MTP head detected (unusable for speculation) — fetching int8 heads from $MTP_REPO"
        huggingface-cli download "$MTP_REPO" --local-dir "$DEST" --include "*mtp*"
    else
        echo "[MTP] ok (int8 head, or no MTP head present)"
    fi
else
    [ -d "$DEST" ] || { echo "ERROR: --skip-download but $DEST does not exist" >&2; exit 1; }
    echo "  using existing model at $DEST"
fi

echo
echo "=== build (ARCH=$ARCH) + architecture self-test ==="
ARCH="$ARCH" ./setup.sh

echo
echo "=== 1/3: disk benchmark (parallel 19 MB random reads, the way the engine reads experts) ==="
gcc -O2 -fopenmp iobench.c -o iobench
# out-NNNNN.safetensors = a real (large) expert shard; excludes the small
# out-mtp-*/out-idx-* shards so the benchmark reads something representative.
shard=$(ls "$DEST"/out-[0-9]*.safetensors 2>/dev/null | head -1 || true)
if [ -n "$shard" ]; then
    ./iobench "$shard" 19 64 8 0   # buffered
    ./iobench "$shard" 19 64 8 1   # O_DIRECT
else
    echo "  (no out-NNNNN.safetensors shard found at top level of $DEST — skipping)"
fi

echo
echo "=== 2/3: decode speed (real generation, NGEN=$NGEN tokens; watch tok/s and expert hit-rate) ==="
./coli run --model "$DEST" --ngen "$NGEN" "Hello! Please introduce yourself in one sentence."

echo
echo "=== 3/3: quality benchmark (MMLU/HellaSwag/ARC, BENCH_LIMIT=$BENCH_LIMIT examples/task) ==="
echo "    (raise BENCH_LIMIT for a more accurate score — this can take a while on slow disks)"
./coli bench --model "$DEST" --limit "$BENCH_LIMIT"

echo
echo "==== done. Report your numbers (machine, ARCH=$ARCH, disk, RAM, tok/s, hit-rate) in an issue. ===="
