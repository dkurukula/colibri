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
# Before running the model, it also checks RAM/swap: heavy swapping can make
# the whole machine unresponsive (not just colibrì), so low total RAM,
# already-high swap usage, or a --ram budget bigger than what's actually free
# are caught up front, explained, and gated behind a fix/continue-anyway/abort
# choice (or, with -y, the safe fix is applied automatically) instead of
# silently letting the box start thrashing.
#
# Usage:
#   scripts/quickstart.sh                      # interactive, asks before downloading
#   scripts/quickstart.sh -y                   # non-interactive, accepts every default
#   scripts/quickstart.sh --dir /data/glm52    # different download location
#   scripts/quickstart.sh --ram 24             # cap the engine's RAM budget (GB)
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
RAM_ARG="${RAM:-}"                  # empty = let the engine auto-budget (88% of available RAM)
YES=0
SKIP_DOWNLOAD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DEST="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --repo) REPO="$2"; shift 2;;
        --ram) RAM_ARG="$2"; shift 2;;
        --skip-download) SKIP_DOWNLOAD=1; shift;;
        -y|--yes) YES=1; shift;;
        -h|--help) sed -n '2,29p' "$0"; exit 0;;
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

mem_info(){ # prints "total_kb avail_kb swap_total_kb swap_free_kb" (best effort, 0s if unknown)
    if [ -r /proc/meminfo ]; then
        awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2}
             END{printf "%d %d %d %d", t+0, a+0, st+0, sf+0}' /proc/meminfo
    elif [ "$(uname -s 2>/dev/null)" = "Darwin" ] && command -v sysctl >/dev/null 2>&1; then
        local total_kb page free_pages avail_kb sw swtot_m swused_m swtot_kb swused_kb swfree_kb
        total_kb=$(( $(sysctl -n hw.memsize) / 1024 ))
        page=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
        free_pages=$(vm_stat 2>/dev/null | awk '/Pages free/{gsub("\\.","",$3); print $3}')
        avail_kb=$(( ${free_pages:-0} * page / 1024 ))
        sw=$(sysctl -n vm.swapusage 2>/dev/null || echo "")
        swtot_m=$(printf '%s' "$sw" | sed -n 's/.*total = \([0-9.]*\)M.*/\1/p')
        swused_m=$(printf '%s' "$sw" | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')
        swtot_kb=$(awk -v m="${swtot_m:-0}" 'BEGIN{printf "%d", m*1024}')
        swused_kb=$(awk -v m="${swused_m:-0}" 'BEGIN{printf "%d", m*1024}')
        swfree_kb=$(( swtot_kb - swused_kb ))
        printf '%d %d %d %d' "$total_kb" "$avail_kb" "$swtot_kb" "$swfree_kb"
    else
        printf '0 0 0 0'
    fi
}

# Heavy swapping doesn't just slow colibri down — it can make the WHOLE
# machine unresponsive (UI stalls, other apps get OOM-killed, disk thrashes).
# Check for that risk BEFORE loading a 744B model, explain it, and offer a
# fix instead of a wall of text nobody reads until the desktop has frozen.
ram_swap_gate(){
    local total_kb avail_kb swtot_kb swfree_kb
    read -r total_kb avail_kb swtot_kb swfree_kb <<<"$(mem_info)"
    if [ "$total_kb" = 0 ]; then
        echo "[RAM] could not read memory info on this platform — skipping the swap safety check"
        return
    fi
    local total_gb=$((total_kb/1024/1024)) avail_gb=$((avail_kb/1024/1024)) swtot_gb=$((swtot_kb/1024/1024))
    local swused_kb=$((swtot_kb - swfree_kb)); [ "$swused_kb" -lt 0 ] && swused_kb=0
    local swused_pct=0
    [ "$swtot_kb" -gt 0 ] && swused_pct=$((swused_kb*100/swtot_kb))

    echo "  RAM : ${total_gb} GB total, ${avail_gb} GB available right now"
    [ "$swtot_gb" -gt 0 ] && echo "  swap: ${swtot_gb} GB total, ${swused_pct}% already in use"

    local risky=0 reasons=()
    if [ "$total_gb" -lt 16 ]; then
        risky=1; reasons+=("total RAM (${total_gb} GB) is below colibri's 16 GB floor")
    fi
    if [ "$swtot_gb" -gt 0 ] && [ "$swused_pct" -ge 40 ]; then
        risky=1; reasons+=("swap is already ${swused_pct}% full — this machine is under memory pressure BEFORE colibri even starts")
    fi
    if [ -n "$RAM_ARG" ] && [ "$RAM_ARG" -gt "$avail_gb" ] 2>/dev/null; then
        risky=1; reasons+=("--ram ${RAM_ARG} exceeds the ${avail_gb} GB actually available right now")
    fi
    [ "$risky" = 0 ] && return

    local safe_gb=$(( avail_gb * 80 / 100 ))
    [ "$safe_gb" -lt 4 ] && safe_gb=4

    echo
    echo "WARNING: heavy swapping is likely and can make this machine unresponsive"
    echo "         (UI lag, other apps killed by the OOM killer, disk thrashing):"
    local r; for r in "${reasons[@]}"; do echo "  - $r"; done
    echo

    if [ "$YES" = 1 ] || [ ! -t 0 ]; then
        echo "-y given: applying the safe fix automatically -> --ram ${safe_gb}"
        RAM_ARG="$safe_gb"
        return
    fi

    echo "Options:"
    echo "  [1] use a safer budget: --ram ${safe_gb}  (recommended)"
    echo "  [2] continue anyway, unchanged  (risk of a frozen system)"
    echo "  [3] abort"
    local choice
    read -r -p "Choice [1]: " choice || true
    case "${choice:-1}" in
        2) echo "continuing unchanged — you asked for it.";;
        3) echo "aborted."; exit 1;;
        *) RAM_ARG="$safe_gb"; echo "using --ram ${safe_gb}";;
    esac
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
echo "=== RAM/swap safety check ==="
ram_swap_gate
ram_flag=(); [ -n "$RAM_ARG" ] && ram_flag=(--ram "$RAM_ARG")

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
./coli run --model "$DEST" "${ram_flag[@]}" --ngen "$NGEN" "Hello! Please introduce yourself in one sentence."

echo
echo "=== 3/3: quality benchmark (MMLU/HellaSwag/ARC, BENCH_LIMIT=$BENCH_LIMIT examples/task) ==="
echo "    (raise BENCH_LIMIT for a more accurate score — this can take a while on slow disks)"
./coli bench --model "$DEST" "${ram_flag[@]}" --limit "$BENCH_LIMIT"

echo
echo "==== done. Report your numbers (machine, ARCH=$ARCH, disk, RAM, tok/s, hit-rate) in an issue. ===="
