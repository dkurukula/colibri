#!/usr/bin/env bash
# Verify the engine actually loads and runs a REAL GLM-5.2 checkpoint end to end —
# real config, real tokenizer, real int4/int8 quantized weights — not just the
# synthetic fixtures used elsewhere (bench_cpu_tiers.sh, bench_repin.sh).
#
# The full checkpoint is ~370 GB, too large to fetch just to sanity-check the
# loader/kernels. Since colibri loads embed/lm_head/attention/dense-FFN/router/
# shared-experts fully resident at startup and only streams routed MoE experts
# lazily at inference time (see model_init in glm.c), a real model with
# num_hidden_layers truncated down to first_k_dense_replace needs NO routed
# expert tensors at all — just the dense-resident set for those layers. That's
# a real, complete, runnable (if artificially short) slice of the real model,
# downloadable in single-digit GB instead of hundreds.
#
# What this proves: real config.json parses (shape/field validation), the real
# tokenizer round-trips, real int4/int8 (.qs sidecar) weights load, and real
# attention (MLA) + dense FFN + lm_head run without crashing/NaN through the
# ACTUAL compiled kernels — including a byte-identical-output cross-check
# between the AVX2/AVX-512 build and the AVX-only (Ivy Bridge) build on the
# SAME real weights, with TEMP=0 (greedy) for determinism.
#
# Usage:
#   scripts/verify_real_model.sh [huggingface-repo-id]
# Defaults to the model README.md documents (jlnsrk/GLM-5.2-colibri-int4).
#
# Tunables (env):
#   DIR=/tmp/glm_real_verify   where the real shards are downloaded/cached
#   ARCHES="native ivybridge"  CPU tiers to build and cross-check
#   PROMPT="The capital of France is"
#   NGEN=24
#   QEMU=0            set to 1 to also cross-check under qemu -cpu IvyBridge
#                     (real weights make this SLOW — expect ~15min, vs seconds
#                     on the tiny synthetic fixture bench_cpu_tiers.sh uses)
#   KEEP=0            set to 1 to keep $DIR afterward (default: left in place,
#                     re-run reuses the cached download — delete it yourself
#                     with `rm -rf $DIR` when done, it's several GB)
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE"

REPO="${1:-jlnsrk/GLM-5.2-colibri-int4}"
DIR="${DIR:-/tmp/glm_real_verify}"
ARCHES=(${ARCHES:-native ivybridge})
PROMPT="${PROMPT:-The capital of France is}"
NGEN="${NGEN:-24}"
QEMU="${QEMU:-0}"
KEEP="${KEEP:-0}"
BIN_DIR="$CODE/.verify_bins"

log(){ printf '%s\n' "$*"; }
mkdir -p "$DIR" "$BIN_DIR"

# ---------- 1) find which real shards hold the dense-resident tensors ----------
FOUND="$DIR/found_shards.json"
if [ ! -f "$FOUND" ]; then
    log "[locate] scanning $REPO's shard headers (range requests only, no weight data)..."
    python3 tools/find_real_shards.py "$REPO" --out "$FOUND"
fi
FIRST_DENSE=$(python3 -c "import json;print(json.load(open('$FOUND'))['first_k_dense_replace'])")
SHARDS=$(python3 -c "import json;print(' '.join(json.load(open('$FOUND'))['shards']))")
log "[locate] need shards: $SHARDS (covers layers 0..$((FIRST_DENSE-1)))"

# ---------- 2) download just those shards + config/tokenizer ----------
NEED_FILES="config.json generation_config.json tokenizer.json tokenizer_config.json $SHARDS"
MISSING=0
for f in $NEED_FILES; do [ -f "$DIR/$f" ] || MISSING=1; done
if [ "$MISSING" = 1 ]; then
    log "[download] fetching $(echo $NEED_FILES | wc -w) files from $REPO..."
    hf download "$REPO" $NEED_FILES --local-dir "$DIR" >/tmp/verify_real_model_dl.log 2>&1 || {
        tail -30 /tmp/verify_real_model_dl.log >&2; exit 1; }
    rm -rf "$DIR/.cache"
fi
du -sh "$DIR" 2>/dev/null | sed 's/^/[download] on disk: /'

# ---------- 3) truncate the config to just the dense prefix (no MoE tensors needed) ----------
python3 -c "
import json
p='$DIR/config.json'
c=json.load(open(p))
c['num_hidden_layers']=$FIRST_DENSE
json.dump(c,open(p,'w'))
"

# ---------- 4) build every requested tier, run the SAME real prompt greedily ----------
declare -A OUT
for arch in "${ARCHES[@]}"; do
    log "[build] ARCH=$arch"
    make -s glm ARCH="$arch"
    mv glm "$BIN_DIR/glm-$arch"
    log "[run] ARCH=$arch"
    out=$(PROMPT="$PROMPT" NGEN="$NGEN" TEMP=0 SNAP="$DIR" RAM_GB=6 CTX=256 "$BIN_DIR/glm-$arch" 4 8 8 2>&1)
    kernel=$(printf '%s\n' "$out" | grep -oE 'idot: [a-z0-9-]+' | head -1)
    gen=$(printf '%s\n' "$out" | sed -n '/^'"$(printf '%s' "$PROMPT" | sed 's/[.[\*^$/]/\\&/g')"'/p')
    OUT[$arch]="$gen"
    log "  $kernel"
    log "  $gen"
    if [ -z "$gen" ]; then log "FAIL: ARCH=$arch produced no output — see:"; printf '%s\n' "$out" >&2; exit 1; fi
done

log ""
log "=== cross-tier output (must be byte-identical, TEMP=0 greedy) ==="
REF_ARCH="${ARCHES[0]}"; REF="${OUT[$REF_ARCH]}"; PASS=1
for arch in "${ARCHES[@]}"; do
    if [ "${OUT[$arch]}" = "$REF" ]; then
        log "  ok  $arch matches $REF_ARCH"
    else
        log "  FAIL $arch differs from $REF_ARCH"
        PASS=0
    fi
done

# ---------- 5) optional: prove the AVX-only build needs no AVX2 (real weights, real time) ----------
if [ "$QEMU" = 1 ] && command -v qemu-x86_64-static >/dev/null 2>&1; then
    log ""
    log "=== qemu-user -cpu IvyBridge on real weights (slow: no JIT for AVX-heavy loops) ==="
    make -s glm ARCH=ivybridge; mv glm "$BIN_DIR/glm-ivybridge"
    make -s glm ARCH=x86-64-v3; mv glm "$BIN_DIR/glm-x86-64-v3"
    if OMP_NUM_THREADS=1 PROMPT="$PROMPT" NGEN=1 TEMP=0 SNAP="$DIR" RAM_GB=6 CTX=64 \
        qemu-x86_64-static -cpu IvyBridge "$BIN_DIR/glm-x86-64-v3" 4 8 8 >/tmp/qemu_v3_real.log 2>&1; then
        log "FAIL: AVX2 build did NOT SIGILL under emulated Ivy Bridge (expected — AVX2 isn't in Ivy Bridge)"; PASS=0
    else
        grep -qi "illegal instruction" /tmp/qemu_v3_real.log && log "  ok  AVX2 build SIGILLs under emulated Ivy Bridge, as expected"
    fi
    if out=$(OMP_NUM_THREADS=1 PROMPT="$PROMPT" NGEN="$NGEN" TEMP=0 SNAP="$DIR" RAM_GB=6 CTX=256 \
        qemu-x86_64-static -cpu IvyBridge "$BIN_DIR/glm-ivybridge" 4 8 8 2>&1); then
        gen=$(printf '%s\n' "$out" | sed -n '/^'"$(printf '%s' "$PROMPT" | sed 's/[.[\*^$/]/\\&/g')"'/p')
        if [ "$gen" = "$REF" ]; then log "  ok  ivybridge build runs under emulated Ivy Bridge, output matches $REF_ARCH"
        else log "  FAIL ivybridge build ran under qemu but output differs"; PASS=0; fi
    else
        log "  FAIL ivybridge build crashed under emulated Ivy Bridge (should run: AVX+SSSE3 only)"; PASS=0
    fi
fi

rm -rf "$BIN_DIR"
log ""
if [ "$KEEP" != 1 ]; then log "(real-model download cached at $DIR — delete with 'rm -rf $DIR' when done, ~$(du -sh "$DIR" 2>/dev/null | cut -f1))"; fi
if [ "$PASS" = 1 ]; then log "==== REAL MODEL VERIFIED ===="; else log "==== FAILED ===="; exit 1; fi
