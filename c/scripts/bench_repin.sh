#!/usr/bin/env bash
# Reproducible benchmark for the load-aware live re-pin gate (arXiv:2607.10183
# "ATSInfer", Algorithm 3; see repin.h / README "Live tier adaptation").
#
# The old --repin policy swapped experts back into the RAM pin table every N
# emitted tokens, unconditionally, even when nothing had actually changed —
# paying real disk-I/O cost (and, on GPU builds, VRAM re-upload cost) on
# schedule instead of only when throughput actually regresses. REPIN_EPS>0
# gates that swap on a real deviation from a rolling tok/s baseline instead.
#
# This script proves the gate does what it claims WITHOUT a live serve
# session or the real 744B checkpoint: it uses REPLAY mode's REPIN_BENCH=<n>
# hook to chop a deterministic token replay into fixed windows, feed each
# window's real measured tok/s through the exact same repin_pass() a chat
# turn would use, and count how many windows actually paid a disk-read swap
# — comparing REPIN_EPS=0 (legacy, unconditional) against REPIN_EPS=0.15
# (load-aware, the project's default) on the IDENTICAL token sequence.
#
# Usage (from c/ or anywhere):
#   scripts/bench_repin.sh
#
# Fixtures are generated on first run with the project's own tooling and then
# reused (glm_bench_medium/ — gitignored). One-time need for fixture
# generation only:
#   pip install torch transformers safetensors
#
# Tunables (env):
#   ARCH=native      CPU tier to build (see bench_cpu_tiers.sh for the list)
#   WINDOW=20        tokens per re-pin check window (REPIN_BENCH=$WINDOW)
#   REPIN_N=15       min tokens between checks (REPIN=$REPIN_N); must be
#                    <= WINDOW so every window is actually eligible to check
#   PIN_GB=0.02      RAM pin budget: small on purpose, so the gate's decision
#                    always has a real disk-backed swap available to make
#   N_TOKENS=600     length of the synthetic replay sequence
#   REPS=5           trials per REPIN_EPS value (median swap count reported)
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE"

ARCH="${ARCH:-native}"
WINDOW="${WINDOW:-20}"
REPIN_N="${REPIN_N:-15}"
PIN_GB="${PIN_GB:-0.02}"
N_TOKENS="${N_TOKENS:-600}"
REPS="${REPS:-5}"

log(){ printf '%s\n' "$*"; }

# ---------- 1) fixture: medium (313M) benchmark model, reused if present ----------
if [ ! -d glm_bench_medium ]; then
    log "[fixtures] generating medium (313M) benchmark fixture (tools/make_glm_bench_model.py)..."
    python3 tools/make_glm_bench_model.py --output glm_bench_medium >/tmp/make_glm_bench_model.log 2>&1 || {
        tail -30 /tmp/make_glm_bench_model.log >&2
        echo "need: pip install torch transformers safetensors" >&2
        exit 1
    }
fi

# ---------- 2) a longer synthetic replay reference ----------
# REPLAY mode doesn't check correctness (no oracle comparison, just real
# forward passes), so the continuation tokens only need to be valid vocab
# ids — they don't need to come from an actual generation. The short
# ref_glm.json (prompt + 8 tokens) that make_glm_bench_model.py ships is
# nowhere near enough to fill several REPIN_BENCH windows; this extends the
# same real prompt with a longer, deterministically-seeded random tail.
LONG_REF="glm_bench_medium/ref_repin_bench.json"
if [ ! -f "$LONG_REF" ]; then
    log "[fixtures] generating ${N_TOKENS}-token synthetic replay reference..."
    N_TOKENS="$N_TOKENS" python3 - "$LONG_REF" <<'PY'
import json, os, random, sys
out = sys.argv[1]
ref = json.load(open("glm_bench_medium/ref_glm.json"))
cfg = json.load(open("glm_bench_medium/config.json"))
vocab = cfg["vocab_size"]
prompt = ref["prompt_ids"]
n_new = int(os.environ["N_TOKENS"])
rng = random.Random(1234)
full = list(prompt) + [rng.randrange(vocab) for _ in range(n_new)]
json.dump({"prompt_ids": prompt, "full_ids": full}, open(out, "w"))
PY
fi

# ---------- 3) build ----------
log "[build] ARCH=$ARCH"
make -s glm ARCH="$ARCH"

# ---------- 4) real expert-usage stats, so PIN picks a plausible hot set ----------
# STATS=<file> is only dumped by the default (non-TF/REPLAY/SERVE) validate-
# against-oracle path, so this deliberately runs plain (no TF=1): a real
# generate() pass over the fixture's own ref_glm.json populates m->eusage
# per layer/expert, which pin_load() below then ranks by frequency.
PIN_STATS="glm_bench_medium/pin_repin_bench.stats"
if [ ! -f "$PIN_STATS" ]; then
    log "[fixtures] generating expert-usage stats (real pass, for PIN)..."
    REF="$CODE/glm_bench_medium/ref_glm.json" STATS="$CODE/$PIN_STATS" \
        SNAP=glm_bench_medium ./glm 32 8 8 >/tmp/bench_repin_stats.log 2>&1 || {
        tail -30 /tmp/bench_repin_stats.log >&2
        exit 1
    }
fi

# ---------- 5) run the REPIN_EPS sweep ----------
trial(){
    local eps="$1"
    REF="$CODE/$LONG_REF" REPLAY=1 REPIN_BENCH="$WINDOW" REPIN="$REPIN_N" REPIN_EPS="$eps" \
        PIN="$CODE/$PIN_STATS" PIN_GB="$PIN_GB" SNAP=glm_bench_medium ./glm 32 8 8 2>&1 \
        | grep -oE '^REPIN bench: [0-9]+/[0-9]+ windows swapped' | grep -oE '[0-9]+/[0-9]+' || true
}
median_swaps(){
    local eps="$1" i out swapped windows swaps=()
    for i in $(seq 1 "$REPS"); do
        out=$(trial "$eps")
        [ -z "$out" ] && { echo "no REPIN bench output (window=$WINDOW too small vs N_TOKENS=$N_TOKENS?)" >&2; exit 1; }
        swapped="${out%%/*}"; windows="${out##*/}"
        swaps+=("$swapped")
    done
    local sorted=($(printf '%s\n' "${swaps[@]}" | sort -n))
    local mid=$(( ${#sorted[@]} / 2 ))
    printf '%s %s\n' "${sorted[$mid]}" "$windows"
}

log ""
log "=== repin gate: REPIN_EPS=0 (legacy, unconditional) vs 0.15 (load-aware) ==="
log "    window=$WINDOW tok, min-interval=$REPIN_N tok, PIN_GB=$PIN_GB, ${REPS}x trials, median reported"
read legacy_swaps windows <<<"$(median_swaps 0)"
read aware_swaps  windows <<<"$(median_swaps 0.15)"

log ""
printf '| REPIN_EPS | windows swapped (median of %s) |\n|---|---|\n' "$REPS"
printf '| 0    (legacy)     | %s/%s |\n' "$legacy_swaps" "$windows"
printf '| 0.15 (load-aware) | %s/%s |\n' "$aware_swaps" "$windows"
log ""
if [ "$legacy_swaps" -gt 0 ]; then
    reduction=$(( 100 * (legacy_swaps - aware_swaps) / legacy_swaps ))
    log "load-aware gate avoided ${reduction}% of the disk-I/O swaps legacy paid on the identical token sequence"
else
    log "legacy fired 0 swaps in this configuration — widen WINDOW/N_TOKENS or shrink REPIN_N to get a meaningful comparison"
fi
if [ "$aware_swaps" -ge "$legacy_swaps" ]; then
    log "WARNING: load-aware did not swap less than legacy — this fixture's routing may be too uniform to show a deviation; see script header tunables"
    exit 1
fi
