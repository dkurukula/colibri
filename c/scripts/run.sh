#!/usr/bin/env bash
# GLM-5.2 pipeline (int4, streaming, 15 GB RAM) — all inside WSL, model on ext4.
#   usage from c/:  scripts/run.sh ["prompt"] [n_token]
# Does: (1) waits for the move to ext4, (2) resumes the conversion until it's done,
#     (3) builds the engine, (4) generates text while staying within the RAM budget.
set -euo pipefail

DIR="${COLI_MODEL:-/home/vincenzo/glm52_i4}"          # int4 model on ext4 (NOT /mnt/c!)
REPO="${COLI_REPO:-zai-org/GLM-5.2-FP8}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAM_GB="${RAM_GB:-15}"
PROMPT="${1:-Hi, who are you?}"
NGEN="${2:-64}"

cd "$CODE"

# 0) sanity: the model must be on ext4, not on 9p/Windows
case "$DIR" in /mnt/*) echo "ERROR: $DIR is on /mnt (9p/Windows). Put it on ext4."; exit 1;; esac

# 1) if a move-rsync is still alive, wait for it
while pgrep -f "rsync.*glm52_i4" >/dev/null 2>&1; do
    echo "[1/4] waiting for the move to ext4... ($(du -sh "$DIR" 2>/dev/null | cut -f1))"; sleep 20
done
echo "[1/4] move complete: $(du -sh "$DIR" | cut -f1), shards $(ls "$DIR"/*.safetensors 2>/dev/null | wc -l)"

# 2) resumes+completes the conversion (resumable: skips shards already done)
echo "[2/4] converting (resumes where it left off): output -> $DIR"
python3 tools/convert_fp8_to_int4.py --repo "$REPO" --outdir "$DIR" --ebits 4 --io-bits 8

# 3) the engine needs tokenizer.json + config.json in the model dir
for f in config.json tokenizer.json; do
    [ -f "$DIR/$f" ] || { echo "ERROR: missing $DIR/$f"; exit 1; }
done
echo "[3/4] building the engine"; make -s glm

# 4) real generation, with auto-cap from the RAM budget and RSS heartbeat on stderr
echo "[4/4] generating (RAM_GB=$RAM_GB, NGEN=$NGEN)"; echo "------"
SNAP="$DIR" RAM_GB="$RAM_GB" PROMPT="$PROMPT" NGEN="$NGEN" ./glm 64
