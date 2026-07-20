<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — piccolo motore, modello immenso">
</p>

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ pronto in 32s · residente 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and only ~11 GB of those change from token to token (the routed experts). So:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays **resident in RAM at int4** (~9.9 GB);
- the **21,504 routed experts** (75 MoE layers × 256 experts + the MTP head, ~19 MB each at int4) live **on disk** (~370 GB) and are **streamed on demand**, with a per-layer LRU cache, an optional pinned hot-store, and the OS page cache as a free L2.

The engine is a single C file (`c/glm.c`, ~2,400 lines) plus small headers. No BLAS, no Python at runtime, no GPU required (an opt-in CUDA tier for pinned experts exists — see below).

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — validated token-exact against a `transformers` oracle (teacher-forcing 32/32, greedy 20/20 on a tiny-random model with the real architecture).
- **MLA attention** (q/kv-LoRA, interleaved partial RoPE) with **compressed KV-cache**: 576 floats/token instead of 32,768 (57× smaller — GLM-5.2 has 64 heads and no GQA).
- **DeepSeek-V3-style sigmoid router** (noaux_tc, routed_scaling_factor), shared expert, first-3-dense layers.
- **Native MTP speculative decoding** — GLM-5.2's own multi-token-prediction head (layer 78) drafts tokens that the main model verifies in one batched forward. **The head must be int8** (the converter does this by default): at int4 draft acceptance collapses to 0–4% and speculation never engages; at int8 it's 39–59% acceptance, **2.2–2.8 tokens/forward** (community-measured, [#8](https://github.com/JustVugg/colibri/issues/8)). Lossless — *and stays lossless under sampling* via rejection sampling. Honest caveat from the same measurement: on a **cold** cache each verified draft routes to extra experts (~660 → ~1100 expert-loads/token), so speculation can be a net *time* loss until the cache/pin warms up — the adaptive guard and `DRAFT=0` are there for that.
- **Grammar-forced speculative drafts** (`GRAMMAR=file.gbnf`, [#48](https://github.com/JustVugg/colibri/issues/48)) — on constrained-output workloads (JSON/NDJSON, function calling, structured extraction) the grammar itself is a third draft source: wherever it admits exactly **one** legal byte (braces, quotes, key names, enum bodies), that forced span is tokenized and injected as pre-accepted drafts with ~1.0 acceptance — no draft head, no lookup table, and it engages even with the int4 MTP head from [#8](https://github.com/JustVugg/colibri/issues/8). It never constrains sampling: forced spans are verified in the same batch-union forward as any draft, so a wrong or out-of-sync grammar cannot change the output — worst case is rejected drafts, and an adaptive guard turns the source off below 50% acceptance. Byte-level GBNF subset (literals, char classes, `| ( ) ? * +`, comments); `GRAMMAR_DRAFT=n` caps the forced span per forward (default 24). Composes with `DRAFT`/MTP, which fill the free-text gaps between forced spans.
- **True sampling** — temperature + nucleus, defaults tuned for int4 reality (0.7 / 0.90; the official 1.0 / 0.95 samples quantization noise from the tail).
- **Integer-dot kernels** (Q8_0-style int8 activations, AVX2 `maddubs`): int8 matmuls 1.4–2.5× faster (119 GFLOP/s measured), int4 1.8× in batch — routing decided per shape by measurement (int4 single-row stays f32: it measured slower).
- **MLA weight absorption** (DeepSeek trick) for decode: no per-token k/v reconstruction — the query absorbs `kv_b`, context is projected after attention. Validated exact: TF 32/32 and generation 20/20 with absorption forced everywhere.
- **Async expert readahead**: while one block of experts is being multiplied, the kernel is already reading the next (`WILLNEED`).
- **Quantization kernels**: int8 / packed int4 / packed int2, per-row scales, AVX2, dequant-on-use. Packing validated bit-identical to the int8 container.
- **DSA sparse attention** — GLM-5.2's lightning indexer, faithful to the reference `glm_moe_dsa` modeling: per-layer top-2048 causal key selection (full/shared indexer layers), auto-detected from the `out-idx-*` weights (`--indexer` converter mode, ~189 MB extracted from the FP8 repo). Validated exact: forcing the selection to keep every key reproduces dense attention token-for-token. `DSA=0` disables, `DSA_TOPK` overrides.
- **KV-cache persistence** — conversations reopen **warm** across engine restarts: serve mode appends the compressed MLA KV to `.coli_kv` after every turn (~182 KB/token, crash-safe) and resumes it at startup with zero re-prefill. Validated byte-identical to an uninterrupted session. `KVSAVE=0` disables.
- **Router-lookahead prefetch** (`PILOT=1`, experimental) — the next layer's routing is 71.6% predictable from the current layer's post-attention state (measured); a dedicated I/O thread prefetches those experts while the current layer computes.
- **Batch-union MoE**: in prefill (and MTP verification), each unique expert of the batch is read once and applied to every position that routes to it.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety**: the expert cache is auto-sized from `MemAvailable` at startup — an honest peak projection (working set, KV, MTP row, reconstruction buffers) so the kernel OOM-killer never fires.
- **Offline FP8→int4 converter** (`c/tools/convert_fp8_to_int4.py`): downloads one shard at a time (~5 GB), dequants (128×128 block scales), requantizes to the engine's container, deletes the shard — the 756 GB FP8 checkpoint never needs to exist on disk at once. Resumable.

## Honest numbers (WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (VHDX random) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation (int8 head) | 2.2–2.8 tok/forward measured ([#8](https://github.com/JustVugg/colibri/issues/8)) |

This is not fast. It is a 744B frontier-class model **answering correctly on a machine that costs less than one H100 fan**. Warm cache, pinned hot experts and MTP push the useful-response latency down considerably; the physics of the disk does the rest.

### SSD note
Cold starts are heavy on random reads (~11 GB/token), but reads don't meaningfully wear an SSD — colibrì's streaming is read-only. The real concerns under heavy use are (1) **swap traffic** if the system runs out of RAM (writes do wear the drive — keep a sane `--ram` budget; colibrì's auto-budget is designed to stay clear of swap) and (2) **sustained thermals**: hours at full read duty cycle will heat cheaper drives. Monitor drive temperature and health.

## Download the model

**Fastest path — one command** downloads the real 744B GLM-5.2 int4 model and
benchmarks it, handling the fiddly parts automatically:

- confirms (or lets you change) the download location
- checks free disk space *before* touching the network
- detects and fixes the int4-vs-int8 MTP head gotcha below (no manual file-size checking)
- builds for your CPU
- checks RAM/swap headroom before loading the model (more below)
- runs the full benchmark

```bash
cd c
make quickstart                    # interactive: confirms before downloading anything
make quickstart ARGS="-y"          # non-interactive: accepts every default (~/glm52_i4)
```

Useful flags via `ARGS="..."`: `--dir PATH` (download location, default
`~/glm52_i4`), `--arch ivybridge` (AVX-only CPUs — see "CPU tier" below),
`--ram N` (cap the engine's RAM budget in GB), `--skip-download` (model's
already there, just build+benchmark). See the full list with
`bash scripts/quickstart.sh --help`, or read `c/scripts/quickstart.sh` directly.

**Swap safety gate:** heavy swapping doesn't just slow colibrì down, it can
make the *whole machine* unresponsive. Before loading the model,
`make quickstart` checks total RAM, current swap usage, and any `--ram` you
passed. If total RAM is below the 16 GB floor, swap is already >40% full, or
`--ram` exceeds what's actually free right now, it explains why and offers a
choice instead of silently letting the box start thrashing: `[1]` use a
safer auto-computed budget (the default, and what `-y` applies
automatically), `[2]` continue unchanged anyway, or `[3]` abort.

### Manual download

A pre-converted **GLM-5.2 int4** model is also available directly on Hugging Face:

**https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4**

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download jlnsrk/GLM-5.2-colibri-int4 --local-dir ~/glm52_i4

COLI_MODEL=~/glm52_i4 ./coli chat
```

If the MTP files there are still the int4 head (see [#8](https://github.com/JustVugg/colibri/issues/8) — sizes `1765523544/2686077736/536747200` = int4, unusable), grab the **int8 MTP heads** from the community clone by matey-0: **https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp** (`make quickstart` above checks for and fixes this automatically).

This skips the FP8 → int4 conversion step entirely.

Thanks DatPat for your help!

### Quick start

```bash
cd c
./setup.sh                      # checks gcc/OpenMP, builds, self-tests

# ONE command does everything model-side: downloads GLM-5.2-FP8 shard by shard
# (never needs the full 756 GB at once), converts to the int4 container, then
# converts the MTP head for speculative decoding. Resumable at any point.
# Conversion (only) needs python with: pip install torch safetensors huggingface_hub numpy
./coli convert --model /nvme/glm52_i4     # ~400 GB free on a real ext4/NVMe path

# chat — RAM budget, expert cache and MTP are all detected automatically:
COLI_MODEL=/nvme/glm52_i4 ./coli chat
```

Inspect the planned storage hierarchy before loading the model:

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli plan
COLI_MODEL=/nvme/glm52_i4 ./coli plan --gpu 0,1 --ram 128 --vram 48 --json

# apply the bounded plan to the normal runner
COLI_MODEL=/nvme/glm52_i4 ./coli chat --auto-tier
```

`coli plan` reads only safetensors headers and reports the model's exact dense/expert
footprint, runtime RAM reserve, safe expert-cache cap, and bounded VRAM hot tier. Its
versioned JSON output is intended to be shared by the CLI, API server, Web UI, and
desktop shell; it does not allocate model tensors or start inference.
`--auto-tier` applies the same plan to `chat`, `run`, `serve`, and benchmarks. It
sets the RAM budget and context immediately; the VRAM tier is enabled only when
the current `glm` binary is linked with CUDA. Explicit flags and environment
variables keep precedence over automatic values.

Before loading the model, `coli doctor` performs a read-only readiness check and
explains whether the selected Disk/RAM/VRAM placement is runnable:

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli doctor
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --gpu 0 --ram 128 --json
```

Doctor validates the model directory, config, tokenizer, safetensors headers,
engine executable, available RAM, requested NVIDIA devices, CUDA linkage, and the
same placement budget used by `coli plan`. It never starts `glm`, reads tensor
payloads, imports a model framework, or creates a CUDA context. The versioned JSON
report uses stable check IDs for automation. Warnings keep exit status 0; missing
requirements or an unsafe RAM projection return 1, while invalid CLI values return 2.

The engine at runtime is pure C — python is only used by the one-time converter.

### Windows 11 (native, no WSL)

colibrì builds and runs natively on Windows 11 x86-64 with MinGW-w64. The port adds
a `_WIN32` compatibility layer in `c/compat.h` that maps POSIX I/O to the Windows API
(pread → ReadFile+OVERLAPPED, posix_fadvise no-op, aligned allocation, MoveFileEx rename,
GlobalMemoryStatusEx RAM detection). All platform differences stay in `compat.h`; the
engine source is unchanged.

**Toolchain:** GCC via [winlibs](https://winlibs.com/) or MSYS2 MinGW-w64. Tested with
GCC 16.1.0 (x86_64-ucrt-posix-seh).

```powershell
# One-time toolchain install (pick one):
scoop install mingw-winlibs                    # portable, no shell needed
# or: pacman -S mingw-w64-x86_64-gcc make     # via MSYS2

# Build (from c/ directory):
make glm.exe            # GLM-5.2 engine (static, no DLL dependencies)
make olmoe.exe          # OLMoE engine (same shims)
make iobench.exe        # disk I/O benchmark
make test-c             # run C tests
make test-python        # run Python tests (requires python)

# Verify (tiny model, 2.4 MB):
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                # generate tiny oracle
SNAP=./glm_tiny TF=1 ./glm.exe 64 16 16        # expect "32/32 posizioni"

# Run with real model:
SNAP=D:\glm52_i4 ./glm.exe 64 4 16            # batch inference
python coli chat --model D:\glm52_i4            # interactive chat
python coli serve --model D:\glm52_i4            # OpenAI-compatible API
```

**Status:** Phase 1 complete (compiles, correct, static-linked). O_DIRECT (Phase 2),
GPU via `LoadLibrary` on `coli_cuda.dll` (Phases G0–G2), and full-model validation
are separate workstreams. See `PORT_WINDOWS_PLAN.md` for the full plan.

### OpenAI-compatible API

`coli serve` keeps one model process loaded and exposes a text-only OpenAI-compatible
HTTP API. The gateway uses only the Python standard library; inference still runs in
the same dependency-free C engine.

```bash
cd c
COLI_MODEL=/nvme/glm52_i4 COLI_API_KEY=local-secret ./coli serve \
  --host 127.0.0.1 --port 8000 --model-id glm-5.2-colibri

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2-colibri",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Implemented endpoints are `GET /v1/models`, `GET /v1/models/{model}`,
`POST /v1/chat/completions`, and legacy `POST /v1/completions`. Chat and
completion requests support JSON responses, SSE streaming, usage counts,
`max_tokens`/`max_completion_tokens`, `temperature`, and `top_p`. The extension
`enable_thinking: true` enables GLM-5.2's reasoning block; the standard
`reasoning_effort` field also enables it unless set to `none`.

The first version is deliberately text-only and serves one generation at a time:
the 744B model stays in one persistent process, so concurrent HTTP requests queue
instead of loading duplicate model copies. Tools, image/audio input, custom stop
sequences, log probabilities, and token penalties return an explicit error rather
than being silently ignored. The default bind address is localhost; set
`COLI_API_KEY` before exposing the server beyond the machine.

Browser access from the Vite development server and Tauri local origins is enabled
by default. Repeat `--cors-origin https://your-ui.example` to allow another exact
origin, or use `--cors-origin '*'` only on a trusted local network.

The engine owns one mutable KV context, so HTTP generation uses a bounded FIFO
admission queue instead of pretending to run unsafe parallel sequences. Configure it
with `--max-queue N` (default 8) and `--queue-timeout SECONDS` (default 300), or the
`COLI_MAX_QUEUE` / `COLI_QUEUE_TIMEOUT` environment variables. Saturated and timed-out
requests receive OpenAI-shaped HTTP 429 errors before streaming headers are sent.
`GET /health` exposes active/queued/completed/rejected counters, and successful
generation responses include `x-colibri-queue-wait-ms`.

### Isolated KV contexts

`coli serve --kv-slots N` allocates up to 16 independent sequence contexts. Requests
select one with the optional integer `cache_slot` field; ordinary OpenAI clients omit
it and keep the original slot 0 behavior.

```json
{
  "model": "glm-5.2-colibri",
  "messages": [{"role": "user", "content": "Continue this conversation"}],
  "cache_slot": 1
}
```

Each slot owns its token history, compressed MLA/DSA KV memory, MTP window, and
crash-safe persistence file (`.coli_kv`, `.coli_kv.1`, ...). The engine still executes
one sequence at a time; this establishes explicit KV ownership without pretending that
threaded HTTP is continuous batching. RAM admission accounts for every configured slot.
Use `COLI_KV_SLOTS=N` as the environment equivalent. Start with a small value: at the
default 4096-token context, every slot costs hundreds of MB.

### Experimental resident CUDA backend

colibrì includes an opt-in CUDA backend for model-resident tensors. Streaming
experts deliberately remain on the original CPU path for now: copying an expert
from NVMe to the GPU on every use would only replace the disk bottleneck with a
PCIe bottleneck. Resident quantized tensors are uploaded lazily once and reused.

```bash
cd c
make cuda-test CUDA=1                  # q8/q4/q2/f32 kernel correctness
make CUDA=1
# optional dense-path experiment (hot experts are configured below)
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Requirements: Linux, an NVIDIA driver, and a CUDA Toolkit under
`/usr/local/cuda` (override with `CUDA_HOME=/path/to/cuda`). `CUDA_ARCH=native`
builds for the GPU in the current machine; set an explicit architecture when
cross-compiling. Requesting CUDA with a CPU-only binary, an invalid device, or
an unavailable runtime fails at startup instead of silently falling back.

The normal `make` build and runtime behavior are unchanged. CUDA defaults to an
expert-only accelerator: resident dense/attention tensors stay on CPU because
fixture measurements show that moving them does not help while expert I/O is
the bottleneck. `CUDA_DENSE=1` keeps the earlier all-resident experimental path.
A measured `PIN` profile can promote its hottest experts into the persistent
VRAM tier while keeping the rest in RAM:

```bash
STATS=stats.txt SNAP=/nvme/glm52_i4 ./glm 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
# multi-GPU expert tier, 96 GB total budget across six devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=96 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Selected experts are uploaded during startup, so capacity failures occur before
inference and the log reports their exact tensor footprint. The budget is clamped
against free VRAM after reserving the projected dense resident set and 2 GB of
runtime headroom per selected device. With `COLI_GPUS`, `CUDA_EXPERT_GB` is a
total budget across the device set; experts are assigned whole to the
least-loaded device that can hold them. A NUMA-local RAM backing store is not
implemented yet.

Current limitations: devices use independent contexts and synchronous
host-staged activation copies—there is no P2P/NCCL dependency yet. The kernels
are correctness-first custom kernels rather than cuBLAS/Tensor Core kernels.
This draft intentionally makes no end-to-end speedup claim before the full model
is benchmarked.

For a reproducible backend A/B without the full checkpoint, generate the
deterministic 313M-parameter `glm_moe_dsa` fixture and run fixed-token replay:

```bash
cd c
python tools/make_glm_bench_model.py --output /nvme/colibri-bench-medium --device cuda
python tools/benchmark_cuda_fixture.py --model /nvme/colibri-bench-medium --gpu 0
```

The fixture has random weights and is not a language model. It exists only to
preserve the real MLA/MoE/streaming shapes and compare CPU streaming, dense-only
CUDA, CPU hot-store, and CUDA hot-expert execution with identical replay tokens.

### Web interface

`web/` contains a community-contributed browser UI (React + TypeScript, ~390
lines of source, a pure API client — it never touches the engine directly):

```bash
cd web
npm ci && npm run dev        # then point it at an OpenAI-compatible endpoint
```

It speaks the standard OpenAI Chat Completions protocol with SSE streaming, so it
works against the colibrì OpenAI-compatible server (in review, #21) or any other
compatible endpoint. Nothing leaves the endpoint you configure. The terminal
`coli chat` remains the first-class interface.

Useful knobs (env or flags): `--temp T` token sampling temperature (default 0.7 + nucleus 0.90 — tuned for int4; 0 = greedy), `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens per answer (`:piu` in chat continues a truncated one), `--repin N` adapt RAM/VRAM hot experts every N emitted tokens, `AUTOPIN=0` disable the learning cache's auto-pin, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `GRAMMAR=g.gbnf` grammar-forced drafts for constrained JSON/NDJSON output (`GRAMMAR_DRAFT=n` caps the forced span), `TF=1` teacher-forcing validation, `PILOT=1` router-lookahead disk prefetch (experimental — see below), `CAP_RAISE=0` don't auto-grow the expert cache.

**The expert cache auto-sizes to your RAM** (since 2026-07-10): the engine now *raises* the LRU cap to fill your `--ram` budget instead of only lowering it. Before this fix a 128 GB machine ran with the same 8-experts/layer cache as a 16 GB one (issue #12) — **if you benchmarked colibrì before this date, rerun: your numbers were capped.**

**Router-lookahead prefetch** (`PILOT=1`, experimental): GLM-5.2's expert routing is measurably predictable *ahead of time* — applying layer L+1's router to layer L's post-attention state recalls **71.6%** of the true top-8 (vs 41.3% for "same experts as last token"). `PILOT=1` uses this to issue next-layer expert readahead from a dedicated I/O thread while the current layer computes. On our dev box the disk is already ~80% saturated, so it measures neutral; on machines where compute and disk are balanced (like the Ryzen AI 9 in issue #12: 43% disk / 46% matmul) it should overlap real work — measurements welcome.

**The learning cache**: the engine records which experts your usage actually routes to (`.coli_usage` next to the model, updated every turn) and at startup automatically pins the hottest ones in spare RAM. colibrì literally gets faster the more you use it.

**Live tier adaptation** (`--repin N`, opt-in): at safe turn boundaries, a decaying
session heat map replaces cold pinned experts with hotter streamed experts. Replacement
loads the expert from disk into the existing RAM slot; GPU-backed slots immediately
refresh the same VRAM tier budget. A 25% hysteresis and a four-swap limit prevent tier
thrashing. Persistent `.coli_usage` remains the long-term signal and is not decayed.
`--repin N` now also gates the swap on measured load, not just token count (concept from
["Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices"](https://arxiv.org/abs/2607.10183),
arXiv:2607.10183 — its Algorithm 3): N is the minimum check interval, but the engine only
pays the disk cost of a swap when this turn's measured tok/s has drifted from a rolling
baseline by more than `REPIN_EPS` (default 0.15, the paper's own chosen threshold) —
otherwise it just tracks the drift and rechecks later. This also fixes the swap pass
never firing on ordinary (non-continuation) turns — expect up to a few hundred ms of
disk-bound latency on the turn that triggers a swap, exactly as `--repin` always
documented, now actually happening for normal chat instead of only the rare truncated-
response continuation. `REPIN_EPS<=0` restores the old unconditional-every-N-tokens
behavior. The baseline needs two consistent readings before it's trusted (a single noisy
first sample shouldn't anchor every future check), so the earliest possible swap moves
from N tokens to roughly 2N; each `--kv-slots` conversation tracks its own baseline, so
one slot's normal throughput is never misread as a regression relative to another's.

**Conversations reopen warm** (`.coli_kv`, since 2026-07-10): `coli chat` persists the compressed MLA KV-cache to disk after every turn (~182 KB/token, appended incrementally, crash-safe). Close the chat, reopen it tomorrow — the model still remembers the whole conversation and **zero re-prefill happens**: validated byte-identical to an uninterrupted session. `:reset` clears it, `KVSAVE=0` disables it.

## Got a better machine? Try it — here's what to expect

colibrì was built on deliberately humble hardware (12 cores, 25 GB RAM, NVMe behind a WSL2 VHDX that caps random reads at ~1 GB/s). **Every one of those constraints is a knob your machine can turn up.** The engine needs: Linux (or WSL2), macOS, or **Windows 11 natively (MinGW-w64)**; gcc with OpenMP, ≥16 GB RAM, and the ~370 GB int4 model on a local NVMe (ext4/NTFS — never a network/9p mount).

### CPU tier: pick the build that matches your vector ISA

> **Fork note:** this repository is a fork of [JustVugg/colibri](https://github.com/JustVugg/colibri).
> Upstream colibrì assumes an AVX2 CPU (Haswell, 2013+) and falls back to plain
> scalar C otherwise — that's a deliberate upstream choice, not a bug, and it is
> **not** something the original author designed around AVX-only hardware for.
> This fork adds the AVX-only code path below (Sandy/Ivy Bridge, 2011-2012:
> vectorized AVX float matmuls + SSSE3 integer dot product, no AVX2/FMA) so the
> engine also runs vectorized — not scalar — on that older hardware.

| CPU | build | matmul kernel | IDOT kernel |
|---|---|---|---|
| Haswell+ (2013+), AVX2/FMA | `make` (`ARCH=native`, default) | AVX2 FMA | avx2 |
| **Sandy/Ivy Bridge (2011-2012), AVX only** | `make ARCH=ivybridge` | AVX (no FMA) | ssse3 |
| Anything older (SSE-only) | `make ARCH=x86-64` | scalar | scalar |
| Skylake-X+ with AVX-512 VNNI | `make` (`ARCH=native` on that CPU) | AVX2 FMA | avx512-vnni |

```bash
# Ivy Bridge / Sandy Bridge / any AVX-without-AVX2 CPU — same engine, vectorized
# kernels instead of the scalar fallback (256-bit AVX for the float matmuls,
# 128-bit SSSE3 for the integer IDOT dot product):
cd c
make ARCH=ivybridge && ./glm            # native to *this* machine

# building a binary to hand to another AVX-only machine (no AVX2/FMA assumed):
make portable-avx                       # ARCH=sandybridge under the hood
```

`ARCH=native` already autodetects this correctly on real Ivy Bridge/Sandy Bridge
hardware — `ARCH=ivybridge` only matters when cross-building a binary for a
different, AVX-only machine. Startup logs the kernel actually picked
(`idot: avx2` / `idot: ssse3` / `idot: scalar` / `idot: avx512-vnni`).

### Running GLM-5.2 on Ivy Bridge hardware

The model itself doesn't care which CPU tier built the engine — it's the same
744B GLM-5.2 int4 checkpoint either way. What you need, concretely:

| requirement | how much | why |
|---|---|---|
| disk (local ext4/NTFS, **not** `/mnt/c` or network/9p) | **~370 GB** for the int4 model, **~400 GB free** while converting locally (deletes each FP8 shard as it goes) | streamed on demand at inference time |
| RAM | **16 GB floor**, 25 GB+ comfortable | ~9.9 GB dense stays resident; the rest is expert cache — more RAM = higher hit-rate = fewer disk reads/token. Ivy Bridge-era machines are exactly the ones most likely to be RAM-constrained too — `make quickstart`'s swap safety gate (below) catches that before it makes the machine unresponsive. |
| CPU | any Sandy/Ivy Bridge-class x86-64 (2011+, has AVX) | gets the vectorized AVX+SSSE3 kernels below instead of the scalar fallback |

The single-command path from "Download the model" above already does all of
this correctly on Ivy Bridge — just tell it which `ARCH` to build:

```bash
cd c
make quickstart ARGS="--arch ivybridge -y"    # download + build (AVX-only) + benchmark, one shot
```

Or step by step, if you'd rather see/control each part:

```bash
cd c
make ARCH=ivybridge                     # AVX-only build (no AVX2/FMA required)

# get the model — pick ONE (default path ~/glm52_i4; override if you keep models elsewhere):
pip install -U "huggingface_hub[cli]"                              # (a) pre-converted, skips conversion
huggingface-cli download jlnsrk/GLM-5.2-colibri-int4 --local-dir ~/glm52_i4
# — or —
./coli convert --model ~/glm52_i4                                  # (b) convert FP8 yourself, needs ~400 GB free

# run it — RAM budget, expert cache and MTP are all detected automatically:
COLI_MODEL=~/glm52_i4 ./coli chat --ram 16     # set --ram to whatever you actually have free
```

Expect it to be **disk-bound, not CPU-bound** at this scale (see the
back-of-envelope table below). The AVX kernels mainly matter for the matmul
share of that time — roughly 2× the scalar fallback on int8, ~6× on int4
(see the reproducible benchmark table above) — and for keeping the engine
off the "illegal instruction" crash an AVX2-only build would hit on this
hardware.

Reproduce this: `make bench-cpu-tiers` builds all four tiers, runs real forward
passes (prefill + autoregressive decode) against generated tiny/medium
random-weight fixtures, asserts every tier agrees, and prints the kernel
timing table above (see `c/scripts/bench_cpu_tiers.sh`; one-time need for
fixture generation: `pip install torch transformers safetensors`). With
`qemu-user-static` installed it additionally proves the Ivy Bridge build
actually runs under an emulated Ivy Bridge CPU while an AVX2 build SIGILLs
under the same CPU model.

### Benchmark the full model

`make quickstart` (see "Download the model" above) already downloads the
model *and* runs this benchmark in one shot — this section is for benchmarking
a model you already have, or for running each step individually with more
control (e.g. a higher `--limit` on the quality suite).

One copy-paste block, four steps, using the **real 744B GLM-5.2 model** (not a
fixture) end to end: build for your CPU, measure your disk the way the engine
actually reads it, measure decode speed, then run the quality suite. Set
`MODEL` and `ARCH` once at the top and the rest is unchanged for any machine
— `ARCH=ivybridge` (or `sandybridge`) if that's your CPU tier, `native`
otherwise.

```bash
cd c
MODEL=~/glm52_i4                             # your downloaded/converted int4 model (see "Download the model" above)
ARCH=native                                  # or: ivybridge / sandybridge / x86-64-v3

# 0) build + architecture self-test (expects 32/32):
ARCH="$ARCH" ./setup.sh

# 1) measure YOUR disk the way the engine uses it (parallel 19 MB random reads):
gcc -O2 -fopenmp iobench.c -o iobench
./iobench "$MODEL"/out-00069.safetensors 19 64 8 0   # buffered, 8 threads
./iobench "$MODEL"/out-00069.safetensors 19 64 8 1   # O_DIRECT

# 2) chat; watch the per-turn stats line (tok/s, expert hit-rate, RSS):
COLI_MODEL="$MODEL" ./coli chat

# 3) record expert usage, then pin the hottest experts in your spare RAM:
COLI_MODEL="$MODEL" STATS=stats.txt ./coli chat
COLI_MODEL="$MODEL" PIN=stats.txt PIN_GB=20 ./coli chat        # scale PIN_GB to your free RAM

# 4) quality benchmarks (MMLU/HellaSwag/ARC):
COLI_MODEL="$MODEL" ./coli bench
```

Report your numbers (machine, `ARCH`, disk, RAM, tok/s, hit-rate) in an issue
— see the [community benchmarks](#community-benchmarks-measured) below for
the format other people have used.

**Back-of-envelope predictions** (decode is disk-bound: a cold token costs ~11.4 GB of expert reads; MTP speculation roughly halves the effective cost *once the cache is warm*; RAM turns cold reads into free cache hits):

| machine | expected |
|---|---|
| this dev box (WSL2 VHDX, ~1 GB/s, 25 GB RAM) | ~0.05–0.1 tok/s cold — proven baseline |
| native Linux, PCIe4 NVMe (~3–5 GB/s random), 32 GB | ~0.5–1 tok/s |
| PCIe5 NVMe or 2×NVMe RAID0 (~8–12 GB/s), 64 GB (PIN ~40 GB of hot experts) | ~2–4 tok/s |
| 128–256 GB RAM, 12 cores (hot experts cached) | ~2–4 tok/s — matmul-bound: ~80 GFLOP/token vs ~250 GFLOP/s of our AVX2 kernels |
| same RAM + 24–32 cores, or AVX-512/VNNI kernels | ~5–15 tok/s — interactive; kernel work is the multiplier |

These are estimates, not measurements — if you run colibrì on serious hardware, **please open an issue with your numbers**: real datapoints from better machines are exactly what this project needs next.

### Community benchmarks (measured)

Real numbers from real machines, stock build (`setup.sh`, gcc 13), greedy decoding, `--ngen 32`, MTP active:

| machine | disk (iobench, 19 MB × 64, 8 threads) | config | measured |
|---|---|---|---|
| Intel Core Ultra 7 270K Plus (24 threads) · WSL2 · 24 GB RAM · NVMe VHDX ([#2](https://github.com/JustVugg/colibri/issues/2)) | 1.96 GB/s buffered · 2.74 GB/s O_DIRECT | default | 0.07 tok/s · expert hit 3–4% · RSS 14.1 GB |
| 〃 | 〃 | `--topp 0.7` | **0.11 tok/s** · expert hit 11% · RSS 14.7 GB |
| Apple M5 Max (18 cores) · macOS · 128 GB unified · internal SSD ([#4](https://github.com/JustVugg/colibri/issues/4), [#5](https://github.com/JustVugg/colibri/issues/5)) | 14.2 GB/s O_DIRECT | default, MTP off | **1.06 tok/s** · expert hit 23% · RSS 21.8 GB |
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVME SSD | — | `MTP=1 DIRECT=1` | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |
| Ryzen AI 9 HX 370 (Framework 13) · Arch Linux · 128 GB · WD SN850X, BTRFS zstd ([#12](https://github.com/JustVugg/colibri/issues/12)) | — | int8 MTP head · `--cap 32` · 46.7 GB auto-learned PIN | **0.37 tok/s** · expert hit 66% · MTP acceptance 52% (2.59 tok/fw) · RSS 105 GB |
| Ryzen 9 9950X (32 threads) · Linux · 123 GB · Crucial P3 QLC Gen3 ([#31](https://github.com/JustVugg/colibri/issues/31)) | 1.51 GB/s buffered | default, 2 runs from cold | 0.10 tok/s · hit 53% · profile 66% disk |
| 〃 same machine, model moved to a Samsung 9100 PRO PCIe 5.0 ([#31](https://github.com/JustVugg/colibri/issues/31)) | **8.81 GB/s** O_DIRECT | 〃 (usage history retained) | **0.28 tok/s** · hit 57% · profile flips: 32% disk / **57% matmul** |
| Ryzen AI Max+ 395 (Framework Desktop) · Ubuntu · 128 GB LPDDR5x · Intel Optane 905p PCIe 3.0 ([#39](https://github.com/JustVugg/colibri/issues/39)) | 3.27 GB/s buffered | int8 MTP head · fresh history (pure LRU, auto-raised cap 65) | 0.16 tok/s · hit 57% · profile 49% disk / 47% matmul |
| 〃 five runs later — learned pin 47.6 GB ([#39](https://github.com/JustVugg/colibri/issues/39)) | 〃 | `--temp 0.7 --topp 0.7` | **0.40 tok/s** · hit 71% · fastest non-Apple datapoint |
| **Dell PowerEdge R720 · Linux · Ivy Bridge (AVX, no AVX2/FMA) · 134.6 GB RAM** | — | `CAP_RAISE=0 ./coli run --ram 50 --cap 8` | 0.12 tok/s · expert hit 11.6% · RSS 24.54 GB |
| 〃 quality benchmark | — | `./coli bench --ram 50 --cap 8` | `hellaswag` 30.0% acc / 50.0% acc_norm · `arc_challenge` 70.0% acc / 60.0% acc_norm · `mmlu` 50.0% acc / 50.0% acc_norm · **MEDIA acc_norm 53.3%** · score wall 16137s · RSS 22.02 GB · expert hit 1% · 120 requests / 16121.6s (~0.0074 req/s) |

Takeaways: with 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so decode stays cold even on a disk 2–2.7× faster than the dev box — **on small-RAM machines the RAM cap, not the disk, is the binding constraint**, exactly as the table above predicts; `--topp 0.7` alone bought a clean 1.6× end-to-end speedup. The M5 Max datapoint lands right on the table's second row: **~1 tok/s of a 744B model on a laptop SSD** — and its 14 GB/s disk shifts the bottleneck back to RAM budget and kernels. The Framework 13 rows are the cache thesis proven end-to-end on one machine: 0.29 → 0.37 tok/s (hit 28% → 66%, speculation finally engaging at 52% acceptance) just by giving the cache its RAM — int8 MTP head + a bigger cap + the learned pin. The cap part is now automatic (cap auto-raise, 2026-07-10). The 9950X pair is the cleanest bottleneck experiment yet — same machine, same history, only the disk swapped: ×5.8 disk bandwidth bought ×2.9 tokens, and the profile **flipped from 66% disk to 57% matmul**. Past ~5 GB/s the disk stops being the story and the CPU (or the CUDA expert tier) becomes it.

The R720 rows are the first real-hardware confirmation of this fork's AVX-only kernels (built with `ARCH=ivybridge`, `idot: ssse3`) on genuine Ivy Bridge silicon — see `Containerfile.r720` for the exact reproducible setup (`make quickstart ARGS='-y --dir /model --arch ivybridge --ram 48'`). 134 GB of RAM couldn't lift the 1% expert hit-rate on the quality run, because the benchmark harness sweeps one forward per answer option — a different access pattern from decode's single generation path, so its low hit-rate isn't comparable to the decode row above it. At 0.12 tok/s decode, this machine is disk-bound rather than CPU-bound, same as the Ivy Bridge section above predicts — its ~13-year-old spinning/SAS storage, not the AVX kernels, is the ceiling here.

## Quality benchmark — help wanted

We have never measured how much the int4 quantization costs in accuracy — the harness is built and wired, but scoring is one forward per answer option, and on the dev box's ~1 GB/s disk a full run takes the better part of a day. **This is the single most valuable thing a faster machine can contribute.** The code is here and ready; one command runs it end to end (it auto-downloads the datasets on first use):

```bash
cd c
./coli bench                                   # hellaswag, arc_challenge, mmlu — 40 questions each
./coli bench hellaswag --limit 200             # one task, more questions
./coli bench mmlu arc_challenge --ram 100      # pick tasks, set a RAM budget
```

It prints per-task accuracy (log-likelihood scoring, EleutherAI-harness style). Published full-precision GLM-5.2 scores on these tasks sit around 85–95%; if our int4 container lands within a few points, the quantization is validated — if it doesn't, we know to invest in mixed / grouped-scale quantization. **If you have the hardware to run this, please open an issue with the numbers** — it's the measurement the project is missing.

## Supporting the project

colibrì is a one-person project, written and tested entirely on a 12-core laptop with 25 GB of RAM — the numbers above are the ceiling of what I can measure at home. If this project is useful or interesting to you and you'd like to support its development (better test hardware translates *directly* into a faster engine for everyone: real NVMe scaling data, bigger pinned caches, int2/int3 quality sweeps on real benchmarks), you can:

- ⭐ star the repo and share it;
- 🐛 open issues with benchmark numbers from your hardware;
- 💬 reach out via GitHub issues if you'd like to sponsor development or donate hardware.

Every contribution, from a datapoint to a disk, moves the ceiling.

## Repo layout

```
Makefile                  root build/check entry point
c/
├── glm.c                 single-file GLM engine
├── st.h, tok.h, json.h   runtime headers
├── backend_cuda.*        optional CUDA tier
├── Makefile              build and local checks
├── coli                  user-facing CLI
├── openai_server.py      OpenAI-compatible HTTP gateway
├── setup.sh              one-command local setup
├── tools/                offline conversion, fixtures and benchmarks
├── scripts/              long-running conversion helpers
└── tests/                dependency-free C and Python tests
web/                      browser UI (pure OpenAI-API client, community-maintained)
```

The runtime path intentionally stays flat and readable: `glm.c` plus its small
headers. Auxiliary Python and shell tooling is grouped separately and is never a
runtime dependency of the engine.

From the repository root, `make`, `make check`, and `make clean` delegate to the
engine Makefile. Existing commands run from `c/` continue to work unchanged.

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand flowers a day. This engine keeps a 744-billion-parameter giant alive on hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
