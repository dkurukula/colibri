# Contributing

Keep changes focused and preserve Colibri's dependency-free default CPU path.

## Branches

- **`main`** is the stable branch. It's what users clone, and it stays known-good
  (engine always passes the token-exact oracle: `SNAP=./glm_tiny TF=1 ./glm 64 16 16`).
- **`dev`** is the integration branch. **Open your PR against `dev`.** Reviewed PRs
  land there first; once a batch is tested and stable, the maintainer fast-forwards
  it into `main`. This keeps `main` clean instead of taking every PR one at a time.

Every PR — on either branch — is reviewed for a clean build (0 warnings), the oracle
(32/32 TF + 20/20 greedy), and its own targeted validation before merge.

## Local checks

Run the lightweight checks locally:

```sh
make check
```

`make -C c check` remains available for scripts that already run from the
engine directory.

This performs one portable CPU build, C unit tests, and Python standard-library
tests. It does not download a model or require CUDA.

CUDA changes should additionally be checked on a CUDA-capable Linux host:

```sh
make -C c cuda-test CUDA_ARCH=native
```

Kernel/SIMD changes (anything touching the `__AVX2__`/`__AVX__`/`__SSSE3__`/
`__ARM_NEON` branches in `glm.c`) should additionally be checked with:

```sh
make bench-cpu-tiers
```

This builds `native`/`x86-64-v3`/`ivybridge`/`x86-64` (AVX-512, AVX2+FMA,
AVX-only+SSSE3, scalar), asserts all four agree on real forward passes against
the project's own tiny/medium random-weight oracles (generated on first run;
needs `pip install torch transformers safetensors`), and times the quantized
matmul kernel per tier. If `qemu-user-static` is installed it also runs an
extra check under an emulated Ivy Bridge CPU. See
`c/scripts/bench_cpu_tiers.sh` for details, including why a same-match-rate,
different-specific-token result on int4 is a pass (floating-point
non-associativity on an already-wrong argmax tie), not a failure.

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.
