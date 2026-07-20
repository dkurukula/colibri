#ifndef COLIBRI_REPIN_H
#define COLIBRI_REPIN_H

#include <math.h>
#include <stdint.h>

/* Load-aware gate for live re-pinning (arXiv:2607.10183 "ATSInfer", Algorithm 3).
 * A purely periodic trigger (every repin_n emitted tokens) is load-UNAWARE: it pays the
 * swap cost (disk reads, optional VRAM re-upload) on schedule even when nothing actually
 * changed, and reacts no faster to a real regression (disk contention, thermal throttling)
 * than to noise. This gate keeps repin_n as the minimum check interval (the paper's tau)
 * but only lets the swap through when the measured tok/s has drifted from a rolling
 * baseline by more than eps (the paper's epsilon; default 0.15, their own chosen value) —
 * eps=0 restores the old unconditional-every-N-tokens behavior.
 *
 * Pure function of its arguments: no globals, no Model, no disk I/O, so it's directly
 * unit-testable (see tests/test_repin.c) independent of the rest of the engine.
 *
 * toks_per_s: this turn's measured decode throughput — the TPOT-equivalent load signal.
 * last_repin/baseline/have_baseline: caller-owned state, updated in every branch (including
 * "checked but skipped") so the next call sees a consistent minimum-interval clock. */
static int repin_gate(uint64_t *last_repin, uint64_t n_emit, int repin_n,
                       double toks_per_s, double eps,
                       double *baseline, int *have_baseline){
    if(repin_n<=0) return 0;
    if(n_emit - *last_repin < (uint64_t)repin_n) return 0;
    if(toks_per_s>0 && eps>0){
        double tpot = 1000.0/toks_per_s;             /* ms/token, same unit as the paper's TPOT */
        if(!*have_baseline){ *baseline=tpot; *have_baseline=1; *last_repin=n_emit; return 0; }
        double dev = fabs(tpot-*baseline)/(*baseline);
        if(dev < eps){
            *baseline = 0.8*(*baseline) + 0.2*tpot;   /* EMA: still track slow drift */
            *last_repin = n_emit;                      /* rate-limit the next check, not just swaps */
            return 0;
        }
        *baseline = tpot;   /* re-baseline: about to act on this deviation */
    }
    *last_repin = n_emit;
    return 1;
}

#endif
