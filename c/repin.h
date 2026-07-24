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
 * baseline by more than eps (the paper's epsilon; default 0.15, their own chosen value).
 * eps<=0 restores the old unconditional-every-N-tokens behavior outright — that is an
 * explicit "gate disabled" request, distinct from toks_per_s<=0 (this turn produced no
 * usable throughput reading, e.g. a 0-token turn near the context limit) which the gate
 * treats as inconclusive and skips, NOT as evidence to swap on.
 *
 * warm: samples seen since the baseline was last (re)established. 0 = none yet, 1 = one
 * sample (averaged into baseline, not yet trusted), >=2 = trusted, deviation checks begin.
 * A single noisy first sample would otherwise anchor every future check, so two samples
 * are averaged before the gate starts comparing against it. Firing a swap re-enters this
 * same warm-up (baseline is set to the just-measured value but warm resets to 1, not
 * "trusted") instead of hard-committing to that one (possibly still-transient) reading —
 * deviation is direction-agnostic (fabs), so without this, a recovery right after the swap
 * would itself look like a second deviation and fire again.
 *
 * Pure function of its arguments: no globals, no Model, no disk I/O, so it's directly
 * unit-testable (see tests/test_repin.c) independent of the rest of the engine. Caller owns
 * last_repin/baseline/warm and should keep them scoped per independent workload (e.g. per
 * KV/conversation slot) — a single shared instance across unrelated concurrent sessions
 * would read one session's throughput as a deviation from another's baseline. */
static int repin_gate(uint64_t *last_repin, uint64_t n_emit, int repin_n,
                       double toks_per_s, double eps,
                       double *baseline, int *warm){
    if(repin_n<=0) return 0;
    if(n_emit - *last_repin < (uint64_t)repin_n) return 0;
    *last_repin = n_emit;              /* interval elapsed: every path below counts as "checked" */

    if(eps<=0) return 1;               /* gate explicitly disabled: legacy unconditional swap */
    if(toks_per_s<=0) return 0;        /* no usable reading this turn: inconclusive, not "swap" */

    double tpot = 1000.0/toks_per_s;   /* ms/token, same unit as the paper's TPOT */
    if(*warm<2){
        *baseline = (*warm==0) ? tpot : 0.5*(*baseline) + 0.5*tpot;
        (*warm)++;
        return 0;
    }
    double dev = fabs(tpot-*baseline)/(*baseline);
    if(dev < eps){
        *baseline = 0.8*(*baseline) + 0.2*tpot;   /* EMA: still track slow drift */
        return 0;
    }
    *baseline = tpot; *warm = 1;       /* fire: re-enter warm-up, see comment above */
    return 1;
}

#endif
