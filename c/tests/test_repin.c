#include <stdio.h>
#include "../repin.h"

static int fail(const char *message){
    fprintf(stderr,"repin gate test failed: %s\n",message);
    return 1;
}

int main(void){
    uint64_t last; double baseline; int have;

    /* off switch: repin_n<=0 never fires, never touches state */
    last=0; baseline=0; have=0;
    if(repin_gate(&last,100,0,25.0,0.15,&baseline,&have)) return fail("repin_n=0 should be off");
    if(last!=0 || have!=0) return fail("off switch touched state");

    /* below the minimum interval: no check yet, even with a huge deviation */
    last=0; baseline=0; have=0;
    if(repin_gate(&last,5,16,25.0,0.15,&baseline,&have)) return fail("fired before minimum interval");
    if(last!=0) return fail("interval gate advanced last_repin early");

    /* first check past the interval: bootstraps the baseline, does not swap yet */
    last=0; baseline=0; have=0;
    if(repin_gate(&last,16,16,25.0,0.15,&baseline,&have)) return fail("bootstrap should not swap");
    if(!have || baseline!=1000.0/25.0 || last!=16) return fail("bootstrap did not record baseline");

    /* small deviation (<15%) after a baseline exists: skip, but EMA-track and advance the clock */
    last=16; baseline=40.0; have=1;                 /* baseline tpot = 40 ms/tok (25 tok/s) */
    if(repin_gate(&last,32,16,23.0,0.15,&baseline,&have))  /* 1000/23 = 43.5ms, dev ~8.6% */
        return fail("small deviation should not swap");
    if(last!=32) return fail("skip-but-checked branch did not advance last_repin");
    if(baseline==40.0) return fail("EMA baseline did not move at all");

    /* large deviation (>=15%) after a baseline exists: fires, and re-baselines to current */
    last=32; baseline=40.0; have=1;
    double tpot_now=1000.0/15.0;                      /* 15 tok/s vs 25 tok/s baseline: ~40% slower */
    if(!repin_gate(&last,48,16,15.0,0.15,&baseline,&have)) return fail("large deviation should swap");
    if(last!=48) return fail("swap branch did not advance last_repin");
    if(baseline!=tpot_now) return fail("swap branch should re-baseline to the current tpot");

    /* eps=0 restores legacy behavior: unconditional swap once the interval elapses, even
     * with ~zero measured deviation (25 tok/s == the 40ms/tok baseline exactly) that
     * eps=0.15 would clearly have skipped */
    last=0; baseline=40.0; have=1;
    if(!repin_gate(&last,16,16,25.0,0.0,&baseline,&have)) return fail("eps=0 should swap even with no deviation");
    if(last!=16) return fail("eps=0 swap branch did not advance last_repin");

    /* toks_per_s<=0 (nothing emitted this turn) with eps>0: still swap once due, don't
     * corrupt the baseline with a division by a non-positive throughput */
    last=0; baseline=7.0; have=1;
    if(!repin_gate(&last,16,16,0.0,0.15,&baseline,&have)) return fail("toks_per_s<=0 should still swap when due");
    if(baseline!=7.0) return fail("toks_per_s<=0 must not perturb the baseline");
    last=0;
    if(!repin_gate(&last,16,16,-1.0,0.15,&baseline,&have)) return fail("negative toks_per_s should still swap when due");

    puts("repin gate tests: ok");
    return 0;
}
