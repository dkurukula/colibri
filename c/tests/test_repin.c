#include <stdio.h>
#include "../repin.h"

static int fail(const char *message){
    fprintf(stderr,"repin gate test failed: %s\n",message);
    return 1;
}

int main(void){
    uint64_t last; double baseline; int warm;

    /* off switch: repin_n<=0 never fires, never touches state */
    last=0; baseline=0; warm=0;
    if(repin_gate(&last,100,0,25.0,0.15,&baseline,&warm)) return fail("repin_n=0 should be off");
    if(last!=0 || warm!=0) return fail("off switch touched state");

    /* below the minimum interval: no check yet, even with a huge deviation */
    last=0; baseline=0; warm=0;
    if(repin_gate(&last,5,16,25.0,0.15,&baseline,&warm)) return fail("fired before minimum interval");
    if(last!=0) return fail("interval gate advanced last_repin early");

    /* eps<=0: legacy unconditional swap once due, regardless of toks_per_s or existing
     * baseline/warm state (neither is read nor touched) */
    last=0; baseline=99.0; warm=1;
    if(!repin_gate(&last,16,16,3.0,0.0,&baseline,&warm)) return fail("eps<=0 should swap unconditionally when due");
    if(last!=16) return fail("eps<=0 branch did not advance last_repin");
    if(baseline!=99.0 || warm!=1) return fail("eps<=0 branch should not touch baseline/warm");
    last=0;
    if(!repin_gate(&last,16,16,-1.0,-0.5,&baseline,&warm)) return fail("negative eps should behave like eps=0");

    /* toks_per_s<=0 with eps>0: no usable reading -> skip, don't touch baseline/warm, but
     * still advance the check clock (this used to silently fall through to an unconditional
     * swap, treating "no data" as "swap"; that was the bug) */
    last=0; baseline=5.0; warm=1;
    if(repin_gate(&last,16,16,0.0,0.15,&baseline,&warm)) return fail("toks_per_s<=0 must not swap");
    if(last!=16) return fail("toks_per_s<=0 branch did not advance last_repin");
    if(baseline!=5.0 || warm!=1) return fail("toks_per_s<=0 branch perturbed baseline/warm");
    last=0;
    if(repin_gate(&last,16,16,-2.0,0.15,&baseline,&warm)) return fail("negative toks_per_s must not swap");

    /* warm-up: the first TWO valid readings establish the baseline (averaged), no swap
     * during warm-up regardless of how different they are from each other */
    last=0; baseline=0; warm=0;
    if(repin_gate(&last,16,16,25.0,0.15,&baseline,&warm)) return fail("1st warm-up sample should not swap");
    if(warm!=1 || baseline!=40.0) return fail("1st warm-up sample should set baseline directly (40ms/tok @ 25tok/s)");
    last=16;
    if(repin_gate(&last,32,16,1000.0/44.0,0.15,&baseline,&warm)) return fail("2nd warm-up sample should not swap");
    if(warm!=2 || baseline!=42.0) return fail("2nd warm-up sample should average into baseline ((40+44)/2=42)");

    /* post warm-up, small deviation (<15%): skip, EMA-track, warm stays "trusted" */
    last=32; baseline=42.0; warm=2;
    if(repin_gate(&last,48,16,24.0,0.15,&baseline,&warm))  /* 24 tok/s = 41.67ms/tok, ~0.8% off 42.0 */
        return fail("small deviation should not swap");
    if(warm!=2) return fail("small deviation should not re-enter warm-up");
    if(baseline==42.0) return fail("EMA baseline did not move at all on a real (if small) reading");

    /* post warm-up, large deviation (>=15%): fires, and re-enters warm-up (does NOT lock
     * baseline to the single degraded reading as if it were now "normal") */
    last=48; baseline=40.0; warm=2;
    double degraded=1000.0/15.0;                    /* 15 tok/s vs 25 tok/s baseline: ~67% slower */
    if(!repin_gate(&last,64,16,15.0,0.15,&baseline,&warm)) return fail("large deviation should swap");
    if(last!=64) return fail("swap branch did not advance last_repin");
    if(baseline!=degraded) return fail("swap branch should set baseline to the just-measured (degraded) reading");
    if(warm!=1) return fail("swap branch should re-enter warm-up (warm=1), not stay 'trusted'");

    /* the oscillation check this exists to prevent: right after a swap fires on a slowdown,
     * throughput recovering to its old normal value must NOT itself look like a second
     * deviation and fire again — because we're back in warm-up, it just gets folded into
     * re-establishing the baseline instead */
    if(repin_gate(&last,80,16,25.0,0.15,&baseline,&warm))
        return fail("recovery right after a swap must not immediately fire again");
    if(warm!=2) return fail("recovery sample should complete warm-up");
    if(baseline!=0.5*degraded+0.5*40.0) return fail("recovery sample should average with the post-swap baseline");

    puts("repin gate tests: ok");
    return 0;
}
