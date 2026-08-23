/* CUDA_MISS_GPU async pipeline integration test: coli_miss_gpu_issue /
 * coli_miss_gpu_take (see colibri.c, right after coli_miss_gpu_try) and the
 * underlying coli_cuda_miss_issue/take in backend_cuda.cu.
 *
 * Sibling of test_cuda_miss_gpu.c, same fixture and hand-computed reference
 * (mk_expert, x/eg/eu/ed) reused on purpose: if the async path and the
 * synchronous coli_miss_gpu_try ever disagree on the same inputs, the bug is
 * in the async plumbing (pinned staging, per-slot device scratch, stream
 * sequencing), not in the shared coli_cuda_expert_mlp/quant_matmul kernels
 * (already covered by test_backend_cuda.cu and test_cuda_miss_gpu.c).
 *
 * The property this file exists to catch that test_cuda_miss_gpu.c cannot:
 * two slots with DIFFERENT experts issued back-to-back, both still pending,
 * must resolve to their OWN expert's numbers at take() time — a single
 * aliased pinned buffer or shared device scratch between slots would corrupt
 * one or both silently (right output shape, wrong numbers), not crash.
 *
 * Skips (exit 77) if no CUDA device is available, matching test_cuda_miss_gpu.c.
 * Requires COLI_CUDA. NOT part of `make check`: needs a real CUDA toolchain +
 * GPU. As of this writing this file has been COMPILE-CHECKED ONLY (no CUDA
 * device was available in the environment that authored it) — treat a first
 * real run on hardware as this test's actual validation, not a formality.
 *
 * Build (see also `make miss-gpu-test`, same recipe with this file swapped in):
 *   nvcc -c backend_cuda.cu -o backend_cuda.o
 *   gcc -DCOLI_CUDA -O2 -fopenmp tests/test_cuda_miss_issue.c backend_cuda.o \
 *       -lcudart -lm -fopenmp -o tests/test_cuda_miss_issue
 */
#ifndef COLI_CUDA
#define COLI_CUDA
#endif
#include <math.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int close_enough(const float *got, const float *want, int n, const char *label){
    for(int i=0;i<n;i++) if(fabsf(got[i]-want[i])>1e-4f){
        fprintf(stderr,"%s mismatch at %d: got %.6f want %.6f\n",label,i,got[i],want[i]);
        return 0;
    }
    return 1;
}

static void mk_expert(ESlot *e, const float *gate, const float *up, const float *down, int D, int I){
    memset(e,0,sizeof *e);
    e->g.fmt=0; e->g.O=I; e->g.I=D; e->g.qf=(float*)gate;
    e->u.fmt=0; e->u.O=I; e->u.I=D; e->u.qf=(float*)up;
    e->d.fmt=0; e->d.O=D; e->d.I=I; e->d.qf=(float*)down;
}

static void ref_row(const float *xr, float *want){
    float a=xr[0], b=xr[1];
    float sa=a/(1.0f+expf(-a))*a, sb=b/(1.0f+expf(-b))*b;
    want[0]=sa; want[1]=sb; want[2]=sa+sb; want[3]=sa-sb;
}
/* Same down-projection shape as ref_row but scaled 2x on the first two
 * outputs — matches expert_c's ed2 below, so a slot fed the WRONG expert's
 * weights produces a value that fails against the RIGHT reference instead of
 * accidentally matching it (both references share the same h0/h1 magnitudes
 * by construction, only the down-proj differs). */
static void ref_row_2x(const float *xr, float *want){
    float a=xr[0], b=xr[1];
    float sa=a/(1.0f+expf(-a))*a, sb=b/(1.0f+expf(-b))*b;
    want[0]=2*sa; want[1]=2*sb; want[2]=sa+sb; want[3]=sa-sb;
}

int main(void){
    int device=0;
    if(!coli_cuda_init(&device,1)){ fprintf(stderr,"no CUDA device — skipping\n"); return 77; }

    const float x[4]  = {1,-2,3,-4};
    const float eg[8] = {1,0,0,0, 0,1,0,0};   /* gate: O=2,I=4 */
    const float eu[8] = {1,0,0,0, 0,1,0,0};   /* up:   O=2,I=4 */
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};      /* down: O=4,I=2 */
    const float ed2[8]= {2,0, 0,2, 1,1, 1,-1};      /* same gate/up, scaled down-proj */

    ESlot expert_a, expert_c;
    mk_expert(&expert_a,eg,eu,ed,4,2);
    mk_expert(&expert_c,eg,eu,ed2,4,2);

    /* ---- basic issue/take roundtrip, single slot ---- */
    {
        float want[4]; ref_row(x,want);
        if(!coli_miss_gpu_issue(&expert_a,x,1,device,0)){
            fprintf(stderr,"issue failed on a fresh ring, slot 0\n"); return 1; }
        const float *y=coli_miss_gpu_take(0,device);
        if(!y){ fprintf(stderr,"take returned NULL after a successful issue\n"); return 1; }
        if(!close_enough(y,want,4,"basic issue/take, slot 0")) return 1;
    }

    /* ---- take() with nothing pending must return NULL, not crash ---- */
    if(coli_miss_gpu_take(0,device)!=NULL){
        fprintf(stderr,"take returned non-NULL with nothing pending\n"); return 1; }
    if(coli_miss_gpu_take(1,device)!=NULL){
        fprintf(stderr,"take on slot 1 returned non-NULL with nothing pending\n"); return 1; }

    /* ---- double buffering: two DIFFERENT experts in flight at once, taken
     * out of order relative to issue — the core property this file exists
     * to check. If slot 0 and slot 1 share any state (pinned staging,
     * device scratch), one of these two comparisons fails even though both
     * issues report success. ---- */
    {
        float want_a[4], want_c[4]; ref_row(x,want_a); ref_row_2x(x,want_c);
        if(!coli_miss_gpu_issue(&expert_a,x,1,device,0)){
            fprintf(stderr,"issue failed, slot 0 (double-buffer case)\n"); return 1; }
        if(!coli_miss_gpu_issue(&expert_c,x,1,device,1)){
            fprintf(stderr,"issue failed, slot 1 (double-buffer case)\n"); return 1; }
        /* take slot 1 FIRST, opposite of issue order, to catch a FIFO/index
         * mixup between the two slots' streams. */
        const float *yc=coli_miss_gpu_take(1,device);
        if(!yc){ fprintf(stderr,"take(1) returned NULL\n"); return 1; }
        if(!close_enough(yc,want_c,4,"double-buffer, slot 1 (expert_c)")) return 1;
        const float *ya=coli_miss_gpu_take(0,device);
        if(!ya){ fprintf(stderr,"take(0) returned NULL\n"); return 1; }
        if(!close_enough(ya,want_a,4,"double-buffer, slot 0 (expert_a)")) return 1;
    }

    /* ---- reissuing into a still-pending slot must fail, not silently drop
     * or corrupt the outstanding one ---- */
    {
        if(!coli_miss_gpu_issue(&expert_a,x,1,device,0)){
            fprintf(stderr,"issue failed setting up the still-pending case\n"); return 1; }
        if(coli_miss_gpu_issue(&expert_c,x,1,device,0)){
            fprintf(stderr,"issue into an already-pending slot was NOT rejected\n"); return 1; }
        /* the original (expert_a) issue must still be intact and takeable */
        float want[4]; ref_row(x,want);
        const float *y=coli_miss_gpu_take(0,device);
        if(!y){ fprintf(stderr,"take failed after a rejected re-issue attempt\n"); return 1; }
        if(!close_enough(y,want,4,"still-pending slot survives a rejected re-issue")) return 1;
    }

    /* ---- ring wraparound with refreshed weights (mirrors test_cuda_miss_gpu.c's
     * expert_c case): slot 0 must serve expert_c's numbers, not a stale
     * expert_a upload from earlier in this file. ---- */
    {
        float want[4]; ref_row_2x(x,want);
        if(!coli_miss_gpu_issue(&expert_c,x,1,device,0)){
            fprintf(stderr,"issue failed on wraparound refresh\n"); return 1; }
        const float *y=coli_miss_gpu_take(0,device);
        if(!y){ fprintf(stderr,"take failed on wraparound refresh\n"); return 1; }
        if(!close_enough(y,want,4,"ring wraparound refresh, slot 0")) return 1;
    }

    printf("test_cuda_miss_issue: ok\n");
    return 0;
}
