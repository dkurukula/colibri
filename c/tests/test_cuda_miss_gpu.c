/* CUDA_MISS_GPU integration test: miss_ring_init / coli_miss_gpu_try (see
 * colibri.c, right after qt_cuda_update — the closest existing precedent is
 * repin_do_swaps' VRAM-slot refresh, a few thousand lines further down).
 *
 * Same numeric fixture as test_backend_cuda.cu's coli_cuda_expert_mlp case
 * (tiny 4-in/2-out gate+up, 2-in/4-out down, hand-computed silu reference) —
 * reused here on purpose: if this and that test ever disagree, the bug is in
 * how this file's ESlot/QT plumbing feeds coli_cuda_expert_mlp, not in the
 * kernel itself (already covered by test_backend_cuda.cu).
 *
 * No real model needed: builds fake ESlots by hand, exactly the shape the
 * decode miss loop in moe() would hand to coli_miss_gpu_try.
 *
 * Skips (exit 77) if no CUDA device is available, matching test_backend_cuda.cu.
 * Requires COLI_CUDA (this file only compiles under -DCOLI_CUDA — see the
 * Makefile's miss-gpu-test target). NOT part of `make check`: needs a real
 * CUDA toolchain + GPU, like cuda-test.
 *
 * Build (see also `make miss-gpu-test`):
 *   nvcc -c backend_cuda.cu -o backend_cuda.o
 *   gcc -DCOLI_CUDA -O2 -fopenmp tests/test_cuda_miss_gpu.c backend_cuda.o \
 *       -lcudart -lm -fopenmp -o tests/test_cuda_miss_gpu
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

/* Builds a fake CPU-resident (fmt=0, f32) expert directly from raw weight
 * arrays — the same shape expert_host_ensure() would leave in e->g/e->u/e->d
 * for a real streamed-from-disk miss, minus the disk I/O. */
static void mk_expert(ESlot *e, const float *gate, const float *up, const float *down, int D, int I){
    memset(e,0,sizeof *e);
    e->g.fmt=0; e->g.O=I; e->g.I=D; e->g.qf=(float*)gate;
    e->u.fmt=0; e->u.O=I; e->u.I=D; e->u.qf=(float*)up;
    e->d.fmt=0; e->d.O=D; e->d.I=I; e->d.qf=(float*)down;
}

int main(void){
    int device=0;
    if(!coli_cuda_init(&device,1)){ fprintf(stderr,"no CUDA device — skipping\n"); return 77; }

    const float x[8]  = {1,-2,3,-4, 2,1,-1,0.5f};   /* two decode rows worth of input, D=4 */
    const float eg[8] = {1,0,0,0, 0,1,0,0};          /* gate: O=2,I=4 */
    const float eu[8] = {1,0,0,0, 0,1,0,0};          /* up:   O=2,I=4 */
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};       /* down: O=4,I=2 */

    ESlot expert_a, expert_b;
    mk_expert(&expert_a,eg,eu,ed,4,2);

    /* row 0 of x, D=4 — same silu(gate(x))*up(x) -> down(...) math as
     * test_backend_cuda.cu's coli_cuda_expert_mlp case, one row at a time
     * (coli_miss_gpu_try is decode-only: S=1, one row per call). */
    float y[4], want[4];
    for(int row=0; row<2; row++){
        const float *xr=x+row*4;
        float a=xr[0], b=xr[1];        /* gate/up are both the same [I,I] identity-ish projection here */
        float sa=a/(1.0f+expf(-a))*a, sb=b/(1.0f+expf(-b))*b;
        want[0]=sa; want[1]=sb; want[2]=sa+sb; want[3]=sa-sb;
        if(!coli_miss_gpu_try(&expert_a,y,xr,device)){ fprintf(stderr,"coli_miss_gpu_try failed on a fresh ring\n"); return 1; }
        if(!close_enough(y,want,4,"first expert, row")) return 1;
    }
    if(g_miss_gpu_hits!=2 || g_miss_gpu_fallback!=0 || g_miss_gpu_shape_mismatch!=0){
        fprintf(stderr,"unexpected counters after 2 same-shape calls: hits=%llu fallback=%llu mismatch=%llu\n",
            (unsigned long long)g_miss_gpu_hits,(unsigned long long)g_miss_gpu_fallback,(unsigned long long)g_miss_gpu_shape_mismatch);
        return 1;
    }

    /* A THIRD call, same shape, DIFFERENT weights: the ring (N=2) must wrap
     * around and correctly refresh a slot it has already used once, not just
     * serve a stale first upload. Same gate/up as expert_a (so h0=silu(x0)*x0,
     * h1=silu(x1)*x1 exactly as already verified above) — only down changes,
     * to isolate "did the refresh actually take" from any new gate/up math. */
    const float ed2[8]={2,0, 0,2, 1,1, 1,-1};   /* y = [2*h0, 2*h1, h0+h1, h0-h1] */
    ESlot expert_c; mk_expert(&expert_c,eg,eu,ed2,4,2);
    { const float *xr=x;
      float a=xr[0], b=xr[1];
      float sa=a/(1.0f+expf(-a))*a, sb=b/(1.0f+expf(-b))*b;
      want[0]=2*sa; want[1]=2*sb; want[2]=sa+sb; want[3]=sa-sb;
      if(!coli_miss_gpu_try(&expert_c,y,xr,device)){ fprintf(stderr,"coli_miss_gpu_try failed on ring wraparound\n"); return 1; }
      if(!close_enough(y,want,4,"third expert (ring wraparound), row")) return 1;
    }

    /* Defensive guard: a DIFFERENT-shaped expert (D=8 instead of 4) must be
     * rejected (fall back to caller), never pushed into the fixed-size ring
     * slot — this is the exact GPU-memory-safety property qt_shape_eq exists
     * for (see tests/test_qt_shape_eq.c for the pure-logic version of this). */
    const float wide_g[16]={0}, wide_u[16]={0}, wide_d[16]={0};
    mk_expert(&expert_b,wide_g,wide_u,wide_d,8,2);
    if(coli_miss_gpu_try(&expert_b,y,x,device)){
        fprintf(stderr,"mismatched-shape expert was NOT rejected — GPU memory safety violated\n");
        return 1;
    }
    if(g_miss_gpu_shape_mismatch!=1){
        fprintf(stderr,"shape-mismatch guard did not fire (counter=%llu)\n",(unsigned long long)g_miss_gpu_shape_mismatch);
        return 1;
    }
    /* the ring itself must be untouched by the rejected attempt — same-shape
     * expert immediately after must still work */
    { const float *xr=x;
      float a=xr[0], b=xr[1];
      float sa=a/(1.0f+expf(-a))*a, sb=b/(1.0f+expf(-b))*b;
      want[0]=sa; want[1]=sb; want[2]=sa+sb; want[3]=sa-sb;
      if(!coli_miss_gpu_try(&expert_a,y,xr,device)){ fprintf(stderr,"ring unusable after a rejected shape mismatch\n"); return 1; }
      if(!close_enough(y,want,4,"post-mismatch recovery, row")) return 1;
    }

    printf("test_cuda_miss_gpu: ok (hits=%llu fallback=%llu shape_mismatch=%llu)\n",
        (unsigned long long)g_miss_gpu_hits,(unsigned long long)g_miss_gpu_fallback,(unsigned long long)g_miss_gpu_shape_mismatch);
    return 0;
}
