/* Microbenchmark for the COLI_MISS_CPU_EVERY prototype: does concurrent
 * CPU+GPU expert execution contend on shared host DRAM bandwidth enough to
 * matter on this hardware? Not a colibri.c integration test -- standalone,
 * against quant.h's matmul_i4 (self-contained, "no Model or QT dependency"
 * per its own header comment) and raw CUDA calls, so it needs nothing this
 * repo doesn't already have and isn't gated on any of colibri.c's state.
 *
 * Three loops, N iterations each, one int4-packed expert-sized buffer
 * (representative dims, see D/I below -- sized to land near the ~18.9MB/
 * expert figure established for the real GLM-5.2 deployment this targets):
 *   1. PCIe only:  pinned-stage + cudaMemcpyAsync H2D + sync, per iteration.
 *   2. CPU only:   matmul_i4 (OMP-parallel over O) on the main thread.
 *   3. Concurrent: issue the async H2D (non-blocking), run matmul_i4 on a
 *      DIFFERENT buffer on the main thread while it's in flight, then sync.
 *
 * Rule: loop3 ~= max(loop1, loop2)  -> concurrency is ~free, the split idea
 *                                      has legs.
 *       loop3 ~= loop1 + loop2      -> they contend, it doesn't.
 *
 * Skips (exit 77) if no CUDA device is available, matching the other CUDA
 * tests in this directory.
 *
 * Build (plain C + libcudart -- no device kernels here, just the CUDA
 * runtime API, so no nvcc/C++ needed; quant.h uses C11 _Thread_local, which
 * isn't valid under nvcc's C++ mode):
 *   gcc -O3 -fopenmp tests/bench_miss_cpu_gpu_split.c -o tests/bench_miss_cpu_gpu_split \
 *     -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -lcudart -lm
 */
#include "../quant.h"
#include <cuda_runtime.h>
#include <time.h>

static int cuda_ok(cudaError_t e, const char *what){
    if(e!=cudaSuccess){ fprintf(stderr,"%s: %s\n", what, cudaGetErrorString(e)); return 0; }
    return 1;
}

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec/1e9;
}

int main(void){
    int ndev=0;
    if(cudaGetDeviceCount(&ndev)!=cudaSuccess || ndev<1){ fprintf(stderr,"no CUDA device — skipping\n"); return 77; }

    /* Representative GLM-5.2-scale expert: D*I ~= 12.6M so 3 tensors at
     * ~0.5 byte/param (int4) land near the ~18.9MB/expert figure already
     * established for the real deployment (colibri.c's CUDA_MISS_GPU doc
     * comment). Exact production dims aren't needed -- what matters is a
     * real order-of-magnitude PCIe transfer and a real matmul_i4 call on
     * THIS card/CPU, not matching the model bit-for-bit. */
    const int D = 4096, I = 3072;   /* gate/up: O=I(3072),I=D(4096); down: O=D,I=I */
    const int rb_gu = (D+1)/2, rb_d = (I+1)/2;
    const size_t gu_bytes = (size_t)I*rb_gu, d_bytes = (size_t)D*rb_d;
    const size_t total_weight_bytes = 2*gu_bytes + d_bytes;   /* gate+up+down, matches qt_bytes' int4 formula */
    printf("expert size: gate/up %.2f MB each, down %.2f MB, total %.2f MB\n",
        gu_bytes/1e6, d_bytes/1e6, (total_weight_bytes + (size_t)(2*I+D)*4)/1e6);

    const int N = 40;   /* iterations per loop */

    uint8_t *h_gate=(uint8_t*)malloc(gu_bytes), *h_up=(uint8_t*)malloc(gu_bytes), *h_down=(uint8_t*)malloc(d_bytes);
    float *h_scale_gu=(float*)malloc((size_t)I*sizeof(float)), *h_scale_d=(float*)malloc((size_t)D*sizeof(float));
    float *h_x=(float*)malloc((size_t)D*sizeof(float)), *h_y=(float*)malloc((size_t)I*sizeof(float));
    if(!h_gate||!h_up||!h_down||!h_scale_gu||!h_scale_d||!h_x||!h_y){ fprintf(stderr,"host alloc failed\n"); return 1; }
    for(size_t i=0;i<gu_bytes;i++) h_gate[i]=(uint8_t)(i*2654435761u);
    for(size_t i=0;i<gu_bytes;i++) h_up[i]=(uint8_t)(i*2246822519u);
    for(size_t i=0;i<d_bytes;i++) h_down[i]=(uint8_t)(i*3266489917u);
    for(int i=0;i<I;i++) h_scale_gu[i]=1.0f;
    for(int i=0;i<D;i++) h_scale_d[i]=1.0f;
    for(int i=0;i<D;i++) h_x[i]=(float)((i%17)-8)*0.1f;

    uint8_t *pin_a, *pin_b; float *pin_scale_a, *pin_scale_b;
    if(!cuda_ok(cudaMallocHost((void**)&pin_a, gu_bytes), "pinned alloc a") ||
       !cuda_ok(cudaMallocHost((void**)&pin_b, gu_bytes), "pinned alloc b") ||
       !cuda_ok(cudaMallocHost((void**)&pin_scale_a, (size_t)I*sizeof(float)), "pinned scale alloc a") ||
       !cuda_ok(cudaMallocHost((void**)&pin_scale_b, (size_t)I*sizeof(float)), "pinned scale alloc b")) return 1;
    memcpy(pin_a, h_gate, gu_bytes); memcpy(pin_b, h_gate, gu_bytes);
    memcpy(pin_scale_a, h_scale_gu, (size_t)I*sizeof(float)); memcpy(pin_scale_b, h_scale_gu, (size_t)I*sizeof(float));

    void *d_w=NULL; float *d_scale=NULL;
    if(!cuda_ok(cudaMalloc(&d_w, gu_bytes), "device weight alloc") ||
       !cuda_ok(cudaMalloc((void**)&d_scale, (size_t)I*sizeof(float)), "device scale alloc")) return 1;

    cudaStream_t stream;
    if(!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create")) return 1;

    /* ---- loop 1: PCIe only ---- */
    double t0 = now_s();
    for(int it=0; it<N; it++){
        if(!cuda_ok(cudaMemcpyAsync(d_w, pin_a, gu_bytes, cudaMemcpyHostToDevice, stream), "h2d") ||
           !cuda_ok(cudaMemcpyAsync(d_scale, pin_scale_a, (size_t)I*sizeof(float), cudaMemcpyHostToDevice, stream), "scale h2d") ||
           !cuda_ok(cudaStreamSynchronize(stream), "sync")) return 1;
    }
    double t_pcie = (now_s()-t0) / N;

    /* ---- loop 2: CPU only (matmul_i4, S=1, full OMP team over O=I) ---- */
    t0 = now_s();
    for(int it=0; it<N; it++){
        matmul_i4(h_y, h_x, h_gate, h_scale_gu, /*S=*/1, D, I);
    }
    double t_cpu = (now_s()-t0) / N;

    /* ---- loop 3: concurrent -- issue async H2D, run matmul_i4 on the main
     * thread against a DIFFERENT host buffer while it's in flight, then sync ---- */
    t0 = now_s();
    for(int it=0; it<N; it++){
        if(!cuda_ok(cudaMemcpyAsync(d_w, pin_b, gu_bytes, cudaMemcpyHostToDevice, stream), "h2d (concurrent)") ||
           !cuda_ok(cudaMemcpyAsync(d_scale, pin_scale_b, (size_t)I*sizeof(float), cudaMemcpyHostToDevice, stream), "scale h2d (concurrent)"))
            return 1;
        matmul_i4(h_y, h_x, h_up, h_scale_gu, /*S=*/1, D, I);   /* different weight buffer than the async copy's source */
        if(!cuda_ok(cudaStreamSynchronize(stream), "sync (concurrent)")) return 1;
    }
    double t_concurrent = (now_s()-t0) / N;

    double t_max = t_pcie>t_cpu ? t_pcie : t_cpu;
    double t_sum = t_pcie + t_cpu;
    printf("\nper-iteration (ms), N=%d:\n", N);
    printf("  loop1 PCIe only:        %.3f ms\n", t_pcie*1000);
    printf("  loop2 CPU only:         %.3f ms\n", t_cpu*1000);
    printf("  loop3 concurrent:       %.3f ms\n", t_concurrent*1000);
    printf("  max(loop1,loop2):       %.3f ms  (no-contention prediction)\n", t_max*1000);
    printf("  loop1+loop2:            %.3f ms  (full-contention prediction)\n", t_sum*1000);
    double frac = (t_concurrent - t_max) / (t_sum - t_max);
    printf("  loop3 position: %.0f%% of the way from no-contention to full-contention\n", frac*100);

    return 0;
}
