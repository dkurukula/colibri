/* Correctness + timing comparison: coli_cuda_expert_mlp (quant_matmul, one
 * block per (output,row)) vs coli_cuda_expert_mlp_tiled (quant_matmul_tiled,
 * weight row shared across QMT_TILE_S rows per block) -- see both kernels'
 * doc comments in backend_cuda.cu for why the tiled variant exists: a real
 * cold-prefill miss batch routes hundreds of rows to one expert, and
 * quant_matmul's per-block weight re-read multiplies GMEM traffic by the row
 * count, measured live to make CUDA_MISS_GPU slower than CPU at that scale
 * (see colibri.c's CUDA_MISS_GPU nr<=64 cap and its commit message).
 *
 * Same fixture shape as bench_tensor_core.cu (D=6144, I=2048 -- this model's
 * real hidden_size/moe_intermediate_size), same timing methodology (warm-up
 * call, N-iteration average). Sweeps row counts from decode (1) through the
 * existing S<=64 resident-tier regime up to several hundred, the regime a
 * long cold prefill's popular experts actually land in.
 *
 * Skips (exit 77) if no CUDA device is available.
 * Build:
 *   nvcc -O3 -std=c++17 -arch=native backend_cuda.cu tests/bench_expert_mlp_tiled.cu \
 *       -o tests/bench_expert_mlp_tiled -lcudart
 */
#include "../backend_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double run(int (*fn)(ColiCudaTensor*,ColiCudaTensor*,ColiCudaTensor*,float*,const float*,int),
                   ColiCudaTensor *g,ColiCudaTensor *u,ColiCudaTensor *d,
                   const float *x,float *y,int rows,int iterations){
    if(!fn(g,u,d,y,x,rows))std::exit(2);
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;i++)if(!fn(g,u,d,y,x,rows))std::exit(2);
    auto end=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(end-begin).count()/iterations;
}

int main(){
    constexpr int D=6144,I=2048;               /* real hidden_size / moe_intermediate_size */
    constexpr int MAXROWS=1024;
    int device=0;if(!coli_cuda_init(&device,1))return 77;
    std::vector<unsigned char> hidden((size_t)I*D/2),down((size_t)D*I/2);
    std::vector<float> hs(I),ds(D),x((size_t)MAXROWS*D),a((size_t)MAXROWS*D),b((size_t)MAXROWS*D);
    for(size_t i=0;i<hidden.size();i++)hidden[i]=(unsigned char)((i*17+29)&255);
    for(size_t i=0;i<down.size();i++)down[i]=(unsigned char)((i*13+41)&255);
    for(int i=0;i<I;i++)hs[i]=0.006f+(i%11)*0.0002f;
    for(int i=0;i<D;i++)ds[i]=0.006f+(i%7)*0.0002f;
    for(size_t i=0;i<x.size();i++)x[i]=std::sin((float)(i+1)*0.013f)*2.f;
    ColiCudaTensor *g=nullptr,*u=nullptr,*d=nullptr;
    if(!coli_cuda_tensor_upload(&g,hidden.data(),hs.data(),2,D,I,device)||
       !coli_cuda_tensor_upload(&u,hidden.data(),hs.data(),2,D,I,device)||
       !coli_cuda_tensor_upload(&d,down.data(),ds.data(),2,I,D,device))return 2;
    int any_bad=0;
    for(int rows: {1,8,32,64,128,256,512,1024}){
        double base_ms=run(coli_cuda_expert_mlp,g,u,d,x.data(),a.data(),rows,5);
        double tiled_ms=run(coli_cuda_expert_mlp_tiled,g,u,d,x.data(),b.data(),rows,5);
        double se=0,ref=0;
        for(size_t i=0;i<(size_t)rows*D;i++){double diff=b[i]-a[i];se+=diff*diff;ref+=(double)a[i]*a[i];}
        double rms=std::sqrt(se/(ref+1e-20));
        int bad=rms>1e-3;                       /* reassociation tolerance, not bit-identical */
        any_bad|=bad;
        std::printf("rows=%4d  untiled_ms=%8.3f  tiled_ms=%8.3f  speedup=%.3fx  rms=%.7f%s\n",
                    rows,base_ms,tiled_ms,base_ms/tiled_ms,rms,bad?"  MISMATCH":"");
    }
    coli_cuda_tensor_free(g);coli_cuda_tensor_free(u);coli_cuda_tensor_free(d);coli_cuda_shutdown();
    return any_bad?1:0;
}
