/* qt_shape_eq() is the guard CUDA_MISS_GPU's transient expert ring (see
 * miss_ring_init/coli_miss_gpu_try in colibri.c, right after qt_cuda_update)
 * uses before ever pushing a new expert's weights into a fixed-size device
 * slot. A device tensor is allocated ONCE at a given fmt/I/O and
 * coli_cuda_tensor_update() never reallocates it -- so this guard is the only
 * thing standing between "every routed expert is byte-uniform in size" (true
 * in practice, see repin.h) actually holding for a given model, and silently
 * pushing a mismatched-size payload into a fixed device buffer.
 *
 * Kept CUDA-independent specifically so it's testable here, without a GPU or
 * a CUDA toolchain -- see coli_miss_gpu_try's own comment for why. */
#include <assert.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static QT mk(int fmt, int I, int O){
    QT t; memset(&t,0,sizeof t); t.fmt=fmt; t.I=I; t.O=O; return t;
}

int main(void){
    QT ref = mk(2, 6144, 2048);   /* a plausible real int4 gate/up projection shape */

    { QT same = mk(2, 6144, 2048);
      assert(qt_shape_eq(&ref,&same) && "identical fmt/I/O must compare equal"); }

    { QT diff_fmt = mk(1, 6144, 2048);
      assert(!qt_shape_eq(&ref,&diff_fmt) && "different fmt (int8 vs int4) must not compare equal"); }

    { QT diff_I = mk(2, 6145, 2048);
      assert(!qt_shape_eq(&ref,&diff_I) && "different I must not compare equal"); }

    { QT diff_O = mk(2, 6144, 2049);
      assert(!qt_shape_eq(&ref,&diff_O) && "different O must not compare equal"); }

    { QT diff_both = mk(1, 2048, 6144);   /* e.g. down_proj's shape, transposed vs gate/up */
      assert(!qt_shape_eq(&ref,&diff_both) && "down_proj-shaped tensor must not compare equal to gate/up-shaped"); }

    /* fields the guard must NOT look at: two tensors that differ only in gs
     * (group size, meaningful for fmt=4) or cuda-side state still compare
     * equal on fmt/I/O alone -- gs mismatches for the ring's fixed fmt=4 case
     * are out of scope for this guard (grouped int4 experts are a real format
     * this project supports, but CUDA_MISS_GPU targets the common byte-uniform
     * per-row-scale formats first; see PR description). */
    { QT gs_diff = ref; gs_diff.gs = 128;
      assert(qt_shape_eq(&ref,&gs_diff) && "gs is not part of the fmt/I/O identity this guard checks"); }

    printf("qt_shape_eq tests: ok\n");
    return 0;
}
