/* test_expert_slab_shrink.c -- a cache slot that once held a wider expert must
 * shrink back down when reused for a narrower one, or its actual RSS silently
 * drifts past whatever cap_for_ram() assumed when it sized the slot count.
 *
 * Ported from upstream JustVugg/colibri commit 7c97625 (#856), scoped down:
 * that commit also rewrote cap_for_ram()'s capacity math to price each expert
 * row at its own width instead of the container's widest (new functions
 * expert_bytes_row/expert_cache_row_bytes this fork doesn't have and isn't
 * adopting here -- our cap_for_ram() never had the "charge everyone the
 * widest width" regression that motivated it, so that half doesn't apply).
 * This file keeps only the shrink-on-load fix itself: expert_load_impl() and
 * uring_load_add() reallocating DOWN as well as up. Measures wide/narrow
 * byte sizes empirically via expert_load() rather than expert_bytes_row(),
 * so it needs nothing beyond what this fork already has.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Structured like a real GLM-5.2 container: sparse layers 1..4 are int4
 * routed experts, layer 5 is an int8 MTP head -- the exact mix that lets a
 * cache slot bleed from wide to narrow through the ws[]<->LRU swap in moe().
 * Big enough that the shrink's 25% relative threshold dominates its 64 KB
 * floor, or nothing would ever shrink and the test would pass by never
 * exercising the path. */
enum { N_LAYERS = 5, FIRST_DENSE = 1, N_EXPERTS = 2, O = 384, I = 768 };

static const char *SUF[3] = { "gate_proj", "up_proj", "down_proj" };

typedef struct { char name[128]; int64_t nbytes; const char *dtype; } Ent;
static Ent ents[512]; static int nent;

static void add(const char *fmt, int64_t nbytes, const char *dtype, int l, int e, const char *suf){
    Ent *t=&ents[nent++];
    snprintf(t->name,sizeof t->name,fmt,l,e,suf);
    t->nbytes=nbytes; t->dtype=dtype;
}

static void add_expert_dims(int l, int e, int int8, int inter, int hid){
    for(int k=0;k<3;k++){
        int rows = (k==2) ? hid : inter, cols = (k==2) ? inter : hid;   /* k==2 is down_proj */
        add("model.layers.%d.mlp.experts.%d.%s.weight",
            int8 ? (int64_t)rows*cols : (int64_t)rows*((cols+1)/2), "U8", l, e, SUF[k]);
        add("model.layers.%d.mlp.experts.%d.%s.weight.qs",
            (int64_t)rows*4, "F32", l, e, SUF[k]);
    }
}
static void add_expert(int l, int e, int int8){ add_expert_dims(l,e,int8,O,I); }

static int write_ents(const char *dir){
    char path[256]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    char *hdr=malloc(1<<20); int hn=0; int64_t off=0;
    hn+=sprintf(hdr+hn,"{");
    for(int i=0;i<nent;i++){
        hn+=sprintf(hdr+hn,"%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
            i?",":"", ents[i].name, ents[i].dtype,
            (long long)ents[i].nbytes/(ents[i].dtype[0]=='F'?4:1),
            (long long)off,(long long)(off+ents[i].nbytes));
        off+=ents[i].nbytes;
    }
    hn+=sprintf(hdr+hn,"}");
    while(hn%8) hdr[hn++]=' ';

    FILE *f=fopen(path,"wb"); if(!f){ free(hdr); return 0; }
    uint64_t hl=(uint64_t)hn; fwrite(&hl,8,1,f); fwrite(hdr,1,(size_t)hn,f);
    void *zero=calloc(1,65536);
    for(int64_t left=off;left>0;){ size_t n=left>65536?65536:(size_t)left; fwrite(zero,1,n,f); left-=(int64_t)n; }
    free(zero); free(hdr); fclose(f);
    return 1;
}

static int write_container(const char *dir){
    nent=0;
    for(int l=FIRST_DENSE;l<N_LAYERS;l++)
        for(int e=0;e<N_EXPERTS;e++) add_expert(l,e,0);          /* routed rows: int4 */
    for(int e=0;e<N_EXPERTS;e++) add_expert(N_LAYERS,e,1);       /* MTP row:     int8 */
    return write_ents(dir);
}

int main(void){
    const char *dir="tests/tmp_slab_shrink";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    if(!write_container(dir)){ printf("FAIL: could not write the fixture container\n"); return 1; }

    Model m; memset(&m,0,sizeof m);
    Cfg *c=&m.c;
    c->n_layers=N_LAYERS; c->first_dense=FIRST_DENSE; c->n_experts=N_EXPERTS;
    c->hidden=I; c->moe_inter=O;
    m.ebits=4; m.has_mtp=1;
    m.L=calloc(N_LAYERS+1,sizeof(Layer));
    for(int i=FIRST_DENSE;i<N_LAYERS;i++) m.L[i].sparse=1;
    st_init(&m.S,dir);        /* void: it exits on a malformed container */
    if(!st_find(&m.S,"model.layers.1.mlp.experts.0.gate_proj.weight")){
        printf("FAIL: st_init did not index the fixture\n"); return 1; }

    /* Measure the widths this container really holds, empirically, via the
     * same expert_load() the shrink patch is in -- no expert_bytes_row(). */
    ESlot probe_w; memset(&probe_w,0,sizeof probe_w);
    CHECK(expert_load(&m,N_LAYERS,0,&probe_w,0,0)==0);
    int64_t wide = probe_w.slab_cap;
    ESlot probe_n; memset(&probe_n,0,sizeof probe_n);
    CHECK(expert_load(&m,FIRST_DENSE,0,&probe_n,0,0)==0);
    int64_t narrow = probe_n.slab_cap;
    printf("  routed row slab_cap %lld B\n  MTP row slab_cap    %lld B\n",
        (long long)narrow,(long long)wide);
    CHECK(narrow > 0);
    CHECK(wide > narrow);                    /* int8 head is wider than int4 rows */
    compat_aligned_free(probe_w.slab); free(probe_w.fslab);
    compat_aligned_free(probe_n.slab); free(probe_n.fslab);

    /* ---- a reused slot must shrink back down --------------------------- */
    ESlot s; memset(&s,0,sizeof s);
    CHECK(expert_load(&m,N_LAYERS,0,&s,0,0)==0);      /* wide row first */
    int64_t cap_after_wide = s.slab_cap;
    CHECK(cap_after_wide >= wide);
    CHECK(expert_load(&m,FIRST_DENSE,0,&s,0,0)==0);   /* now the same slot, narrow row */
    int64_t cap_after_narrow = s.slab_cap;
    printf("  slab_cap after MTP %lld B -> after routed %lld B\n",
        (long long)cap_after_wide,(long long)cap_after_narrow);
    CHECK(cap_after_narrow < cap_after_wide);         /* it came down */
    CHECK(cap_after_narrow >= narrow);                /* and still fits what it holds */

    /* Negative control: the shrink must never make a slot too small for its
     * own expert. Reload the wide row into the shrunken slot and require it
     * to grow back. */
    CHECK(expert_load(&m,N_LAYERS,1,&s,0,0)==0);
    CHECK(s.slab_cap >= wide);
    compat_aligned_free(s.slab); free(s.fslab);

    /* ---- the migration itself, as moe() actually performs it ------------ */
    /* colibri.c's "promozione LRU" swap is  ESlot tmp=*dst; *dst=m->ws[q];
     * m->ws[q]=tmp;  -- the cache slot's OLD contents travel back into ws[].
     * So a wide slab reaches a narrow row on the THIRD step, not the first,
     * which is why a two-load test above would miss the swap-mediated path. */
    {
        ESlot ws, mtp_cache, main_cache;
        memset(&ws,0,sizeof ws); memset(&mtp_cache,0,sizeof mtp_cache);
        memset(&main_cache,0,sizeof main_cache);
        ESlot tmp;
        /* 1: MTP miss loads wide into ws, promoted into an empty MTP cache slot */
        CHECK(expert_load(&m,N_LAYERS,0,&ws,0,0)==0);
        tmp=mtp_cache; mtp_cache=ws; ws=tmp;
        /* 2: another MTP miss; the promotion now EVICTS, handing the wide slab back */
        CHECK(expert_load(&m,N_LAYERS,1,&ws,0,0)==0);
        tmp=mtp_cache; mtp_cache=ws; ws=tmp;
        CHECK(ws.slab_cap >= wide);            /* ws is carrying a wide slab now */
        /* 3: a MAIN row's miss reuses that slot -- the bleed, if nothing shrinks */
        CHECK(expert_load(&m,FIRST_DENSE,1,&ws,0,0)==0);
        tmp=main_cache; main_cache=ws; ws=tmp;
        printf("  slab that reached the int4 row: %lld B (a wide one is %lld B)\n",
            (long long)main_cache.slab_cap,(long long)wide);
        CHECK(main_cache.slab_cap < wide);     /* the narrow row is not paying int8 */
        CHECK(main_cache.slab_cap >= narrow);  /* and still holds what it holds */
        compat_aligned_free(ws.slab); free(ws.fslab);
        compat_aligned_free(mtp_cache.slab); free(mtp_cache.fslab);
        compat_aligned_free(main_cache.slab); free(main_cache.fslab);
    }

    /* ---- arena slices must never be shrunk ------------------------------ */
    /* pin_arena_bind hands slots interior pointers into ONE per-layer
     * allocation. Freeing one would corrupt the heap, and they're already
     * per-layer width so they have nothing to shrink. Prove the guard. */
    {
        size_t stride=(size_t)wide+8192;
        uint8_t *arena=NULL;
        CHECK(posix_memalign((void**)&arena,4096,stride)==0);
        float *farena=calloc(1<<16,sizeof(float));
        ESlot a; memset(&a,0,sizeof a);
        a.slab=arena; a.aslab=arena; a.slab_cap=(int64_t)stride;
        a.fslab=farena; a.afslab=farena; a.fslab_cap=1<<16;
        CHECK(expert_load(&m,FIRST_DENSE,0,&a,0,0)==0);   /* narrow expert in a wide arena slice */
        printf("  arena slot: slab %s, cap %lld (was %lld)\n",
            a.slab==arena?"untouched":"MOVED", (long long)a.slab_cap,(long long)stride);
        CHECK(a.slab==arena);                  /* not freed, not reallocated */
        CHECK(a.slab_cap==(int64_t)stride);    /* and its capacity was left alone */
        CHECK(a.fslab==farena);
        free(farena); compat_aligned_free(arena);
    }

    { char path[256]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
      remove(path); rmdir(dir); }

    printf("test_expert_slab_shrink: %s\n", fails?"FAILED":"ok");
    return fails?1:0;
}
