/* autopin_preserve_lru() / autopin_lru_reserve() guard the RAM budget split
 * between automatic history pinning and the adaptive LRU expert cache (see
 * both functions in colibri.c, right after expert_avail()). Before this,
 * autopin could take up to half the expert RAM budget before the LRU was
 * ever sized -- at scale (hundreds of thousands of routed selections, a few
 * hours of chat) that starves the LRU cache and increases expert disk I/O
 * on exactly the systems this budget logic exists to protect.
 *
 * Pure functions, no Model/GPU/disk needed -- testable here directly. */
#include <math.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures;

static void check_autopin(const char *label, double planned, double available,
                          double lru, double expected){
    double got=autopin_preserve_lru(planned,available,lru);
    if(fabs(got-expected)>1e-9){
        fprintf(stderr,"FAIL %-36s -> %.3f (want %.3f)\n",label,got,expected);
        failures++;
    }
}

static void check_lru_reserve(const char *label, double available, double slot,
                              int requested, int expected_cap, double expected_bytes){
    int cap=-1;
    double got=autopin_lru_reserve(available,slot,requested,&cap);
    if(cap!=expected_cap || fabs(got-expected_bytes)>1e-9){
        fprintf(stderr,"FAIL %-36s -> cap %d, %.3f (want %d, %.3f)\n",
                label,cap,got,expected_cap,expected_bytes);
        failures++;
    }
}

int main(void){
    /* autopin_preserve_lru: planned pin, expert-available, lru-reserve -> pin bytes granted */
    check_autopin("nine slots preserve eight",4.5,9.0,8.0,1.0);
    check_autopin("sixteen slots keep half",8.0,16.0,8.0,8.0);
    check_autopin("twenty slots keep plan",10.0,20.0,8.0,10.0);
    check_autopin("insufficient for requested LRU",4.0,6.0,8.0,0.0);
    check_autopin("small plan capped",2.0,9.0,8.0,1.0);
    check_autopin("zero plan",0.0,9.0,8.0,0.0);

    /* autopin_lru_reserve: expert-available, bytes/slot, requested cap -> (cap, bytes) */
    check_lru_reserve("requested cap fits",16.0,1.0,8,8,8.0);
    check_lru_reserve("reserve only affordable cap",9.0,2.0,8,4,8.0);
    check_lru_reserve("zero slot size",9.0,0.0,8,0,0.0);
    check_lru_reserve("zero requested cap",9.0,1.0,0,0,0.0);

    if(failures){
        fprintf(stderr,"autopin/LRU budget tests: %d FAILURE(S)\n",failures);
        return 1;
    }
    printf("test_autopin_lru: ok\n");
    return 0;
}
