// LD_PRELOAD shim: report an emulated free/total VRAM to llama.cpp's fitter (any CUDA runtime major).
// FAKE_CUDA_FREE_MIB / FAKE_CUDA_TOTAL_MIB; unset = pass through. Build: gcc -shared -fPIC -O2 -o fakemem.so fakemem.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdlib.h>
typedef int cudaError_t;
typedef cudaError_t (*fn_t)(size_t *, size_t *);
cudaError_t cudaMemGetInfo(size_t *f, size_t *t) {
    const char *ef = getenv("FAKE_CUDA_FREE_MIB"), *et = getenv("FAKE_CUDA_TOTAL_MIB");
    if (ef && et) {
        *f = (size_t)atoll(ef) << 20;
        *t = (size_t)atoll(et) << 20;
        return 0;
    }
    static fn_t real = 0;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "cudaMemGetInfo");
    if (!real) {
        const char *libs[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so", 0};
        for (int i = 0; libs[i] && !real; i++) {
            void *h = dlopen(libs[i], RTLD_LAZY | RTLD_NOLOAD);
            if (h) real = (fn_t)dlsym(h, "cudaMemGetInfo");
        }
    }
    return real ? real(f, t) : 999;
}
