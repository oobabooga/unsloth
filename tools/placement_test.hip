// Where do the pages actually land: the VRAM carve-out, or system RAM?
//
// Allocates the same size twice, once per allocator, and reads amdgpu's own
// mem_info_vram_used plus /proc/meminfo MemAvailable before and after. Touching
// the buffer first, because an untouched mapping proves nothing about placement.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static long read_first_ulong(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static long vram_used_bytes() {
    static char path[256] = {0};
    if (!path[0]) {
        for (int i = 0; i < 8; i++) {
            char cand[256];
            snprintf(cand, sizeof(cand), "/sys/class/drm/card%d/device/mem_info_vram_used", i);
            FILE * f = fopen(cand, "r");
            if (f) { fclose(f); snprintf(path, sizeof(path), "%s", cand); break; }
        }
    }
    return path[0] ? read_first_ulong(path) : -1;
}

static long gtt_used_bytes() {
    static char path[256] = {0};
    if (!path[0]) {
        for (int i = 0; i < 8; i++) {
            char cand[256];
            snprintf(cand, sizeof(cand), "/sys/class/drm/card%d/device/mem_info_gtt_used", i);
            FILE * f = fopen(cand, "r");
            if (f) { fclose(f); snprintf(path, sizeof(path), "%s", cand); break; }
        }
    }
    return path[0] ? read_first_ulong(path) : -1;
}

static long mem_available_kb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) { sscanf(line + 13, "%ld", &v); break; }
    }
    fclose(f);
    return v;
}

__global__ void touch(uint32_t * p, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = (uint32_t)i;
}

struct Snap { long vram, gtt, avail; };

static Snap snap() {
    Snap s;
    s.vram  = vram_used_bytes();
    s.gtt   = gtt_used_bytes();
    s.avail = mem_available_kb();
    return s;
}

static void report(const char * what, Snap a, Snap b, double gib) {
    printf("  %-26s asked for %.1f GiB\n", what, gib);
    printf("      VRAM used   %+8.2f GiB\n", (b.vram - a.vram) / 1073741824.0);
    printf("      GTT used    %+8.2f GiB\n", (b.gtt - a.gtt) / 1073741824.0);
    printf("      MemAvailable%+8.2f GiB\n", (b.avail - a.avail) / 1048576.0);
    fflush(stdout);
}

static void run(bool managed, double gib, int device) {
    const size_t bytes = (size_t)(gib * 1073741824.0) & ~(size_t)4095;
    const size_t n     = bytes / sizeof(uint32_t);

    Snap before = snap();

    uint32_t * p = nullptr;
    hipError_t err = managed ? hipMallocManaged(&p, bytes, hipMemAttachGlobal)
                             : hipMalloc(&p, bytes);
    if (err != hipSuccess) {
        printf("  %-26s ALLOC FAILED: %s\n", managed ? "hipMallocManaged" : "hipMalloc",
               hipGetErrorString(err));
        (void)hipGetLastError();
        fflush(stdout);
        return;
    }
    if (managed) {
        (void)hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, device);
        (void)hipGetLastError();
    }

    // Placement is only real once the pages are backed, so write every one.
    hipLaunchKernelGGL(touch, dim3(4096), dim3(256), 0, 0, p, n);
    (void)hipDeviceSynchronize();

    Snap after = snap();
    report(managed ? "hipMallocManaged+coarse" : "hipMalloc", before, after, gib);

    (void)hipFree(p);
    (void)hipDeviceSynchronize();
}

int main(int argc, char ** argv) {
    const int device = 0;
    (void)hipSetDevice(device);

    double gib = argc > 1 ? atof(argv[1]) : 8.0;

    Snap s = snap();
    printf("baseline: VRAM used %.2f GiB, GTT used %.2f GiB, MemAvailable %.2f GiB\n\n",
           s.vram / 1073741824.0, s.gtt / 1073741824.0, s.avail / 1048576.0);

    run(false, gib, device);
    printf("\n");
    run(true, gib, device);
    return 0;
}
