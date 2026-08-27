// Does hipMallocManaged memory hold its contents on gfx1151?
//
// ggml enables it purely through the allocator: with GGML_CUDA_ENABLE_UNIFIED_MEMORY
// present, ggml_cuda_device_malloc() calls hipMallocManaged() + hipMemAdvise(
// hipMemAdviseSetCoarseGrain) instead of hipMalloc(). Everything else -- the
// hipMemcpy that uploads weights, the kernels that read them -- is unchanged.
// So this reproduces exactly that: upload a known pattern with hipMemcpy H2D,
// read it back from a kernel, and count the elements that came back wrong.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHK(expr)                                                              \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            printf("    hip error: %s -> %s\n", #expr, hipGetErrorString(_e));  \
            fflush(stdout);                                                    \
            return false;                                                      \
        }                                                                      \
    } while (0)

__host__ __device__ static inline uint32_t pattern(uint64_t i) {
    return (uint32_t)((i * 2654435761ull) ^ (i >> 13) ^ 0x9E3779B9ull);
}

__global__ void k_verify(const uint32_t * p, uint64_t n,
                         unsigned long long * bad, unsigned long long * sum) {
    uint64_t idx    = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long b = 0, s = 0;
    for (uint64_t i = idx; i < n; i += stride) {
        uint32_t got = p[i];
        s += got;
        if (got != pattern(i)) b++;
    }
    atomicAdd(bad, b);
    atomicAdd(sum, s);
}

__global__ void k_write(uint32_t * p, uint64_t n) {
    uint64_t idx    = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = idx; i < n; i += stride) p[i] = pattern(i) ^ 0xA5A5A5A5u;
}

enum AllocKind { PLAIN, MANAGED, MANAGED_COARSE };

static const char * kind_name(AllocKind k) {
    switch (k) {
        case PLAIN:          return "hipMalloc";
        case MANAGED:        return "hipMallocManaged";
        default:             return "hipMallocManaged+coarse";
    }
}

// fill_mode: 0 = hipMemcpy H2D (what ggml does), 1 = plain CPU stores into the
// pointer (only legal for managed memory, and the path a coherence bug would hit
// hardest).
static bool run_case(AllocKind kind, size_t bytes, int fill_mode, int device) {
    const uint64_t n = bytes / sizeof(uint32_t);
    printf("  [%s] %.2f GiB, fill=%s\n", kind_name(kind), bytes / 1073741824.0,
           fill_mode ? "cpu-store" : "hipMemcpy");
    fflush(stdout);

    size_t mfree = 0, mtotal = 0;
    if (hipMemGetInfo(&mfree, &mtotal) == hipSuccess) {
        printf("    before: hipMemGetInfo free=%.2f GiB total=%.2f GiB\n",
               mfree / 1073741824.0, mtotal / 1073741824.0);
    }

    uint32_t * p   = nullptr;
    hipError_t err = hipSuccess;
    double     t0  = 0;

    if (kind == PLAIN) {
        err = hipMalloc(&p, bytes);
    } else {
        err = hipMallocManaged(&p, bytes, hipMemAttachGlobal);
        if (err == hipSuccess && kind == MANAGED_COARSE) {
            hipError_t a = hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, device);
            printf("    hipMemAdvise(SetCoarseGrain) -> %s\n", hipGetErrorString(a));
            (void)hipGetLastError();
        }
    }
    if (err != hipSuccess) {
        printf("    ALLOC FAILED: %s\n", hipGetErrorString(err));
        (void)hipGetLastError();
        fflush(stdout);
        return true;  // an allocation that cannot be made is a finding, not an error
    }
    printf("    alloc ok, ptr=%p\n", (void *)p);
    fflush(stdout);

    unsigned long long expect = 0;

    if (fill_mode == 0) {
        const size_t     chunk = (64u << 20) / sizeof(uint32_t);
        std::vector<uint32_t> stage(chunk);
        for (uint64_t off = 0; off < n; off += chunk) {
            uint64_t m = std::min<uint64_t>(chunk, n - off);
            for (uint64_t j = 0; j < m; j++) {
                uint32_t v = pattern(off + j);
                stage[j]   = v;
                expect += v;
            }
            hipError_t ce = hipMemcpy(p + off, stage.data(), m * sizeof(uint32_t),
                                      hipMemcpyHostToDevice);
            if (ce != hipSuccess) {
                printf("    hipMemcpy H2D failed at offset %llu: %s\n",
                       (unsigned long long)off, hipGetErrorString(ce));
                (void)hipGetLastError();
                hipFree(p);
                fflush(stdout);
                return true;
            }
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            uint32_t v = pattern(i);
            p[i]       = v;
            expect += v;
        }
    }
    CHK(hipDeviceSynchronize());

    unsigned long long * d_bad = nullptr;
    unsigned long long * d_sum = nullptr;
    CHK(hipMalloc(&d_bad, sizeof(unsigned long long)));
    CHK(hipMalloc(&d_sum, sizeof(unsigned long long)));
    CHK(hipMemset(d_bad, 0, sizeof(unsigned long long)));
    CHK(hipMemset(d_sum, 0, sizeof(unsigned long long)));

    t0 = 0;
    hipLaunchKernelGGL(k_verify, dim3(4096), dim3(256), 0, 0, p, n, d_bad, d_sum);
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());
    (void)t0;

    unsigned long long h_bad = 0, h_sum = 0;
    CHK(hipMemcpy(&h_bad, d_bad, sizeof(h_bad), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(&h_sum, d_sum, sizeof(h_sum), hipMemcpyDeviceToHost));

    printf("    GPU read back: mismatched=%llu / %llu (%.4f%%), sum %s\n", h_bad,
           (unsigned long long)n, n ? 100.0 * h_bad / n : 0.0,
           h_sum == expect ? "MATCHES" : "DIFFERS");
    if (h_sum != expect) {
        printf("      sum got=%llu want=%llu\n", h_sum, expect);
    }

    // Now the other direction: kernel writes, host reads (sampled).
    hipLaunchKernelGGL(k_write, dim3(4096), dim3(256), 0, 0, p, n);
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());

    unsigned long long back_bad = 0, checked = 0;
    if (kind == PLAIN) {
        const size_t     chunk = (64u << 20) / sizeof(uint32_t);
        std::vector<uint32_t> stage(chunk);
        for (uint64_t off = 0; off < n; off += chunk) {
            uint64_t m = std::min<uint64_t>(chunk, n - off);
            if (hipMemcpy(stage.data(), p + off, m * sizeof(uint32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                (void)hipGetLastError();
                break;
            }
            for (uint64_t j = 0; j < m; j += 4096) {
                checked++;
                if (stage[j] != (pattern(off + j) ^ 0xA5A5A5A5u)) back_bad++;
            }
        }
    } else {
        for (uint64_t i = 0; i < n; i += 4096) {
            checked++;
            if (p[i] != (pattern(i) ^ 0xA5A5A5A5u)) back_bad++;
        }
    }
    printf("    host read back after kernel write: mismatched=%llu / %llu sampled\n",
           back_bad, checked);

    CHK(hipFree(d_bad));
    CHK(hipFree(d_sum));
    CHK(hipFree(p));
    fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    int device = 0;
    CHK(hipSetDevice(device));

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
        printf("device: %s (%s)\n", prop.name, prop.gcnArchName);
        printf("  totalGlobalMem            = %.2f GiB\n", prop.totalGlobalMem / 1073741824.0);
        printf("  integrated                = %d\n", prop.integrated);
        printf("  managedMemory             = %d\n", prop.managedMemory);
        printf("  concurrentManagedAccess   = %d\n", prop.concurrentManagedAccess);
        printf("  directManagedMemAccessFromHost = %d\n", prop.directManagedMemAccessFromHost);
        printf("  pageableMemoryAccess      = %d\n", prop.pageableMemoryAccess);
        printf("  canMapHostMemory          = %d\n", prop.canMapHostMemory);
    }
    const char * xnack = getenv("HSA_XNACK");
    printf("  HSA_XNACK env             = %s\n", xnack ? xnack : "(unset)");
    printf("\n");
    fflush(stdout);

    std::vector<double> sizes;
    for (int i = 1; i < argc; i++) sizes.push_back(atof(argv[i]));
    if (sizes.empty()) sizes = {1.0, 8.0, 24.0, 40.0, 56.0, 72.0, 88.0};

    for (double gib : sizes) {
        size_t bytes = (size_t)(gib * 1073741824.0);
        bytes &= ~(size_t)4095;
        printf("== %.2f GiB ==\n", gib);
        run_case(PLAIN, bytes, 0, device);
        run_case(MANAGED, bytes, 0, device);
        run_case(MANAGED_COARSE, bytes, 0, device);
        run_case(MANAGED_COARSE, bytes, 1, device);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
