// ballast <keep_free_gib>: hold device memory until killed (test harness only).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
int main(int argc, char** argv) {
    double keep = atof(argv[1]) * (1ull << 30);
    size_t f = 0, t = 0; hipMemGetInfo(&f, &t);
    printf("before: free %.2f GiB total %.2f GiB\n", f / 1073741824.0, t / 1073741824.0);
    std::vector<void*> held; size_t chunk = 1ull << 30;
    while (true) { hipMemGetInfo(&f, &t); if (f <= keep + chunk) break; void* p; if (hipMalloc(&p, chunk) != hipSuccess) break; hipMemset(p, 1, chunk); held.push_back(p); }
    hipMemGetInfo(&f, &t);
    printf("held %zu GiB, free now %.2f GiB\n", held.size(), f / 1073741824.0); fflush(stdout);
    while (true) sleep(60);
}
