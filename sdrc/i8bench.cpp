// HIP INT8 linear as sd.cpp's fallback runs it (src/core/cuda/int8.cu): hipblasGemmEx int8 x int8 -> full INT32
// accumulator, then a separate scale/bias epilogue to FP32. Times both and reports the INT32 temporary size.
// usage: i8bench <out.jsonl>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

__global__ void dequant_vec4(const int4 * acc, const float * xs, const float4 * ws, const float4 * bias, float4 * out, int n4, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= n4 || r >= rows) return;
    const int4 a = acc[(size_t) r * n4 + c];
    const float s = xs[r];
    const float4 w = ws[c], b = bias[c];
    out[(size_t) r * n4 + c] = make_float4(a.x * s * w.x + b.x, a.y * s * w.y + b.y, a.z * s * w.z + b.z, a.w * s * w.w + b.w);
}

#define CK(x) do { auto e_ = (x); if (e_ != 0) { fprintf(fo, "{\"error\": \"%s = %d\"}\n", #x, (int) e_); fclose(fo); return 1; } } while (0)

int main(int argc, char ** argv) {
    FILE * fo = fopen(argc > 1 ? argv[1] : "i8bench.jsonl", "w");
    hipblasHandle_t h; CK(hipblasCreate(&h));
    struct S { int M, N, K; } shapes[] = {{4096, 3072, 3072}, {4096, 12288, 3072}, {19108, 14336, 4096}};
    for (auto sh : shapes) {
        const int M = (sh.M + 3) / 4 * 4, N = sh.N, K = sh.K;
        int8_t *A, *W; int32_t *C; float *xs, *ws, *bias, *out;
        CK(hipMalloc(&A, (size_t) M * K)); CK(hipMalloc(&W, (size_t) N * K)); CK(hipMalloc(&C, (size_t) M * N * 4));
        CK(hipMalloc(&xs, M * 4)); CK(hipMalloc(&ws, N * 4)); CK(hipMalloc(&bias, N * 4)); CK(hipMalloc(&out, (size_t) M * N * 4));
        CK(hipMemset(A, 1, (size_t) M * K)); CK(hipMemset(W, 1, (size_t) N * K)); CK(hipMemset(xs, 0, M * 4));
        CK(hipMemset(ws, 0, N * 4)); CK(hipMemset(bias, 0, N * 4));
        const int32_t alpha = 1, beta = 0;
#if HIP_VERSION >= 60500000
        const auto ct = HIPBLAS_COMPUTE_32I;
#else
        const auto ct = HIPBLAS_R_32I;
#endif
        auto gemm = [&]() { return hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, W, HIPBLAS_R_8I, K, A, HIPBLAS_R_8I, K,
                                                 &beta, C, HIPBLAS_R_32I, N, ct, HIPBLAS_GEMM_DEFAULT); };
        auto epi = [&]() { dequant_vec4<<<dim3((N / 4 + 255) / 256, M), 256>>>((const int4 *) C, xs, (const float4 *) ws, (const float4 *) bias, (float4 *) out, N / 4, M); };
        CK(gemm()); epi(); CK(hipDeviceSynchronize());
        int32_t probe = 0; CK(hipMemcpy(&probe, C, 4, hipMemcpyDeviceToHost));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        std::vector<float> tg, te;
        for (int it = 0; it < 15; ++it) {
            float ms;
            CK(hipEventRecord(e0)); CK(gemm()); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); tg.push_back(ms);
            CK(hipEventRecord(e0)); epi(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); te.push_back(ms);
        }
        std::sort(tg.begin(), tg.end()); std::sort(te.begin(), te.end());
        const double g = tg[tg.size() / 2], e = te[te.size() / 2];
        fprintf(fo, "{\"M\": %d, \"N\": %d, \"K\": %d, \"gemm_ms\": %.4f, \"gemm_ms_min_max\": [%.4f, %.4f], \"epilogue_ms\": %.4f, "
                    "\"epilogue_ms_min_max\": [%.4f, %.4f], \"epilogue_share\": %.4f, \"int32_temp_MiB\": %.1f, \"gemm_TOPS\": %.2f, "
                    "\"epilogue_GBps\": %.1f, \"check_C0_eq_K\": %s}\n",
                sh.M, N, K, g, tg.front(), tg.back(), e, te.front(), te.back(), e / (g + e), (double) M * N * 4 / 1048576.0,
                2.0 * M * N * K / (g * 1e9), (double) M * N * 8 / (e * 1e6), probe == K ? "true" : "false");
        fflush(fo);
        hipFree(A); hipFree(W); hipFree(C); hipFree(xs); hipFree(ws); hipFree(bias); hipFree(out);
    }
    hipblasDestroy(h);
    fclose(fo);
    return 0;
}
