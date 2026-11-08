// sdxp FA probe: uniform attention (Q = 0) over N keys with V in [0.5, 1.5], GGML_PREC_F32.
// Exact answer: out[d,h,i] = mean_j V[d,j,h]. FP16 accumulation of sum_j V passes 65504 near N = 65536.
// usage: sdxp-fa-probe <device> <out.jsonl>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    const char * dev_name = argc > 1 ? argv[1] : "ROCm0";
    FILE * fo = fopen(argc > 2 ? argv[2] : "fa_probe.jsonl", "w");
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (!dev) { fprintf(fo, "{\"error\": \"no device %s\"}\n", dev_name); fclose(fo); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    const int D = 128, H = 8, NQ = 4096;
    for (int N : {4096, 32768, 61440, 73728, 147456}) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, H, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N, H, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N, H, 1);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> hq((size_t) D * NQ * H, 0.0f);
        std::vector<ggml_fp16_t> hk((size_t) D * N * H), hv((size_t) D * N * H);
        std::vector<double> mean((size_t) D * H, 0.0);
        for (int h = 0; h < H; ++h) {
            for (int j = 0; j < N; ++j) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx = d + (size_t) D * (j + (size_t) N * h);
                    hk[idx] = ggml_fp32_to_fp16(((j * 31 + d * 17 + h * 7) % 200) / 100.0f - 1.0f);
                    const float vv = 0.5f + ((j * 7 + d * 13 + h) % 101) / 100.0f;
                    hv[idx] = ggml_fp32_to_fp16(vv);
                    mean[d + (size_t) D * h] += ggml_fp16_to_fp32(hv[idx]);
                }
            }
        }
        for (auto & m : mean) m /= N;
        ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, hk.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, hv.data(), 0, ggml_nbytes(v));
        ggml_backend_graph_compute(be, gf);  // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        const ggml_status st = ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> ho((size_t) D * H * NQ);
        ggml_backend_tensor_get(o, ho.data(), 0, ggml_nbytes(o));
        size_t nonfinite = 0, bad_tokens = 0;
        double max_err = 0.0, max_abs = 0.0;
        for (int i = 0; i < NQ; ++i) {
            for (int h = 0; h < H; ++h) {
                bool bad = false;
                for (int d = 0; d < D; ++d) {
                    const float x = ho[d + (size_t) D * (h + (size_t) H * i)];
                    if (!std::isfinite(x)) { ++nonfinite; bad = true; continue; }
                    const double e = fabs(x - mean[d + (size_t) D * h]);
                    if (e > max_err) max_err = e;
                    if (fabs(x) > max_abs) max_abs = fabs(x);
                    if (e > 0.05) bad = true;
                }
                bad_tokens += bad;
            }
        }
        fprintf(fo, "{\"device\": \"%s\", \"N\": %d, \"NQ\": %d, \"D\": %d, \"H\": %d, \"status\": %d, \"ms\": %.3f, "
                    "\"nonfinite\": %zu, \"bad_token_heads\": %zu, \"of\": %d, \"max_abs_err\": %.6g, \"max_abs_out\": %.6g}\n",
                dev_name, N, NQ, D, H, (int) st, ms, nonfinite, bad_tokens, NQ * H, max_err, max_abs);
        fflush(fo);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    fclose(fo);
    return 0;
}
