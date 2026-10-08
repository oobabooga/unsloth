#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "runtime/tiling.h"
sd::Tensor<float> process_tiles_2d_old(const sd::Tensor<float>&, int, int, int, int, int, float, bool, bool, const TileProcessCallback&, bool);
sd::Tensor<float> process_tiles_2d_new(const sd::Tensor<float>&, int, int, int, int, int, float, bool, bool, const TileProcessCallback&, bool);

int main() {
    std::mt19937 rng(123);
    int cases = 0, fails = 0, both_throw = 0;
    struct Shape { int w, h; std::vector<int64_t> rest; };
    std::vector<Shape> shapes = {{40, 24, {16, 1}}, {64, 64, {4, 1}}, {37, 29, {3, 5}}, {60, 34, {31, 16}}, {17, 50, {2, 3, 2}}, {8, 8, {4, 2}}};
    for (auto& s : shapes)
    for (int scale : {1, 2, 4, 8})
    for (int mode = 0; mode < 2; mode++)             // 0 decode-like (output larger), 1 encode-like (input larger)
    for (int tile : {4, 7, 16, 32, 1000})
    for (float ov : {0.f, 0.125f, 0.25f, 0.5f})
    for (int circ = 0; circ < 4; circ++) {
        if (scale == 1 && mode == 1) continue;
        int small_w = s.w, small_h = s.h;
        int in_w = mode == 0 ? small_w : small_w * scale, in_h = mode == 0 ? small_h : small_h * scale;
        int out_w = mode == 0 ? small_w * scale : small_w, out_h = mode == 0 ? small_h * scale : small_h;
        std::vector<int64_t> shape = {in_w, in_h};
        for (auto r : s.rest) shape.push_back(r);
        auto in = sd::Tensor<float>::zeros(shape);
        std::normal_distribution<float> nd(0.f, 3.f);
        for (int64_t i = 0; i < in.numel(); i++) in.data()[i] = nd(rng);
        int out_c = 3;
        auto cb = [&](const sd::Tensor<float>& t) {
            int64_t tw = t.shape()[0], th = t.shape()[1];
            int64_t ow = mode == 0 ? tw * scale : tw / scale, oh = mode == 0 ? th * scale : th / scale;
            std::vector<int64_t> os = {ow, oh};
            for (size_t d = 2; d < t.shape().size(); d++) os.push_back(t.shape()[d]);
            if (os.size() > 2) os.back() = out_c;  // change channel count on the last dim
            auto o = sd::Tensor<float>::zeros(os);
            int64_t planes_out = o.numel() / (ow * oh), planes_in = t.numel() / (tw * th);
            for (int64_t p = 0; p < planes_out; p++)
                for (int64_t y = 0; y < oh; y++)
                    for (int64_t x = 0; x < ow; x++) {
                        int64_t sx = mode == 0 ? x / scale : x * scale, sy = mode == 0 ? y / scale : y * scale;
                        float a = t.data()[(p % planes_in) * tw * th + sy * tw + sx];
                        float b = t.data()[((p + 1) % planes_in) * tw * th + (th - 1 - sy) * tw + (tw - 1 - sx)];
                        o.data()[p * ow * oh + y * ow + x] = std::tanh(a) * 1.7f + b * 0.3f + 0.01f * (float)(x - y);
                    }
            return o;
        };
        bool cx = circ & 1, cy = circ & 2;
        sd::Tensor<float> a, b; std::string ea, eb;
        try { a = process_tiles_2d_old(in, out_w, out_h, scale, tile, tile, ov, cx, cy, cb, true); } catch (std::exception& e) { ea = e.what(); }
        try { b = process_tiles_2d_new(in, out_w, out_h, scale, tile, tile, ov, cx, cy, cb, true); } catch (std::exception& e) { eb = e.what(); }
        cases++;
        if (!ea.empty() || !eb.empty()) { both_throw += (ea == eb); if (ea != eb) { fails++; printf("THROW MISMATCH old='%s' new='%s'\n", ea.c_str(), eb.c_str()); } continue; }
        bool same = a.shape() == b.shape() && a.numel() == b.numel() && memcmp(a.data(), b.data(), a.numel() * sizeof(float)) == 0;
        if (!same) { fails++; if (fails < 10) printf("DIFF shape=%dx%d scale=%d mode=%d tile=%d ov=%g circ=%d\n", s.w, s.h, scale, mode, tile, ov, circ); }
    }
    printf("cases=%d compared=%d both_threw_same=%d bitwise_diffs=%d\n", cases, cases - both_throw, both_throw, fails);
    return fails != 0;
}
