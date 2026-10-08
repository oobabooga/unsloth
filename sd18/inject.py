# Adds a whole-graph test of MUL_MAT(F16 weight) + bias ADD [+ MUL scale + ADD residual | SWIGLU]
# to test-backend-ops, i.e. the graph shapes the patch 0003 cuBLAS epilogue fusion matches.
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as f:
    src = f.read()

STRUCT = r'''
// sd18 check: whole graph, so a backend may fuse the elementwise tail into the matmul.
// mode 0: mm + bias, 1: residual + (mm + bias) * scale, 2: swiglu(mm + bias)
struct test_sd18_mm_epilogue : public test_case {
    const int mode;
    const int64_t k;
    const int64_t n;
    const int64_t m;
    const bool inplace;

    test_sd18_mm_epilogue(int mode, int64_t k, int64_t n, int64_t m, bool inplace)
        : mode(mode), k(k), n(n), m(m), inplace(inplace) {}

    std::string vars() override { return VARS_TO_STR5(mode, k, n, m, inplace); }
    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "SD18_MM_EPILOGUE"; }
    bool run_whole_graph() override { return true; }
    double max_nmse_err() override { return 5e-4; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
        ggml_tensor * out = inplace ? ggml_add_inplace(ctx, mm, b) : ggml_add(ctx, mm, b);
        if (mode == 1) {
            ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_tensor * r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, m);
            out = ggml_add(ctx, ggml_mul(ctx, out, s), r);
        } else if (mode == 2) {
            out = ggml_swiglu(ctx, out);
        }
        ggml_set_name(out, "out");
        return out;
    }
};

'''

REG = '''
    for (int mode : {0, 1, 2}) {
        for (bool inplace : {false, true}) {
            test_cases.emplace_back(new test_sd18_mm_epilogue(mode, 256, 384, 512, inplace));
            test_cases.emplace_back(new test_sd18_mm_epilogue(mode, 320, 128, 77, inplace));
            test_cases.emplace_back(new test_sd18_mm_epilogue(mode, 1024, 768, 300, inplace));
            test_cases.emplace_back(new test_sd18_mm_epilogue(mode, 128, 64, 4, inplace));
        }
    }
'''

pat = re.compile(r"static std::vector<std::unique_ptr<test_case>> make_test_cases_eval\(\) \{\r?\n"
                 r"    std::vector<std::unique_ptr<test_case>> test_cases;\r?\n")
matches = list(pat.finditer(src))
assert len(matches) == 1, len(matches)
mt = matches[0]
src = src[:mt.start()] + STRUCT + src[mt.start():mt.end()] + REG + src[mt.end():]
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("injected SD18_MM_EPILOGUE into", path)
