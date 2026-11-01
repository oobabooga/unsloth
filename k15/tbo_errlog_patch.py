# tbo_errlog_patch.py TREE (k15 q4_1 ticket, coordinator 11-01): a TEST-ONLY change to TREE/tests/test-backend-ops.cpp, applied after the
# tree check, before the build. GGML_TBO_LOG_ERR=1 prints "[OP] ERR_ALL = x" for every compared output (not only failures);
# GGML_TBO_SEED=N seeds the uniform inputs from N, the tensor's name and shape and the chunk start (identical inputs for the same case in
# two builds), instead of std::random_device. Unset: the stock behaviour. Exit 1 when an anchor is missing.
import sys
p = sys.argv[1] + "/tests/test-backend-ops.cpp"
s = open(p, encoding="utf-8").read()
a1 = "static void init_tensor_uniform(ggml_tensor * tensor"
a2 = "            thread_local std::default_random_engine gen(std::random_device{}());"
a3 = "            double err = ud->tc->err(f1.data(), f2.data(), f1.size());"
for a in (a1, a2, a3):
    if s.count(a) != 1:
        print("tbo_errlog_patch: anchor not found once: " + a.strip()); sys.exit(1)
s = s.replace(a1, '''static uint32_t tbo_seed_hash(const ggml_tensor * t, size_t start) {
    uint32_t h = 2166136261u;
    for (const char * c = t->name; *c; ++c) { h ^= (uint8_t) *c; h *= 16777619u; }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) { h ^= (uint32_t) t->ne[i]; h *= 16777619u; }
    h ^= (uint32_t) start; h *= 16777619u;
    return h;
}

''' + a1, 1)
s = s.replace(a2, '''            static const char * tbo_seed = getenv("GGML_TBO_SEED");
            thread_local std::default_random_engine gen_r(std::random_device{}());
            std::default_random_engine gen_s(tbo_seed ? (uint32_t) atoi(tbo_seed) * 2654435761u ^ tbo_seed_hash(tensor, start) : 0u);
            std::default_random_engine & gen = tbo_seed ? gen_s : gen_r;''', 1)
s = s.replace(a3, a3 + '''
            static const bool tbo_log_err = getenv("GGML_TBO_LOG_ERR") != nullptr;
            if (tbo_log_err) {
                printf("[%s] ERR_ALL = %.9f ", ggml_op_desc(t1), err);
            }''', 1)
open(p, "w", encoding="utf-8").write(s)
print("tbo_errlog_patch: applied to " + p)
