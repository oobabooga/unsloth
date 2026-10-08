import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {128, 2, 3, 1}, {128, 2, 3, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, {128, 4, 3, 1})); // strided dst\n"
assert s.count(anchor) == 1
extra = "    // sd17 extra: row-copy kernel coverage\n"
for dt in ("GGML_TYPE_F32", "GGML_TYPE_F16"):
    for ne0 in (4, 8, 12, 36, 64, 128, 132, 260):
        for ps in ("{0, 2, 1, 3}", "{0, 2, 3, 1}", "{0, 3, 1, 2}", "{0, 1, 3, 2}"):
            extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{{ne0}, 5, 3, 2}}, {{-1,-1,-1,-1}}, {ps}));\n"
        extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{{ne0}, 5, 3, 2}}, {{-1,-1,-1,-1}}, {{0, 2, 1, 3}}, {{0, 2, 1, 3}}));\n"
        extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{{ne0}, 2, 3, 1}}, {{{ne0}, 2, 3, 1}}, {{0, 0, 0, 0}}, {{0, 0, 0, 0}}, false, {{{ne0}, 4, 3, 1}}));\n"
    for ne0 in (6, 10, 30):  # not a multiple of 4: must fall back
        extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{{ne0}, 5, 3, 2}}, {{-1,-1,-1,-1}}, {{0, 2, 1, 3}}));\n"
    extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{128, 42, 4777, 1}}, {{-1,-1,-1,-1}}, {{0, 2, 1, 3}}));\n"
    extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{64, 32, 4096, 1}}, {{-1,-1,-1,-1}}, {{0, 2, 1, 3}}));\n"
    extra += f"    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, {dt}, {{64, 4096, 32, 1}}, {{-1,-1,-1,-1}}, {{0, 2, 1, 3}}));\n"
s = s.replace(anchor, anchor + extra)
# perf-mode copies of the H3-sized permute
panchor = "    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {8192, 512, 2, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));\n"
assert s.count(panchor) == 1
pextra = ("    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {128, 42, 19108, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));\n"
          "    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F16,  {128, 42, 19108, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));\n"
          "    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F16,  {64, 32, 16384, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));\n")
s = s.replace(panchor, panchor + pextra)
open(p, "w").write(s)
print("added", extra.count("emplace_back"), "eval cases and", pextra.count("emplace_back"), "perf cases")
