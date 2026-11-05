"""Harness only: add RMS_NORM perf cases (per-head Q/K norm shapes and one long-row control) to
test-backend-ops' make_test_cases_perf(). Applied identically to base and head builds."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
anchor = "static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n"
assert s.count(anchor) == 1, "anchor not found"
cases = (
    "    // sdrc harness: per-head Q/K RMS norm rows (ne0 = head dim) and a long-row control\n"
    "    for (int64_t hd : {64, 128}) {\n"
    "        test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {hd, 32, 4096, 1}, false, 1e-6f));\n"
    "        test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {hd, 24, 1024, 1}, false, 1e-6f));\n"
    "    }\n"
    "    test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {3072, 4096, 1, 1}, false, 1e-6f));\n"
)
p.write_text(s.replace(anchor, anchor + cases), encoding="utf-8")
print("added RMS_NORM perf cases")
