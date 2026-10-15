# AMD CI: llama.cpp IQ1_M MMQ on gfx1151

Builds public ggml-org/llama.cpp at BASE_SHA (base) and BASE_SHA + iq1m.patch (head)
with ggml-hip for gfx1151 in one job on one host, then runs test-backend-ops for both
via amd_ci/lib/differential.py (probe amd_ci/probes/llama_tbo_probe.py, criteria
amd_ci/criteria/llama_iq1m_mmq.py). The verdict lands in the iq1m-mmq-gpu artifact
(gpu/VERDICT.md, gpu/verdict.json).
