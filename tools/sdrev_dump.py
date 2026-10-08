# Review-only: make sd-cli dump the raw float latent, decoded video and audio waveform when
# SDREV_DUMP=<dir> is set, so builds can be compared bit for bit. Usage: python3 sdrev_dump.py <src-root>
import sys, os
p = os.path.join(sys.argv[1], "src/pipeline/video.cpp")
s = open(p).read()
helper = '''
namespace {
    template <typename Tn>
    void sdrev_dump(const char* name, const Tn& t) {
        const char* d = std::getenv("SDREV_DUMP");
        if (d == nullptr || t.empty()) return;
        std::string path = std::string(d) + "/" + name;
        if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(t.data(), sizeof(float), (size_t)t.numel(), f); std::fclose(f); }
    }
}
'''
anchor = "namespace sd::pipeline {"
assert s.count(anchor) == 1
s = s.replace(anchor, "#include <cstdio>\n#include <string>\n" + helper + "\n" + anchor, 1)
a1 = '        LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);\n'
assert s.count(a1) == 1
s = s.replace(a1, a1 + '        sdrev_dump("latent.f32", video_latent);\n        sdrev_dump("vid.f32", vid);\n', 1)
a2 = "                auto waveform = sd->decode_ltx_audio_latent(audio_latent);\n"
assert s.count(a2) == 1
s = s.replace(a2, a2 + '                sdrev_dump("audio_latent.f32", audio_latent);\n                sdrev_dump("audio.f32", waveform);\n', 1)
open(p, "w").write(s)
print("patched", p)
