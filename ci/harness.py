# Make sd-cli also write the H3 audio when frames go to an image sequence (test harness only).
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);\n        return sucessful_reults != 0;\n'
new = '        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);\n        write_audio_sidecar(get_video_audio_sidecar_path(cli_params));\n        return sucessful_reults != 0;\n'
assert s.count(old) == 1, "harness insertion point not found"
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("harness applied to", p)
