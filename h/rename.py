import sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
out = zipfile.ZipFile(dst, "w"); n = 0
for i in zipfile.ZipFile(src).infolist():
    name = i.filename; base = name.rsplit("/", 1)[-1]
    if base.lower().startswith("libomp140."):
        name = name[: len(name) - len(base)] + "libomp.dll"; n += 1
    out.writestr(name, b"")
out.close(); print(dst, "renamed entries:", n)
assert n == 1
