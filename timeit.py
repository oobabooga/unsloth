"""timeit.py OUT CMD...: run CMD, write 'wall user sys' (children rusage) to OUT, exit with CMD's code."""
import resource, subprocess, sys, time
out, cmd = sys.argv[1], sys.argv[2:]
t0 = time.monotonic()
rc = subprocess.run(cmd).returncode
wall = time.monotonic() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
open(out, "w").write(f"{wall:.2f} {ru.ru_utime:.2f} {ru.ru_stime:.2f}\n")
sys.exit(rc)
