"""pre-commit with an optional simulated CPU count (AB_CPUS), so partitioning matches a bigger host."""
import os, sys
import pre_commit.xargs as x
n = os.environ.get("AB_CPUS")
if n:
    x.cpu_count = lambda: int(n)
from pre_commit.main import main
raise SystemExit(main())
