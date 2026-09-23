"""Compile one shard of a torch extension's objects into ccache, then stop before linking.

Usage, from the package's source directory:

  prebuilt_wheels_shard.py SHARD SHARDS

flash-attn is 97 nvcc invocations of several minutes each, longer in total than GitHub's
six-hour job limit. The warm jobs each run setup.py with torch's ninja step replaced by this
one, which compiles every SHARDS-th object starting at SHARD, so ccache holds all of them by
the time the build job runs the real `bdist_wheel` and gets cache hits.

The command lines are torch's own: setup.py still writes build.ninja, and only the set of
targets ninja is asked for changes. That is what makes the objects hit in the build job.
"""

import os
import runpy
import subprocess
import sys

import torch.utils.cpp_extension as cpp_extension


def main() -> None:
    shard, shards = int(sys.argv[1]), int(sys.argv[2])
    if not 0 <= shard < shards:
        raise SystemExit(f"shard {shard} is outside 0..{shards - 1}")

    def compile_shard(build_directory, verbose, error_prefix):
        listing = subprocess.run(
            ["ninja", "-t", "targets", "all"],
            cwd = build_directory,
            capture_output = True,
            text = True,
            check = True,
        ).stdout
        objects = sorted(
            {
                line.split(":")[0]
                for line in listing.splitlines()
                if line.split(":")[0].endswith(".o")
            }
        )
        mine = objects[shard::shards]
        print(f"shard {shard + 1} of {shards}: {len(mine)} of {len(objects)} objects", flush = True)
        jobs = os.environ.get("MAX_JOBS", "1")
        subprocess.run(["ninja", "-v", "-j", jobs, *mine], cwd = build_directory, check = True)
        # Nothing to link: the objects are what ccache keeps.
        raise SystemExit(0)

    cpp_extension._run_ninja_build = compile_shard
    sys.argv = ["setup.py", "build_ext"]
    runpy.run_path("setup.py", run_name = "__main__")
    raise SystemExit("setup.py returned without reaching the ninja step")


if __name__ == "__main__":
    main()
