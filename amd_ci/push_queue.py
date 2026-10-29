#!/usr/bin/env python3
"""Host-wide FIFO for pushing AMD CI branches, so queued runs wait their turn instead of evicting each other.

GitHub keeps one running job plus ONE waiter per concurrency group and a third arrival cancels the
waiter. Every exclusive GPU job joins `amd-ci-gfx1151-gpu`, so sessions pushing at once evict each
other's queued measurements. This queue sits in front of the push:

  exclusive  at most one of this host's exclusive jobs is pushed-but-not-running at a time; the next
             is pushed only once the previous one is running or done
  shared     (`scaffold.py --shared-gpu`, suites-only, Windows) up to --max-shared per OS in flight
  evicted    a GPU job cancelled before any of its steps ran goes back to the FRONT of the queue
  failed     a push that fails frees its slot at once; three failures (or a deleted dir) end it

  python amd_ci/push_queue.py submit --dir ci_pr123 --branch amd-ci-pr123   # enqueue, dispatch, wait
  python amd_ci/push_queue.py submit ... --no-wait                          # enqueue and return
  python amd_ci/push_queue.py tick                                          # one dispatch pass
  python amd_ci/push_queue.py status

State lives in a shared lock dir (gpu_queue's when importable, else AMD_CI_QUEUE_DIR or
~/.cache/amd_ci_queue), so every session and workspace on the host sees one queue. Pushing a branch by
hand still works; it just skips the queue.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HOST_REPO = "oobabooga/unsloth"
REMOTE = "https://github.com/oobabooga/unsloth.git"
GPU_GROUP = "amd-ci-gfx1151-gpu"
MAX_SHARED = int(os.environ.get("AMD_CI_QUEUE_MAX_SHARED", "12"))  # of ~15 runners per OS
POLL_S = int(os.environ.get("AMD_CI_QUEUE_POLL_S", "120"))
MAX_ATTEMPTS = int(os.environ.get("AMD_CI_QUEUE_MAX_ATTEMPTS", "20"))
MAX_PUSH_FAILURES = 3
GH_TIMEOUT_S = 60
PUSH_TIMEOUT_S = 300


def queue_dir() -> Path:
    env = os.environ.get("AMD_CI_QUEUE_DIR")
    if env:
        d = Path(env)
    else:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            import gpu_queue  # noqa: PLC0415  (shared host lock dir when the repo ships it)
            d = Path(gpu_queue.lock_dir()) / "amd_ci_queue"
        except Exception:  # noqa: BLE001
            d = Path.home() / ".cache" / "amd_ci_queue"
    d.mkdir(parents = True, exist_ok = True)
    try:
        os.chmod(d, 0o777)
    except OSError:
        pass
    return d


@contextmanager
def locked_state():
    d = queue_dir()
    try:
        fd = os.open(d / "queue.lock", os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        # Created 0644 by another user (umask): flock only needs an open descriptor, read-only is enough.
        fd = os.open(d / "queue.lock", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        path = d / "queue.json"
        state = json.loads(path.read_text(encoding = "utf-8")) if path.exists() else {"tickets": []}
        yield state
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent = 1), encoding = "utf-8")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o666)
        except OSError:
            pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def tick_lock():
    """One tick at a time host-wide, without blocking: True when this process holds it. The state
    lock is only held to read and to write back, never across a `gh` call or a push."""
    fd = os.open(queue_dir() / "tick.lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def workflow_traits(directory: Path) -> tuple[str, str]:
    """(mode, os) from the branch's workflow: exclusive iff any job joins the shared GPU group."""
    texts = [p.read_text(encoding = "utf-8") for p in (directory / ".github" / "workflows").glob("*.yml")]
    if not texts:
        raise SystemExit(f"{directory} has no .github/workflows/*.yml; scaffold it first")
    text = "\n".join(texts)
    mode = "exclusive" if re.search(rf"group:\s*{GPU_GROUP}\b", text) else "shared"
    # The Windows template also carries a tiny linux-control job; the branch counts as Windows.
    osn = "windows" if re.search(r"runs-on:\s*\[\s*self-hosted\s*,\s*Windows\b", text) else "linux"
    return mode, osn


def group_jobs(directory: Path) -> list[str]:
    """Run-time names of the jobs that join the GPU group. A run turns in_progress as soon as ANY
    job starts (the suites job has no group), so only these say whether the GPU job left the
    group's queue or was evicted from it."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return []
    names = []
    for p in sorted((directory / ".github" / "workflows").glob("*.yml")):
        try:
            doc = yaml.safe_load(p.read_text(encoding = "utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        for key, job in (doc.get("jobs") or {}).items():
            conc = (job or {}).get("concurrency")
            group = conc.get("group") if isinstance(conc, dict) else conc
            if str(group or "").strip() == GPU_GROUP:
                names.append(str(job.get("name") or key))
    return names


def _ticket_gpu_jobs(ticket: dict, jobs: list[dict]) -> list[dict]:
    names = set(ticket.get("gpu_jobs") or [])
    if names:
        return [j for j in jobs if j.get("name") in names]
    return [j for j in jobs if re.search(r"GPU|exclusive", j.get("name") or "")]


def _job_ran_steps(job: dict) -> bool:
    return any(s.get("conclusion") in ("success", "failure") for s in job.get("steps") or [])


def plan(tickets: list[dict], max_shared: int = MAX_SHARED) -> list[str]:
    """Ticket ids to push now, oldest first. Pure: selftest drives it directly.

    exclusive: nothing pushed while one of ours is still `pushed` (waiting in GitHub's group).
    shared: per OS, pushed + running stays at or under max_shared.
    """
    live = [t for t in tickets if t["state"] in ("waiting", "pushed", "running")]
    order = sorted(live, key = lambda t: (t.get("priority", 0), t["created"]))
    out: list[str] = []
    excl_blocked = any(t["mode"] == "exclusive" and t["state"] == "pushed" for t in live)
    inflight: dict[str, int] = {}
    for t in live:
        if t["mode"] == "shared" and t["state"] in ("pushed", "running"):
            inflight[t["os"]] = inflight.get(t["os"], 0) + 1
    for t in order:
        if t["state"] != "waiting":
            continue
        if t["mode"] == "exclusive":
            if not excl_blocked:
                out.append(t["id"])
                excl_blocked = True
        elif inflight.get(t["os"], 0) < max_shared:
            out.append(t["id"])
            inflight[t["os"]] = inflight.get(t["os"], 0) + 1
    return out


def _requeue(ticket: dict) -> None:
    ticket["attempts"] = ticket.get("attempts", 0) + 1
    if ticket["attempts"] >= MAX_ATTEMPTS:
        ticket["state"], ticket["conclusion"] = "done", "evicted"
    else:
        ticket["state"], ticket["priority"] = "waiting", -1


def apply_run(ticket: dict, run: dict | None, gpu_steps_ran: bool | None, jobs: list[dict] | None = None) -> None:
    """Advance a pushed / running ticket from its latest run. An eviction requeues at the front.

    With `jobs` (the run's jobs) an exclusive ticket is judged by its GPU-group jobs: `running` only
    once one of them has started, and evicted when one was cancelled without running a step, even
    if the suites job ran. Without them, the run's own status decides (a run with no other job)."""
    if run is None:
        return
    ticket["run_id"] = run.get("databaseId")
    gpu = _ticket_gpu_jobs(ticket, jobs) if jobs and ticket.get("mode") == "exclusive" else []
    if gpu:
        if any(j.get("conclusion") == "cancelled" and not _job_ran_steps(j) for j in gpu):
            _requeue(ticket)   # evicted from the group; the suites job may still be running
        elif run.get("status") == "completed":
            ticket["state"], ticket["conclusion"] = "done", run.get("conclusion")
        elif any(j.get("status") in ("in_progress", "completed") for j in gpu):
            ticket["state"] = "running"
        return
    if run.get("status") in ("in_progress",):
        ticket["state"] = "running"
    elif run.get("status") == "completed":
        if run.get("conclusion") == "cancelled" and gpu_steps_ran is False and ticket["state"] == "pushed":
            _requeue(ticket)
        else:
            ticket["state"], ticket["conclusion"] = "done", run.get("conclusion")


def _gh_json(args: list[str]):
    try:
        p = subprocess.run(["gh", *args], capture_output = True, text = True, timeout = GH_TIMEOUT_S)
        return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def latest_run(branch: str, since: float) -> dict | None:
    runs = _gh_json(["run", "list", "-R", HOST_REPO, "-b", branch, "-L", "10",
                     "--json", "databaseId,status,conclusion,createdAt"]) or []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since - 30))
    runs = [r for r in runs if r.get("createdAt", "") >= stamp]
    return max(runs, key = lambda r: r["createdAt"]) if runs else None


def run_jobs(run_id) -> list[dict] | None:
    got = _gh_json(["run", "view", str(run_id), "-R", HOST_REPO, "--json", "jobs"])
    return got.get("jobs") if got else None


def gpu_steps_ran(run_id) -> bool | None:
    jobs = _gh_json(["run", "view", str(run_id), "-R", HOST_REPO, "--json", "jobs"])
    if not jobs:
        return None
    # The suites job stays green when only the GPU job is evicted, so judge the GPU job alone.
    all_jobs = jobs.get("jobs", [])
    gpu = [j for j in all_jobs if "GPU" in (j.get("name") or "")] or all_jobs
    return any(s.get("conclusion") in ("success", "failure") for j in gpu for s in j.get("steps") or [])


def _push_failed(ticket: dict, why: str) -> bool:
    ticket["push_failures"] = ticket.get("push_failures", 0) + 1
    ticket["last_error"] = why[-300:]
    if ticket["push_failures"] >= MAX_PUSH_FAILURES or not Path(ticket["dir"]).is_dir():
        ticket["state"], ticket["conclusion"] = "done", "push_failed"
    print(f"push failed for {ticket['branch']}: {why[-300:]}")
    return False


def push(ticket: dict) -> bool:
    d = Path(ticket["dir"])
    if not d.is_dir():
        return _push_failed(ticket, f"{d} no longer exists")
    git = ["git", "-C", str(d)]
    try:
        if not (d / ".git").exists():
            subprocess.run(git + ["init", "-q"], check = True, timeout = 60)
            subprocess.run(git + ["checkout", "-q", "-b", ticket["branch"]], check = True, timeout = 60)
            subprocess.run(git + ["add", "-A"], check = True, timeout = 120)
            subprocess.run(git + ["commit", "-q", "-m", f"AMD CI {ticket['branch']}"], check = True, timeout = 120)
        elif ticket.get("pushes", 0):
            subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "Rerun from the AMD CI queue"],
                           check = True, timeout = 120)
        p = subprocess.run(git + ["push", "-q", ticket.get("remote", REMOTE), f"HEAD:refs/heads/{ticket['branch']}"],
                           capture_output = True, text = True, timeout = PUSH_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        return _push_failed(ticket, str(e))
    if p.returncode != 0:
        return _push_failed(ticket, p.stderr.strip())
    ticket["pushes"] = ticket.get("pushes", 0) + 1
    ticket["state"], ticket["pushed_at"] = "pushed", time.time()
    print(f"{time.strftime('%H:%M')} pushed {ticket['branch']} ({ticket['mode']}, {ticket['os']})")
    return True


def dispatch(tickets: list[dict], max_shared: int = MAX_SHARED, push_fn = None) -> None:
    """Push what plan() picks. A failed push gives up its slot in the same pass (the next waiter
    is tried), so one bad ticket never blocks the queue behind it; after MAX_PUSH_FAILURES (or a
    missing dir) it is finished as `push_failed`."""
    push_fn = push_fn or push
    by_id = {t["id"]: t for t in tickets}
    failed: set[str] = set()
    for _ in range(len(tickets) + 1):
        retry = False
        for tid in plan([t for t in tickets if t["id"] not in failed], max_shared):
            if not push_fn(by_id[tid]):
                failed.add(tid)
                retry = True
        if not retry:
            break


def tick(max_shared: int = MAX_SHARED) -> None:
    with tick_lock() as mine:
        if not mine:
            return   # another session is ticking; it advances every ticket
        with locked_state() as state:
            tickets = json.loads(json.dumps(state["tickets"]))
        for t in tickets:
            if t["state"] in ("pushed", "running"):
                run = latest_run(t["branch"], t["pushed_at"])
                jobs = run_jobs(run["databaseId"]) if run and t.get("mode") == "exclusive" else None
                ran = gpu_steps_ran(run["databaseId"]) if not jobs and run and run.get("status") == "completed" \
                    and run.get("conclusion") == "cancelled" else None
                apply_run(t, run, ran, jobs)
        dispatch(tickets, max_shared)
        with locked_state() as state:
            mine_ids = {t["id"] for t in tickets}
            state["tickets"] = tickets + [t for t in state["tickets"] if t["id"] not in mine_ids]
            cutoff = time.time() - 7 * 86400
            state["tickets"] = [t for t in state["tickets"] if t["state"] != "done" or t["created"] > cutoff]


def cmd_submit(a) -> int:
    d = Path(a.dir).resolve()
    mode, osn = workflow_traits(d)
    tid = uuid.uuid4().hex[:10]
    with locked_state() as state:
        state["tickets"].append({"id": tid, "dir": str(d), "branch": a.branch, "remote": a.remote,
                                 "mode": mode, "os": osn, "state": "waiting", "created": time.time(),
                                 "gpu_jobs": group_jobs(d),
                                 "user": getpass.getuser(), "attempts": 0})
    print(f"queued {a.branch} as {tid} ({mode}, {osn})")
    while True:
        tick(a.max_shared)
        with locked_state() as state:
            t = next(x for x in state["tickets"] if x["id"] == tid)
        if t["state"] == "done":
            print(f"{a.branch}: {t.get('conclusion')} run={t.get('run_id')}"
                  + (f" ({t['last_error']})" if t.get("conclusion") == "push_failed" else ""))
            return 0 if t.get("conclusion") == "success" else 1
        if a.no_wait and t["state"] != "waiting":
            return 0
        time.sleep(POLL_S)


def cmd_status(_a) -> int:
    with locked_state() as state:
        tickets = list(state["tickets"])
    live = sorted((t for t in tickets if t["state"] != "done"),
                  key = lambda t: (t.get("priority", 0), t["created"]))
    print(f"queue dir {queue_dir()}  live {len(live)}  done (7 d) {len(tickets) - len(live)}")
    for i, t in enumerate(live):
        print(f"  {i + 1:>2} {t['state']:<8} {t['mode']:<9} {t['os']:<7} {t['branch']}  "
              f"user={t['user']} attempts={t.get('attempts', 0)} run={t.get('run_id')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    s = sub.add_parser("submit")
    s.add_argument("--dir", required = True)
    s.add_argument("--branch", required = True)
    s.add_argument("--remote", default = REMOTE)
    s.add_argument("--no-wait", action = "store_true", help = "return once pushed (keeps polling while it waits its turn)")
    s.add_argument("--max-shared", type = int, default = MAX_SHARED)
    t = sub.add_parser("tick")
    t.add_argument("--max-shared", type = int, default = MAX_SHARED)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "submit":
        return cmd_submit(a)
    if a.cmd == "tick":
        tick(a.max_shared)
        return 0
    return cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
