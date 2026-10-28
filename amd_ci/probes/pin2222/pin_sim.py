"""Real pinned-allocation scenarios for unsloth-zoo PR 2222 (copy of scripts/sim2222_addendum/pin_sim.py plus env fields). Usage: pin_sim.py <scenario> <arm>.
Settings come from PYTORCH_ALLOC_CONF and the cap from UNSLOTH_GC_PINNED_MAX_GB (set by the driver)."""
import json, os, sys, threading

MIB = 1 << 20
scen, arm = sys.argv[1], sys.argv[2]
import torch
from unsloth_zoo import gradient_checkpointing as gc

dev_src = torch.randint(0, 255, (16 * MIB,), dtype = torch.uint8, device = "cuda")
torch.cuda.synchronize()
CAP = int(float(os.environ["UNSLOTH_GC_PINNED_MAX_GB"]) * (1 << 30))
events, track = [], [0]


def stats():
    s = torch.cuda.host_memory_stats()
    return int(s["allocated_bytes.current"]), int(s["allocated_bytes.peak"])


def note(ev, **kw):
    h, p = stats()
    track[0] = max(track[0], h)
    events.append(dict(ev = ev, held_mib = round(h / MIB, 3), peak_mib = round(p / MIB, 3), **kw))


def ours(nbytes, tag = ""):
    """What the offload pool does: _new_host_buffer, then a non_blocking D2H copy into it."""
    buf = gc._new_host_buffer(nbytes, torch.uint8)
    pinned = bool(buf.is_pinned())
    if pinned: buf.copy_(dev_src[:nbytes], non_blocking = True)
    torch.cuda.synchronize()
    note(f"ours {nbytes / MIB:g}MiB{tag}", pinned = pinned)
    return buf


def release(lst):
    """Release the last pinned buffer as _release_pool_pinned does, then drop it."""
    idx = [i for i, b in enumerate(lst) if b.is_pinned()]
    if not idx:
        note("release: none pinned"); return
    buf = lst.pop(idx[-1])
    n = gc._storage_bytes(buf)
    gc._release_pinned(n)
    del buf
    note(f"release ours {n / MIB:g}MiB")


def foreign(nbytes):
    t = torch.empty(nbytes, dtype = torch.uint8, device = "cpu", pin_memory = True)
    note(f"foreign pin {nbytes / MIB:g}MiB")
    return t


def free_foreign(F, empty = True):
    F.clear()
    torch.cuda.synchronize()
    if empty: torch._C._host_emptyCache()
    note("free foreign" + (" + empty_host_cache" if empty else ""))


def _settings():
    g = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if g is None: return None
    try: return g()
    except Exception as e: return f"error: {type(e).__name__}: {e}"


def _policy():
    f = getattr(gc, "_host_alloc_policy", None)
    if f is None: return None
    try: return [str(x) for x in f()]
    except Exception as e: return f"error: {type(e).__name__}: {e}"


L, F = [], []
if scen in ("1", "1b"):
    # Fallback review scenario 1 (1b = unit test sequence, no empty_host_cache).
    F.append(foreign(6 * MIB))
    L += [ours(3 * MIB), ours(3 * MIB)]
    release(L)
    free_foreign(F, empty = scen == "1")
    L.append(ours(5 * MIB))
elif scen == "2":
    # Codex scenario 3: a 3 MiB request reuses a cached 3.5 MiB block.
    F.append(foreign(6 * MIB))
    L += [ours(int(3.5 * MIB)), ours(int(3.5 * MIB))]
    release(L)
    L.append(ours(3 * MIB))
    free_foreign(F, empty = True)
    L += [ours(3 * MIB), ours(MIB // 2)]
elif scen == "3":
    # Scenario 2: A reserves 2 MiB and stalls before torch pins it; B asks for 3 MiB meanwhile.
    L.append(ours(3 * MIB))
    F.append(foreign(5 * MIB))
    real_empty = torch.empty
    inside, go = threading.Event(), threading.Event()

    def stalled_empty(*args, pin_memory = False, **kwargs):
        if pin_memory and threading.current_thread().name == "A" and not inside.is_set():
            inside.set()
            go.wait(10)
        return real_empty(*args, pin_memory = pin_memory, **kwargs)

    torch.empty = stalled_empty
    out = {}
    a = threading.Thread(target = lambda: out.update(a = gc._new_host_buffer(2 * MIB, torch.uint8)), name = "A")
    b = threading.Thread(target = lambda: out.update(b = gc._new_host_buffer(3 * MIB, torch.uint8)), name = "B")
    a.start()
    assert inside.wait(10), "A never reached the pinned allocation"
    b.start()
    b.join(1)
    b_done_while_a_stalled = not b.is_alive()
    note("B while A stalled", b_finished = b_done_while_a_stalled,
         b_pinned = bool(out["b"].is_pinned()) if "b" in out else None)
    go.set()
    a.join(10); b.join(10)
    torch.empty = real_empty
    note("A and B done", a_pinned = bool(out["a"].is_pinned()), b_pinned = bool(out["b"].is_pinned()))
    L += [out["a"], out["b"]]
else:
    raise SystemExit(f"unknown scenario {scen}")

held, peak = stats()
decisions = [e["pinned"] for e in events if "pinned" in e]
if scen == "3": decisions += [events[-1]["a_pinned"], events[-1]["b_pinned"]]
print("RESULT " + json.dumps(dict(
    scenario = scen, arm = arm, torch = torch.__version__, conf = os.environ.get("PYTORCH_ALLOC_CONF"),
    hip = torch.version.hip, arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
    alloc_settings = _settings(), host_policy = _policy(),
    zoo_file = gc.__file__, cap_mib = CAP / MIB, module_cap = getattr(gc, "GC_PINNED_CAP", None),
    peak_mib = round(peak / MIB, 3), tracked_max_mib = round(track[0] / MIB, 3), final_held_mib = round(held / MIB, 3),
    over_cap = peak > CAP, decisions = decisions, events = events,
)), flush = True)
