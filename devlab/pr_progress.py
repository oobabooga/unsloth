# Extracted verbatim from llama_cpp.py: _psutil_process_stat, _read_io_bytes, _read_rss_bytes, load_progress
import os
import re
from pathlib import Path
from typing import Optional

_EXTRA_WEIGHT_FILE_FLAGS = frozenset(
    {"--mmproj", "-mm", "--model-draft", "--spec-draft-model", "-md"}
)

def _extra_weight_file_bytes(argv) -> int:
    """Sizes of the projector and draft model a llama-server command line names, in either the
    ``--flag value`` or the ``--flag=value`` form."""
    if not isinstance(argv, (list, tuple)):
        return 0
    paths: list[str] = []
    tokens = [str(token) for token in argv]
    for index, token in enumerate(tokens):
        flag, eq, value = token.partition("=")
        if flag not in _EXTRA_WEIGHT_FILE_FLAGS:
            continue
        if eq:
            paths.append(value)
        elif index + 1 < len(tokens):
            paths.append(tokens[index + 1])
    total = 0
    for value in paths:
        try:
            path = Path(value)
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total

_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class LlamaCppBackend:
    @staticmethod
    def _read_rss_bytes(pid: int) -> Optional[int]:
        """Resident set size of ``pid`` in bytes, from /proc/<pid>/status (Linux).
        0 when the status has no VmRSS line (zombie / kernel thread); None where
        /proc is unavailable (macOS/Windows) or the value is unreadable."""
        if not os.path.isdir("/proc"):
            return LlamaCppBackend._psutil_process_stat(pid, "rss")
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # IndexError guards a "VmRSS:" line with no value column.
                        return int(line.split()[1]) * 1024  # kB -> bytes
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return 0  # readable but no VmRSS line

    @staticmethod
    def _read_io_bytes(pid: int) -> Optional[int]:
        """Bytes ``pid`` has read through read() calls so far, page cache hits included.

        What tracks a load that streams weights straight to VRAM: the process reads every tensor
        once, while its RSS stays at the few hundred MiB it keeps on the host. Linux reads ``rchar``
        from /proc/<pid>/io; Windows has psutil's read_bytes (ReadTransferCount). None where neither
        exists (macOS) or the process is gone."""
        if os.path.isdir("/proc"):
            try:
                with open(f"/proc/{pid}/io", "r", encoding = "utf-8") as f:
                    for line in f:
                        if line.startswith("rchar:"):
                            return int(line.split()[1])
            except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
                return None
            return None
        return LlamaCppBackend._psutil_process_stat(pid, "io")

    @staticmethod
    def _psutil_process_stat(pid: int, kind: str) -> Optional[int]:
        """RSS or read bytes of ``pid`` through psutil, for hosts without /proc."""
        try:
            import psutil

            process = psutil.Process(pid)
            if kind == "rss":
                return int(process.memory_info().rss)
            counters = process.io_counters()
            return int(getattr(counters, "read_chars", None) or counters.read_bytes)
        except Exception:
            return None

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        During warm-up llama-server reads the weight files before it can answer, a window
        where status only reports ``loading`` and the UI spinner looks stuck for minutes on
        large MoEs. ``bytes_loaded`` is the larger of two samples of the llama-server process:
        the bytes it has read (a load that streams tensors to VRAM keeps almost none of them on
        the host) and its resident memory (what a memory-mapped load pages in, which read()
        counters never see). Measured against the shard sizes plus a vision projector or draft
        model named on its command line. Returns ``None`` when no load is in flight.

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # max(bytes read, resident memory), never decreasing
                "bytes_total":  int,   # weight files the server loads
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }

        Returns ``None`` where neither sample is available or the process is gone.
        """
        proc = self._process
        if proc is None:
            return None
        pid = proc.pid
        if pid is None:
            return None

        # Sum shard sizes (primary + any extras alongside).
        bytes_total = 0
        gguf_path = self._gguf_path
        if gguf_path:
            primary = Path(gguf_path)
            try:
                if primary.is_file():
                    bytes_total += primary.stat().st_size
            except OSError:
                pass
            # Extra shards share the primary's prefix before the shard index.
            try:
                parent = primary.parent
                stem = primary.name
                m = _SHARD_RE.match(stem)
                prefix = m.group(1) if m else None
                if prefix and parent.is_dir():
                    prefix_lower = prefix.lower()
                    for sibling in parent.iterdir():
                        if (
                            sibling.is_file()
                            and sibling.name.lower().startswith(prefix_lower)
                            and sibling.name != stem
                            and sibling.suffix.lower() == ".gguf"
                        ):
                            try:
                                bytes_total += sibling.stat().st_size
                            except OSError:
                                pass
            except OSError:
                pass
        bytes_total += _extra_weight_file_bytes(getattr(proc, "args", None))

        rss = LlamaCppBackend._read_rss_bytes(pid)
        read = LlamaCppBackend._read_io_bytes(pid)
        if rss is None and read is None:
            return None
        bytes_loaded = max(rss or 0, read or 0)

        # RSS climbs as weights page in, then drops once -ngl offloads them to
        # VRAM and the mmap pages are freed. Hold a per-process high-water mark
        # so the bar never regresses to ~8% mid-load (#5740).
        hwm_pid, hwm = getattr(self, "_load_rss_hwm", (None, 0))
        hwm = bytes_loaded if hwm_pid != pid else max(hwm, bytes_loaded)
        self._load_rss_hwm = (pid, hwm)
        bytes_loaded = hwm

        phase = "ready" if self._healthy else "mmap"
        fraction = 0.0
        if bytes_total > 0:
            # Reads past the weights (headers, a second pass) must not push the figure past the total.
            bytes_loaded = min(bytes_loaded, bytes_total)
            fraction = min(1.0, bytes_loaded / bytes_total)
        # Once llama-server is healthy the load is complete by definition. With
        # layers offloaded to VRAM (-ngl) the process releases the mmap'd weight
        # pages, so VmRSS sinks back well below the shard total; the raw RSS
        # fraction would then report a partial (~8%) load indefinitely and freeze
        # a fraction-driven progress bar even though the model is ready (#5740).
        if self._healthy:
            if bytes_total > 0:
                bytes_loaded = bytes_total
            fraction = 1.0
        return {
            "phase": phase,
            "bytes_loaded": bytes_loaded,
            "bytes_total": bytes_total,
            "fraction": round(fraction, 4),
        }
