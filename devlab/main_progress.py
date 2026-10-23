# Extracted verbatim from llama_cpp.py: _read_rss_bytes, load_progress
import os
import re
from pathlib import Path
from typing import Optional

_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


class LlamaCppBackend:
    @staticmethod
    def _read_rss_bytes(pid: int) -> Optional[int]:
        """Resident set size of ``pid`` in bytes, from /proc/<pid>/status (Linux).
        0 when the status has no VmRSS line (zombie / kernel thread); None where
        /proc is unavailable (macOS/Windows) or the value is unreadable."""
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # IndexError guards a "VmRSS:" line with no value column.
                        return int(line.split()[1]) * 1024  # kB -> bytes
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return 0  # readable but no VmRSS line

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        During warm-up llama-server mmaps weight shards into page cache before
        pushing layers to VRAM, a window where status only reports ``loading``
        and the UI spinner looks stuck for minutes on large MoEs. Samples
        ``/proc/<pid>/status VmRSS`` against the sum of GGUF shard sizes for a
        real progress bar. Returns ``None`` when no load is in flight.

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # VmRSS of the llama-server
                "bytes_total":  int,   # sum of shard file sizes
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }

        Linux-only; returns ``None`` where ``/proc/<pid>/status`` is unavailable.
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

        # VmRSS of the llama-server; None where /proc is unavailable.
        bytes_loaded = LlamaCppBackend._read_rss_bytes(pid)
        if bytes_loaded is None:
            return None

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
