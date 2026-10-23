# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-launch log files for sidecar servers, quoted in the error when one fails to start."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import IO, Optional

from loggers import get_logger

logger = get_logger(__name__)

_TAIL_CHARS = 1500


def logs_dir(kind: str) -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "logs" / kind


def open_child_log(kind: str, port: int) -> "tuple[Optional[Path], Optional[IO[bytes]]]":
    """Open a log for a child's output, or ``(None, None)`` so a logging error never blocks a start."""
    try:
        directory = logs_dir(kind)
        directory.mkdir(parents = True, exist_ok = True)
        path = directory / f"{kind}-{int(time.time())}-port-{port}.log"
        handle = open(path, "wb")
    except OSError as exc:
        logger.warning("Could not open the %s log file: %s", kind, exc)
        return None, None
    try:
        from utils.log_retention import prune_log_dir

        # After the open, so the cap counts this file and can never delete it.
        prune_log_dir(directory, f"{kind}-*.log", protect = path)
    except Exception:
        pass
    return path, handle


def read_tail(path: Optional[Path], limit: int = _TAIL_CHARS) -> str:
    """The last ``limit`` printable characters of ``path``, or "" when it cannot be read."""
    if path is None:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit * 4))
            data = handle.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors = "replace")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text.strip()[-limit:]


_ERROR_LINE = re.compile(
    r"\b(error|fatal|failed|abort|assert|unknown argument|invalid)\b", re.IGNORECASE
)
_MAX_ERROR_LINES = 5


def error_lines(path: Optional[Path]) -> "list[str]":
    """The first error lines, since a rejected argument is printed before a usage dump that a tail cuts off."""
    if path is None:
        return []
    found: "list[str]" = []
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                line = raw.decode("utf-8", errors = "replace").strip()
                if line and _ERROR_LINE.search(line):
                    found.append(line[:300])
                    if len(found) >= _MAX_ERROR_LINES:
                        break
    except OSError:
        return []
    return found


def exit_details(returncode: Optional[int], path: Optional[Path]) -> str:
    """A suffix naming the exit status, the errors and last output, and the log file."""
    parts = []
    if returncode is not None:
        parts.append(f"It exited with status {returncode}.")
    errors = error_lines(path)
    if errors:
        parts.append("Errors reported:\n" + "\n".join(errors))
    tail = read_tail(path, limit = 600 if errors else _TAIL_CHARS)
    if tail:
        parts.append(f"Last output:\n{tail}")
    if path is not None:
        parts.append(f"Full log: {path}")
    return (" " + "\n".join(parts)) if parts else ""
