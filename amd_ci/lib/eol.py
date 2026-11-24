#!/usr/bin/env python3
"""Line-ending normalisation for anything compared across Linux and Windows runners.

Git for Windows checks text files out with CRLF (core.autocrlf=true is its installer default), so
the same commit hashes differently on the two OSes and a diff written on Linux stops applying.
scaffold.py's `.gitattributes` (`* -text`) keeps the CI branch byte-exact and states.py clones the
PR repo with core.autocrlf=false, but a probe that compares bytes it did not check out itself (a
pip-installed file, a file another tool wrote) still has to normalise. Use these:

  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
  from eol import sha256_lf, read_text_lf, same_text
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def lf(data: bytes) -> bytes:
    """CRLF and lone CR as LF."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_lf(path: Path | str) -> str:
    """sha256 of the file with its line endings normalised: equal on a CRLF and an LF checkout."""
    return hashlib.sha256(lf(Path(path).read_bytes())).hexdigest()


def read_text_lf(path: Path | str, encoding: str = "utf-8") -> str:
    """The file's text with LF endings whatever the checkout wrote (and whatever the platform reads)."""
    return lf(Path(path).read_bytes()).decode(encoding)


def same_text(a: Path | str, b: Path | str) -> bool:
    """The two files hold the same text, line endings aside."""
    return lf(Path(a).read_bytes()) == lf(Path(b).read_bytes())
