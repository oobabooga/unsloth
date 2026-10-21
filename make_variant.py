"""Rewrite .pre-commit-config.yaml in place for one A/B variant."""
import re, sys
from pathlib import Path

variant, path = sys.argv[1], Path(sys.argv[2])
text = path.read_text()
start = text.index("      - id: ruff-format-with-kwargs\n")
end = text.index("      # Re-pins allowScripts")
block = text[start:end]
exclude = re.search(r"^        exclude: (.*)$", block, re.M).group(1)

if variant == "main":
    new = block
elif variant == "single_par":
    new = block.replace("        require_serial: true\n", "")
    assert new != block
elif variant == "split":
    new = f"""      - id: kwargs-spacing-pre
        name: Kwarg spacing, before ruff format
        entry: python scripts/enforce_kwargs_spacing.py --pre
        language: python
        types: [python]
        exclude: {exclude}
      - id: ruff-format-with-kwargs
        name: Ruff format
        entry: ruff format
        language: python
        types: [python]
        exclude: {exclude}
        additional_dependencies:
          - ruff==0.6.9
        require_serial: true
      - id: kwargs-spacing-post
        name: Kwarg spacing, after ruff format
        entry: python scripts/enforce_kwargs_spacing.py
        language: python
        types: [python]
        exclude: {exclude}
"""
else:
    raise SystemExit(f"unknown variant {variant}")
path.write_text(text[:start] + new + text[end:])
