"""Validate every mask-demo block in docstrings and RST docs against the library.

The Sphinx directive in docs/_ext/mask_grid.py performs the same validation at doc build
time; this test catches demo drift in the much more frequently run test suite.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "_ext"))

from mask_grid import MaskDemoDirective, validate_demo  # noqa: E402

DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+mask-demo::\s*$")
OPTION_RE = re.compile(r"^\s*:[\w-]+:")


def extract_demo_blocks(path):
    """Yield (line_number, content) for each mask-demo block in the file."""
    lines = path.read_text().split("\n")
    i = 0
    while i < len(lines):
        m = DIRECTIVE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        block = []
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() == "":
                block.append("")
            elif len(line) - len(line.lstrip()) <= indent:
                break
            else:
                block.append(line)
            j += 1
        while block and block[-1] == "":
            block.pop()
        content_lines = [ln for ln in block if not OPTION_RE.match(ln)]
        nonblank = [ln for ln in content_lines if ln.strip()]
        if nonblank:
            common = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
            yield i + 1, "\n".join(
                ln[common:] if ln.strip() else "" for ln in content_lines
            )
        i = j


def collect_demos():
    files = sorted(
        list((ROOT / "src" / "rlemasklib").glob("*.py"))
        + list((ROOT / "src" / "rlemasklib").glob("*.pyx"))
        + list((ROOT / "docs").rglob("*.rst"))
    )
    for path in files:
        if "_build" in path.parts:
            continue
        for lineno, content in extract_demo_blocks(path):
            yield pytest.param(
                content, id=f"{path.relative_to(ROOT)}:{lineno}"
            )


@pytest.mark.parametrize("content", collect_demos())
def test_mask_demo(content):
    directive = MaskDemoDirective.__new__(MaskDemoDirective)
    masks, template_lines = directive._parse_content(content)
    errors = validate_demo(template_lines, masks)
    assert not errors, "\n".join(errors)
