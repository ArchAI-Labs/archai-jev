"""Turn the `pyproject.toml` of this checkout into the one of a GPU wheel (spec 011, T14).

A GPU wheel is the same module (`archai_jev`) under another distribution name, so that it is a
separate project on PyPI (`pip install archai-jev-vulkan`) and the CPU wheels keep their name and
their size. Only `name` and `description` change; the Cargo feature is chosen when building
(`maturin build --release --features vulkan`).

    python ci/gpu/make_variant.py vulkan            # rewrites pyproject.toml in place (a CI checkout)
    python ci/gpu/make_variant.py vulkan --check    # prints the new text, writes nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_NAME = "archai-jev"
#: variant -> (feature of Cargo, what the description says)
VARIANTS = {
    "vulkan": (
        "vulkan",
        "Vulkan GPU build (NVIDIA, AMD, Intel). Replaces archai-jev; do not install both.",
    ),
}


def variant_text(text: str, variant: str) -> str:
    """``pyproject.toml`` as the wheel ``archai-jev-<variant>`` needs it.

    Raises:
        ValueError: unknown variant, or the file is not shaped as expected (a changed line must
            be found exactly once, so that a rename of a key is noticed here and not on PyPI).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(VARIANTS)}")
    _, note = VARIANTS[variant]
    name_line = f'name = "{BASE_NAME}"'
    if text.count(name_line) != 1:
        raise ValueError(f"expected exactly one line {name_line!r} in pyproject.toml")
    match = re.search(r'^description = "(?P<text>[^"]*)"$', text, re.MULTILINE)
    if match is None or len(re.findall(r"^description = ", text, re.MULTILINE)) != 1:
        raise ValueError(
            "expected exactly one single-line description in pyproject.toml"
        )
    description = f'description = "{match["text"]} {note}"'
    out = text.replace(name_line, f'name = "{BASE_NAME}-{variant}"')
    return out.replace(match.group(0), description)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument(
        "--check", action="store_true", help="print the result instead of writing it"
    )
    ap.add_argument("--file", type=Path, default=ROOT / "pyproject.toml")
    args = ap.parse_args(argv)
    new = variant_text(args.file.read_text(encoding="utf-8"), args.variant)
    if args.check:
        sys.stdout.write(new)
    else:
        args.file.write_text(new, encoding="utf-8", newline="\n")
        print(f"{args.file}: now {BASE_NAME}-{args.variant}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
