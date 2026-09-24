#!/usr/bin/env python3
"""Render the abstract as the plain text arXiv's form wants, and check it fits.

arXiv's abstract field takes plain text and caps it at 1920 characters. The
paper's abstract is LaTeX and was, at 2551 characters, well over -- which the
submission form discovers, not the author. So this converts rather than
transcribes: the text that goes in the form comes out of `main.tex`, and if it
does not fit, that is a fact about `main.tex` to be fixed there.

  ./abstract_plain.py            print the plain text and its length
  ./abstract_plain.py --check    exit non-zero if it exceeds the limit, or if
                                 the copy pasted into SUBMISSION.md has drifted

The conversion is deliberately narrow. It unwraps the handful of macros this
abstract actually uses and would raise on anything else rather than silently
emit a stray backslash into a published abstract.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LIMIT = 1920
HERE = Path(__file__).resolve().parent

# Applied in order. Everything here is a macro the abstract uses today; a new
# one shows up as a leftover backslash and is caught below.
RULES = [
    (r"\\(?:textbf|emph|texttt)\{([^{}]*)\}", r"\1"),
    (r"\$1\.6\\times10\^\{-5\}\$", "1.6e-5"),
    (r"\$([^$]*)\$", r"\1"),
    (r"\\times", "x"),
    (r"\\%", "%"),
    (r"\\,", " "),
    (r"---", "--"),
    (r"68--69", "68-69"),
]


def plain(tex: str) -> str:
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if body is None:
        raise SystemExit("no abstract environment in main.tex")
    s = body.group(1)
    for pattern, repl in RULES:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    if "\\" in s or "{" in s or "}" in s:
        raise SystemExit(f"unconverted LaTeX left in the abstract:\n{s}")
    return s


def pasted() -> str | None:
    """The copy in SUBMISSION.md, so the two cannot drift apart unnoticed.

    Fences are anchored to line starts and the opener is allowed a language
    tag. Without both, ```` ```bash ```` fails to match as an opener while its
    closing fence matches, the pairing slips by one, and every "block" found is
    the prose between two code blocks -- which is how the first version of this
    silently found no abstract at all and checked nothing.
    """
    md = HERE / "SUBMISSION.md"
    if not md.exists():
        return None
    blocks = re.findall(r"^```[A-Za-z]*\n(.*?)\n^```\s*$", md.read_text(),
                        re.S | re.M)
    found = [b.strip() for b in blocks if b.strip().startswith("We stud")]
    if not found:
        raise SystemExit(
            "SUBMISSION.md has no abstract block to compare against. Either it "
            "was removed, or this extraction is broken -- do not read the "
            "absence as agreement.")
    return found[0]


def main() -> int:
    text = plain((HERE / "main.tex").read_text())
    check = "--check" in sys.argv

    if not check:
        print(text)
        print(f"\n{len(text)} characters (arXiv limit {LIMIT}, "
              f"{LIMIT - len(text)} to spare)", file=sys.stderr)
        return 0

    if len(text) > LIMIT:
        print(f"abstract is {len(text)} characters, {len(text) - LIMIT} over "
              f"arXiv's {LIMIT}; shorten it in main.tex", file=sys.stderr)
        return 1
    p = pasted()
    if p is None:
        print(f"abstract fits: {len(text)}/{LIMIT} characters "
              "(no SUBMISSION.md to cross-check)", file=sys.stderr)
        return 0
    if p != text:
        print("SUBMISSION.md's abstract has drifted from main.tex; "
              "replace it with the output of ./abstract_plain.py",
              file=sys.stderr)
        # Show the first sentence that differs, from its first differing
        # character, so the hint points at the change rather than past it.
        for a, b in zip(p.split(". "), text.split(". ")):
            if a == b:
                continue
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
            print(f"  at character {i} of a sentence:\n"
                  f"    SUBMISSION.md: {a[max(0, i - 20):i + 50]}\n"
                  f"    main.tex:      {b[max(0, i - 20):i + 50]}", file=sys.stderr)
            break
        return 1
    print(f"abstract fits: {len(text)}/{LIMIT} characters, "
          "SUBMISSION.md matches", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
