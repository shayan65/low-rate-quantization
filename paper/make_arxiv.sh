#!/usr/bin/env bash
# Assemble the arXiv submission tarball, and prove it builds standalone.
#
# arXiv compiles what you upload, in a directory that contains only what you
# upload. The failure mode is a file that is present in the working tree and
# absent from the tarball -- a generated table, a figure -- which builds here
# and fails there. So this script does not just tar the directory: it copies
# the declared file list into a scratch directory, builds *there*, and refuses
# to leave a tarball behind if that build is not clean.
#
# The bibliography is a `thebibliography` environment inside main.tex, so no
# .bbl or .bib is needed. No custom class, no fonts, no shell-escape.
#
# Usage:  ./make_arxiv.sh  [output.tar.gz]

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/arxiv-submission.tar.gz}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The other way to fail a submission is at the metadata form rather than the
# build: an abstract over arXiv's 1920-character limit, or a SUBMISSION.md whose
# pasted copy no longer matches the paper.
"$here/abstract_plain.py" --check

# Everything the document \input{}s or \includegraphics{}es, and nothing else.
mkdir -p "$work/generated"
cp "$here/main.tex" "$here/fig_method.tex" "$work/"
cp "$here"/generated/*.tex "$here"/generated/*.pdf "$work/generated/"

# Build in the scratch directory, so a file missing from the list fails here
# rather than on arXiv.
builder=""
for cand in tectonic pdflatex; do
    command -v "$cand" >/dev/null 2>&1 && { builder="$cand"; break; }
done
if [ -z "$builder" ]; then
    echo "no TeX engine found (tectonic or pdflatex); not writing $out" >&2
    exit 1
fi

mkdir -p "$work/out"
if [ "$builder" = tectonic ]; then
    ( cd "$work" && tectonic -X compile main.tex --outdir out --keep-logs ) >/dev/null 2>&1
else
    ( cd "$work" && pdflatex -interaction=nonstopmode -output-directory=out main.tex \
      && pdflatex -interaction=nonstopmode -output-directory=out main.tex ) >/dev/null 2>&1
fi

log="$work/out/main.log"
for bad in "undefined" "Overfull" "Underfull"; do
    if grep -q "$bad" "$log"; then
        echo "standalone build is not clean ($bad); not writing $out" >&2
        grep -n "$bad" "$log" | head >&2
        exit 1
    fi
done

pages="$(grep -o '([0-9]* pages' "$log" | head -1 | tr -d '(')"
rm -rf "$work/out"
tar -czf "$out" -C "$work" .
echo "wrote $out  ($pages, $(du -h "$out" | cut -f1))"
tar -tzf "$out" | sed 's|^\./||' | grep . | sort
