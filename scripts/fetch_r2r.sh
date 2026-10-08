#!/usr/bin/env bash
# Fetch Reveal2Revise (R2R) at the pinned SHA for research comparison.
#
# R2R (Pahde et al., MICCAI 2023) has NO licence. It is NOT vendored under
# third_party/: that would redistribute unlicensed code. This script clones it
# into git-ignored third_party_fetched/reveal2revise/ and verifies the SHA.
# User decision 2026-10-08: research and comparison use accepted; whether R2R
# rows appear in any publication is decided separately at publishing time.
# Pending: email the authors to ask for a licence.
#
# Usage: bash scripts/fetch_r2r.sh   (run from the repo root)
set -euo pipefail

REPO="https://github.com/maxdreyer/Reveal2Revise.git"
PIN="f00ce0614947b48bb8320beeff64c700c6d94c09"
DEST="third_party_fetched/reveal2revise"

if [ -d "$DEST/.git" ]; then
    echo "Fetching in existing $DEST"
    git -C "$DEST" fetch --quiet origin
else
    echo "Cloning $REPO"
    mkdir -p third_party_fetched
    git clone --quiet "$REPO" "$DEST"
fi

git -C "$DEST" checkout --quiet "$PIN"
GOT="$(git -C "$DEST" rev-parse HEAD)"
if [ "$GOT" != "$PIN" ]; then
    echo "SHA mismatch: want $PIN, have $GOT" >&2
    exit 1
fi
echo "Reveal2Revise at $GOT (no licence; research-comparison use only)"
