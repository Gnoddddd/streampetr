#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
STREAM_DIR="$PROJECT_ROOT/repos/StreamPETR"
STREAM_COMMIT="95f64702306ccdb7a78889578b2a55b5deb35b2a"

mkdir -p "$PROJECT_ROOT/repos"
if [[ ! -d "$STREAM_DIR/.git" ]]; then
  git clone https://github.com/exiawsh/StreamPETR.git "$STREAM_DIR"
fi
git -C "$STREAM_DIR" fetch --all --tags
git -C "$STREAM_DIR" checkout "$STREAM_COMMIT"

echo "StreamPETR pinned at $(git -C "$STREAM_DIR" rev-parse HEAD)"
