#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Install the elevation_mapping_cupy package from the official GitHub repository
# using uv. Designed to run inside Docker during the image build so the
# resulting image already has the Python package available.
#
set -euo pipefail

# Reuse the same cache path convention as install-python.sh for better layer caching.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Ensure uv is available (install-python.sh should already have done this, but
# add a defensive check so the script can be run independently if needed).
if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "ERROR: uv is required but not found in PATH"
    exit 1
  fi
fi

export UV_SYSTEM_PYTHON=1

echo "Installing elevation_mapping_cupy_core from GitHub..."
uv pip install --system "elevation_mapping_cupy_core @ git+https://github.com/leggedrobotics/elevation_mapping_cupy_core"

echo "Elevation mapping installation completed."
