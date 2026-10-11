#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Install Python dependencies using uv in a virtual environment.
# Designed to run inside Docker during image build.
# Follows best practices from: https://docs.astral.sh/uv/guides/integration/docker/
#
set -euo pipefail

# Root of the tree that holds `docker/requirements/`. During the Docker build the
# Dockerfile copies `docker/requirements/` to /tmp/docker/requirements and calls
# this script with /tmp.
WORKSPACE_ROOT="${1:-/tmp}"

# Platform selection — injected by Dockerfile ENV PLATFORM=${PLATFORM}.
# When running outside Docker, set PLATFORM manually or let it default to desktop.
PLATFORM="${PLATFORM:-desktop}"
if [[ "$PLATFORM" == "agx" ]]; then
  # NVIDIA Jetson AGX Orin (aarch64, JetPack 6 / L4T r36.4)
  PYTORCH_REQ="requirements-agx-pytorch.txt"
else
  # desktop (default) — x86_64 workstation with a discrete NVIDIA GPU
  PYTORCH_REQ="requirements-desktop-pytorch.txt"
fi

# Platform-neutral packages are shared by both images, so there is only one
# file. It is installed as a SECOND resolution pass, after the platform file:
# keeping the two apart is what lets each use its own index configuration
# without the platform wheel index leaking into ordinary PyPI packages.
COMMON_REQ="requirements-common.txt"
echo "Platform: $PLATFORM — using $PYTORCH_REQ and $COMMON_REQ"

# Set up uv cache directory (best practice for Docker builds)
# This allows caching to work even if /root/.cache is not mounted
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
# The CUDA wheels torch pulls in are hundreds of MB each and download
# concurrently; uv's 30 s default times them out on an ordinary link.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
mkdir -p "$UV_CACHE_DIR"

# Install uv if not already present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  # uv installer installs to $HOME/.local/bin, which is /root/.local/bin for root user
  # Reference: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
  (curl -LsSf https://astral.sh/uv/install.sh | sh)
  # Add uv to PATH for current session (works for both root and non-root)
  export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure uv is in PATH (in case it was installed but PATH wasn't updated)
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation failed or not in PATH"
    exit 1
  fi
fi

# Determine Python version (default to 3.10)
PYTHON_VERSION="${2:-3.10}"
PYTHON_CMD="python${PYTHON_VERSION}"

# Check if Python version is available
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "ERROR: Python $PYTHON_VERSION not found. Available:"
  ls -1 /usr/bin/python* 2>/dev/null || true
  exit 1
fi

# Verify Python is available
FULL_PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
echo "Installing Python packages to system Python ($FULL_PYTHON_VERSION)"
echo "Using system Python is safe in Docker containers since they're already isolated."

# Use system Python by default (no virtual environment)
# Reference: https://docs.astral.sh/uv/guides/integration/docker/#using-the-pip-interface
export UV_SYSTEM_PYTHON=1

# Install requirements files if they exist
# Best practice: copy requirements separately from source code for better caching
# Reference: https://docs.astral.sh/uv/guides/integration/docker/#installing-requirements
REQUIREMENTS_DIR="$WORKSPACE_ROOT/docker/requirements"
# Index strategy for the platform file, which draws on two indexes.
#
# By default uv only considers versions from the FIRST index that carries a
# given package name, to avoid dependency-confusion attacks. On desktop that
# breaks the build: download.pytorch.org carries a `cuda-python`, but not the
# pinned version, so resolution stops there instead of falling back to PyPI.
# Allowing uv to look across both indexes fixes it, and torch still resolves to
# the CUDA build because +cu128 sorts above the plain PyPI version.
#
# The Jetson file deliberately keeps the strict default. Its packages are pinned
# to exact versions that PyPI also publishes as aarch64 CPU-only wheels, so a
# cross-index match there could silently produce a torch with no GPU support.
INDEX_ARGS=()
if [[ "$PLATFORM" == "desktop" ]]; then
  INDEX_ARGS+=(--index-strategy unsafe-best-match)
fi

if [[ -f "$REQUIREMENTS_DIR/$PYTORCH_REQ" ]]; then
  echo "Installing $PYTORCH_REQ to system Python..."
  uv pip install --system "${INDEX_ARGS[@]}" -r "$REQUIREMENTS_DIR/$PYTORCH_REQ"
fi

if [[ -f "$REQUIREMENTS_DIR/$COMMON_REQ" ]]; then
  echo "Installing $COMMON_REQ to system Python..."
  uv pip install --system -r "$REQUIREMENTS_DIR/$COMMON_REQ"
fi

# Make uv accessible system-wide (for non-root users in container)
if [[ -f "$HOME/.local/bin/uv" ]] && [[ ! -f /usr/local/bin/uv ]]; then
  echo "Making uv accessible system-wide..."
  cp "$HOME/.local/bin/uv" /usr/local/bin/uv
fi

echo "Python dependencies installed successfully to system Python"
