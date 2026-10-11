#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Resolve and install ROS package dependencies for the workspace.
# Designed to run inside Docker during image build so runtime containers
# already contain all required system dependencies.
#
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: install-rosdeps.sh must run as root."
  exit 1
fi

WORKSPACE_ROOT="${1:-/tmp/workspace}"
SRC_DIR="$WORKSPACE_ROOT/src"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Skipping rosdep install: no src directory at $SRC_DIR"
  exit 0
fi

if ! command -v rosdep >/dev/null 2>&1; then
  echo "ERROR: rosdep is not installed. Run install-ros2.sh first."
  exit 1
fi

# Determine the ROS distro by sourcing the first available setup file if
# ROS_DISTRO is not already exported (e.g., by install-ros2.sh).
if [[ -z "${ROS_DISTRO:-}" ]]; then
  if ROS_SETUP_PATH="$(find /opt/ros -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | head -n1)"; then
    if [[ -f "$ROS_SETUP_PATH/setup.bash" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$ROS_SETUP_PATH/setup.bash"
      set -u
    fi
  fi
fi

echo "Running rosdep update/install for workspace at $SRC_DIR"
apt-get update
rosdep update --rosdistro "${ROS_DISTRO:-}"
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:-}/setup.bash"
set -u
# `cmake_clang_tools` is an optional lint helper that has no rosdep rule on
# public indexes; skipping it keeps the build working on a stock ROS install.
rosdep install \
  --rosdistro "${ROS_DISTRO:-}" \
  --from-paths "$SRC_DIR" \
  --ignore-src \
  -r \
  -y \
  --skip-keys "cmake_clang_tools"
