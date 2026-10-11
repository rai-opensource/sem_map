#!/usr/bin/env bash

# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
set -euo pipefail

# Puts the machine into its highest-performance state and starts the CUDA MPS
# daemon, so several sem_map processes can share one GPU efficiently.
#
# Usage:
#   ./scripts/enable_max_mode.sh             # Jetson: max power mode + fan + MPS
#   ./scripts/enable_max_mode.sh --desktop   # desktop/x86 GPU: MPS daemon only
#
# The Jetson-specific steps (nvpmodel, jetson_clocks) do not exist on a desktop
# machine, so pass --desktop there to skip them.
DESKTOP=false
if [ "${1:-}" = "--desktop" ]; then
echo "[enable_max_mode] Desktop mode: only starting CUDA MPS daemon..."
DESKTOP=true
fi

if [ "$DESKTOP" = "false" ]; then
echo "[enable_max_mode] Setting Jetson to maximum power mode..."
# make sure that jetson is set to maximum power mode
sudo nvpmodel -m 0

echo "[enable_max_mode] Setting fan to maximum speed..."
# make sure that fan is set to maximum speed
sudo jetson_clocks --fan
fi

echo "[enable_max_mode] Preparing CUDA MPS environment and starting daemon..."
# Set MPS directories (must be set before mkdir)
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
# Start the daemon
sudo mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
sudo chown "$USER" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
# to quit the daemon - echo quit | nvidia-cuda-mps-control

echo "[enable_max_mode] Done. MPS daemon started."
