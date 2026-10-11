#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Install the ZED SDK only (no ROS2 wrapper).

set -euo pipefail

# Ensure CUDA tools are on PATH when running under sudo/root
if [ -d "/usr/local/cuda/bin" ]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [ -d "/opt/cuda/bin" ]; then
  export PATH="/opt/cuda/bin:${PATH}"
fi

# This script must be run as root for system wide installation.
if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: ZED ROS2 wrapper install must be run as root, please run:"
  echo "  sudo $0"
  exit 1
fi

# Check if CUDA is available (required for x86_64 ZED SDK)
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. ZED SDK requires CUDA to work properly on x86_64."
    echo "Please install CUDA toolkit first:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo "Aborting installation."
    exit 1
fi

# Install required packages
apt-get update && apt-get install -y --no-install-recommends \
  python3-rosdep \
  python3-rosinstall-generator \
  python3-vcstool \
  python3-rosinstall \
  build-essential \
  cmake \
  git \
  wget \
  zstd

# Create a temporary directory for ZED SDK installation
TMP_DIR="/tmp/zed_sdk_install"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
pushd "${TMP_DIR}" > /dev/null

# Detect system architecture
zed_sdk_version="5.1"
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
  echo "Detected x86_64/AMD64 system, downloading Ubuntu 22.04 ZED SDK with CUDA 12 support..."
  SDK_URL="https://download.stereolabs.com/zedsdk/${zed_sdk_version}/cu12/ubuntu22"
  INSTALL_OPTS=(-- silent runtime_only skip_cuda skip_od_module skip_drivers skip_tools skip_python)
elif [[ "$ARCH" == "aarch64" ]]; then
  echo "Detected ARM64 system, downloading Jetson Tegra ZED SDK..."
  SDK_URL="https://download.stereolabs.com/zedsdk/${zed_sdk_version}/l4t36.4/jetsons"
  INSTALL_OPTS=(-- silent runtime_only skip_cuda skip_od_module skip_drivers skip_tools skip_python)
else
  echo "ERROR: Unsupported architecture: $ARCH"
  exit 1
fi

echo "Downloading ZED SDK from: $SDK_URL"
wget -O zed_sdk.run "$SDK_URL"
chmod +x zed_sdk.run

echo "Installing ZED SDK with options: $INSTALL_OPTS"
./zed_sdk.run "${INSTALL_OPTS[@]}"

popd > /dev/null
rm -rf "${TMP_DIR}"

# Verify ZED SDK installation
if [[ -f "/usr/local/zed/zed-config.cmake" ]]; then
  echo "ZED SDK installation verified successfully."
else
  echo "ERROR: ZED SDK installation failed. Check the logs above."
  exit 1
fi

echo "ZED SDK installation complete."
