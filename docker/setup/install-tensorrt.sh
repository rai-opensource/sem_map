#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Install TensorRT for both AMD64 (x86_64) and ARM64 (AGX Orin) systems.

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
  echo "ERROR: TensorRT install must be run as root, please run:"
  echo "  sudo $0"
  exit 1
fi

# Detect system architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Create a temporary directory for TensorRT installation
TMP_DIR="/tmp/tensorrt_install"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
pushd "${TMP_DIR}" > /dev/null

# Install based on architecture
if [[ "$ARCH" == "x86_64" ]]; then
  echo "Installing TensorRT for AMD64 (x86_64) system..."

  # The nvcr.io/nvidia/cuda base image already has the NVIDIA apt repository
  # configured, so we can install TensorRT directly without downloading a local
  # .deb repo package.
  #
  # Pinned because the repo is unversioned and now serves TRT 11, which dropped
  # the FP16/INT8 BuilderFlags and EXPLICIT_BATCH this code uses. 10.8.0 is the
  # floor for Blackwell — 10.7 rejects sm_120. The aarch64 branch below stays on
  # 10.7 (JetPack 6 is CUDA 12.6). Override with the TRT_PIN_VERSION build arg.
  TRT_PIN_VERSION="${TRT_PIN_VERSION:-10.8.0.*}"
  echo "Pinning TensorRT apt packages to ${TRT_PIN_VERSION}"
  cat > /etc/apt/preferences.d/tensorrt-pin <<EOF
Package: libnvinfer* libnvinfer-plugin* libnvonnxparsers* libnvparsers* libnvinfer-bin python3-libnvinfer* tensorrt*
Pin: version ${TRT_PIN_VERSION}
Pin-Priority: 1001
EOF

  apt-get update

  echo "Installing TensorRT libraries (pinned to ${TRT_PIN_VERSION})..."
  # --allow-downgrades lets this replace a newer TRT (e.g. 11.x) already present.
  apt-get install -y --no-install-recommends --allow-downgrades \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    libnvinfer-bin \
    python3-libnvinfer

  # Create symlink for trtexec if it doesn't exist
  if [ ! -e /usr/local/bin/trtexec ]; then
    if [ -f /usr/src/tensorrt/bin/trtexec ]; then
      ln -s /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
      echo "Created symlink for trtexec"
    else
      echo "WARNING: trtexec not found at /usr/src/tensorrt/bin/trtexec"
    fi
  fi

elif [[ "$ARCH" == "aarch64" ]]; then
  echo "Installing TensorRT for ARM64 (AGX Orin) system..."

  # TensorRT 10.7.0 for AGX Orin with CUDA 12.6
  OS="ubuntu2204"
  VERSION="10.7.0"
  TAG="${VERSION}-cuda-12.6"
  DEB_FILE="nv-tensorrt-local-tegra-repo-${OS}-${TAG}_1.0-1_arm64.deb"
  DEB_URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${VERSION}/local_repo/${DEB_FILE}"

  echo "Downloading TensorRT repository package..."
  wget -O "${DEB_FILE}" "${DEB_URL}"

  echo "Installing TensorRT repository..."
  dpkg -i "${DEB_FILE}"

  # Copy keyring
  KEYRING_FILE="/var/nv-tensorrt-local-tegra-repo-${OS}-${TAG}/nv-tensorrt-local-tegra-C50F04B9-keyring.gpg"
  if [ -f "${KEYRING_FILE}" ]; then
    cp "${KEYRING_FILE}" /usr/share/keyrings/
  else
    echo "WARNING: Could not find keyring file at ${KEYRING_FILE}"
  fi

  apt-get update

  echo "Installing TensorRT..."
  apt-get install -y --no-install-recommends tensorrt

else
  echo "ERROR: Unsupported architecture: $ARCH"
  echo "This script supports x86_64 (AMD64) and aarch64 (ARM64/AGX Orin) only."
  popd > /dev/null
  rm -rf "${TMP_DIR}"
  exit 1
fi

popd > /dev/null
rm -rf "${TMP_DIR}"

# Verify TensorRT installation
echo ""
echo "Verifying TensorRT installation..."
if command -v trtexec &> /dev/null; then
  echo "✓ trtexec found in PATH"
elif [ -f /usr/local/bin/trtexec ]; then
  echo "✓ trtexec found at /usr/local/bin/trtexec"
elif [ -f /usr/src/tensorrt/bin/trtexec ]; then
  echo "✓ trtexec found at /usr/src/tensorrt/bin/trtexec"
else
  echo "WARNING: trtexec not found. TensorRT may not be fully installed."
fi

# Check for Python bindings
if python3 -c "import tensorrt" 2>/dev/null; then
  echo "✓ Python TensorRT bindings available"
  python3 -c "import tensorrt as trt; print(f'  TensorRT version: {trt.__version__}')" 2>/dev/null || true
else
  echo "WARNING: Python TensorRT bindings not found. You may need to install python3-libnvinfer-dev"
fi

# Check for libraries
if [ -f /usr/lib/x86_64-linux-gnu/libnvinfer.so ] || [ -f /usr/lib/aarch64-linux-gnu/libnvinfer.so ]; then
  echo "✓ TensorRT libraries installed"
else
  echo "WARNING: TensorRT libraries not found in expected location"
fi

echo ""
echo "TensorRT installation complete."
