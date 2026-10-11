#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Install cuDSS (CUDA Sparse Direct Solver) library for CUDA 12.x
# Required by PyTorch 2.8+ for sparse matrix operations
# libcudss.so is a separate library from cuSPARSELt

set -euo pipefail

# This script must be run as root for system wide installation.
if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: cuDSS install must be run as root, please run:"
  echo "  sudo $0"
  exit 1
fi

# Detect CUDA version
if [ -f /usr/local/cuda/version.txt ]; then
  CUDA_VERSION_RAW=$(cat /usr/local/cuda/version.txt | grep "CUDA Version" | awk '{print $NF}')
elif command -v nvcc &> /dev/null; then
  CUDA_VERSION_RAW=$(nvcc --version | grep "release" | awk '{print $NF}' | cut -d',' -f1)
else
  echo "ERROR: Could not detect CUDA version. Please set CUDA_VERSION environment variable."
  echo "Using default CUDA version: 12.6"
  CUDA_VERSION_RAW="12.6"
fi

# Normalize CUDA version: remove "V" prefix and extract major.minor (e.g., "V12.6.68" -> "12.6")
CUDA_VERSION=$(echo "$CUDA_VERSION_RAW" | sed 's/^V//' | cut -d'.' -f1-2)

echo "Detected CUDA version: $CUDA_VERSION_RAW (normalized to: $CUDA_VERSION)"

# Use CUDA_VERSION from environment if set (for Docker builds)
export CUDA_VERSION="${CUDA_VERSION:-12.6}"

# Create temporary directory
TMP_DIR="/tmp/cudss_install"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
pushd "${TMP_DIR}" > /dev/null

# Determine architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# cuDSS download URLs
# For Jetson (aarch64), use linux-aarch64
# For SBSA (server ARM64), use linux-sbsa
# For x86_64, use linux-x86_64
if [[ ${CUDA_VERSION:0:4} =~ "13" ]]; then
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
        CUDSS_NAME="libcudss-linux-x86_64-0.7.1.4_cuda13-archive"
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='aarch64'
        CUDSS_NAME="libcudss-linux-aarch64-0.7.1.4_cuda13-archive"
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
elif [[ ${CUDA_VERSION:0:4} =~ ^12\.[0-9]$ ]]; then
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
        CUDSS_NAME="libcudss-linux-x86_64-0.7.1.4_cuda12-archive"
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='aarch64'
        CUDSS_NAME="libcudss-linux-aarch64-0.7.1.4_cuda12-archive"
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
else
    echo "ERROR: Unsupported CUDA version: ${CUDA_VERSION}"
    echo "This script supports CUDA 12.x and CUDA 13.x"
    popd > /dev/null
    rm -rf "${TMP_DIR}"
    exit 1
fi

CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${arch_path}/${CUDSS_NAME}.tar.xz"
echo "Downloading cuDSS for CUDA ${CUDA_VERSION} (${arch_path})..."
curl --retry 3 -OLs "${CUDSS_URL}"

# Extract and install
echo "Extracting cuDSS..."
tar xf ${CUDSS_NAME}.tar.xz

# Determine CUDA installation path
if [ -d "/usr/local/cuda" ]; then
    CUDA_ROOT="/usr/local/cuda"
elif [ -d "/opt/cuda" ]; then
    CUDA_ROOT="/opt/cuda"
else
    echo "ERROR: Could not find CUDA installation directory"
    popd > /dev/null
    rm -rf "${TMP_DIR}"
    exit 1
fi

echo "Installing cuDSS to ${CUDA_ROOT}..."
# Ensure directories exist
mkdir -p ${CUDA_ROOT}/include/
mkdir -p ${CUDA_ROOT}/lib64/
cp -a ${CUDSS_NAME}/include/* ${CUDA_ROOT}/include/ 2>/dev/null || true
cp -a ${CUDSS_NAME}/lib/* ${CUDA_ROOT}/lib64/

# Update library cache
ldconfig

popd > /dev/null
rm -rf "${TMP_DIR}"

# Verify installation
if [ -f "${CUDA_ROOT}/lib64/libcudss.so.0" ] || [ -f "${CUDA_ROOT}/lib64/libcudss.so" ] || \
   [ -f "${CUDA_ROOT}/lib/libcudss.so.0" ] || [ -f "${CUDA_ROOT}/lib/libcudss.so" ]; then
    echo "✓ cuDSS installed successfully"
    echo "  Found: $(find ${CUDA_ROOT} -name "libcudss.so*" 2>/dev/null | head -1)"
else
    echo "WARNING: cuDSS installation may have failed - libraries not found"
    echo "  Searched in: ${CUDA_ROOT}/lib64/ and ${CUDA_ROOT}/lib/"
fi

echo ""
echo "cuDSS installation complete."
