#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Install cuSPARSELt library for CUDA 11.8, 12.x, and 13.x
# Required by PyTorch 2.9+ for sparse matrix operations
# Based on official PyTorch installation script:
# https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cusparselt.sh

set -euo pipefail

# This script must be run as root for system wide installation.
if [[ "$EUID" -ne 0 ]]; then
  echo "ERROR: cuSPARSELt install must be run as root, please run:"
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
#   exit 1
fi

# Normalize CUDA version: remove "V" prefix and extract major.minor (e.g., "V12.6.68" -> "12.6")
CUDA_VERSION=$(echo "$CUDA_VERSION_RAW" | sed 's/^V//' | cut -d'.' -f1-2)

echo "Detected CUDA version: $CUDA_VERSION_RAW (normalized to: $CUDA_VERSION)"

# Use CUDA_VERSION from environment if set (for Docker builds)
export CUDA_VERSION="${CUDA_VERSION:-12.6}"

# Create temporary directory
TMP_DIR="/tmp/cusparselt_install"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
pushd "${TMP_DIR}" > /dev/null

# Determine architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html
# Version mapping based on official PyTorch script:
# https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cusparselt.sh
if [[ ${CUDA_VERSION:0:4} =~ "13" ]]; then
    # Determine architecture path: sbsa for ARM64 (Jetson), x86_64 for x86_64
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='sbsa'
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.8.0.4_cuda13-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
    echo "Downloading cuSPARSELt for CUDA ${CUDA_VERSION} (${arch_path})..."
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
elif [[ ${CUDA_VERSION:0:4} =~ ^12\.[5-9]$ ]]; then
    # Determine architecture path: sbsa for ARM64 (Jetson), x86_64 for x86_64
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='sbsa'
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.7.1.0-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
    echo "Downloading cuSPARSELt for CUDA ${CUDA_VERSION} (${arch_path})..."
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
elif [[ ${CUDA_VERSION:0:4} == "12.4" ]]; then
    # Determine architecture path: sbsa for ARM64 (Jetson), x86_64 for x86_64
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='sbsa'
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.6.2.3-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
    echo "Downloading cuSPARSELt for CUDA ${CUDA_VERSION} (${arch_path})..."
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
elif [[ ${CUDA_VERSION:0:4} =~ ^12\.[1-3]$ ]]; then
    # Determine architecture path: sbsa for ARM64 (Jetson), x86_64 for x86_64
    if [[ $ARCH == x86_64* ]] || [[ $ARCH == amd64* ]]; then
        arch_path='x86_64'
    elif [[ $ARCH == aarch64* ]]; then
        arch_path='sbsa'
    else
        echo "ERROR: Unsupported architecture: $ARCH"
        popd > /dev/null
        rm -rf "${TMP_DIR}"
        exit 1
    fi
    # For CUDA 12.1-12.3, use version 0.5.2.1 (older but compatible)
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.5.2.1-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
    echo "Downloading cuSPARSELt for CUDA ${CUDA_VERSION} (${arch_path})..."
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
    CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.4.0.7-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/${CUSPARSELT_NAME}.tar.xz"
    echo "Downloading cuSPARSELt for CUDA ${CUDA_VERSION}..."
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
else
    echo "ERROR: Unsupported CUDA version: ${CUDA_VERSION}"
    echo "This script supports CUDA 11.8, CUDA 12.1-12.9, and CUDA 13.x"
    popd > /dev/null
    rm -rf "${TMP_DIR}"
    exit 1
fi

# Extract and install
echo "Extracting cuSPARSELt..."
tar xf ${CUSPARSELT_NAME}.tar.xz

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

echo "Installing cuSPARSELt to ${CUDA_ROOT}..."
# Ensure directories exist
mkdir -p ${CUDA_ROOT}/include/
mkdir -p ${CUDA_ROOT}/lib64/
cp -a ${CUSPARSELT_NAME}/include/* ${CUDA_ROOT}/include/
cp -a ${CUSPARSELT_NAME}/lib/* ${CUDA_ROOT}/lib64/

# Update library cache
ldconfig

popd > /dev/null
rm -rf "${TMP_DIR}"

# Verify installation
# Note: The library is named libcusparseLt.so (capital L), not libcusparse_lt.so
if [ -f "${CUDA_ROOT}/lib64/libcusparseLt.so" ] || [ -f "${CUDA_ROOT}/lib/libcusparseLt.so" ] || \
   [ -f "${CUDA_ROOT}/lib64/libcusparse_lt.so" ] || [ -f "${CUDA_ROOT}/lib/libcusparse_lt.so" ]; then
    echo "✓ cuSPARSELt installed successfully"
    # Note: libcudss.so.0 is NOT part of cuSPARSELt - it's a separate CUDA library
    # that may be part of the CUDA toolkit itself or needs separate installation
    if find ${CUDA_ROOT} -name "libcudss.so*" 2>/dev/null | grep -q .; then
        echo "✓ libcudss.so found"
    else
        echo "NOTE: libcudss.so not found in cuSPARSELt (this is expected - it's a separate library)"
    fi
else
    echo "WARNING: cuSPARSELt installation may have failed - libraries not found"
fi

echo ""
echo "cuSPARSELt installation complete."
