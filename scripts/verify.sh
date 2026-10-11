#!/usr/bin/env bash

# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# verify.sh — Environment and Library Verification Summary
#
# This script should verify the following on the host system:
# - Ubuntu version
# - Disk storage information
# - Available GPU(s) and their VRAM
# - NVIDIA CUDA driver version
# - Python version
# - NumPy version
# - PyTorch version and whether CUDA is enabled
# - ONNX Runtime version and whether CUDA is enabled
# - TensorRT version and whether the TensorRT CLI is available
#
# It only reads the system - nothing is installed or changed.
#
# Usage (from the repo root, or inside the container):
#   bash scripts/verify.sh
#   bash scripts/verify.sh --python .venv      # inspect a specific virtualenv
#
# Please run this script with bash (not sh).

if [ -z "$BASH_VERSION" ]; then
    echo "Please run this script with bash, not sh. For example: bash scripts/verify.sh"
    exit 1
fi

# Do not use strict -e to allow continuing if some checks fail
set -o pipefail

PYTHON_BIN="python3"

# Parse arguments (supports: --python <path-to-venv-or-python>)
candidate=""
while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            if [ -n "${2:-}" ]; then
                candidate="$2"
                shift
            else
                echo "Error: --python requires a path to a venv or python executable"
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--python <venv-dir|python-path>]"
            exit 1
            ;;
    esac
    shift
done

# Resolve provided candidate to a python executable
if [ -n "$candidate" ]; then
    if [ -d "$candidate" ]; then
        if [ -x "$candidate/bin/python3" ]; then
            PYTHON_BIN="$candidate/bin/python3"
        elif [ -x "$candidate/bin/python" ]; then
            PYTHON_BIN="$candidate/bin/python"
        else
            echo "Warning: directory '$candidate' does not look like a virtualenv (no bin/python[3]). Falling back to python3."
        fi
    elif [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
    else
        echo "Warning: '$candidate' is not an executable or directory. Falling back to python3."
    fi
fi

get_ubuntu_version() {
    if [ -f /etc/os-release ]; then
        grep 'PRETTY_NAME=' /etc/os-release | cut -d'=' -f2 | tr -d '"'
    elif command -v lsb_release >/dev/null 2>&1; then
        lsb_release -d | cut -f2 2>/dev/null || echo "Unknown"
    else
        echo "Unknown"
    fi
}

get_storage_info() {
    echo "==================== Storage Info ===================="
    echo "Disk usage summary:"
    df -h | grep -E '^/dev/' | awk '{print $1, $2, $3, $4, $5, $6}' | column -t 2>/dev/null || echo "No /dev/ filesystems found"
}

get_gpu_info() {
    arch=$(uname -m 2>/dev/null || echo "")
    # The container has /etc/nv_tegra_release but not tegrastats.
    if [ "$arch" = "aarch64" ] && { [ -f /etc/nv_tegra_release ] || command -v tegrastats >/dev/null 2>&1; }; then
        echo "======================= GPU =========================="
        model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo "NVIDIA Jetson")
        # Total RAM stands in for VRAM: tegrastats if present, else /proc/meminfo.
        total_mb=""
        if command -v tegrastats >/dev/null 2>&1; then
            ts_line=$(timeout 2s tegrastats 2>/dev/null | head -n1)
            total_mb=$(echo "$ts_line" | sed -n 's/.*RAM [^\/]*\/[[:space:]]*\([0-9]\+\)MB.*/\1/p')
        fi
        if [ -z "$total_mb" ]; then
            total_mb=$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null)
        fi
        if [ -n "$total_mb" ]; then
            echo "GPU: ${model} (Integrated) | VRAM: Shared (System RAM: ${total_mb} MB)"
        else
            echo "GPU: ${model} (Integrated) | VRAM: Shared"
        fi
        l4t_ver=$(get_l4t_version)
        jp_ver=$(get_jetpack_version "$l4t_ver")
        echo "JetPack: $jp_ver (L4T: r$l4t_ver)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "======================= GPU =========================="
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | awk -F, '{gsub(/^\s+|\s+$/, "", $1); gsub(/^\s+|\s+$/, "", $2); print "GPU:",$1" | VRAM:",$2}' || echo "not found"
    else
        echo "======================= GPU =========================="
        echo "nvidia-smi not found (no NVIDIA GPU or drivers not installed)"
    fi
}

get_cuda_driver_version() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 2>/dev/null || echo "not found"
    elif [ -f /proc/driver/nvidia/version ]; then
        sed -n '1s/.*Kernel Module  \([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null || echo "not found"
    else
        echo "not found"
    fi
}

get_cuda_toolkit_version() {
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version | grep -o 'V[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?' | head -n1 | sed 's/^V//' 2>/dev/null || echo "not found"
    else
        echo "not found"
    fi
}

# Jetson: L4T -> JetPack mapping and helpers.
#
# Kept in sync with NVIDIA_JETPACK in jetson-stats (jtop),
# jtop/core/jetson_variables.py, which is the table jtop itself reports from.
# JetPack 4 is omitted: it is Ubuntu 18.04 and cannot run this package.
declare -A L4T_TO_JETPACK=(
    # JetPack 7
    ["39.2.2"]="7.2.2"
    ["39.2.1"]="7.2.1"
    ["39.2.0"]="7.2 GA"
    ["39.0.0"]="7.2 EA"
    ["38.4.0"]="7.1"
    ["38.2.2"]="7.0"
    ["38.2.1"]="7.0"
    ["38.2.0"]="7.0"
    # JetPack 6
    ["36.5.2"]="6.2.3"
    ["36.5.0"]="6.2.2"
    ["36.4.7"]="6.2.1"
    ["36.4.4"]="6.2.1"
    ["36.4.3"]="6.2"
    ["36.4.2"]="6.1 (rev1)"
    ["36.4.0"]="6.1"
    ["36.3.0"]="6.0"
    ["36.2.0"]="6.0 DP"
    ["36.0.0"]="6.0 EA"
    # JetPack 5
    ["35.6.5"]="5.1.7"
    ["35.6.4"]="5.1.6"
    ["35.6.3"]="5.1.5"
    ["35.6.2"]="5.1.5"
    ["35.6.1"]="5.1.5"
    ["35.6.0"]="5.1.4"
    ["35.5.0"]="5.1.3"
    ["35.4.1"]="5.1.2"
    ["35.3.1"]="5.1.1"
    ["35.3.0"]="5.1.1 PRE"
    ["35.2.1"]="5.1"
    ["35.1.0"]="5.0.2 GA"
    ["34.1.1"]="5.0.1 DP"
    ["34.1.0"]="5.0 DP"
    ["34.0.1"]="5.0 PRE-DP"
)

get_l4t_version() {
    if [[ -f /etc/nv_tegra_release ]]; then
        local l4t_line=$(head -n 1 /etc/nv_tegra_release)
        local l4t_release=$(echo "$l4t_line" | sed 's/.*# R\([0-9]*\).*/\1/')
        local l4t_revision=$(echo "$l4t_line" | sed 's/.*REVISION: \([0-9.]*\).*/\1/')
        echo "${l4t_release}.${l4t_revision}"
    else
        echo "Unknown"
    fi
}

get_jetpack_version() {
    local l4t_version="$1"
    if [[ -n "${L4T_TO_JETPACK[$l4t_version]:-}" ]]; then
        echo "${L4T_TO_JETPACK[$l4t_version]}"
    else
        echo "Unknown"
    fi
}

get_python_version() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" --version 2>&1 | sed 's/Python //'
    else
        echo "not found"
    fi
}

get_python_path() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        command -v "$PYTHON_BIN"
    else
        echo "not found"
    fi
}

get_numpy_version() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" - <<'PY'
try:
    import numpy as np
    print(np.__version__)
except Exception:
    print("not found")
PY
    else
        echo "not found"
    fi
}

get_pytorch_info() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" - <<'PY'
# Report *why* an import failed rather than a bare "not found". A missing
# shared library (e.g. libopenblas.so.0, which the Jetson torch wheels need)
# looks identical to "not installed" otherwise, which makes it much harder to
# diagnose than it should be.
try:
    import torch
    ver = torch.__version__
    cuda = torch.cuda.is_available()
    print(f"{ver} (CUDA: {cuda})")
except ImportError as e:
    print(f"INSTALLED BUT FAILING TO IMPORT: {e}")
except Exception as e:
    print(f"error: {type(e).__name__}: {e}")
PY
    else
        echo "not found"
    fi
}

get_onnxruntime_info() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" - <<'PY'
try:
    import onnxruntime as ort
    ver = ort.__version__
    providers = set(ort.get_available_providers())
    cuda = 'CUDAExecutionProvider' in providers
    trt = 'TensorrtExecutionProvider' in providers
    print(f"{ver} (CUDA: {cuda}, TensorRT: {trt})")
except ImportError as e:
    # A NumPy 1.x-built wheel under NumPy 2.x fails here; say so explicitly.
    print(f"INSTALLED BUT FAILING TO IMPORT: {e}")
except Exception as e:
    print(f"error: {type(e).__name__}: {e}")
PY
    else
        echo "not found"
    fi
}

get_tensorrt_version() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" - <<'PY'
try:
    import tensorrt as trt
    print(trt.__version__)
except Exception:
    print("not found")
PY
    else
        echo "not found"
    fi
}

get_tensorrt_cli() {
    # trtexec often ships without being on PATH; look for it in the usual place
    # so the check does not report a false negative.
    local PATH="$PATH"
    if ! command -v trtexec >/dev/null 2>&1 && [ -x /usr/src/tensorrt/bin/trtexec ]; then
        PATH="/usr/src/tensorrt/bin:$PATH"
    fi

    if command -v trtexec >/dev/null 2>&1; then
        # Try to parse version from trtexec banner output, fall back to availability
        ver=$(trtexec 2>&1 | sed -n 's/.*\[TensorRT v\([^]]*\)\].*/\1/p' | head -n1)
        if [ -z "$ver" ]; then
            ver=$(trtexec --help 2>&1 | sed -n 's/.*\[TensorRT v\([^]]*\)\].*/\1/p' | head -n1)
        fi
        if [ -n "$ver" ]; then
            # If version is a 6-digit numeric like 101303, format to 10.13.3
            if [[ "$ver" =~ ^[0-9]{6}$ ]]; then
                major="${ver:0:2}"
                minor="${ver:2:2}"
                patch_raw="${ver:4:2}"
                # strip leading zero from patch
                patch=$((10#$patch_raw))
                echo "${major}.${minor}.${patch}"
            else
                echo "$ver"
            fi
        else
            echo "available"
        fi
    else
        echo "not found"
    fi
}

# Collect values
ubuntu_version=$(get_ubuntu_version)
cuda_driver_version=$(get_cuda_driver_version)
cuda_toolkit_version=$(get_cuda_toolkit_version)
python_version=$(get_python_version)
python_path=$(get_python_path)
numpy_version=$(get_numpy_version)
pytorch_info=$(get_pytorch_info)
onnxruntime_info=$(get_onnxruntime_info)
tensorrt_version=$(get_tensorrt_version)
tensorrt_cli=$(get_tensorrt_cli)

echo "==================== System Info ===================="
echo "Ubuntu Version: $ubuntu_version"
echo

get_gpu_info
echo

echo "====================== CUDA ========================="
echo "CUDA Toolkit Version: $cuda_toolkit_version"
echo "CUDA Driver Version: $cuda_driver_version"
echo "TensorRT Version: $tensorrt_version"
echo "TensorRT CLI: $tensorrt_cli"
echo

echo "================ Python Environment ================"
echo "Python Version: $python_version"
echo "Python Path: $python_path"
echo "NumPy Version: $numpy_version"
echo "PyTorch: $pytorch_info"
echo "ONNX Runtime: $onnxruntime_info"
echo

get_storage_info
echo
echo "====================================================="
