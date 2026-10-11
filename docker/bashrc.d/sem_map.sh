# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Interactive shell defaults for the sem_map container.
#
# Installed to /etc/profile.d/sem_map.sh and sourced from the container user's
# .bashrc at image build time. This used to be appended by the entrypoint on
# every container start, which meant it could be written twice or drift between
# the image and a running container.
#
# shellcheck shell=bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export XDG_RUNTIME_DIR="/tmp/user/$(id -un)"
mkdir -m 0700 -p "$XDG_RUNTIME_DIR" 2>/dev/null || true
export NO_AT_BRIDGE=1

# Enable user site-packages so `pip install --user` works without sudo.
export PYTHONUSERBASE="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"

export UV_CACHE_DIR="$HOME/.cache/uv"
mkdir -p "$UV_CACHE_DIR" 2>/dev/null || true

# CUDA toolchain convenience.
# /usr/local/cuda is a symlink on Jetson and resolves to cuda-12.6.
if [ -d /usr/local/cuda/bin ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [ -d /usr/local/cuda/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
# Jetson-specific CUDA targets directory.
if [ -d /usr/local/cuda/targets/aarch64-linux/lib ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}"
fi
# Note: cuDNN (bundled in PyTorch's wheel, needed by onnxruntime-gpu) is
# registered with ldconfig at image build time rather than exported here.

# TensorRT toolchain convenience (trtexec).
if [ -d /usr/src/tensorrt/bin ]; then
  export PATH="/usr/src/tensorrt/bin:$PATH"
fi

# ZED SDK CMake config — required for find_package(ZED) in ROS 2 builds.
# Only present when the image was built with INSTALL_ZED=true.
if [ -d /usr/local/zed ]; then
  export CMAKE_PREFIX_PATH="/usr/local/zed${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi

# Repo location. docker-compose.yaml bind-mounts the repository at the same
# absolute path inside the container as on the host, and passes that path in as
# WORKSPACE_DIR, so relative symlinks and absolute paths in configs, artifact
# caches and rosbag metadata resolve identically in both worlds.
export WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/sem_map}"

# Repo environment (ROS overlay, PYTHONPATH, RMW config, ...).
alias sem_map_setup='source "$WORKSPACE_DIR/scripts/setup_env.sh"'
if [[ $- == *i* ]] && [ -f "$WORKSPACE_DIR/scripts/setup_env.sh" ]; then
  source "$WORKSPACE_DIR/scripts/setup_env.sh"
fi

# Keep bash history in the (bind-mounted) repo so it survives container removal.
if [ -d "$WORKSPACE_DIR" ]; then
  export HISTFILE="$WORKSPACE_DIR/.bash_history"
fi

# ── Prompt ────────────────────────────────────────────────────────────────────
if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
  if [ -f /usr/lib/git-core/git-sh-prompt ]; then
    . /usr/lib/git-core/git-sh-prompt
    unset GIT_PS1_SHOWDIRTYSTATE GIT_PS1_SHOWSTASHSTATE GIT_PS1_SHOWUNTRACKEDFILES
    # Pass the parentheses and colors AS the format string to __git_ps1 so an
    # empty "()" is not printed outside of a git repo.
    PS1='\[\033[01;32m\]\u\[\033[00m\] \[\033[01;33m\]➜\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\]$(__git_ps1 " (\[\033[01;31m\]%s\[\033[01;34m\])")\[\033[00m\] \[\033[01;37m\]\$\[\033[00m\] '
  else
    PS1='\[\033[01;32m\]\u\[\033[00m\] \[\033[01;33m\]➜\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\] \[\033[01;37m\]\$\[\033[00m\] '
  fi
else
  PS1='\u ➜ \w \$ '
fi
