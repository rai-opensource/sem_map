# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# sem_map environment setup.
#
# Source this file (do not execute it) to prepare a shell for building and
# running sem_map:
#
#     source scripts/setup_env.sh
#
# It sources ROS 2, picks a middleware, sets colcon's defaults so a plain
# `colcon build` is correctly configured, and sources the resulting install
# space so `ros2 launch sem_map ...` works.
#
# shellcheck shell=bash

# Safety check: ensure this script is being sourced and not executed directly
if [ "${BASH_SOURCE[0]}" -ef "$0" ] ; then
  echo "ERROR: $(basename "${BASH_SOURCE[0]}") must be sourced, not executed directly!"
  exit 1
fi

# Repo root = parent of the directory containing this script (scripts/..).
# This repo IS the ROS package: package.xml lives at the repo root, and there is
# no enclosing colcon workspace. Build/install artifacts are kept inside the repo.
SEM_MAP_ROOT=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
export SEM_MAP_ROOT

if [ -d "/opt/ros/humble" ]; then
    ROS_DISTRO="humble"
    echo "Using ROS distribution: $ROS_DISTRO"

    # shellcheck disable=SC1091
    source "/opt/ros/humble/setup.bash"

    # Source a local virtualenv if one exists at the repo root
    if [ -f "$SEM_MAP_ROOT/.venv/bin/activate" ]; then
        echo "Sourcing venv from: $SEM_MAP_ROOT/.venv/bin/activate"
        # shellcheck disable=SC1091
        source "$SEM_MAP_ROOT/.venv/bin/activate"
        export COLCON_PYTHON_EXECUTABLE="$SEM_MAP_ROOT/.venv/bin/python"
    fi

    # Source the install space produced by `colcon build` (if already built)
    if [ -f "$SEM_MAP_ROOT/install/setup.bash" ]; then
        echo "Sourcing install space from: $SEM_MAP_ROOT/install/setup.bash"
        # shellcheck disable=SC1091
        source "$SEM_MAP_ROOT/install/setup.bash"
    fi

    # ── Middleware selection ──────────────────────────────────────────────
    # ROS 2 itself defaults to FastDDS (rmw_fastrtps_cpp). We select CycloneDDS
    # instead: it copes better with the large image and point cloud messages
    # this package publishes, and needs no extra daemon.
    #
    # Choose a different one by exporting SEM_MAP_DDS before sourcing this file:
    #   cyclonedds  (default here)
    #   fastdds     ROS 2's own default, if you would rather stay on it
    #   zenoh       needs rmw_zenoh_cpp; set ZENOH_ROUTER_IP for a remote router
    SEM_MAP_DDS=${SEM_MAP_DDS:-cyclonedds}
    case "$SEM_MAP_DDS" in
        cyclonedds)
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            # SEM_MAP_CYCLONEDDS_URI lets a robot environment file (e.g.
            # scripts/ros/b2w_env.sh) supply its own peers config and survive
            # scripts that re-source this file, such as record_sem_map.sh.
            export CYCLONEDDS_URI="${SEM_MAP_CYCLONEDDS_URI:-file://$SEM_MAP_ROOT/scripts/config/cyclonedds-config.xml}"
            echo "Using CycloneDDS"
            ;;
        fastdds|fastrtps)
            export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
            echo "Using FastDDS (the ROS 2 default)"
            ;;
        zenoh)
            # shellcheck disable=SC1091
            source "$SEM_MAP_ROOT/scripts/ros/configure_zenoh.sh" "${ZENOH_ROUTER_IP:-127.0.0.1}"
            ;;
        *)
            echo "Warning: unknown SEM_MAP_DDS='$SEM_MAP_DDS'. Using CycloneDDS."
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export CYCLONEDDS_URI="${SEM_MAP_CYCLONEDDS_URI:-file://$SEM_MAP_ROOT/scripts/config/cyclonedds-config.xml}"
            ;;
    esac

    # ROS domain ID isolates ROS 2 graphs on the same network.
    # Override by exporting ROS_DOMAIN_ID before sourcing this file.
    export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

    # Put the Python package on the path. The CLI tools, the training code and
    # the labelling apps all `import sem_map`, which lives in scripts/. colcon
    # installs it under share/, but neither this file nor install/setup.bash
    # puts it on PYTHONPATH, so the documented commands -- for example
    # `python scripts/sem_map/cli/run_export_all_trt.py` -- would otherwise die
    # with ModuleNotFoundError. The source tree is used rather than the install
    # space so edits take effect without rebuilding.
    export PYTHONPATH="$SEM_MAP_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

    # Give `colcon build` its defaults: Release, merged install space. Without
    # this, colcon would default to an empty CMAKE_BUILD_TYPE and silently
    # produce an unoptimised binary. Run colcon from the repo root.
    # Re-source this file afterwards to pick up the new install space.
    if [ -f "$SEM_MAP_ROOT/colcon-defaults.yaml" ]; then
        export COLCON_DEFAULTS_FILE="$SEM_MAP_ROOT/colcon-defaults.yaml"
    fi

    # Enable command-line completion for ROS 2 CLI tools and colcon
    if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
    fi

else
    echo "ROS 2 Humble not found at /opt/ros/humble - skipping ROS setup"
fi

# X display for RViz / OpenCV windows. Override by exporting DISPLAY beforehand.
export DISPLAY=${DISPLAY:-:0}

# CUDA / MPS settings. MPS lets several CUDA processes share one GPU
# efficiently; see scripts/enable_max_mode.sh to start the MPS daemon.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}

# Aliases
alias sem_map_record="$SEM_MAP_ROOT/scripts/ros/record_sem_map.sh"

# ── Optional: vision-LLM class filter ─────────────────────────────────────
# sem_map can optionally query an OpenAI-compatible vision LLM to filter or
# relabel semantic classes. This is off unless the three variables below are
# set. Point SEM_MAP_LLM_BASE_URL at any OpenAI-compatible /v1 endpoint (a
# hosted provider or a local server), pick a vision-capable model, and supply
# the matching API key. Keep real keys out of version control - export them
# from your own shell profile or a git-ignored file instead.
#
# export SEM_MAP_LLM_BASE_URL=""   # e.g. https://<provider>/v1/
# export SEM_MAP_LLM_MODEL=""      # a vision-capable model name
# export SEM_MAP_LLM_API_KEY=""    # your API key
