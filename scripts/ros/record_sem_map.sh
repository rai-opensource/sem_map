#!/bin/bash

# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# === ROS2 Bag Recording Script for Semantic Mapping ===
# Records the topics listed in config/ros/record_sem_map_topics.yaml in MCAP
# format.
#
# Usage:
#   ./scripts/ros/record_sem_map.sh [suffix]
#
# Environment overrides:
#   OUTPUT_DIR   output directory      (default: <repo>/recordings)
#   BAG_NAME     bag name              (default: sem_map_<timestamp>[_<suffix>])
#   TOPICS_YAML  topic list to record  (default: <repo>/config/ros/record_sem_map_topics.yaml)
#   STORAGE_CONFIG  mcap storage options (default: <repo>/config/ros/mcap_zstd.yaml)
#                   Set to "" to record uncompressed.

# === Configuration ===
# Determine repo root relative to this script (scripts/ros/.. -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default output directory (can be overridden)
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/recordings}"

# Optional suffix from command line argument
SUFFIX="${1:-}"

# Build bag name with optional suffix
if [ -n "$SUFFIX" ]; then
    BAG_NAME="${BAG_NAME:-sem_map_$(date +%Y%m%d_%H%M%S)_${SUFFIX}}"
else
    BAG_NAME="${BAG_NAME:-sem_map_$(date +%Y%m%d_%H%M%S)}"
fi

# === Environment Setup ===
if [ -f "$REPO_ROOT/scripts/setup_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/setup_env.sh"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# === Topic List ===
# Topics are defined in config/ros/record_sem_map_topics.yaml so they can be
# edited without touching this script. Override with TOPICS_YAML=/path/to.yaml.
TOPICS_YAML="${TOPICS_YAML:-$REPO_ROOT/config/ros/record_sem_map_topics.yaml}"

if [ ! -f "$TOPICS_YAML" ]; then
    echo "ERROR: topics YAML not found: $TOPICS_YAML" >&2
    exit 1
fi

# Read all topic lists from YAML, one topic per line, into ALL_TOPICS.
mapfile -t ALL_TOPICS < <(python3 - "$TOPICS_YAML" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
for t in cfg.get("topics") or []:
    print(t)
PY
)

if [ "${#ALL_TOPICS[@]}" -eq 0 ]; then
    echo "ERROR: no topics loaded from $TOPICS_YAML" >&2
    exit 1
fi

# === Storage Options ===
# Per-chunk zstd, the same options sem_map_replay_record.launch.py records with,
# so a live bag and a replay bag are stored the same way. Foxglove and rosbag2
# both read chunk compression without unpacking the file first.
STORAGE_CONFIG="${STORAGE_CONFIG-$REPO_ROOT/config/ros/mcap_zstd.yaml}"

STORAGE_ARGS=()
if [ -n "$STORAGE_CONFIG" ]; then
    if [ -f "$STORAGE_CONFIG" ]; then
        STORAGE_ARGS+=(--storage-config-file "$STORAGE_CONFIG")
    else
        echo "WARNING: storage config not found, recording uncompressed: $STORAGE_CONFIG" >&2
        STORAGE_CONFIG=""
    fi
fi

# === Recording Command ===
BAG_PATH="$OUTPUT_DIR/$BAG_NAME"

echo "=========================================="
echo "ROS2 Bag Recording - Semantic Mapping"
echo "=========================================="
echo "Output:     $BAG_PATH.mcap"
echo "Format:     MCAP${STORAGE_CONFIG:+ (zstd, from $(basename "$STORAGE_CONFIG"))}"
echo "Topics:     ${#ALL_TOPICS[@]} (from $TOPICS_YAML)"
echo ""
echo "Press Ctrl+C to stop recording"
echo "=========================================="
echo ""

# Record with MCAP storage format
ros2 bag record \
    --output "$BAG_PATH" \
    --storage mcap \
    "${STORAGE_ARGS[@]}" \
    "${ALL_TOPICS[@]}"

echo ""
echo "Recording stopped. Bag saved to: $BAG_PATH"
