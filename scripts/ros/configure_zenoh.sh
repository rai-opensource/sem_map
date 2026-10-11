#!/bin/bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Configure Zenoh for ROS2 communication

# This script should only be sourced, not executed
if [ "${BASH_SOURCE[0]}" -ef "$0" ] ; then
  echo "ERROR: $(basename "${BASH_SOURCE[0]}") should only be sourced, not run!"
  exit 1
fi

# Determine script directory (the config template lives in ../config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to localhost. Pass a router IP as $1 for a multi-machine setup.
ZENOH_ROUTER_IP="${1:-127.0.0.1}"

if [[ $# -eq 0 ]] ; then
  echo "Configuring Zenoh to communicate with localhost only."
elif [[ $# -eq 1 ]] ; then
  echo "Configuring Zenoh to communicate with router at: ${ZENOH_ROUTER_IP}"
else
  echo "Usage: source configure_zenoh.sh [router_ip]"
  echo "Only one router IP is supported"
  return 1
fi

# Set RMW implementation to Zenoh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Generate config file in user-writable location
ZENOH_CONFIG_DIR="${HOME}/.tmp"
mkdir -p "$ZENOH_CONFIG_DIR"
ZENOH_CONFIG_FILE="$ZENOH_CONFIG_DIR/sem-map-zenoh.json5"
ZENOH_CONFIG_FILE_TMP="$ZENOH_CONFIG_FILE.tmp"

python3 "$SCRIPT_DIR/render_zenoh_config.py" \
  --router-ip "${ZENOH_ROUTER_IP}" \
  --output-file "$ZENOH_CONFIG_FILE_TMP"

# Point ROS2 nodes to the generated config
export ZENOH_SESSION_CONFIG_URI="$ZENOH_CONFIG_FILE"

# If config changed, restart the ROS2 daemon to clear cached values
if ! cmp --silent "$ZENOH_CONFIG_FILE_TMP" "$ZENOH_CONFIG_FILE" 2>/dev/null ; then
  ros2 daemon stop > /dev/null 2>&1 || true
fi

# Move the temp file to the final location
mv -f "$ZENOH_CONFIG_FILE_TMP" "$ZENOH_CONFIG_FILE"

start_zenohd() {
  # Start the zenohd router in the background if it's not already started
  PID_FILE="/tmp/zenohd.pid"
  LOG_FILE="/tmp/zenohd.log"

  # Check if zenohd is already running by looking for the process
  if pgrep -x rmw_zenohd > /dev/null 2>&1; then
    EXISTING_PID=$(pgrep -x rmw_zenohd)
    echo "zenohd router is already running (PID: $EXISTING_PID)"
    echo "$EXISTING_PID" > "$PID_FILE"
  else
    # Run zenohd in the background if not running
    echo "Starting zenohd router in the background. Logs: $LOG_FILE"

    # Launch the process in the background without setsid
    # This keeps it simple and allows proper PID tracking
    nohup ros2 run rmw_zenoh_cpp rmw_zenohd > "$LOG_FILE" 2>&1 &

    # Capture the actual PID
    ZENOH_PID=$!
    echo "$ZENOH_PID" > "$PID_FILE"

    # Give it a moment to start
    sleep 0.5

    # Verify it's running
    if kill -0 "$ZENOH_PID" 2>/dev/null; then
      echo "zenohd router started successfully (PID: $ZENOH_PID)"
    else
      echo "Warning: zenohd router may have failed to start. Check logs at $LOG_FILE"
    fi
  fi
}

start_zenohd

echo "Zenoh configured successfully"
