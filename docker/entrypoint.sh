#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Container entrypoint for the sem_map image.
#
# This runs as the unprivileged container user (the image sets USER), not root.
# Everything that needs root — creating the user, sudoers, device groups, the
# setgid bits on dist-packages, the shell environment, the rosdep cache — happens
# once at image build time rather than on every container start.
#
# That split is deliberate. `docker exec` does not run the entrypoint, so an
# entrypoint that starts as root and drops privileges only for its own child
# leaves every `docker compose exec` shell as root, and anything built from one of
# those shells lands on the host owned by root. Baking the user into the image
# makes `exec` unprivileged too, with no `-u` flag to remember.
#
# Optional environment (docker-compose.yaml sets it):
#   WORKSPACE_DIR   path to the repo inside the container, bind-mounted at the
#                   same absolute path as on the host so symlinks resolve
#
# Host-vs-container UID matching is numeric, so only the numbers have to line up.
# If your host user is not 1000:1000, rebuild with
#   --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
# See docker/Dockerfile.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/sem_map}"

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "WARNING: WORKSPACE_DIR=$WORKSPACE_DIR does not exist inside the" \
       "container. Is the repository bind-mounted? See docker-compose.yaml." >&2
fi

if [[ $# -eq 0 ]]; then
  set -- /bin/bash
fi

cd "$WORKSPACE_DIR" 2>/dev/null || cd "$HOME"
exec "$@"
