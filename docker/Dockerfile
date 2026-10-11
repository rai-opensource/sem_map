# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# sem_map development / runtime image.
#
# Builds for two targets from one file:
#   PLATFORM=desktop  x86_64 workstation with a discrete NVIDIA GPU  (default)
#   PLATFORM=agx      NVIDIA Jetson AGX Orin, JetPack 6 / L4T r36.4
#
# Build from the REPOSITORY ROOT (the build context must be the repo root, since
# the rosdep step binds package.xml from it):
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t sem_map:desktop .
# or just use the docker-compose.yaml at the repo root:
#   docker compose --profile desktop build
#
# ── Base image selection ───────────────────────────────────────────────────────
# BuildKit resolves both FROM directives at parse time; only the selected stage
# is actually pulled/used. Both images are public on nvcr.io (no login needed).
ARG PLATFORM=desktop

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0           AS base-agx
FROM nvcr.io/nvidia/cuda:12.8.2-devel-ubuntu22.04 AS base-desktop

FROM base-${PLATFORM} AS workspace

# Prevent interactive prompts during installation
ARG DEBIAN_FRONTEND=noninteractive

# Set up locale and timezone
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
      locales \
      tzdata && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG="en_US.UTF-8"

# Install essential system utilities and development tools
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
      sudo \
      git \
      curl \
      wget \
      ca-certificates \
      gnupg \
      lsb-release \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      python-is-python3 \
      software-properties-common \
      build-essential \
      cmake \
      # The Jetson torch wheels link against the system OpenBLAS and the base
      # l4t-jetpack image does not ship it, so `import torch` dies with
      # "libopenblas.so.0: cannot open shared object file" without this.
      libopenblas0 \
      bash-completion \
      vim \
      tree \
      iputils-ping \
      net-tools \
      ffmpeg \
      x11-apps \
      dbus-x11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Re-declare ARGs after FROM (ARGs don't survive FROM boundaries) and bake
# PLATFORM into ENV so RUN steps (notably install-python.sh, which picks the
# matching requirements files) can read it.
ARG PLATFORM=desktop
ENV PLATFORM=${PLATFORM}

# Optional Stereolabs ZED SDK. Off by default: it is a ~1 GB download from
# stereolabs.com and is only needed if you run a ZED camera live. Enable with
#   docker build --build-arg INSTALL_ZED=true ...
# or by setting INSTALL_ZED=true in the environment used by docker-compose.
ARG INSTALL_ZED=false

RUN mkdir -p /tmp/scripts/setup

# ── NVIDIA / ROS2 / Python toolchain ───────────────────────────────────────────
# Each install script is COPYed immediately before it runs so that editing one
# script does not invalidate the cached layers of the others.

# TensorRT pin for the desktop build; BuildKit exposes ARGs to RUN as env vars.
# The aarch64 branch of the script installs a fixed version and ignores this.
ARG TRT_PIN_VERSION=10.8.0.*

COPY docker/setup/install-tensorrt.sh /tmp/scripts/setup/install-tensorrt.sh
RUN /tmp/scripts/setup/install-tensorrt.sh

COPY docker/setup/install-ros2.sh /tmp/scripts/setup/install-ros2.sh
RUN /tmp/scripts/setup/install-ros2.sh

COPY docker/setup/install-zed-sdk.sh /tmp/scripts/setup/install-zed-sdk.sh
RUN if [ "${INSTALL_ZED}" = "true" ]; then \
      /tmp/scripts/setup/install-zed-sdk.sh; \
    else \
      echo "INSTALL_ZED=${INSTALL_ZED} — skipping ZED SDK install."; \
    fi

COPY docker/setup/install-cusparselt.sh /tmp/scripts/setup/install-cusparselt.sh
RUN /tmp/scripts/setup/install-cusparselt.sh

COPY docker/setup/install-cudss.sh /tmp/scripts/setup/install-cudss.sh
RUN /tmp/scripts/setup/install-cudss.sh

# Python deps. install-python.sh reads $WORKSPACE_ROOT/docker/requirements/ and
# selects the desktop/agx requirement pair from $PLATFORM.
COPY docker/setup/install-python.sh /tmp/scripts/setup/install-python.sh
COPY docker/requirements/ /tmp/docker/requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    /tmp/scripts/setup/install-python.sh /tmp

# cuDNN arrives inside PyTorch's nvidia-cudnn-cu12 wheel, not as a system
# package, so the dynamic linker cannot find it. onnxruntime-gpu then fails to
# load libonnxruntime_providers_cuda.so and silently falls back to CPU
# inference. Registering the wheel's lib directory with ldconfig fixes it for
# every process; an LD_LIBRARY_PATH export would not, because `runuser` strips
# LD_* variables when the entrypoint drops privileges.
RUN CUDNN_LIB="$(python3 -c 'import glob;print(next(iter(glob.glob("/usr/*/lib/python3*/dist-packages/nvidia/cudnn/lib")+glob.glob("/usr/lib/python3*/dist-packages/nvidia/cudnn/lib")), ""))')" && \
    if [ -n "$CUDNN_LIB" ]; then \
      echo "$CUDNN_LIB" > /etc/ld.so.conf.d/nvidia-cudnn.conf && ldconfig && \
      echo "registered cuDNN at $CUDNN_LIB"; \
    else \
      echo "WARNING: bundled cuDNN not found; onnxruntime will run on CPU" >&2; \
    fi

COPY docker/setup/install-elevation-mapping.sh /tmp/scripts/setup/install-elevation-mapping.sh
RUN /tmp/scripts/setup/install-elevation-mapping.sh

# ── ROS dependencies for this package ──────────────────────────────────────────
# This repository *is* the ROS2 package (package.xml sits at the repo root), so
# there is no multi-package src/ tree to crawl. We bind just package.xml into a
# throwaway overlay layout — /tmp/workspace/src/sem_map/package.xml — which is
# exactly what `rosdep install --from-paths /tmp/workspace/src` expects to find.
# Binding only package.xml (rather than the whole repo) also means edits to
# source files do not bust this cache layer.
COPY docker/setup/install-rosdeps.sh /tmp/scripts/setup/install-rosdeps.sh
RUN --mount=type=bind,source=package.xml,target=/tmp/workspace/src/sem_map/package.xml \
    /tmp/scripts/setup/install-rosdeps.sh /tmp/workspace

# ── Container user ─────────────────────────────────────────────────────────────
# A fixed non-root user, baked in at build time.
#
# Bind-mount ownership is purely numeric: the host only ever sees UIDs and GIDs,
# so the *name* is cosmetic and only UID/GID have to line up with the host user.
# 1000:1000 is the first user on a single-user Linux install, which covers both
# supported platforms. Build for a host that differs:
#
#   docker compose --profile desktop build \
#       --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
#
# The home directory is /home/${CONTAINER_USER} and must NOT contain the
# bind-mounted repository. Deriving it from the *host* username (as this image
# used to) puts a repo at /home/<user>/... inside it, and the ownership fixup
# then recursively chowns the entire tree — build output, models and datasets —
# on every single container start.
ARG CONTAINER_USER=dev
ARG HOST_UID=1000
ARG HOST_GID=1000
ENV CONTAINER_USER=${CONTAINER_USER}

RUN set -eux; \
    if ! getent group "${HOST_GID}" >/dev/null; then \
      groupadd --gid "${HOST_GID}" "${CONTAINER_USER}"; \
    fi; \
    # --no-log-init is not optional here. /var/log/lastlog and /var/log/faillog
    # are indexed BY UID, so useradd extends them to cover the new UID. They are
    # sparse on disk, but Docker does not preserve sparseness when it exports a
    # layer, so on a host with a large UID -- an LDAP or Active Directory account
    # is typically in the millions -- this single layer materialises as hundreds
    # of gigabytes. Building with a domain UID produced a 249 GB image before
    # this flag was added; with it, the same layer is a few megabytes.
    useradd --no-log-init \
            --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home \
            --home-dir "/home/${CONTAINER_USER}" --shell /bin/bash "${CONTAINER_USER}"; \
    # Passwordless sudo. No account password is ever set: the container is a
    # single-user sandbox reachable only through `docker exec`, so a password
    # would add no real security while a hardcoded one would actively hurt.
    install -d -m 0755 /etc/sudoers.d; \
    printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${CONTAINER_USER}" \
      > "/etc/sudoers.d/90-${CONTAINER_USER}"; \
    chmod 0440 "/etc/sudoers.d/90-${CONTAINER_USER}"; \
    # sudo for admin, video/render for the GPU and DRI nodes. Host GIDs for the
    # device groups do not always match the container's (a stock Ubuntu host uses
    # render=104, this base image uses 108). Where that matters, add the host's
    # numeric GID with `group_add:` in docker-compose.yaml; the agx profile is
    # privileged and bind-mounts /dev, so it is unaffected.
    for grp in sudo video render; do \
      if getent group "$grp" >/dev/null; then usermod -aG "$grp" "${CONTAINER_USER}"; fi; \
    done; \
    # `pip install -e` ignores PYTHONUSERBASE and writes into the system tree, so
    # make it group-writable. Setting setgid on the directory itself is O(1); we
    # deliberately do NOT recurse, which would be very slow on an image this size.
    for d in "/usr/local/lib/python3.10/dist-packages" \
             "/usr/lib/python3.10/dist-packages" \
             "/usr/lib/python3/dist-packages" \
             "/usr/local/bin"; do \
      if [ -d "$d" ]; then chgrp "${HOST_GID}" "$d"; chmod 2775 "$d"; fi; \
    done; \
    # The ZED installer, when enabled, leaves /usr/local/zed root-only (700),
    # which hides its CMake config from non-root builds.
    if [ -d /usr/local/zed ]; then chmod -R a+rX /usr/local/zed; fi

# Interactive shell environment. Baked here rather than appended by the
# entrypoint on every start, so it cannot drift or be written twice.
COPY docker/bashrc.d/sem_map.sh /etc/profile.d/sem_map.sh
RUN set -eux; \
    chmod 0644 /etc/profile.d/sem_map.sh; \
    printf '\n# Container defaults (see /etc/profile.d/sem_map.sh)\n. /etc/profile.d/sem_map.sh\n' \
      >> "/home/${CONTAINER_USER}/.bashrc"; \
    touch "/home/${CONTAINER_USER}/.hushlogin"; \
    install -d -o "${HOST_UID}" -g "${HOST_GID}" \
      "/home/${CONTAINER_USER}/.local" \
      "/home/${CONTAINER_USER}/.cache/uv" \
      "/home/${CONTAINER_USER}/.ros"; \
    chown -R "${HOST_UID}:${HOST_GID}" "/home/${CONTAINER_USER}"

# rosdep's cache, populated at build time so it is not rebuilt per container.
RUN su -s /bin/bash - "${CONTAINER_USER}" -c "rosdep update" || \
    echo "WARNING: rosdep update failed at build time; it will retry on first use" >&2

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Everything from here runs as the unprivileged user — including `docker exec`,
# which takes the image's configured user. That is the whole point: `docker exec`
# does NOT run the entrypoint, so a root ENTRYPOINT that drops privileges only
# for its own child leaves every `exec` shell as root, and everything built from
# one of those shells lands on the host owned by root.
USER ${CONTAINER_USER}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
