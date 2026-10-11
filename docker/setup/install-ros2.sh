#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Script to install and set up ROS2 for Docker containers and direct hardware installations.
# Handles ROS2 package installation, rosdep configuration, and system setup.
#
set -euo pipefail

# Verify root privileges before proceeding
if [[ "$EUID" -ne 0 ]] ; then
  echo "ERROR: ros install must be run as root, please run:"
  echo "  sudo $0"
  exit 1
fi

# Install curl if not present
apt-get update && apt-get install -y curl

# Clean up any existing ROS2 files
rm -f /etc/apt/sources.list.d/ros2.list

# Source the os-release file to get the Ubuntu version
source "/etc/os-release"

# Get the latest ros-apt-source version
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')

# Download and install ros-apt-source package
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
apt-get install -y /tmp/ros2-apt-source.deb

# Refresh package lists
apt-get update

# Determine ROS2 distribution based on Ubuntu version
if [[ "$VERSION_ID" == "24.04" ]]; then
  readonly TARGET_ROS_DISTRO="jazzy"
elif [[ "$VERSION_ID" == "22.04" ]]; then
  readonly TARGET_ROS_DISTRO="humble"
else
  echo "ERROR: Unsupported Ubuntu version: $VERSION_ID"
  echo "This script supports Ubuntu 22.04 (Humble) and 24.04 (Jazzy)"
  exit 1
fi

echo "Installing ROS2 $TARGET_ROS_DISTRO for Ubuntu $VERSION_ID ($UBUNTU_CODENAME)"

# Install ROS2 desktop and additional tools
apt-get install -y --no-install-recommends \
  python3-colcon-common-extensions \
  python3-colcon-clean \
  python3-rosdep \
  libpcap-dev \
  ros-$TARGET_ROS_DISTRO-desktop \
  ros-$TARGET_ROS_DISTRO-topic-tools \
  ros-$TARGET_ROS_DISTRO-xacro \
  ros-$TARGET_ROS_DISTRO-rosbag2-storage-mcap \
  ros-$TARGET_ROS_DISTRO-foxglove-bridge \
  ros-$TARGET_ROS_DISTRO-robot-state-publisher \
  ros-$TARGET_ROS_DISTRO-joint-state-publisher \
  ros-$TARGET_ROS_DISTRO-rmw-cyclonedds-cpp \
  ros-$TARGET_ROS_DISTRO-rmw-zenoh-cpp \
  ros-$TARGET_ROS_DISTRO-grid-map-msgs \
  ros-$TARGET_ROS_DISTRO-sophus \

# Load ROS2 environment variables
set +u
source "/opt/ros/$TARGET_ROS_DISTRO/setup.bash"
set -u

# Set up rosdep package manager
rm -rf /etc/ros/rosdep/sources.list.d  # Clean up any previous installations
rosdep init
rosdep update
