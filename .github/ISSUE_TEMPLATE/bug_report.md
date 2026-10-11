---
name: Bug Report
about: Report a bug or issue with the project
title: '[BUG] '
labels: bug
assignees: ''
---
<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Run '...'
2. Play bag / launch '...'
3. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened.

## Environment

Please paste the output of `./scripts/verify.sh`, or fill in:

- Platform: [x86_64 + discrete GPU, or Jetson AGX Orin]
- OS / ROS distro: [e.g., Ubuntu 22.04, ROS 2 Humble]
- GPU and driver: [e.g., RTX 4090, driver 560]
- CUDA / TensorRT: [e.g., 12.8 / 10.8.0 on x86_64, or 12.6 / 10.7.0 on a Jetson]
- Running in the provided container? [yes / no — if no, how was it installed]
- Project version or commit: [e.g., v0.1.0 or commit hash]

## Configuration

- Robot config used: [e.g., `robot:=grandtour_real`, or your own]
- Seg layer / model: [e.g., `dinov3s_sem_seg_snow_v1`]
- Backend: [TensorRT / ONNX Runtime / PyTorch]
- Upsampler: [bilinear / jafar_partial / jafar_full]
- Model artifacts: [released via `scripts/artifacts/download.sh`, or self-exported]

## Data

- Input source: [live camera / rosbag / video file]
- If a public dataset, which one: [e.g., GrandTour SNOW-2]

## Logs or Error Messages

```text
Paste the node output, stack trace, or launch log here
```

## Additional Context

Screenshots, RViz captures, config files, or anything else that helps.
