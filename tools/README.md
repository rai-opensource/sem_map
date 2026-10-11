<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# tools

Standalone scripts, run directly with `python tools/<...>.py`, not through ROS.
Every one takes `--help`.

## `mcap/` — working with recordings

| Script | What it does |
| --- | --- |
| `generate_rgb_from_mcap.py` | Save camera frames from a recording as PNGs. Also used by the training pipeline. |
| `filter_mcap.py` | Remove old SemMap results from a recording, so you can run it again cleanly. |
| `add_tf_to_mcap.py` | Add a fixed sensor position to a recording. |
| `remove_tf_from_mcap.py` | Remove everything involving one frame from a recording. |

```bash
python tools/mcap/generate_rgb_from_mcap.py \
    --bag recordings/my_bag.mcap \
    --topic /zed/zed_node/rgb/color/rect/image \
    --out /tmp/frames
```

Needs `mcap`, `mcap-ros2-support`, `opencv-python` and `numpy`. The two scripts
that edit sensor positions also need ROS 2 sourced.

## `grandtour/` — the demo dataset

Download and convert recordings from the public GrandTour dataset. See
[`grandtour/README.md`](grandtour/README.md).
