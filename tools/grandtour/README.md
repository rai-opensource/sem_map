<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# GrandTour dataset

Download and convert recordings from the public
[GrandTour dataset](https://grand-tour.leggedrobotics.com/) (ETH Zurich). It is
distributed through the Kleinkram platform, which needs a free account.

## Setup

```bash
pip install kleinkram
klein login        # opens a browser
```

Converting recordings also needs `rosbags`, `mcap`, `numpy`, `opencv-python`
and `tqdm`.

## Usage

Recordings are named `ETH-1`, `SNOW-2`, `SPX-1` and so on. The full list is in
`grandtour.json`.

```bash
# See what is available (offline)
python mission_cli.py --list

# See the files in one recording
python mission_cli.py ETH-1

# Download it
python mission_cli.py ETH-1 --download --dest ./data

# Convert to the format ROS 2 reads
python mission_cli.py ETH-1 --convert-to-ros2 --dest ./data

# ...and delete the originals to save space
python mission_cli.py ETH-1 --convert-to-ros2 --dest ./data --delete-bags

# If a name is not found, list what the server actually has
python mission_cli.py --debug
```

By default this downloads the odometry, sensor positions, LiDAR and camera. To
get other sensors, edit `DEFAULT_FILE_PATTERNS` at the top of `mission_cli.py`.

`bag_conversion.py` does the conversion and can be run on its own:

```bash
python bag_conversion.py --inbag /path/to/recording_dir
```

Besides the standard message types, it splits the dataset's robot-state messages
into ordinary ROS 2 topics. Set `BATCH_MODE` at the top of the file to convert a
single recording instead of a directory.

## Using it with SemMap

Replay the converted files with the launch files in `launch/`, using
`config/robots/grandtour_real.yaml`. See [`../../docs/GRANDTOUR.md`](../../docs/GRANDTOUR.md)
for a full walkthrough.
