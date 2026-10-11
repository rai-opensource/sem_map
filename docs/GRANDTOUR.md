<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Demo: SemMap on the GrandTour dataset

This runs SemMap end to end on public data, using the
[GrandTour dataset](https://grand-tour.leggedrobotics.com/) from ETH Zurich.

We use the **SNOW-2** recording (Jungfraujoch, 4 minutes, snow and rock). The
`snow_v1` model that ships with this release was trained on it, so the same data
shows both how to run the system and how to train it.

The demo uses the camera only, so you get segmentation and traversability images
but no 2.5D map. Adding the map is covered [at the end](#adding-the-25d-map).

Run everything inside the container (see the README quick start).

## 1. Get set up

```bash
pip install kleinkram        # the dataset client
klein login                  # free account at https://datasets.leggedrobotics.com
```

Then get the models, if you have not already:

```bash
scripts/artifacts/download.sh
python scripts/sem_map/cli/run_export_all_trt.py --fp16
```

The second command takes a few minutes and produces files that only work on this
machine's GPU.

## 2. Download the recording

```bash
python tools/grandtour/mission_cli.py --list          # see all 49 recordings
python tools/grandtour/mission_cli.py SNOW-2 --download \
    --dest data/public_datasets/grandtour \
    --convert-to-ros2
```

This downloads the camera, LiDAR and odometry files and converts them to the
format ROS 2 reads. The demo only needs the camera; the rest is there if you
want the map later.

Expect 10-30 minutes depending on your connection. Check it worked:

```bash
ros2 bag info -s mcap data/public_datasets/grandtour/SNOW-2/*zed2i_images.mcap
```

You should see `/boxi/zed2i/left/image_raw/compressed`, which is the topic the
demo reads.

## 3. Run it

In one terminal:

```bash
source scripts/setup_env.sh
ros2 launch sem_map sem_map_replay.launch.py robot:=grandtour_real
```

Wait until it reports that it is ready — loading the models takes a few seconds.
Then in a second terminal:

```bash
source scripts/setup_env.sh
ros2 bag play --clock -s mcap data/public_datasets/grandtour/SNOW-2/*zed2i_images.mcap
```

`--clock` is required, because the node follows the recording's clock. Add
`-r 0.5` to play at half speed if your GPU cannot keep up.

The recording is 4 minutes and plays in real time. It is working if these keep
printing:

```bash
ros2 topic hz /sem_map/traversability
ros2 topic hz /sem_map/segmentation
```

In `/sem_map/traversability`, bright means drivable (snow-covered ground) and
dark means not (rock, buildings, sky). Open `rviz2` and add Image displays for
both topics to watch it.

To save the results instead of watching them:

```bash
ros2 launch sem_map sem_map_replay_record.launch.py \
    bag:=data/public_datasets/grandtour/SNOW-2/<file>.mcap robot:=grandtour_real
```

This plays the recording, records the output to `./recordings/`, and stops at
the end.

## 4. Train the model yourself (optional)

The same recording reproduces the `snow_v1` model, which is where the interesting
part is: it is a single layer trained on a handful of labelled frames.

```bash
python scripts/sem_map/cli/pipeline_train_tui.py
```

Point it at the downloaded file and step through: pull out frames, pick a varied
subset, label them automatically with SAM 3 using the words in
`config/seg_layers/coarse_3.yaml` (`snowy floor`, `fence`), train, export.

Picking and training take minutes. Labelling takes longer and depends on how
many frames you use. You will end up with a new model file in `artifacts/` and a
validation score printed for each training round.

If you give it a different name, add that name to `config/models.yaml` and select
it with `seg_layer:` in your robot config.

## Adding the 2.5D map

`config/robots/grandtour_real.yaml` has the map turned off because the demo only
replays the camera. To turn it on, edit that file:

```yaml
pose_topic: "/gt_box/dlio/odom"          # check names with `ros2 bag info`
pointcloud_topic: "/gt_box/livox/undistorted"
pose_frame: "base"
world_frame: "odom"
base_frame: "base"
enable_height_map: true
enable_bev_fusion: true
```

Play all of the recording's files together, so the images, poses and LiDAR share
one clock. Check the frame names with `ros2 run tf2_tools view_frames` — they
have to match what the recording actually contains.

## If something goes wrong

| What you see | Usually means |
|---|---|
| Nothing published, node just waits | Topic name mismatch. Compare `ros2 bag info` with your robot config. |
| `Failed to deserialize TensorRT engine`, or an error naming a missing `.plan` | Models missing, or built for a different GPU or TensorRT version. Re-run `run_export_all_trt.py`. |
| Frames dropped as "too old" | Playing faster than the GPU can process. Use `-r 0.5`. |
| Everything one colour | Wrong model for the scene. Check `seg_layer:`. |
| Nothing happens, no errors | Missing `--clock` on `ros2 bag play`. |
