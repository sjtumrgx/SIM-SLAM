# FAST-LIO2 Input Contract for Isaac Go2W Simulation

This note records the Stage 0 contract inspection for the checked-out submodule:

```text
ros2_ws/src/FAST_LIO -> cdb8cf568110c8f4cae96e772fd9da453fe13034 (ROS2)
ros2_ws/src/FAST_LIO/include/ikd-Tree -> e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4
ros2_ws/src/livox_ros_driver2 -> 378d1c7c9d33ec44d0683aef885b6ed0cce9612c
```

## Message Types

`ros2_ws/src/FAST_LIO/src/laserMapping.cpp` subscribes to:

- `livox_ros_driver2::msg::CustomMsg` when `preprocess.lidar_type == 1` (`AVIA`).
- `sensor_msgs::msg::PointCloud2` for other lidar types.

For v1, the repository-owned Isaac path uses PointCloud2 with `preprocess.lidar_type: 2` (`VELO16`) unless Stage 0 is re-run and a safer type is proven.

## PointCloud2 Field Contract

From `ros2_ws/src/FAST_LIO/src/preprocess.h`:

- Velodyne-style PointCloud2 uses fields:
  - `x`, `y`, `z`: float
  - `intensity`: float
  - `time`: float
  - `ring`: uint16
- Ouster-style PointCloud2 uses field:
  - `t`: uint32
  - plus `ring` uint8, `reflectivity` uint16, `ambient` uint16, and `range` uint32

From `ros2_ws/src/FAST_LIO/src/preprocess.cpp`:

- `timestamp_unit` scales PointCloud2 point time into milliseconds internally:
  - `0` seconds -> `time * 1e3`
  - `1` milliseconds -> `time * 1`
  - `2` microseconds -> `time * 1e-3`
  - `3` nanoseconds -> `time * 1e-6`
- Velodyne handler reads `pl_orig.points[i].time`.
- Ouster handler reads `pl_orig.points[i].t`.
- Missing or zero point time causes FAST-LIO2 to fall back to scan-angle timing in some handlers; this is not accepted as v1 completion unless explicitly validated.

## v1 Adapter Contract

`/points_fast_lio` must provide at least:

- `x`, `y`, `z` float fields
- `intensity` float field
- `time` float field for `lidar_type: 2`
- `ring` uint16 field if FAST-LIO Velodyne handler requires scan-line routing

The current checker validates `time`/`t` timing and, when invoked with `--lidar-type 2`, requires `x/y/z/intensity` float32 plus `ring` uint16. The adapter uses the same strict contract before publishing. It can derive `intensity`, `time`, and `ring` only when `derive_intensity_if_missing:=true`, `derive_time_if_missing:=true`, and `derive_ring_if_missing:=true`. Route A enables those derivations by default for the current Isaac `x/y/z` simulation path, but this is a first-pass simulation assumption: passing the checker proves the PointCloud2 field/timing contract, not final scan-order semantics or map quality.

## Config Fields

Repo-owned config: `ros2_ws/src/deploy_policy/config/fast_lio/isaac_go2w.yaml`

Critical fields:

- `common.lid_topic: /points_fast_lio`
- `common.imu_topic: /imu`
- `common.time_sync_en: false`
- `preprocess.lidar_type: 2`
- `preprocess.scan_line: 32`
- `preprocess.scan_rate: 10`
- `preprocess.timestamp_unit: 0`
- `mapping.extrinsic_est_en: false`
- `mapping.extrinsic_T/R`: fixed LiDAR-in-IMU guess to verify after final sensor placement

## Verification Gate

The Isaac runner must set RTX LiDAR `scanRateBaseHz` to the same full-scan rate
used by the adapter/checker/FAST-LIO config and publish `/points_raw` through an
RTX LiDAR render product attached to `RtxLidarROS2PublishPointCloud`; an
unconnected `ROS2RtxLidarHelper` node is not sufficient. The ROS launch wrapper
publishes static aliases for this FAST-LIO fork's hard-coded
`camera_init`/`body` frames so RViz can show the canonical
`map`/`odom`/`base_link`/sensor frame tree.

Run:

```bash
ros2 run deploy_policy check_pointcloud_timing.py \
  --topic /points_fast_lio \
  --clock-topic /clock \
  --scan-rate 10.0 \
  --timestamp-unit 0 \
  --lidar-type 2 \
  --max-clock-skew-sec 0.1
```

This gate must pass before FAST-LIO2 output is treated as valid.
