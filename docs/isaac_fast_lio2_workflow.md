# Isaac Sim → ROS 2 → FAST-LIO2 Workflow

This runbook implements the approved v1 scope from `.omx/plans/implementation-isaacsim-ros2-fast-lio2.md`.

## Scope

- Default robot: Go2W.
- Default scene: `assets/Map/robocon2026.usd` or the existing `assets/Simulation/sim.usd` composition.
- Sensor path: Isaac RTX LiDAR + IMU.
- SLAM path: FAST-LIO2 using `/points_fast_lio` + `/imu`.
- Not included in v1: real hardware, new RL training, multi-robot, high-fidelity radar, Docker/CI, major USD remodel.

## Hard Completion Gate

FAST-LIO2 is not considered integrated until `/points_fast_lio` contains a FAST-LIO-compatible per-point `time` or `t` field with the documented `timestamp_unit`. A topic-only smoke test is not enough.

Isaac timestamp metadata is only acceptable if it is exposed or rewritten into the actual PointCloud2 point field accepted by FAST-LIO2 preprocess code.

## Shell Model

Use separate shells as already recommended by the README.

### Isaac Shell

```bash
conda activate env_isaaclab
# or use Isaac Lab's launcher, depending on your installation
```

### ROS Shell

```bash
source /opt/ros/humble/setup.zsh
cd /path/to/RC2026_SIM/ros2_ws
source .venv-ros2-policy/bin/activate 2>/dev/null || true
source install/setup.zsh 2>/dev/null || true
```

Do not blindly mix the Isaac Conda Python environment with the system ROS 2 Python environment. The policy controllers need one Python 3.10 runtime that can import both apt ROS 2 Humble `rclpy` and `torch`; see the README's `2.1. 准备策略控制器 Python 运行时` section.

## Stage 0: Livox Submodule and FAST-LIO Input Contract

```bash
cd /path/to/RC2026_SIM
# FAST_LIO 已 vendored；该命令只初始化 livox_ros_driver2
git submodule update --init --recursive
cd ros2_ws
source /opt/ros/humble/setup.zsh
rosdep install --from-paths src --ignore-src -r -y
colcon list | grep -Ei 'fast|lio|livox|deploy_policy'
```

Before implementing or changing the adapter, inspect the vendored FAST-LIO2 code and record:

- accepted point message types,
- accepted field name: `time` or `t`,
- accepted datatype,
- `timestamp_unit` mapping,
- `lidar_type`, `scan_line`, `scan_rate`, `time_sync_en`,
- FAST-LIO2 log strings for missing time/frame/sync errors.

## Stage 1: Build ROS Package

```bash
cd /path/to/RC2026_SIM/ros2_ws
source /opt/ros/humble/setup.zsh
colcon build --symlink-install --packages-select deploy_policy
source install/setup.zsh
ros2 pkg executables deploy_policy
```

Expected new executables include:

- `check_pointcloud_timing.py`
- `isaac_pointcloud_time_adapter.py`

## Stage 2: Verify Installed Isaac Node Schema

Run from the Isaac shell:

```bash
cd /path/to/RC2026_SIM
python scripts/ros2/check_isaac_ros2_node_schema.py
```

If this fails, pin the implementation to your installed Isaac Sim version before editing the runner. The repository README targets Isaac Sim 4.5 / 5.x, while online `latest` docs may describe newer node names.

## Stage 3: Start Isaac Runner

Run from the Isaac shell:

```bash
cd /path/to/RC2026_SIM
python scripts/ros2/isaac_fast_lio2_go2w_scene.py \
  --scene assets/Map/robocon2026.usd \
  --robot assets/Go2W/go2w_ros2.usd \
  --scan-rate 10.0
```

The runner sets RTX LiDAR `scanRateBaseHz` from `--scan-rate`, creates an RTX
LiDAR render product, and attaches the official `RtxLidarROS2PublishPointCloud`
writer before publishing `/points_raw`. It also publishes the Isaac transform
tree with `isaac:nameOverride` aliases for `base_link`, `imu_link`, and
`lidar_link`; run the schema checker first because node names are
installed-version dependent.

Then check in the ROS shell:

```bash
ros2 topic list
ros2 topic echo /clock --once
ros2 topic echo /joint_states --once
ros2 topic echo /imu --once
ros2 topic echo /points_raw --once
ros2 topic hz /imu
ros2 topic hz /points_raw
```

## Stage 4: Start PointCloud2 Timing Adapter

Start strict mode first. This republishes only if the incoming cloud already satisfies FAST-LIO timing requirements.

```bash
ros2 launch deploy_policy isaac_fast_lio_inputs.launch.py \
  input_topic:=/points_raw \
  output_topic:=/points_fast_lio \
  timestamp_unit:=0 \
  lidar_type:=2 \
  scan_rate_hz:=10.0 \
  derive_time_if_missing:=false
```

If Stage 0 proves scan ordering and scan rate are valid, derived timing can be enabled explicitly:

```bash
ros2 launch deploy_policy isaac_fast_lio_inputs.launch.py derive_time_if_missing:=true \
  derive_ring_if_missing:=true
```

Do not enable derived timing unless the scan-order assumption has been validated for the selected RTX LiDAR mode.

## Stage 5: Run Timing Gate

```bash
ros2 run deploy_policy check_pointcloud_timing.py \
  --topic /points_fast_lio \
  --clock-topic /clock \
  --scan-rate 10.0 \
  --timestamp-unit 0 \
  --lidar-type 2 \
  --max-clock-skew-sec 0.1
```

Pass means:

- `time` or `t` exists,
- for `lidar_type: 2`, `x/y/z/intensity` are float32 and `ring` exists with `uint16` datatype,
- datatype matches the FAST-LIO2 preprocess contract,
- values are finite and non-zero,
- span is plausible for the scan period,
- header stamp aligns with `/clock`.

Failure blocks FAST-LIO2 completion.

## Stage 6: Control Go2W

```bash
ros2 launch deploy_policy go2w_controller.launch.py \
  use_sim_time:=true \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Verify:

```bash
ros2 topic echo /imu --once
ros2 topic hz /points_fast_lio
```

IMU and cloud values should change during motion.

## Stage 7: Launch FAST-LIO2

```bash
ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py
```

Check outputs. Topic names may vary by the vendored FAST-LIO2 fork:

```bash
ros2 topic list | grep -E 'lio|cloud|odom|path|map|Odometry'
ros2 topic echo /Odometry --once 2>/dev/null || true
```

FAST-LIO2 logs must not contain missing-time, frame, sync, empty-cloud, or preprocess errors.
The launch file also publishes static TF aliases that connect this FAST-LIO
fork's hard-coded `camera_init`/`body` frames to the canonical
`map`/`odom`/`base_link` and sensor frames:

```text
camera_init -> map -> odom
camera_init -> body -> base_link -> {imu_link,lidar_link}
```

## Stage 8: RViz Proof

```bash
rviz2 -d /path/to/RC2026_SIM/ros2_ws/src/deploy_policy/rviz/fast_lio_isaac_go2w.rviz
```

Expected view:

- TF tree with FAST-LIO fork frames. This checked-out fork publishes `camera_init` → `body`, so the provided RViz config uses `camera_init` as the fixed frame.
- live point cloud / registered cloud,
- odometry/path changing during robot motion.

## Troubleshooting

### FAST-LIO2 says `Failed to find match for field 'time'`

The PointCloud2 field contract is not satisfied. Run:

```bash
ros2 run deploy_policy check_pointcloud_timing.py --topic /points_fast_lio --dry-run-schema --json
```

Then fix the adapter or FAST-LIO2 config. Do not claim completion.

### Timing span is implausible

Check `timestamp_unit` and `scan_rate`. For FAST-LIO-style PointCloud2, common units are:

- `0`: seconds
- `1`: milliseconds
- `2`: microseconds
- `3`: nanoseconds

### Isaac node type missing

Run `scripts/ros2/check_isaac_ros2_node_schema.py` and pin node type names to the installed Isaac version. Online `latest` docs can differ from Isaac Sim 4.5/5.x.

### Go2 launch policy path fails

Use Go2W for v1. If switching to Go2, pass the existing flat policy explicitly:

```bash
ros2 launch deploy_policy go2_controller.launch.py \
  use_sim_time:=true \
  policy_path:=/path/to/RC2026_SIM/ros2_ws/src/deploy_policy/policy/go2/flat/exported/policy.pt \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3
```

## Evidence Bundle

Collect these before declaring the workflow complete:

```bash
git submodule status  # should list livox_ros_driver2; FAST_LIO is vendored
colcon list | grep -Ei 'fast|lio|livox|deploy_policy'
ros2 topic list
ros2 topic hz /imu
ros2 topic hz /points_raw
ros2 topic hz /points_fast_lio
ros2 run deploy_policy check_pointcloud_timing.py --topic /points_fast_lio ...
ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py
rviz2 -d ros2_ws/src/deploy_policy/rviz/fast_lio_isaac_go2w.rviz
```
