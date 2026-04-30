# 常见问题排查

本文集中记录 SIM-SLAM / RC2026_SIM 的 Isaac Lab、ROS 2、IsaacSim、Livox 和 FAST-LIO2 常见问题。README 只保留主流程，具体故障排查统一跳转到本文。

## Isaac Lab / Python 环境

### `ModuleNotFoundError: No module named 'isaaclab'`

说明当前 Python 不是 Isaac Lab 环境：

```bash
which python
python -c "import isaaclab; print(isaaclab.__file__)"
```

处理方式：激活 `env_isaaclab`，或使用 Isaac Lab 提供的 `./isaaclab.sh -p` 运行脚本。

### `ModuleNotFoundError: No module named 'Robocon2026'`

通常是没有安装本仓库扩展：

```bash
cd /path/to/SIM-SLAM
python -m pip install -e source/Robocon2026
```

### 任务列表为空或找不到 `Template-*`

检查任务注册：

```bash
python scripts/list_envs.py
python -c "import Robocon2026, gymnasium as gym; print(Robocon2026.__file__)"
```

如果你改了任务命名，需要同步修改 `scripts/list_envs.py` 的过滤条件。

### 资产缺失 / 场景黑屏 / USD 无法打开

确认在仓库根目录运行，并且 `assets/` 已解压：

```bash
pwd
find assets -maxdepth 2 -type f | head
```

Robocon2026 原始 USD 不一定自带灯光；使用仓库 Isaac runner 时会自动补 `/World/ViewerLight`，不要只手动打开地图 USD 后就判断 ROS / LiDAR 链路是否正常。

## ROS 2 / Conda 环境

### ROS 2 命令找不到

当前 shell 没有 source ROS 2：

```bash
source /opt/ros/humble/setup.zsh
ros2 --help
```

建议不要把 `source /opt/ros/humble/setup.*` 写进会影响所有终端的全局配置；用单独 ROS shell 或显式脚本管理。

### `colcon build` 找到 Conda Python 导致失败

退出 Conda 后重新构建：

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
cd /path/to/SIM-SLAM/ros2_ws
rm -rf build install log
colcon build --symlink-install
```

### `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`

这通常不是仓库缺文件，而是 **Python ABI 不匹配**：你在 `env_isaaclab` 等 Conda Python 3.11 shell 里启动了 apt ROS 2 Humble 的 Python 节点，而 Humble 的 `rclpy` C 扩展是给 Ubuntu 22.04 的 Python 3.10 构建的。

处理方式：用 ROS 2 shell + Python 3.10 运行 ROS 节点；策略控制器可使用 README 中的 `.venv-ros2-policy`：

```bash
cd /path/to/SIM-SLAM/ros2_ws
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source .venv-ros2-policy/bin/activate
source install/setup.zsh

ros2 launch deploy_policy go2w_controller.launch.py \
  use_sim_time:=true \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3
```

`go2w_controller.launch.py`、`go2_controller.launch.py`、`armdog_controller.launch.py` 会在启动前检查 `python_executable` 是否能同时导入 `rclpy` 和 `torch`，避免落到难读的底层 C 扩展错误。

### ROS 2 topic 互相看不到

检查 DDS 域和组播：

```bash
echo $ROS_DOMAIN_ID
ros2 multicast receive
# 另开终端：ros2 multicast send
```

多机或多队伍同网段时，建议设置不同 `ROS_DOMAIN_ID`。

## IsaacSim + FAST-LIO2

### IsaacSim 里有 `/points_raw` topic，但 RViz 没点云

先确认 `/points_raw` 是否真的有点：

```bash
ros2 topic echo /points_raw --once
```

如果看到：

```yaml
width: 0
data: []
```

说明只是 topic 存在，RTX LiDAR 没有产生有效 returns。常见原因：

1. Isaac runner 不是最新代码，重启 `scripts/ros2/isaac_fast_lio2_go2w_scene.py`。
2. RTX LiDAR 没有走 render pipeline；headless 也必须 render step。
3. LiDAR prim 位置/朝向不对，没有打到场景几何。
4. 只打开了地图 USD，没有运行仓库 runner 来创建 ROS 2 Bridge 和 RTX LiDAR writer。

### IsaacSim 路线误启动了 `livox_ros_driver2`

IsaacSim 路线不需要真实 Livox driver。正确数据链路是：

```text
IsaacSim /points_raw -> deploy_policy adapter -> /points_fast_lio -> fast_lio_isaac_go2w.launch.py
```

不要使用：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
ros2 launch fast_lio mapping.launch.py config_file:=VLS.yaml
```

这组命令是硬件/通用路线，很容易造成 topic 不匹配。

### FAST-LIO2 收不到 IsaacSim 点云

检查你是否启动了 adapter，并且 FAST-LIO2 是否订阅 `/points_fast_lio`：

```bash
ros2 topic info /points_raw -v
ros2 topic info /points_fast_lio -v
ros2 topic echo /points_fast_lio --once
```

IsaacSim 推荐启动：

```bash
ros2 launch deploy_policy isaac_fast_lio_inputs.launch.py \
  derive_time_if_missing:=true \
  derive_ring_if_missing:=true \
  derive_intensity_if_missing:=true

ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py rviz:=true
```

## 真机 Livox / MID-360 + FAST-LIO2

### `livox_ros_driver2` 报 `bind failed`

通常是 `MID360_config.json` 中的 host IP 不属于本机当前网卡，或端口被占用。

检查本机 IP：

```bash
ip -br addr
```

然后修改：

```text
ros2_ws/src/livox_ros_driver2/config/MID360_config.json
```

确保 `host_net_info` 里的 IP 等于连接雷达的网卡 IP。修改后重启 driver。

### 真机 FAST-LIO2 没有输出地图

按顺序查：

1. `ros2 topic hz /livox/lidar` 是否大于 0；
2. `ros2 topic hz /livox/imu` 是否大于 0；
3. FAST-LIO2 是否使用 `mid360.yaml`，而不是 IsaacSim 的 `isaac_go2w.yaml` 或不匹配的 `VLS.yaml`；
4. 真机启动是否显式 `use_sim_time:=false`；
5. 外参 `extrinsic_T` / `extrinsic_R` 是否合理。

推荐真机启动：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml use_sim_time:=false rviz:=true
```

## FAST-LIO2 输入字段与建图质量

### FAST-LIO2 报 `Failed to find match for field 'time'`

通常是点云缺少逐点时间戳，或 `lidar_type` / `timestamp_unit` 配置与消息字段不匹配。

- Livox 真机：优先使用发布 `livox_ros_driver2/CustomMsg` 的 `msg_*` launch。
- IsaacSim：先用 `isaac_fast_lio_inputs.launch.py` 转成 `/points_fast_lio`。
- 自定义 PointCloud2：确认字段包含 FAST-LIO2 需要的 `time` 或 `t`，并确认单位。

可运行：

```bash
ros2 run deploy_policy check_pointcloud_timing.py --topic /points_fast_lio --dry-run-schema --json
```

### 地图漂移或快速发散

按优先级检查：

1. LiDAR / IMU 时间同步；
2. `lid_topic` / `imu_topic` 是否正确；
3. LiDAR-IMU 外参；
4. IMU 方向和重力方向；
5. 点云时间单位与 `timestamp_unit`；
6. 仿真频率和 ROS 2 `/clock` 是否稳定。
