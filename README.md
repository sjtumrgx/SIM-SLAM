# SIM-SLAM / RC2026_SIM

> ROBOCON 2026 Isaac Lab 仿真、强化学习训练、ROS 2 策略部署与 FAST-LIO2 SLAM 集成示例。
>
> 本仓库当前主体代码来自 `RC2026_SIM`，以 Isaac Lab 外部扩展的形式组织；ROS 2 工作空间用于在 Isaac Sim/Isaac Lab 场景中部署训练策略、接收仿真 IMU / 关节状态 / 点云数据，并接入键盘控制与 FAST-LIO2 建图流程。

![main scene](./README.assets/image-20251109012317799.png)

## 目录

- [项目能做什么](#项目能做什么)
- [仓库结构](#仓库结构)
- [推荐环境矩阵](#推荐环境矩阵)
- [快速开始：Isaac Lab 仿真与训练](#快速开始isaac-lab-仿真与训练)
- [资源文件 assets 的放置方式](#资源文件-assets-的放置方式)
- [常用任务与命令](#常用任务与命令)
- [ROS 2 Humble 安装与 ros2_ws 构建](#ros-2-humble-安装与-ros2_ws-构建)
- [FAST-LIO2 路线总览](#fast-lio2-路线总览)
- [路线 A：IsaacSim + FAST-LIO2](#路线-aisaacsim--fast-lio2)
- [路线 B：真机使用 FAST-LIO2](#路线-b真机使用-fast-lio2)
- [常见问题排查](docs/troubleshooting.md)
- [参考资料](#参考资料)

## 项目能做什么

本项目围绕 ROBOCON 2026 机器人仿真与 SLAM 部署流程，包含四类内容：

1. **Isaac Lab 外部扩展**
   - 扩展路径：`source/Robocon2026/`
   - Python 包名：`Robocon2026`
   - 基于 Isaac Lab Manager-Based RL 环境注册任务。
2. **仿真资产与地图**
   - 代码中默认从仓库根目录的 `assets/` 读取 USD、HDR、材质、PCD 等文件。
   - `assets/Simulation/sim.usd` 可作为 ROS 2 联调场景入口。
3. **强化学习训练与回放**
   - `scripts/rsl_rl/train.py` / `scripts/rsl_rl/play.py`
   - 内置本地 `rsl_rl` 代码，方便与 Isaac Lab 任务配置配套使用。
4. **ROS 2 策略部署与 SLAM**
   - `ros2_ws/src/deploy_policy`：读取 TorchScript 策略，订阅仿真状态，发布关节命令。
   - `ros2_ws/src/FAST_LIO`：仓库内 vendored 的 FAST-LIO2 ROS 2 fork 源码目录；ROS 包名仍叫 `fast_lio`。
   - `ros2_ws/src/livox_ros_driver2`：作为子模块接入 Livox 驱动。

项目中已有的主要任务包括：

| 类别 | 任务 ID | 说明 |
| --- | --- | --- |
| SO-Arm101 夹取 | `Template-Arm-Control-Lift-v0` / `Template-Arm-Control-Lift-Play-v0` | 机械臂夹取并抬升物体 |
| SO-Arm101 视觉蒸馏/微调 | `Template-Arm-Control-Lift-Distillation-Vision-v0` 等 | 视觉输入策略相关实验 |
| 武器组装参考任务 | `Template-Assemble-Weapon-Dual-v0` / `Template-Assemble-Weapon-Single-v0` | 当前 README 中作为参考，建议先用 dummy agent 验证 |
| Unitree Go2 | `Template-Basic-Control-Flat-GO2-v0` / `Rough-GO2-v0` | 四足基础步态控制 |
| Unitree Go2W | `Template-Basic-Control-Flat-GO2W-v0` / `Rough-GO2W-v0` | 轮足基础控制 |
| ArmDog | `Template-Basic-Control-Flat-ArmDog-v0` / `Rough-ArmDog-v0` | Go2W + SO-Arm101 组合机器人 |
| ROBOCON 2026 场景 | `Template-Robocon2026-v0` | 面向整图仿真的综合环境 |

## 仓库结构

```text
.
├── README.md
├── README.assets/                      # README 图片
├── assets/                             # 大体积仿真资产；默认不建议提交 Git
│   ├── ArmDog/ Go2/ Go2W/ SO101/ ...
│   ├── Map/
│   ├── Materials/
│   ├── PCD/map.pcd
│   └── Simulation/sim.usd
├── scripts/
│   ├── list_envs.py                    # 列出已注册 Isaac Lab 任务
│   ├── zero_agent.py                   # 零动作 agent，验证环境配置
│   ├── random_agent.py                 # 随机动作 agent，验证环境配置
│   ├── rsl_rl/train.py                 # 训练入口
│   ├── rsl_rl/play.py                  # 回放入口
│   └── test_core/                      # 场景 / 核心仿真测试脚本
├── source/Robocon2026/
│   ├── setup.py
│   ├── config/extension.toml
│   └── Robocon2026/
│       ├── robots/                     # Go2、Go2W、ArmDog、SO101、Jetbot 等资产配置
│       ├── map/                        # KFS / terrain 场景配置
│       └── tasks/manager_based/        # Isaac Lab ManagerBasedRLEnv 任务
└── ros2_ws/
    └── src/
        ├── deploy_policy/                  # ROS 2 策略部署包
        ├── FAST_LIO/                       # vendored FAST-LIO2 ROS 2 fork；ROS 包名 fast_lio
        └── livox_ros_driver2/              # Livox ROS Driver 2 子模块
```

> 注意：`assets/` 通常超过 1GB，且包含 USD / PCD / HDR / 材质等二进制资产。建议通过网盘或 Git LFS/对象存储分发，不建议直接放入普通 Git 仓库。

## 推荐环境矩阵

| 组件 | 推荐版本 | 说明 |
| --- | --- | --- |
| OS | Ubuntu 22.04 LTS | ROS 2 Humble、Livox ROS Driver 2、FAST-LIO2 ROS 2 fork 的社区组合最常见 |
| GPU | NVIDIA RTX 系列，显存建议 8GB+ | Isaac Sim / Isaac Lab 对驱动与显卡要求较高；训练建议更高显存 |
| NVIDIA Driver | 按 Isaac Sim 官方版本矩阵安装 | 不要只按 PyTorch 版本倒推驱动；优先参考 Isaac Sim 文档 |
| Isaac Sim / Isaac Lab | Isaac Sim 4.5 / 5.x + Isaac Lab 对应版本 | 本仓库 `setup.py` 标注 Python `>=3.10`，classifier 包含 Isaac Sim 4.5 / 5.0 |
| Python | 3.10 或 3.11 | 取决于你安装的 Isaac Lab / Isaac Sim 版本；不要混用多个 Python 解释器 |
| ROS 2 | Humble Hawksbill | Humble 官方 apt 包面向 Ubuntu 22.04 |
| FAST-LIO2 | ROS 2 fork | 已 vendored 到 `ros2_ws/src/FAST_LIO`，保留 GPL-2.0 协议文件 |
| Livox 驱动 | Livox ROS Driver 2 | Humble 下在 `ros2_ws` 用 `colcon build --packages-select livox_ros_driver2 --symlink-install` |

**建议原则：**

- Isaac Lab 训练环境和 ROS 2 部署环境可以放在同一台机器，但最好分成两个 shell 使用：
  - Isaac shell：`conda activate env_isaaclab`
  - ROS shell：`source /opt/ros/humble/setup.zsh && source ros2_ws/install/setup.zsh`
- 不建议在同一个 shell 里同时激活 Isaac 的 Conda 环境和系统 apt 安装的 ROS 2，除非你非常清楚 `PATH` / `PYTHONPATH` / `LD_LIBRARY_PATH` 的优先级。

## 快速开始：Isaac Lab 仿真与训练

以下流程假设你使用 Ubuntu 22.04，并且已经安装好 NVIDIA 驱动。

### 1. 获取仓库和 Livox 子模块

```bash
git clone --recursive https://github.com/sjtumrgx/SIM-SLAM.git
cd SIM-SLAM

# 如果 clone 时没有加 --recursive，后续补执行：
git submodule update --init --recursive
```

### 2. 安装 Isaac Lab

Isaac Lab 安装方式变化较快，建议优先跟随官方文档。社区中最稳妥的做法是：**让 Isaac Lab 拥有自己的 Python/Conda 环境，本仓库作为外部扩展以 editable 模式安装进去**。

示例流程：

```bash
# 仅示例：具体 Python 版本以你使用的 Isaac Lab 文档为准
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip

# 安装 Isaac Lab：请按官方文档选择源码安装或 pip 安装
# 源码方式常见流程示例：
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install rsl_rl

# 验证 Isaac Lab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

如果你使用的是 Isaac Sim App / Conda 方式，也可以用 Isaac Lab 提供的 `isaaclab.sh -p` 调用正确 Python。核心目标是：后续运行本仓库脚本时，`python -c "import isaaclab"` 能成功。

### 3. 安装本仓库 Isaac Lab 扩展

```bash
cd /path/to/SIM-SLAM
conda activate env_isaaclab

# 安装 Robocon2026 扩展
python -m pip install -e source/Robocon2026

# 验证任务注册
python scripts/list_envs.py
```

成功后，你应能看到类似下面的任务列表：

```text
Template-Arm-Control-Lift-v0
Template-Arm-Control-Lift-Play-v0
Template-Basic-Control-Flat-GO2W-v0
Template-Basic-Control-Rough-GO2W-v0
Template-Basic-Control-Flat-GO2-v0
Template-Basic-Control-Rough-GO2-v0
Template-Basic-Control-Flat-ArmDog-v0
Template-Basic-Control-Rough-ArmDog-v0
...
```

### 4. VS Code 代码提示

仓库继承了 Isaac Lab 模板中的 VS Code 任务：

1. 打开 VS Code。
2. `Ctrl + Shift + P`。
3. 选择 `Tasks: Run Task`。
4. 选择 `setup_python_env`。
5. 按提示输入 Isaac Sim / Isaac Lab 安装路径。

成功后会生成 `.vscode/.python.env`，用于补全 Omniverse / Isaac Sim 扩展模块路径。

## 资源文件 assets 的放置方式

当前代码多处以相对路径读取资产，例如：

- `assets/Go2/go2.usd`
- `assets/Go2W/go2w.usd`
- `assets/ArmDog/armdog.usd`
- `assets/SO101/so101.usd`
- `assets/Map/robocon2026.usd`
- `assets/Simulation/sim.usd`
- `assets/PCD/map.pcd`

请将资产解压到仓库根目录的 `assets/` 下：

```text
SIM-SLAM/
├── assets/
│   ├── Go2/
│   ├── Go2W/
│   ├── ArmDog/
│   ├── SO101/
│   ├── Map/
│   └── Simulation/
└── source/
```

资产下载：

```text
百度网盘：
链接: https://pan.baidu.com/s/192W4uDKmPrswztege8Sm-A?pwd=5iff
提取码: 5iff
```

如果出现 `Could not open asset @assets/...@`、模型不可见、材质缺失等问题，优先检查：

```bash
pwd                         # 必须在仓库根目录运行脚本
ls assets/Go2/go2.usd
ls assets/Simulation/sim.usd
```

## 常用任务与命令

### 列出任务

```bash
conda activate env_isaaclab
cd /path/to/SIM-SLAM
python scripts/list_envs.py
```

### 用 dummy agent 验证环境

推荐先用零动作或随机动作验证资产、任务注册、仿真启动是否正常：

```bash
python scripts/zero_agent.py --task Template-Basic-Control-Flat-GO2W-Play-v0
python scripts/random_agent.py --task Template-Basic-Control-Flat-GO2W-Play-v0
```

### 训练策略

```bash
python scripts/rsl_rl/train.py --task Template-Basic-Control-Flat-GO2W-v0
```

常用建议：

```bash
# 无 GUI/headless 训练，适合服务器
python scripts/rsl_rl/train.py --task Template-Basic-Control-Flat-GO2W-v0 --headless

# 指定随机种子，便于复现实验
python scripts/rsl_rl/train.py --task Template-Basic-Control-Flat-GO2W-v0 --seed 42
```

### 回放策略

```bash
python scripts/rsl_rl/play.py --task Template-Basic-Control-Flat-GO2W-Play-v0 \
  --checkpoint /path/to/model.pt
```

### 测试完整场景

```bash
python scripts/test_core/setup_scene.py
```

![basic control](./README.assets/8359e1495b253a6bd45592a9fa76e827.jpg)

![arm control](./README.assets/3c77169a4373174927b6df8be7308698.jpg)

## ROS 2 Humble 安装与 ros2_ws 构建

ROS 2 部分用于策略部署、键盘控制和 SLAM。它不是训练 Isaac Lab 任务的必要条件。

### 1. 推荐安装方式：apt 安装到 `/opt/ros/humble`

ROS 2 Humble 官方二进制包面向 Ubuntu 22.04。推荐使用 apt 安装，因为依赖解析、colcon、rosdep 与大量 C++ 包兼容性最好。

```bash
# 1) locale
locale
sudo apt update
sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2) 添加 universe 与 ROS 2 apt source
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe -y
sudo apt update

export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# 3) 安装 ROS 2 Humble + 开发工具
sudo apt update
sudo apt upgrade -y
sudo apt install ros-humble-desktop ros-dev-tools \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  ros-humble-teleop-twist-keyboard -y

# 4) rosdep 初始化
sudo rosdep init 2>/dev/null || true
rosdep update

# 5) 每次使用 ROS 2 的 shell 里 source，不建议无脑写入全局 .zshrc/.bashrc
source /opt/ros/humble/setup.zsh   # zsh
# source /opt/ros/humble/setup.bash  # bash

# 6) 验证
ros2 run demo_nodes_cpp talker
# 另开终端：source /opt/ros/humble/setup.zsh && ros2 run demo_nodes_py listener
```

> 如果你使用 bash，把上面所有 `setup.zsh` 换成 `setup.bash`。

### 2. 构建本仓库 ros2_ws

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh

# 初始化 Livox 子模块；FAST_LIO 已随主仓库 vendored
cd /path/to/SIM-SLAM
git submodule update --init --recursive

cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select deploy_policy
source install/setup.zsh
```

### 2.1. 准备策略控制器 Python 运行时（`rclpy` + `torch`）

`deploy_policy` 的策略控制器同时需要：

- apt ROS 2 Humble 的 `rclpy`（Ubuntu 22.04 上对应 Python 3.10 C 扩展）；
- PyTorch，用于加载 `policy.pt` TorchScript 策略。

不要直接在 `env_isaaclab` 里启动这些 ROS 2 控制器；Python 3.11 的 Isaac/Conda 环境会加载不了 Humble 的 `rclpy` C 扩展。推荐用 `uv` 为 ROS 侧单独建一个 Python 3.10 venv，并通过 `--system-site-packages` 复用 `/opt/ros/humble` 的 Python 包：

```bash
cd /path/to/SIM-SLAM/ros2_ws
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh

# 本机已验证 uv venv 支持 --system-site-packages；这里显式使用系统 Python 3.10，
# 避免 uv 下载/选择与 apt ROS 2 Humble 不一致的 Python。
uv --version
/usr/bin/python3 - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version
PY

uv venv --python /usr/bin/python3 --system-site-packages .venv-ros2-policy

# 按你的 CUDA/CPU 环境选择 PyTorch backend；下面仅示例 CUDA 12.8。
# 常见可选值包括 cpu、cu128、cu126 等；用 `uv pip install --help` 查看本机 uv 支持项。
uv pip install --python .venv-ros2-policy/bin/python --torch-backend cu128 torch

source .venv-ros2-policy/bin/activate
source install/setup.zsh
python - <<'PY'
import rclpy, torch
print("rclpy:", rclpy.__file__)
print("torch:", torch.__version__)
PY
```

如果你的 `uv` 版本不支持 `--torch-backend`，先升级 `uv`；临时兜底可使用同一个 `.venv-ros2-policy/bin/python` 执行 `python -m pip install torch --index-url <对应 PyTorch wheel 源>`，但仍需保持 Python 3.10 + `--system-site-packages`。

### 3. 启动策略控制器

当前 `deploy_policy` 包中包含 Go2W、Go2、ArmDog 控制脚本与 launch 文件。常用入口：

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh
source .venv-ros2-policy/bin/activate  # 如果按 2.1 创建了 ROS 策略运行时
source install/setup.zsh

# Go2W，默认加载 policy/go2w/rough/exported/policy.pt
ros2 launch deploy_policy go2w_controller.launch.py \
  use_sim_time:=true \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3

# Go2W + FAST-LIO2 联调时推荐先用保守低速安全参数。
# 控制器会在未收到近期 cmd_vel 时保持默认姿态和零轮速，避免启动瞬间动作突变。
ros2 launch deploy_policy go2w_controller.launch.py \
  use_sim_time:=true \
  max_cmd_vel_x:=0.05 \
  max_cmd_vel_y:=0.03 \
  max_cmd_vel_yaw:=0.10 \
  hold_without_cmd_vel:=true \
  cmd_vel_timeout_sec:=0.75 \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3

# 指定策略路径
ros2 launch deploy_policy go2w_controller.launch.py \
  use_sim_time:=true \
  policy_path:=/absolute/path/to/policy.pt \
  python_executable:=$PWD/.venv-ros2-policy/bin/python3

# 键盘控制 cmd_vel
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

仿真侧推荐使用仓库脚本启动 Isaac Sim，而不是只手动打开某个 USD。只打开 `assets/Map/robocon2026.usd` 通常只是加载场地；要自动创建 ROS 2 Bridge、加载 Go2W、创建关节/IMU/LiDAR 发布订阅链路，使用：

```bash
# 终端 A：Isaac shell，不要和 ROS shell 混用
cd /path/to/SIM-SLAM
conda activate env_isaaclab

# Isaac Sim pip/conda 安装使用内置 ROS 2 Humble bridge 时，需要在启动 Python 前设置：
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python - <<'PY'
from pathlib import Path
import isaacsim
print(Path(isaacsim.__file__).resolve().parent / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib")
PY
)"

# 可选但推荐：先检查当前 Isaac Sim 安装中 ROS2/RTX LiDAR 节点 schema
# 该检查会确认 Isaac 5.1 风格的 IsaacReadIMU -> ROS2PublishImu 属性链路，
# 避免把旧示例里的 PublishImu.inputs:targetPrim 写进 ActionGraph。
python scripts/ros2/check_isaac_ros2_node_schema.py

# 打开 Isaac Sim，加载 Robocon2026 场地 + Go2W，并创建 ROS2 Bridge/RTX LiDAR 发布器
python scripts/ros2/isaac_fast_lio2_go2w_scene.py \
  --scene assets/Map/robocon2026.usd \
  --robot assets/Go2W/go2w_ros2.usd \
  --scan-rate 10.0

# 说明：Robocon2026 原始 USD 不自带灯光；该 runner 会在无灯光时自动补一个
# /World/ViewerLight DomeLight，并把初始视角对准场地/机器人，避免 RaytracedLighting 黑屏。
# 如需调试原始 USD，可加 --no-viewer-setup；也可用 --viewer-light-intensity、
# --camera-eye、--camera-target 调整亮度和初始视角。

# CI/远程调试时可只跑 1 步，验证 ActionGraph 能创建且不会立即 schema 报错：
python scripts/ros2/isaac_fast_lio2_go2w_scene.py --headless --max-steps 1
```

如果你已经在 Isaac shell 中 `source /opt/ros/humble/setup.zsh` 并能接受 Isaac/ROS 环境混用，也可以走外部 ROS 2 路径；否则使用上面的 Isaac 内置 Humble bridge 变量。`LD_LIBRARY_PATH` 必须在 `python ...` 启动前设置，不能等 `SimulationApp` 已经启动后再补。

`assets/Go2W/go2w_ros2.usd` 本身已经带有 Go2W 的 ROS IMU / JointState ActionGraph，默认 IMU 传感器 prim 是 `/World/Go2W/base/trunk/imu_link/Imu_Sensor`，关节控制的 articulation root 是 `/World/Go2W/base`。为避免同一个 topic 有两套 publisher/subscriber，仓库 runner 会先禁用 USD 里引用进来的 `ROS_IMU` / `ROS_Joint_States` graph，再显式按 Isaac Sim 5.1 schema 创建一条运行时 `/ActionGraph`：用 `IsaacReadIMU` 读取 `IsaacImuSensor`，再把 `orientation` / `angularVelocity` / `linearAcceleration` 接到 `ROS2PublishImu`。因此不要再使用旧写法 `ROS2PublishImu.inputs:targetPrim`；在 Isaac Sim 5.1 里这个属性不存在，会报 `Attribute named 'inputs:targetPrim' does not refer to a legal og.Attribute`。

然后在 ROS shell 中确认 Isaac Sim / ROS 2 Bridge 正在发布或订阅这些主题：

```bash
ros2 topic list
ros2 topic echo /clock --once
ros2 topic echo /joint_states --once
ros2 topic echo /imu --once
ros2 topic echo /points_raw --once
ros2 topic info /joint_command
```

Isaac 侧脚本主要提供：

| Topic | 类型 | 方向（相对 Isaac） | 说明 |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/msg/Clock` | 发布 | 仿真时间 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 发布 | Go2W 仿真关节状态 |
| `/imu` | `sensor_msgs/msg/Imu` | 发布 | Go2W IMU |
| `/points_raw` | `sensor_msgs/msg/PointCloud2` | 发布 | RTX LiDAR 原始点云，给后续 adapter 使用 |
| `/joint_command` | `sensor_msgs/msg/JointState` | 订阅 | 来自策略控制器的关节命令 |

`deploy_policy` Go2W 控制器主要使用：

| Topic | 类型 | 方向 | 说明 |
| --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 订阅 | 键盘或导航速度指令 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 订阅 | 仿真机器人关节状态 |
| `/imu` | `sensor_msgs/msg/Imu` | 订阅 | 仿真 IMU |
| `/joint_command` | `sensor_msgs/msg/JointState` | 发布 | 策略输出关节命令 |

ArmDog 控制器会根据 `dog_type` 参数使用带后缀的 topic，例如 `joint_command_<dog_type>`、`imu_<dog_type>`、`joint_states_<dog_type>`。

## FAST-LIO2 路线总览

本仓库把 FAST-LIO2 明确分成两条路线，避免把 IsaacSim 仿真链路和真实 Livox 硬件链路混在一起：

| 路线 | 数据来源 | 是否需要 `livox_ros_driver2` | FAST-LIO2 输入 | 推荐启动 |
| --- | --- | --- | --- | --- |
| 路线 A：IsaacSim + FAST-LIO2 | Isaac RTX LiDAR `/points_raw` + Isaac IMU `/imu` | 不需要 | `/points_fast_lio`，由 adapter 从 `/points_raw` 生成 | `deploy_policy` 的 Isaac 专用 launch |
| 路线 B：真机使用 FAST-LIO2 | 真实 Livox / MID-360 UDP 数据 | 需要 | `/livox/lidar` + `/livox/imu` | `livox_ros_driver2` + `fast_lio` 原生 launch |

### FAST-LIO 还是 FAST-LIO2？

本仓库运行链路里说的 SLAM 是 **FAST-LIO2 路线**，但源码目录、ROS 包名和启动包名保留上游命名：

- 源码目录：`ros2_ws/src/FAST_LIO`
- ROS 包名：`fast_lio`
- 通用启动包：`ros2 launch fast_lio mapping.launch.py ...`
- IsaacSim 专用封装：`ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py`

所以命令里看到 `fast_lio` 不代表 FAST-LIO1；当前文档统一称 **FAST-LIO2 ROS 2 fork**。

### 公共构建步骤

两条路线都需要先构建 ROS 2 工作空间中的 FAST-LIO2；IsaacSim 路线还需要 `deploy_policy` 的点云 adapter，真机路线还需要 Livox SDK / driver。

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh

sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libeigen3-dev libpcl-dev libceres-dev \
  libyaml-cpp-dev libboost-all-dev

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select fast_lio deploy_policy
source install/setup.zsh
```

如果 `fast_lio` 找不到，先检查：

```bash
colcon list | grep -i lio
find src/FAST_LIO -maxdepth 2 -name package.xml -print
```

## 路线 A：IsaacSim + FAST-LIO2

适用场景：只用 IsaacSim / Isaac Lab 仿真点云建图，不接真实 MID-360。**这条路线不要启动 `livox_ros_driver2`**。

### A1. 启动 IsaacSim 场景和 ROS 2 Bridge

终端 A 使用 Isaac / Conda 环境，不要和 ROS shell 混用：

```bash
cd /path/to/SIM-SLAM
conda activate env_isaaclab

# Isaac Sim pip/conda 安装使用内置 ROS 2 Humble bridge 时，需要在启动 Python 前设置：
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python - <<'PY'
from pathlib import Path
import isaacsim
print(Path(isaacsim.__file__).resolve().parent / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib")
PY
)"

# 可选但推荐：检查当前 Isaac Sim 安装中的 ROS2/RTX LiDAR node schema。
python scripts/ros2/check_isaac_ros2_node_schema.py

# 加载 Robocon2026 场地 + Go2W，并创建 /clock、/imu、/joint_states、/points_raw 等 ROS topic。
python scripts/ros2/isaac_fast_lio2_go2w_scene.py \
  --scene assets/Map/robocon2026.usd \
  --robot assets/Go2W/go2w_ros2.usd \
  --scan-rate 10.0
```

说明：仓库 runner 会禁用 `go2w_ros2.usd` 中引用进来的重复 ROS graph，重新创建 Isaac Sim 5.1 风格的运行时 `/ActionGraph`，并为 RTX LiDAR 创建 render product。RTX LiDAR 依赖 render pipeline，headless smoke test 也会使用 render step：

```bash
python scripts/ros2/isaac_fast_lio2_go2w_scene.py --headless --max-steps 1
```

默认 RTX LiDAR 点云 writer 使用 full-scan buffer 模式，约按 `--scan-rate 10.0` 发布完整扫描；FAST-LIO2 需要这种输入。如果要复现/对照旧的 per-render partial scan 行为，可显式加 `--partial-scan-pointcloud`，但该模式不适合作为 FAST-LIO2 默认输入。

### A2. 验证 Isaac 原始点云不是空云

终端 B 使用 ROS 2 Humble shell：

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

ros2 topic list
ros2 topic echo /clock --once
ros2 topic echo /imu --once
ros2 topic echo /points_raw --once
ros2 topic hz /points_raw
ros2 run deploy_policy check_pointcloud_timing.py \
  --topic /points_raw \
  --dry-run-schema \
  --timeout-sec 10 \
  --json
```

`/points_raw` 应该是 `sensor_msgs/msg/PointCloud2`，并且 `width > 0`、`data` 非空。当前 Isaac raw cloud 通常只有 `x/y/z` 字段，这是正常的；A3/A4 的 adapter 会补齐 FAST-LIO2 需要的 `intensity/time/ring`。`ros2 topic hz /points_raw` 应接近 runner 的 `--scan-rate`（默认 10 Hz）；如果是 60–70 Hz，通常说明仍在使用 partial-scan writer。`/imu` 的 `linear_acceleration.z` 静止时应接近 `9.8`，否则 FAST-LIO2 初始化会不稳定。

Isaac 侧主要 topic：

| Topic | 类型 | 方向（相对 Isaac） | 说明 |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/msg/Clock` | 发布 | 仿真时间 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 发布 | Go2W 仿真关节状态 |
| `/imu` | `sensor_msgs/msg/Imu` | 发布 | Go2W IMU |
| `/points_raw` | `sensor_msgs/msg/PointCloud2` | 发布 | RTX LiDAR 原始点云，给 adapter 使用 |
| `/joint_command` | `sensor_msgs/msg/JointState` | 订阅 | 来自策略控制器的关节命令 |

### A3. 可选：单独启动 Isaac 点云 adapter 做分阶段调试

FAST-LIO2 不能直接消费本仓库当前 Isaac 原始 `x/y/z` 点云；需要先转成 `/points_fast_lio`，补齐 Velodyne-style `intensity`、逐点 `time` 和 `ring` 字段。

`fast_lio_isaac_go2w.launch.py` 现在默认会启动这个 adapter，并为当前 Isaac `x/y/z` 点云启用派生字段；如果你使用 A4 默认启动方式，**不要再同时运行下面的 standalone adapter**，否则会出现重复 `/points_fast_lio` publisher。只有在分阶段调试时才单独运行 A3；之后启动 A4 时请加 `enable_adapter:=false`。

```bash
ros2 launch deploy_policy isaac_fast_lio_inputs.launch.py \
  input_topic:=/points_raw \
  output_topic:=/points_fast_lio \
  timestamp_unit:=0 \
  lidar_type:=2 \
  scan_rate_hz:=10.0 \
  derive_time_if_missing:=true \
  derive_ring_if_missing:=true \
  derive_intensity_if_missing:=true \
  filter_invalid_xyz:=true \
  max_abs_coordinate:=200.0 \
  scan_line:=32 \
  frame_id:=lidar_link
```

验证字段契约：

```bash
ros2 topic echo /points_fast_lio --once
ros2 run deploy_policy check_pointcloud_timing.py \
  --topic /points_fast_lio \
  --clock-topic /clock \
  --scan-rate 10.0 \
  --timestamp-unit 0 \
  --lidar-type 2 \
  --timeout-sec 10 \
  --json
```

说明：默认不要加过严的 `--max-clock-skew-sec 0.1` 作为首要判据；仿真调度下 clock/header skew 可能较大，但只要字段、类型、逐点时间跨度和 FAST-LIO2 输出链路通过，点云契约本身就是有效的。

完整字段契约见 [`docs/fast_lio2_input_contract.md`](docs/fast_lio2_input_contract.md)。

### A4. 启动 Isaac 专用 FAST-LIO2（默认同时启动 adapter）

```bash
ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py rviz:=true
```

该 launch 默认使用：

- 自动 adapter：`enable_adapter:=true`，`/points_raw` → `/points_fast_lio`
- 派生字段：`derive_time_if_missing:=true`、`derive_ring_if_missing:=true`、`derive_intensity_if_missing:=true`
- 点过滤：`filter_invalid_xyz:=true`、`max_abs_coordinate:=200.0`，避免 NaN/Inf/极大坐标进入 FAST-LIO2 后触发 PCL `VoxelGrid` overflow
- FAST-LIO2 config：`ros2_ws/src/deploy_policy/config/fast_lio/isaac_go2w.yaml`
- 输入点云：`/points_fast_lio`
- 输入 IMU：`/imu`
- 仿真时间：`use_sim_time:=true`
- TF：默认 `publish_sensor_static_tf:=false`，由 Isaac TransformTree 发布机器人内部 link，FAST-LIO2/Route A 只接 `camera_init -> body -> base_link`，避免 `base_link` two parents
- RViz 配置：`ros2_ws/src/deploy_policy/rviz/fast_lio_isaac_go2w.rviz`

如果你已经按 A3 单独启动了 adapter，请用下面的命令避免重复 `/points_fast_lio` publisher：

```bash
ros2 launch deploy_policy fast_lio_isaac_go2w.launch.py rviz:=true enable_adapter:=false
```

常用观察命令：

```bash
ros2 topic hz /points_raw
ros2 topic hz /points_fast_lio
ros2 topic list | grep -E 'cloud|path|odom|map|lio'
ros2 topic info -v /points_fast_lio
ros2 topic echo /Odometry --once
ros2 run tf2_ros tf2_echo camera_init body
```

成功时应看到 `/points_fast_lio` 只有一个 adapter publisher、FAST-LIO2 是 subscriber，并出现 `/cloud_registered`、`/Odometry`、`/path` 等输出。启动初期偶发一次 `No point, skip this scan!` 可以忽略；如果持续出现 `No Effective Points!` 或 RViz 仍只有大箭头/椭圆等异常形状，应按顺序检查 `/points_raw` full-scan 频率、`/points_fast_lio` 契约、`/imu` gravity 和 FAST-LIO2 输出 topic，而不是先调整 RViz 样式。

如果几秒后 base 漂移，并且 RViz 对 `FL_calf` 等所有 link 报 `No transform ... to [camera_init]`，不要再手动补 `base_link -> imu_link/lidar_link` 静态 TF；先确认使用默认 `publish_sensor_static_tf:=false`，并重启 Isaac runner，使其 TransformTree 只发布 articulation root 下的机器人内部 link 树。若同时看到 `VoxelGrid` overflow，说明 `/points_fast_lio` 仍有非法/极大 XYZ，保持 `filter_invalid_xyz:=true max_abs_coordinate:=200.0` 后复测。

如果静止阶段 FAST-LIO2 正常，但启动 Go2W 控制器或键盘 `cmd_vel` 后机器人乱跳并随后出现 `No Effective Points!`，优先按控制器安全链路排查：使用上面的低速参数，检查 `/joint_command` 是否全为有限值、`/joint_states` 是否连续、`/imu` 是否仍合理，再判断 FAST-LIO2 是否真的输入失效。RViz 中 `/cloud_registered` 如果只是颜色偏红/单色，通常是合成 `intensity` 或颜色 transformer 问题；当前 RViz 配置使用 `AxisColor` 给注册点云按 Z 轴上色，并提供默认关闭的 `Adapted FAST-LIO Input Debug` 显示用于查看 `/points_fast_lio`。

完整 IsaacSim runbook 见 [`docs/isaac_fast_lio2_workflow.md`](docs/isaac_fast_lio2_workflow.md)。

## 路线 B：真机使用 FAST-LIO2

适用场景：使用真实 Livox / MID-360 硬件建图。**这条路线不需要 IsaacSim runner，也不要依赖 `/points_raw` adapter**。

### B1. 安装 Livox-SDK2

```bash
cd ~/Downloads
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

### B2. 初始化并构建 Livox ROS Driver 2

`FAST_LIO` 已随主仓库 vendored；Livox driver 作为子模块接入：

```bash
cd /path/to/SIM-SLAM
git submodule update --init --recursive

cd ros2_ws
source /opt/ros/humble/setup.zsh
colcon build --symlink-install --packages-select livox_ros_driver2
source install/setup.zsh
ros2 pkg list | grep livox
```

如果网络无法拉取子模块，可手动克隆：

```bash
cd /path/to/SIM-SLAM/ros2_ws/src
rm -rf livox_ros_driver2
git clone https://github.com/Ericsii/livox_ros_driver2.git livox_ros_driver2
```

### B3. 配置 MID-360 网络

检查本机连接雷达的网卡 IP：

```bash
ip -br addr
```

然后检查 `ros2_ws/src/livox_ros_driver2/config/MID360_config.json` 中 `host_net_info` 的 IP 是否等于本机对应网卡 IP。若 driver 日志出现 `bind failed`，通常就是配置 IP 不在本机网卡上，或端口被占用。

### B4. 启动 Livox driver

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

# MID-360：发布 Livox CustomMsg，适合 FAST-LIO2 逐点时间戳需求。
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

另开终端检查：

```bash
ros2 topic list | grep livox
ros2 topic hz /livox/lidar
ros2 topic echo /livox/imu --once
```

### B5. 启动真机 FAST-LIO2

MID-360 推荐使用仓库内 `mid360.yaml`，不要沿用 IsaacSim 的 `/points_fast_lio` 配置，也不要用不匹配的 `VLS.yaml`：

```bash
cd /path/to/SIM-SLAM/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

ros2 launch fast_lio mapping.launch.py \
  config_file:=mid360.yaml \
  use_sim_time:=false \
  rviz:=true
```

`mid360.yaml` 的关键默认项：

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `common.lid_topic` | `/livox/lidar` | 来自 `livox_ros_driver2 msg_MID360_launch.py` |
| `common.imu_topic` | `/livox/imu` | 来自真实 Livox IMU |
| `preprocess.lidar_type` | `1` | Livox CustomMsg 路线 |
| `preprocess.timestamp_unit` | `3` | 纳秒单位，需与驱动消息一致 |
| `use_sim_time` | `false` | 真机不用 Isaac `/clock` |

常用检查顺序：

```bash
ros2 topic list | grep -E 'livox|cloud|path|odom|map'
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu
rviz2
```

### B6. FAST-LIO2 调参经验

- **先保证时间戳正确，再调外参。** 如果出现 `Failed to find match for field 'time'`，通常说明点云字段或 lidar_type 配置不匹配。
- **Livox 优先使用 CustomMsg。** 普通 `PointCloud2` 可能缺少足够的逐点时间信息。
- **外参已知时不要在线估计。** 将 `extrinsic_est_en` 设为 `false`，减少初始化不稳定。
- **真机不用 `/clock`。** 启动 FAST-LIO2 时显式 `use_sim_time:=false`。
- **网络先于 SLAM 排查。** `bind failed`、无 `/livox/lidar`、topic hz 为 0 时，先查 IP、网卡、防火墙和端口占用。

![slam view 1](./README.assets/image-20251125132947954.png)

![slam view 2](./README.assets/image-20251125133427548.png)

常见故障的详细排查见 [`docs/troubleshooting.md`](docs/troubleshooting.md)。

## 代码格式与提交建议

```bash
pip install pre-commit
pre-commit run --all-files
```

大文件建议：

- README 图片可以保留在 `README.assets/`。
- `assets/`、训练输出、日志、PCD、rosbag 不建议直接提交普通 Git。
- 策略文件 `.pt` 如果很小可跟随仓库；大模型建议 Git LFS 或 Release 附件。

## 致谢

- Isaac Lab / Isaac Sim 项目模板与强化学习接口。
- 重庆邮电大学 HXC 战队在 rcbbs 分享的 ROBOCON 2026 相关模型资产。
- HKU-MARS FAST-LIO / FAST-LIO2 与 ROS 2 社区维护者。
- Livox SDK2 与 Livox ROS Driver 2。

## 参考资料

- Isaac Lab Installation: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>
- Isaac Lab Pip Installation: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- ROS 2 Humble Ubuntu deb packages: <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>
- ROS 2 installation troubleshooting / Conda conflict: <https://docs.ros.org/en/humble/How-To-Guides/Installation-Troubleshooting.html>
- RoboStack Getting Started: <https://robostack.github.io/GettingStarted.html>
- FAST-LIO2 ROS 2 fork used by this project: <https://github.com/Kuriharamio/FAST_LIO/tree/ROS2>
- FAST-LIO upstream: <https://github.com/hku-mars/FAST_LIO>
- Livox ROS Driver 2: <https://github.com/Livox-SDK/livox_ros_driver2>
- Livox-SDK2: <https://github.com/Livox-SDK/Livox-SDK2>
