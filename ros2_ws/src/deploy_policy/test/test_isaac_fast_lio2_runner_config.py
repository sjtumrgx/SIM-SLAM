import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "ros2" / "isaac_fast_lio2_go2w_scene.py").exists():
            return parent
    raise AssertionError("Could not locate repository root containing scripts/ros2")


def _load_runner():
    script_dir = _repo_root() / "scripts" / "ros2"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "isaac_fast_lio2_go2w_scene", script_dir / "isaac_fast_lio2_go2w_scene.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_lidar_mount_is_above_base_link(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["isaac_fast_lio2_go2w_scene.py"])

    args = runner.parse_args()

    assert args.lidar_prim == "/World/Go2W/base/lidar"
    assert args.imu_topic == "imu"
    assert tuple(args.lidar_translation) == (0.0, 0.0, 0.20)


def test_lidar_mount_translation_can_be_overridden(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isaac_fast_lio2_go2w_scene.py",
            "--lidar-translation",
            "0.10",
            "-0.05",
            "0.42",
        ],
    )

    args = runner.parse_args()

    assert tuple(args.lidar_translation) == (0.10, -0.05, 0.42)


def test_rtx_lidar_requires_rendering_even_when_headless():
    runner = _load_runner()

    assert runner._requires_rendering_step() is True


def test_fast_lio_imu_uses_gravity_for_initialization():
    runner = _load_runner()

    assert runner._imu_read_gravity_enabled() is True


def test_imu_topic_can_be_overridden(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["isaac_fast_lio2_go2w_scene.py", "--imu-topic", "imu_debug"],
    )

    args = runner.parse_args()

    assert args.imu_topic == "imu_debug"


def test_imu_reader_uses_latest_physics_sample():
    runner = _load_runner()

    assert runner._imu_use_latest_data_enabled() is True


def test_rtx_lidar_uses_full_scan_writer_by_default(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["isaac_fast_lio2_go2w_scene.py"])

    args = runner.parse_args()

    assert not args.partial_scan_pointcloud
    assert runner._rtx_lidar_pointcloud_writer_name(full_scan=True).endswith("PublishPointCloudBuffer")


def test_partial_scan_writer_remains_available_for_debug(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(sys, "argv", ["isaac_fast_lio2_go2w_scene.py", "--partial-scan-pointcloud"])

    args = runner.parse_args()

    assert args.partial_scan_pointcloud
    assert runner._rtx_lidar_pointcloud_writer_name(full_scan=False).endswith("PublishPointCloud")
