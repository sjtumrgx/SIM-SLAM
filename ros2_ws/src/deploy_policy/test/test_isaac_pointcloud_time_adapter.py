import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sensor_msgs.msg import PointCloud2, PointField

from isaac_pointcloud_time_adapter import IsaacPointCloudTimeAdapter
from pointcloud_timing_core import timing_contract_for_lidar_type, validate_pointcloud_timing


def _xyz_cloud(point_count=5, points=None):
    msg = PointCloud2()
    msg.header.frame_id = "lidar_link"
    if points is None:
        points = [(float(index), 0.0, 0.0) for index in range(point_count)]
    point_count = len(points)
    msg.height = 1
    msg.width = point_count
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * point_count
    msg.data = b"".join(struct.pack("<fff", *point) for point in points)
    msg.is_dense = True
    return msg


def _read_xyz_points(msg):
    return [
        struct.unpack_from("<fff", bytes(msg.data), index * msg.point_step)
        for index in range(msg.width)
    ]


def test_derives_velodyne_fields_from_isaac_xyz_cloud():
    adapter = object.__new__(IsaacPointCloudTimeAdapter)
    adapter.field_name = "time"
    adapter.timestamp_unit = 0
    adapter.scan_rate_hz = 10.0
    adapter.scan_line = 32
    adapter.frame_id = "lidar_link"

    adapted = adapter._append_derived_intensity(_xyz_cloud())
    adapted = adapter._append_derived_time(adapted)
    adapted = adapter._append_derived_ring(adapted)

    field_names = [field.name for field in adapted.fields]
    assert field_names == ["x", "y", "z", "intensity", "time", "ring"]
    assert adapted.point_step == 22
    assert adapted.width == 5

    result = validate_pointcloud_timing(
        adapted,
        timing_contract_for_lidar_type(lidar_type=2, timestamp_unit=0, scan_rate_hz=10.0),
    )
    assert result.ok, result.errors


def test_filters_invalid_or_huge_isaac_xyz_points_before_fast_lio_derivation():
    adapter = object.__new__(IsaacPointCloudTimeAdapter)
    adapter.filter_invalid_xyz = True
    adapter.max_abs_coordinate = 1_000.0

    filtered, dropped = adapter._filter_invalid_xyz_points(
        _xyz_cloud(
            points=[
                (0.0, 0.0, 1.0),
                (float("nan"), 0.0, 1.0),
                (1.0e20, 0.0, 1.0),
                (2.0, 0.0, 1.0),
            ],
        )
    )

    assert dropped == 2
    assert filtered.width == 2
    assert filtered.height == 1
    assert filtered.row_step == filtered.point_step * filtered.width
    assert filtered.is_dense is True
    assert _read_xyz_points(filtered) == [(0.0, 0.0, 1.0), (2.0, 0.0, 1.0)]


def _read_uint16_field(msg, field_name):
    field = next(item for item in msg.fields if item.name == field_name)
    return [
        struct.unpack_from("<H", bytes(msg.data), index * msg.point_step + field.offset)[0]
        for index in range(msg.width)
    ]


def test_derived_ring_uses_configured_scan_line_and_time_uses_frame_id():
    adapter = object.__new__(IsaacPointCloudTimeAdapter)
    adapter.field_name = "time"
    adapter.timestamp_unit = 0
    adapter.scan_rate_hz = 5.0
    adapter.scan_line = 3
    adapter.frame_id = "custom_lidar"

    adapted = adapter._append_derived_intensity(_xyz_cloud(point_count=7))
    adapted = adapter._append_derived_time(adapted)
    adapted = adapter._append_derived_ring(adapted)

    assert adapted.header.frame_id == "custom_lidar"
    assert _read_uint16_field(adapted, "ring") == [0, 1, 2, 0, 1, 2, 0]

    result = validate_pointcloud_timing(
        adapted,
        timing_contract_for_lidar_type(lidar_type=2, timestamp_unit=0, scan_rate_hz=5.0),
    )
    assert result.ok, result.errors
    assert abs(result.span_seconds - 0.2) < 1e-6


def test_rejects_invalid_derived_timing_configuration():
    adapter = object.__new__(IsaacPointCloudTimeAdapter)
    adapter.timestamp_unit = 0
    adapter.scan_rate_hz = 0.0

    try:
        adapter._append_derived_time(_xyz_cloud())
    except ValueError as exc:
        assert "scan_rate_hz must be positive" in str(exc)
    else:
        raise AssertionError("invalid scan_rate_hz should fail")


def test_rejects_invalid_derived_ring_configuration():
    adapter = object.__new__(IsaacPointCloudTimeAdapter)
    adapter.scan_line = 0

    try:
        adapter._append_derived_ring(_xyz_cloud())
    except ValueError as exc:
        assert "scan_line must be positive" in str(exc)
    else:
        raise AssertionError("invalid scan_line should fail")
