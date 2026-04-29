import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pointcloud_timing_core import (
    FLOAT32,
    UINT8,
    UINT16,
    UINT32,
    FieldInfo,
    TimingContract,
    make_pointcloud2_like,
    timing_contract_for_lidar_type,
    validate_pointcloud_timing,
)


def _cloud_with_time(values, *, field_name="time", timestamp_sec=12.0):
    fields = [
        FieldInfo("x", 0, FLOAT32),
        FieldInfo("y", 4, FLOAT32),
        FieldInfo("z", 8, FLOAT32),
        FieldInfo("intensity", 12, FLOAT32),
        FieldInfo(field_name, 16, FLOAT32),
    ]
    points = []
    for index, time_value in enumerate(values):
        points.append((float(index), 0.0, 0.0, 1.0, float(time_value)))
    data = b"".join(struct.pack("<fffff", *point) for point in points)
    return make_pointcloud2_like(fields=fields, data=data, point_step=20, stamp_sec=timestamp_sec)


def _velodyne_cloud(values, *, include_intensity=True):
    fields = [
        FieldInfo("x", 0, FLOAT32),
        FieldInfo("y", 4, FLOAT32),
        FieldInfo("z", 8, FLOAT32),
    ]
    offset = 12
    fmt_prefix = "<fff"
    if include_intensity:
        fields.append(FieldInfo("intensity", offset, FLOAT32))
        offset += 4
        fmt_prefix += "f"
    fields.append(FieldInfo("time", offset, FLOAT32))
    offset += 4
    fields.append(FieldInfo("ring", offset, UINT16))
    fmt = fmt_prefix + "fH"
    points = []
    for index, time_value in enumerate(values):
        base = (float(index), 0.0, 0.0)
        if include_intensity:
            base += (1.0,)
        points.append(base + (float(time_value), index % 32))
    data = b"".join(struct.pack(fmt, *point) for point in points)
    return make_pointcloud2_like(fields=fields, data=data, point_step=offset + 2, stamp_sec=12.0)


def _ouster_cloud(t_values):
    fields = [
        FieldInfo("x", 0, FLOAT32),
        FieldInfo("y", 4, FLOAT32),
        FieldInfo("z", 8, FLOAT32),
        FieldInfo("intensity", 12, FLOAT32),
        FieldInfo("t", 16, UINT32),
        FieldInfo("reflectivity", 20, UINT16),
        FieldInfo("ring", 22, UINT8),
        FieldInfo("ambient", 23, UINT16),
        FieldInfo("range", 25, UINT32),
    ]
    data = b"".join(
        struct.pack("<ffffIHBHI", float(index), 0.0, 0.0, 1.0, int(t_value), 10, index % 32, 20, 1000)
        for index, t_value in enumerate(t_values)
    )
    return make_pointcloud2_like(fields=fields, data=data, point_step=29, stamp_sec=12.0)


def test_accepts_fast_lio_time_field_with_seconds_unit_and_clock_alignment():
    msg = _cloud_with_time([0.0, 0.025, 0.050, 0.075, 0.100], timestamp_sec=42.0)
    result = validate_pointcloud_timing(
        msg,
        TimingContract(timestamp_unit=0, scan_rate_hz=10.0, max_clock_skew_sec=0.02),
        clock_time_sec=42.01,
        fast_lio_log_text="FAST-LIO mapping started",
    )

    assert result.ok, result.errors
    assert result.field_name == "time"
    assert math.isclose(result.span_seconds, 0.100, rel_tol=1e-6)
    assert result.point_count == 5


def test_rejects_cloud_without_fast_lio_time_or_t_field():
    msg = _cloud_with_time([0.0, 0.1], field_name="timestamp")
    result = validate_pointcloud_timing(msg, TimingContract(timestamp_unit=0, scan_rate_hz=10.0))

    assert not result.ok
    assert any("time/t" in error for error in result.errors)


def test_rejects_all_zero_time_values_even_when_field_exists():
    msg = _cloud_with_time([0.0, 0.0, 0.0, 0.0])
    result = validate_pointcloud_timing(msg, TimingContract(timestamp_unit=0, scan_rate_hz=10.0))

    assert not result.ok
    assert any("all zero" in error for error in result.errors)


def test_rejects_timestamp_unit_that_makes_scan_span_implausible():
    # Values are milliseconds for a 10 Hz scan, but contract claims seconds.
    msg = _cloud_with_time([0.0, 25.0, 50.0, 75.0, 100.0])
    result = validate_pointcloud_timing(msg, TimingContract(timestamp_unit=0, scan_rate_hz=10.0))

    assert not result.ok
    assert any("span" in error and "expected" in error for error in result.errors)


def test_rejects_fast_lio_log_missing_time_error_even_if_cloud_shape_passes():
    msg = _cloud_with_time([0.0, 0.025, 0.050, 0.075, 0.100])
    result = validate_pointcloud_timing(
        msg,
        TimingContract(timestamp_unit=0, scan_rate_hz=10.0),
        fast_lio_log_text="Failed to find match for field 'time'.",
    )

    assert not result.ok
    assert any("FAST-LIO" in error for error in result.errors)


def test_rejects_missing_required_ring_field_for_velodyne_contract():
    msg = _cloud_with_time([0.0, 0.025, 0.050, 0.075, 0.100], timestamp_sec=42.0)
    result = validate_pointcloud_timing(
        msg,
        TimingContract(timestamp_unit=0, scan_rate_hz=10.0, required_fields={"ring": 4}),
    )

    assert not result.ok
    assert any("required field 'ring'" in error for error in result.errors)


def test_accepts_velodyne_lidar_type_contract_with_ring_field():
    msg = _velodyne_cloud([0.0, 0.025, 0.050, 0.075, 0.100])
    result = validate_pointcloud_timing(
        msg,
        timing_contract_for_lidar_type(lidar_type=2, timestamp_unit=0, scan_rate_hz=10.0),
    )

    assert result.ok, result.errors
    assert result.field_name == "time"


def test_rejects_velodyne_lidar_type_contract_without_intensity_field():
    msg = _velodyne_cloud([0.0, 0.025, 0.050, 0.075, 0.100], include_intensity=False)
    result = validate_pointcloud_timing(
        msg,
        timing_contract_for_lidar_type(lidar_type=2, timestamp_unit=0, scan_rate_hz=10.0),
    )

    assert not result.ok
    assert any("required field 'intensity'" in error for error in result.errors)


def test_accepts_ouster_lidar_type_contract_with_uint32_t_and_uint8_ring():
    msg = _ouster_cloud([0, 25_000_000, 50_000_000, 75_000_000, 100_000_000])
    result = validate_pointcloud_timing(
        msg,
        timing_contract_for_lidar_type(lidar_type=3, timestamp_unit=3, scan_rate_hz=10.0),
    )

    assert result.ok, result.errors
    assert result.field_name == "t"
