import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sensor_msgs.msg import PointCloud2, PointField

from isaac_pointcloud_time_adapter import IsaacPointCloudTimeAdapter
from pointcloud_timing_core import timing_contract_for_lidar_type, validate_pointcloud_timing


def _xyz_cloud(point_count=5):
    msg = PointCloud2()
    msg.header.frame_id = "lidar_link"
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
    msg.data = b"".join(struct.pack("<fff", float(index), 0.0, 0.0) for index in range(point_count))
    msg.is_dense = True
    return msg


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
