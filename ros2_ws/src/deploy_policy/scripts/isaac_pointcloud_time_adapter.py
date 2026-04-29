#!/usr/bin/env python3
"""Adapt Isaac RTX LiDAR PointCloud2 for FAST-LIO timing requirements.

By default this node only republishes clouds that already contain a FAST-LIO
accepted `time` or `t` field. Derived timing is opt-in and must be justified by
scan-order validation for the selected Isaac RTX LiDAR mode.
"""
from __future__ import annotations

import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

try:
    from pointcloud_timing_core import (
        find_timing_field,
        timing_contract_for_lidar_type,
        validate_pointcloud_timing,
    )
except ModuleNotFoundError:  # pragma: no cover - source-tree package import path
    from scripts.pointcloud_timing_core import (
        find_timing_field,
        timing_contract_for_lidar_type,
        validate_pointcloud_timing,
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean parameter value {value!r}")


class IsaacPointCloudTimeAdapter(Node):
    def __init__(self):
        super().__init__("isaac_pointcloud_time_adapter")
        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("output_topic", "/points_fast_lio")
        self.declare_parameter("field_name", "time")
        self.declare_parameter("timestamp_unit", 0)
        self.declare_parameter("lidar_type", 2)
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("derive_time_if_missing", False)
        self.declare_parameter("derive_ring_if_missing", False)
        self.declare_parameter("scan_line", 32)
        self.declare_parameter("span_tolerance_ratio", 0.20)
        self.declare_parameter("frame_id", "")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.field_name = self.get_parameter("field_name").value
        self.timestamp_unit = int(self.get_parameter("timestamp_unit").value)
        self.lidar_type = int(self.get_parameter("lidar_type").value)
        self.scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        self.derive_time_if_missing = _as_bool(self.get_parameter("derive_time_if_missing").value)
        self.derive_ring_if_missing = _as_bool(self.get_parameter("derive_ring_if_missing").value)
        self.scan_line = int(self.get_parameter("scan_line").value)
        self.span_tolerance_ratio = float(self.get_parameter("span_tolerance_ratio").value)
        self.frame_id = str(self.get_parameter("frame_id").value or "")

        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.subscription = self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.get_logger().info(
            f"Adapting {self.input_topic} -> {self.output_topic}; "
            f"lidar_type={self.lidar_type}, timestamp_unit={self.timestamp_unit}, "
            f"derive_time_if_missing={self.derive_time_if_missing}, "
            f"derive_ring_if_missing={self.derive_ring_if_missing}"
        )

    def _cloud_cb(self, msg: PointCloud2) -> None:
        contract = timing_contract_for_lidar_type(
            lidar_type=self.lidar_type,
            timestamp_unit=self.timestamp_unit,
            scan_rate_hz=self.scan_rate_hz,
            span_tolerance_ratio=self.span_tolerance_ratio,
        )
        result = validate_pointcloud_timing(msg, contract)
        if result.ok:
            if self.frame_id:
                msg.header.frame_id = self.frame_id
            self.publisher.publish(msg)
            return

        has_time_field = find_timing_field(msg.fields, contract.accepted_field_names) is not None
        has_ring_field = any(field.name == "ring" for field in msg.fields)
        if (has_time_field or not self.derive_time_if_missing) and (has_ring_field or not self.derive_ring_if_missing):
            self.get_logger().error(
                "Input cloud is not FAST-LIO timing compatible; not publishing. "
                f"errors={result.errors}"
            )
            return

        try:
            adapted = msg
            if not has_time_field and self.derive_time_if_missing:
                if self.lidar_type != 2:
                    raise ValueError("derived timing is only implemented for FAST-LIO lidar_type=2 (VELO16)")
                adapted = self._append_derived_time(adapted)
            if not has_ring_field and self.derive_ring_if_missing:
                if self.lidar_type != 2:
                    raise ValueError("derived ring is only implemented for FAST-LIO lidar_type=2 (VELO16 uint16 ring)")
                adapted = self._append_derived_ring(adapted)
        except Exception as exc:  # pragma: no cover - defensive ROS runtime path
            self.get_logger().error(f"Failed to derive point timing: {exc}")
            return

        check = validate_pointcloud_timing(adapted, contract)
        if not check.ok:
            self.get_logger().error(f"Derived timing failed validation: {check.errors}")
            return
        if self.frame_id:
            adapted.header.frame_id = self.frame_id
        self.publisher.publish(adapted)

    def _append_derived_ring(self, msg: PointCloud2) -> PointCloud2:
        if self.scan_line <= 0:
            raise ValueError("scan_line must be positive to derive ring")
        point_count = int(msg.width) * int(msg.height)
        old_step = int(msg.point_step)
        new_step = old_step + 2
        raw = bytes(msg.data)
        packer = struct.Struct(">H" if msg.is_bigendian else "<H")
        chunks = []
        for index in range(point_count):
            start = index * old_step
            point = raw[start:start + old_step]
            if len(point) != old_step:
                break
            chunks.append(point + packer.pack(index % self.scan_line))
        adapted = PointCloud2()
        adapted.header = msg.header
        adapted.height = 1
        adapted.width = len(chunks)
        adapted.fields = list(msg.fields)
        field = PointField()
        field.name = "ring"
        field.offset = old_step
        field.datatype = PointField.UINT16
        field.count = 1
        adapted.fields.append(field)
        adapted.is_bigendian = msg.is_bigendian
        adapted.point_step = new_step
        adapted.row_step = new_step * adapted.width
        adapted.data = b"".join(chunks)
        adapted.is_dense = msg.is_dense
        return adapted

    def _append_derived_time(self, msg: PointCloud2) -> PointCloud2:
        if self.scan_rate_hz <= 0:
            raise ValueError("scan_rate_hz must be positive to derive timing")
        point_count = int(msg.width) * int(msg.height)
        if point_count <= 1:
            raise ValueError("need at least 2 points to derive scan-relative timing")
        scan_period = 1.0 / self.scan_rate_hz
        unit_factor = {0: 1.0, 1: 1e-3, 2: 1e-6, 3: 1e-9}[self.timestamp_unit]
        old_step = int(msg.point_step)
        new_step = old_step + 4
        chunks = []
        packer = struct.Struct(">f" if msg.is_bigendian else "<f")
        raw = bytes(msg.data)
        for index in range(point_count):
            start = index * old_step
            point = raw[start:start + old_step]
            if len(point) != old_step:
                break
            seconds = scan_period * index / (point_count - 1)
            chunks.append(point + packer.pack(seconds / unit_factor))

        adapted = PointCloud2()
        adapted.header = msg.header
        if self.frame_id:
            adapted.header.frame_id = self.frame_id
        adapted.height = 1
        adapted.width = len(chunks)
        adapted.fields = list(msg.fields)
        field = PointField()
        field.name = self.field_name
        field.offset = old_step
        field.datatype = PointField.FLOAT32
        field.count = 1
        adapted.fields.append(field)
        adapted.is_bigendian = msg.is_bigendian
        adapted.point_step = new_step
        adapted.row_step = new_step * adapted.width
        adapted.data = b"".join(chunks)
        adapted.is_dense = msg.is_dense
        return adapted


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacPointCloudTimeAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - ROS entrypoint
    main()
