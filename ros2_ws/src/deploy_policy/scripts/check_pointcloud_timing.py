#!/usr/bin/env python3
"""Validate FAST-LIO-compatible per-point timing in a PointCloud2 topic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

try:
    from pointcloud_timing_core import timing_contract_for_lidar_type, validate_pointcloud_timing
except ModuleNotFoundError:  # pragma: no cover - source-tree package import path
    from scripts.pointcloud_timing_core import timing_contract_for_lidar_type, validate_pointcloud_timing


def _load_ros_imports():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from rosgraph_msgs.msg import Clock
        return rclpy, Node, qos_profile_sensor_data, PointCloud2, Clock
    except Exception as exc:  # pragma: no cover - depends on ROS sourced env
        raise RuntimeError(
            "ROS 2 Python modules are unavailable. Source /opt/ros/humble/setup.zsh "
            "and ros2_ws/install/setup.zsh before running this checker."
        ) from exc


def _read_log(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _format_text(result) -> str:
    lines = ["PointCloud2 timing validation"]
    lines.append(f"  ok: {result.ok}")
    lines.append(f"  field: {result.field_name}")
    lines.append(f"  datatype: {result.datatype}")
    lines.append(f"  timestamp_unit: {result.timestamp_unit}")
    lines.append(f"  point_count: {result.point_count}")
    lines.append(f"  span_seconds: {result.span_seconds}")
    lines.append(f"  header_stamp_seconds: {result.header_stamp_seconds}")
    lines.append(f"  clock_time_seconds: {result.clock_time_seconds}")
    if result.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {warning}" for warning in result.warnings)
    if result.errors:
        lines.append("  errors:")
        lines.extend(f"    - {error}" for error in result.errors)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/points_fast_lio", help="PointCloud2 topic to validate")
    parser.add_argument("--clock-topic", default="/clock", help="Clock topic used for sim-time alignment")
    parser.add_argument("--timestamp-unit", type=int, required=False, default=0,
                        help="FAST_LIO timestamp_unit: 0=s, 1=ms, 2=us, 3=ns")
    parser.add_argument("--lidar-type", type=int, default=None,
                        help="FAST_LIO preprocess.lidar_type; 2 requires Velodyne ring field, 3 requires Ouster t/ring fields")
    parser.add_argument("--require-field", action="append", default=[], metavar="NAME[:DATATYPE]",
                        help="Additional required PointCloud2 field. Datatype may be a PointField integer constant.")
    parser.add_argument("--scan-rate", type=float, default=None, help="Expected full-scan rate in Hz")
    parser.add_argument("--measured-publish-period-sec", type=float, default=None,
                        help="Measured full-scan publish period when scan_rate is unavailable")
    parser.add_argument("--span-tolerance-ratio", type=float, default=0.20,
                        help="Allowed fractional tolerance for per-scan time span")
    parser.add_argument("--max-clock-skew-sec", type=float, default=None,
                        help="Maximum allowed cloud header to /clock skew")
    parser.add_argument("--timeout-sec", type=float, default=10.0, help="Time to wait for messages")
    parser.add_argument("--fast-lio-log", default=None, help="Optional FAST-LIO log file to scan for input errors")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--dry-run-schema", action="store_true",
                        help="Print topic PointCloud2 schema and exit after one message")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rclpy, Node, qos_profile_sensor_data, PointCloud2, Clock = _load_ros_imports()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    class CaptureNode(Node):  # pragma: no cover - requires ROS runtime
        def __init__(self):
            super().__init__("check_pointcloud_timing")
            self.cloud = None
            self.clock_sec = None
            self.create_subscription(PointCloud2, args.topic, self._cloud_cb, qos_profile_sensor_data)
            if args.clock_topic:
                self.create_subscription(Clock, args.clock_topic, self._clock_cb, 10)

        def _cloud_cb(self, msg):
            self.cloud = msg

        def _clock_cb(self, msg):
            self.clock_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    rclpy.init(args=None)
    node = CaptureNode()
    deadline = time.monotonic() + args.timeout_sec
    try:
        while rclpy.ok() and time.monotonic() < deadline and node.cloud is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.cloud is None:
            print(f"Timed out waiting for PointCloud2 on {args.topic}", file=sys.stderr)
            return 1
        if args.dry_run_schema:
            schema = [
                {"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count}
                for f in node.cloud.fields
            ]
            payload = {"topic": args.topic, "point_step": node.cloud.point_step, "fields": schema}
            print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
            return 0
        required_fields = {}
        for item in args.require_field:
            if ":" in item:
                name, datatype = item.split(":", 1)
                required_fields[name] = int(datatype)
            else:
                required_fields[item] = None
        contract = timing_contract_for_lidar_type(
            lidar_type=args.lidar_type,
            timestamp_unit=args.timestamp_unit,
            scan_rate_hz=args.scan_rate,
            measured_publish_period_sec=args.measured_publish_period_sec,
            span_tolerance_ratio=args.span_tolerance_ratio,
            max_clock_skew_sec=args.max_clock_skew_sec,
            required_fields=required_fields,
        )
        result = validate_pointcloud_timing(
            node.cloud,
            contract,
            clock_time_sec=node.clock_sec,
            fast_lio_log_text=_read_log(args.fast_lio_log),
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
        else:
            print(_format_text(result))
        return 0 if result.ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
