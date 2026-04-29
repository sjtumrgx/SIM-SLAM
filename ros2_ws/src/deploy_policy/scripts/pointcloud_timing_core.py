#!/usr/bin/env python3
"""FAST-LIO compatible PointCloud2 timing validation helpers.

The functions in this module are intentionally ROS-import free so they can be
unit-tested outside a sourced ROS 2 environment. ROS node scripts should pass in
real ``sensor_msgs.msg.PointCloud2`` objects; tests may pass lightweight objects
with the same attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import struct
from types import SimpleNamespace
from typing import Iterable, Sequence

# sensor_msgs/msg/PointField constants. Re-declared here to keep this module
# importable without ROS installed.
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

_DATATYPE_FORMATS = {
    INT8: ("b", 1),
    UINT8: ("B", 1),
    INT16: ("h", 2),
    UINT16: ("H", 2),
    INT32: ("i", 4),
    UINT32: ("I", 4),
    FLOAT32: ("f", 4),
    FLOAT64: ("d", 8),
}

_TIMESTAMP_UNIT_TO_SECONDS = {
    0: 1.0,       # seconds
    1: 1e-3,      # milliseconds
    2: 1e-6,      # microseconds
    3: 1e-9,      # nanoseconds
}

_FAST_LIO_ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"failed\s+to\s+find\s+match\s+for\s+field\s+['\"]?(?:time|t)['\"]?",
        r"missing\s+(?:point\s+)?time",
        r"timestamp\s+(?:unit|field).*invalid",
        r"time\s+sync.*(?:fail|error|warn)",
        r"frame.*(?:mismatch|error)",
        r"empty\s+cloud",
        r"preprocess.*(?:fail|error)",
    )
]


@dataclass(frozen=True)
class FieldInfo:
    """Minimal PointField-compatible shape."""

    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class TimingContract:
    """FAST-LIO point timing validation contract.

    ``timestamp_unit`` follows FAST_LIO's PointCloud2 convention used by its
    Velodyne-style configs: 0=s, 1=ms, 2=us, 3=ns.
    """

    timestamp_unit: int
    scan_rate_hz: float | None = None
    accepted_field_names: tuple[str, ...] = ("time", "t")
    accepted_datatypes: tuple[int, ...] = (FLOAT32, FLOAT64)
    span_tolerance_ratio: float = 0.20
    measured_publish_period_sec: float | None = None
    max_clock_skew_sec: float | None = None
    require_monotonic: bool = True
    # Additional FAST-LIO PointCloud2 fields required by the selected lidar_type.
    # Mapping is field_name -> expected datatype, or None to accept any datatype.
    required_fields: dict[str, int | None] = field(default_factory=dict)


@dataclass
class TimingValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_name: str | None = None
    datatype: int | None = None
    timestamp_unit: int | None = None
    point_count: int = 0
    span_seconds: float | None = None
    min_time_seconds: float | None = None
    max_time_seconds: float | None = None
    header_stamp_seconds: float | None = None
    clock_time_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "field_name": self.field_name,
            "datatype": self.datatype,
            "timestamp_unit": self.timestamp_unit,
            "point_count": self.point_count,
            "span_seconds": self.span_seconds,
            "min_time_seconds": self.min_time_seconds,
            "max_time_seconds": self.max_time_seconds,
            "header_stamp_seconds": self.header_stamp_seconds,
            "clock_time_seconds": self.clock_time_seconds,
        }


def datatype_size(datatype: int) -> int:
    try:
        return _DATATYPE_FORMATS[datatype][1]
    except KeyError as exc:
        raise ValueError(f"Unsupported PointField datatype {datatype}") from exc


def field_to_info(field_obj: object) -> FieldInfo:
    if isinstance(field_obj, FieldInfo):
        return field_obj
    return FieldInfo(
        name=str(getattr(field_obj, "name")),
        offset=int(getattr(field_obj, "offset")),
        datatype=int(getattr(field_obj, "datatype")),
        count=int(getattr(field_obj, "count", 1)),
    )


def make_pointcloud2_like(*, fields: Sequence[FieldInfo], data: bytes, point_step: int, stamp_sec: float = 0.0,
                          width: int | None = None, height: int = 1, frame_id: str = "lidar_link") -> object:
    """Create a small PointCloud2-like object for tests and offline tools."""
    sec = int(math.floor(stamp_sec))
    nanosec = int(round((stamp_sec - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    point_count = len(data) // point_step if point_step else 0
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id),
        height=height,
        width=width if width is not None else point_count,
        fields=list(fields),
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * (width if width is not None else point_count),
        data=data,
        is_dense=True,
    )


def message_stamp_to_seconds(msg_or_header_or_stamp: object) -> float | None:
    obj = msg_or_header_or_stamp
    if hasattr(obj, "header"):
        obj = getattr(obj, "header")
    if hasattr(obj, "stamp"):
        obj = getattr(obj, "stamp")
    if obj is None:
        return None
    sec = getattr(obj, "sec", None)
    nanosec = getattr(obj, "nanosec", None)
    if sec is None:
        sec = getattr(obj, "secs", None)
    if nanosec is None:
        nanosec = getattr(obj, "nsecs", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) * 1e-9


def find_timing_field(fields: Iterable[object], accepted_names: Sequence[str]) -> FieldInfo | None:
    infos = [field_to_info(item) for item in fields]
    by_name = {item.name: item for item in infos}
    for name in accepted_names:
        if name in by_name:
            return by_name[name]
    return None


def required_fields_for_lidar_type(lidar_type: int | None) -> dict[str, int | None]:
    """Return FAST-LIO PointCloud2 fields required by a selected lidar type.

    Values come from the checked-out FAST_LIO ``src/preprocess.h`` point type
    registrations. The per-point timing field itself is validated separately by
    ``TimingContract.accepted_field_names`` / ``accepted_datatypes``.
    """
    if lidar_type == 2:  # VELO16 / velodyne_ros::Point
        return {
            "x": FLOAT32,
            "y": FLOAT32,
            "z": FLOAT32,
            "intensity": FLOAT32,
            "ring": UINT16,
        }
    if lidar_type == 3:  # OUST64 / ouster_ros::Point
        return {
            "x": FLOAT32,
            "y": FLOAT32,
            "z": FLOAT32,
            "intensity": FLOAT32,
            "reflectivity": UINT16,
            "ring": UINT8,
            "ambient": UINT16,
            "range": UINT32,
        }
    return {}


def timing_contract_for_lidar_type(
    *,
    lidar_type: int | None,
    timestamp_unit: int,
    scan_rate_hz: float | None = None,
    span_tolerance_ratio: float = 0.20,
    measured_publish_period_sec: float | None = None,
    max_clock_skew_sec: float | None = None,
    require_monotonic: bool = True,
    required_fields: dict[str, int | None] | None = None,
) -> TimingContract:
    """Build a strict FAST-LIO timing contract for a PointCloud2 lidar type."""
    lidar_required = required_fields_for_lidar_type(lidar_type)
    if required_fields:
        lidar_required.update(required_fields)

    if lidar_type == 2:
        accepted_field_names = ("time",)
        accepted_datatypes = (FLOAT32,)
    elif lidar_type == 3:
        accepted_field_names = ("t",)
        accepted_datatypes = (UINT32,)
    else:
        accepted_field_names = ("time", "t")
        accepted_datatypes = (FLOAT32, FLOAT64, UINT32)

    return TimingContract(
        timestamp_unit=timestamp_unit,
        scan_rate_hz=scan_rate_hz,
        accepted_field_names=accepted_field_names,
        accepted_datatypes=accepted_datatypes,
        span_tolerance_ratio=span_tolerance_ratio,
        measured_publish_period_sec=measured_publish_period_sec,
        max_clock_skew_sec=max_clock_skew_sec,
        require_monotonic=require_monotonic,
        required_fields=lidar_required,
    )


def _point_count(msg: object) -> int:
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 1) or 1)
    if width > 0 and height > 0:
        return width * height
    point_step = int(getattr(msg, "point_step", 0) or 0)
    data = bytes(getattr(msg, "data", b""))
    return len(data) // point_step if point_step else 0


def extract_field_values(msg: object, field_info: FieldInfo) -> list[float]:
    if field_info.count != 1:
        raise ValueError(f"Timing field {field_info.name!r} must have count=1, got {field_info.count}")
    try:
        fmt_char, size = _DATATYPE_FORMATS[field_info.datatype]
    except KeyError as exc:
        raise ValueError(f"Unsupported timing datatype {field_info.datatype}") from exc

    point_step = int(getattr(msg, "point_step"))
    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    fmt = struct.Struct(endian + fmt_char)
    data = bytes(getattr(msg, "data"))
    count = _point_count(msg)
    values: list[float] = []
    for index in range(count):
        start = index * point_step + field_info.offset
        end = start + size
        if end > len(data):
            break
        values.append(float(fmt.unpack_from(data, start)[0]))
    return values


def convert_time_values_to_seconds(values: Sequence[float], timestamp_unit: int) -> list[float]:
    if timestamp_unit not in _TIMESTAMP_UNIT_TO_SECONDS:
        raise ValueError(f"Unsupported FAST_LIO timestamp_unit {timestamp_unit}; expected one of 0,1,2,3")
    factor = _TIMESTAMP_UNIT_TO_SECONDS[timestamp_unit]
    return [float(value) * factor for value in values]


def expected_scan_span_seconds(contract: TimingContract) -> float | None:
    if contract.scan_rate_hz and contract.scan_rate_hz > 0:
        return 1.0 / float(contract.scan_rate_hz)
    if contract.measured_publish_period_sec and contract.measured_publish_period_sec > 0:
        return float(contract.measured_publish_period_sec)
    return None


def fast_lio_log_errors(log_text: str | None) -> list[str]:
    if not log_text:
        return []
    errors = []
    for pattern in _FAST_LIO_ERROR_PATTERNS:
        match = pattern.search(log_text)
        if match:
            errors.append(f"FAST-LIO log indicates input-contract failure: {match.group(0)}")
    return errors


def validate_pointcloud_timing(msg: object, contract: TimingContract, *, clock_time_sec: float | None = None,
                               fast_lio_log_text: str | None = None) -> TimingValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    field_info = find_timing_field(getattr(msg, "fields", []), contract.accepted_field_names)
    result = TimingValidationResult(
        ok=False,
        errors=errors,
        warnings=warnings,
        field_name=field_info.name if field_info else None,
        datatype=field_info.datatype if field_info else None,
        timestamp_unit=contract.timestamp_unit,
        point_count=_point_count(msg),
        header_stamp_seconds=message_stamp_to_seconds(msg),
        clock_time_seconds=clock_time_sec,
    )

    field_infos = [field_to_info(item) for item in getattr(msg, "fields", [])]
    fields_by_name = {item.name: item for item in field_infos}
    for required_name, required_datatype in contract.required_fields.items():
        required = fields_by_name.get(required_name)
        if required is None:
            errors.append(f"PointCloud2 is missing required field {required_name!r} for the selected FAST-LIO lidar_type.")
        elif required_datatype is not None and required.datatype != required_datatype:
            errors.append(
                f"PointCloud2 required field {required_name!r} datatype {required.datatype} does not match expected {required_datatype}."
            )
        elif required.count != 1:
            errors.append(
                f"PointCloud2 required field {required_name!r} count {required.count} does not match expected 1."
            )

    if field_info is None:
        errors.append(
            "PointCloud2 is missing a FAST-LIO accepted per-point time/t field; "
            "Isaac metadata is insufficient unless exposed as point field 'time' or 't'."
        )
        result.errors.extend(fast_lio_log_errors(fast_lio_log_text))
        return result

    if field_info.datatype not in contract.accepted_datatypes:
        errors.append(
            f"Timing field {field_info.name!r} datatype {field_info.datatype} is not accepted; "
            f"expected one of {list(contract.accepted_datatypes)}."
        )

    try:
        raw_values = extract_field_values(msg, field_info)
        seconds_values = convert_time_values_to_seconds(raw_values, contract.timestamp_unit)
    except ValueError as exc:
        errors.append(str(exc))
        result.errors.extend(fast_lio_log_errors(fast_lio_log_text))
        return result

    result.point_count = len(seconds_values)
    if not seconds_values:
        errors.append("Timing field exists but no point values were readable.")
    elif not all(math.isfinite(value) for value in seconds_values):
        errors.append("Timing field contains non-finite values.")
    else:
        min_time = min(seconds_values)
        max_time = max(seconds_values)
        span = max_time - min_time
        result.min_time_seconds = min_time
        result.max_time_seconds = max_time
        result.span_seconds = span
        if all(abs(value) <= 1e-12 for value in seconds_values):
            errors.append("Timing field values are all zero; this cannot support FAST-LIO undistortion.")
        if any(value < -1e-12 for value in seconds_values):
            errors.append("Timing field contains negative relative times.")
        if contract.require_monotonic:
            monotonic = all(a <= b + 1e-12 for a, b in zip(seconds_values, seconds_values[1:]))
            if not monotonic:
                errors.append("Timing field values are not monotonic within the scan.")
        expected_span = expected_scan_span_seconds(contract)
        if expected_span:
            lower = expected_span * (1.0 - contract.span_tolerance_ratio)
            upper = expected_span * (1.0 + contract.span_tolerance_ratio)
            if not (lower <= span <= upper):
                errors.append(
                    f"Timing span {span:.9f}s is outside expected [{lower:.9f}, {upper:.9f}]s "
                    f"for scan period {expected_span:.9f}s. Check timestamp_unit and scan_rate."
                )
        else:
            warnings.append("No scan_rate_hz or measured_publish_period_sec provided; span plausibility not checked.")

    if result.header_stamp_seconds is None:
        warnings.append("PointCloud2 header stamp is unavailable; /clock alignment not checked.")
    elif clock_time_sec is not None and contract.max_clock_skew_sec is not None:
        skew = abs(result.header_stamp_seconds - clock_time_sec)
        if skew > contract.max_clock_skew_sec:
            errors.append(
                f"PointCloud2 header stamp skew {skew:.9f}s exceeds max_clock_skew_sec "
                f"{contract.max_clock_skew_sec:.9f}s."
            )
    elif clock_time_sec is None:
        warnings.append("No /clock sample provided; header-clock alignment not checked.")

    errors.extend(fast_lio_log_errors(fast_lio_log_text))
    result.ok = not errors
    return result


def append_timing_field(msg: object, *, field_name: str, timestamp_unit: int, scan_rate_hz: float,
                        datatype: int = FLOAT32) -> object:
    """Return a PointCloud2-like object with a derived monotonic timing field.

    This helper exists for the ROS adapter's explicit fallback mode. It assumes
    current point ordering follows scan phase. Callers must only enable it after
    documenting and validating that assumption for the selected RTX LiDAR mode.
    """
    if datatype not in (FLOAT32, FLOAT64):
        raise ValueError("Derived timing field supports FLOAT32 or FLOAT64 only")
    point_count = _point_count(msg)
    if point_count <= 1:
        raise ValueError("Cannot derive scan-relative timing for fewer than 2 points")
    if scan_rate_hz <= 0:
        raise ValueError("scan_rate_hz must be positive")
    factor = _TIMESTAMP_UNIT_TO_SECONDS.get(timestamp_unit)
    if factor is None:
        raise ValueError(f"Unsupported timestamp_unit {timestamp_unit}")

    old_point_step = int(getattr(msg, "point_step"))
    old_data = bytes(getattr(msg, "data"))
    fmt_char, field_size = _DATATYPE_FORMATS[datatype]
    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    packer = struct.Struct(endian + fmt_char)
    new_offset = old_point_step
    new_point_step = old_point_step + field_size
    scan_period = 1.0 / scan_rate_hz
    new_chunks = []
    for index in range(point_count):
        start = index * old_point_step
        point = old_data[start:start + old_point_step]
        if len(point) < old_point_step:
            break
        seconds = scan_period * index / (point_count - 1)
        unit_value = seconds / factor
        new_chunks.append(point + packer.pack(unit_value))

    new_fields = [field_to_info(item) for item in getattr(msg, "fields", [])]
    new_fields.append(FieldInfo(field_name, new_offset, datatype, 1))
    return make_pointcloud2_like(
        fields=new_fields,
        data=b"".join(new_chunks),
        point_step=new_point_step,
        stamp_sec=message_stamp_to_seconds(msg) or 0.0,
        width=point_count,
        height=1,
        frame_id=getattr(getattr(msg, "header", None), "frame_id", "lidar_link"),
    )


__all__ = [
    "INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT32", "FLOAT64",
    "FieldInfo", "TimingContract", "TimingValidationResult",
    "append_timing_field", "convert_time_values_to_seconds", "datatype_size", "expected_scan_span_seconds",
    "extract_field_values", "fast_lio_log_errors", "field_to_info", "find_timing_field",
    "make_pointcloud2_like", "message_stamp_to_seconds", "required_fields_for_lidar_type",
    "timing_contract_for_lidar_type", "validate_pointcloud_timing",
]
