from pathlib import Path

import yaml


RVIZ_FILE = Path(__file__).resolve().parents[1] / "rviz" / "fast_lio_isaac_go2w.rviz"


def _display_by_name(name):
    data = yaml.safe_load(RVIZ_FILE.read_text())
    displays = data["Visualization Manager"]["Displays"]
    return next(display for display in displays if display.get("Name") == name)


def test_fast_lio_cloud_uses_axis_color_not_synthetic_intensity():
    display = _display_by_name("FAST-LIO Cloud")

    assert display["Class"] == "rviz_default_plugins/PointCloud2"
    assert display["Topic"]["Value"] == "/cloud_registered"
    assert display["Color Transformer"] == "AxisColor"
    assert display["Axis"] == "Z"
    assert display["Use rainbow"] is True


def test_adapted_cloud_debug_display_is_available_but_disabled():
    display = _display_by_name("Adapted FAST-LIO Input Debug")

    assert display["Class"] == "rviz_default_plugins/PointCloud2"
    assert display["Topic"]["Value"] == "/points_fast_lio"
    assert display["Enabled"] is False
    assert display["Color Transformer"] == "AxisColor"
