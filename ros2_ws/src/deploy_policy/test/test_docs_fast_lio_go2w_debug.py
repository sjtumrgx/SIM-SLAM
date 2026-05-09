from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
README = REPO_ROOT / "README.md"
TROUBLESHOOTING = REPO_ROOT / "docs" / "troubleshooting.md"
WORKFLOW = REPO_ROOT / "docs" / "isaac_fast_lio2_workflow.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_documents_go2w_safe_low_speed_controller_launch():
    text = _read(README)

    assert "max_cmd_vel_x:=0.05" in text
    assert "hold_without_cmd_vel:=true" in text
    assert "cmd_vel_timeout_sec:=0.75" in text


def test_troubleshooting_documents_motion_triggered_no_effective_points():
    text = _read(TROUBLESHOOTING)

    assert "No Effective Points!" in text
    assert "go2w_controller.launch.py" in text
    assert "/joint_command" in text
    assert "max_cmd_vel_x:=0.05" in text


def test_workflow_documents_rviz_red_cloud_classification():
    text = _read(WORKFLOW)

    assert "Adapted FAST-LIO Input Debug" in text
    assert "AxisColor" in text
    assert "intensity" in text


def test_docs_document_mapping_drift_voxelgrid_and_tf_double_parent_fix():
    text = _read(TROUBLESHOOTING) + _read(WORKFLOW)

    assert "VoxelGrid" in text
    assert "filter_invalid_xyz" in text
    assert "publish_sensor_static_tf" in text
    assert "base_link" in text and "two parents" in text
