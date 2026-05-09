import importlib.util
import sys
from pathlib import Path

from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "fast_lio_isaac_go2w.launch.py"


def _load_launch_module():
    launch_dir = LAUNCH_FILE.parent
    sys.path.insert(0, str(launch_dir))
    spec = importlib.util.spec_from_file_location("fast_lio_isaac_go2w_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _launch_description():
    return _load_launch_module().generate_launch_description()


def _walk_actions(actions):
    for action in actions:
        yield action
        nested = getattr(action, "actions", None)
        if nested is None:
            nested = getattr(action, "_GroupAction__actions", None)
        if nested:
            yield from _walk_actions(nested)


def _text_of(value):
    if hasattr(value, "text"):
        return value.text
    if isinstance(value, tuple):
        return "".join(_text_of(item) for item in value)
    if isinstance(value, list):
        return "".join(_text_of(item) for item in value)
    return str(value)


def _declared_arguments(ld):
    declared = {}
    for action in _walk_actions(ld.entities):
        if isinstance(action, DeclareLaunchArgument):
            declared[action.name] = _text_of(getattr(action, "default_value", None))
    return declared


def _nodes(ld):
    return [action for action in _walk_actions(ld.entities) if isinstance(action, Node)]


def _node_name(node):
    try:
        return getattr(node, "node_name", None)
    except RuntimeError:
        return getattr(node, "_Node__node_name", None)


def _has_launch_config(value, name):
    if hasattr(value, "_IfCondition__predicate_expression"):
        return _has_launch_config(getattr(value, "_IfCondition__predicate_expression"), name)
    if isinstance(value, LaunchConfiguration):
        return _text_of(getattr(value, "variable_name", None)) == name
    if isinstance(value, ParameterValue):
        return _has_launch_config(getattr(value, "value", None), name)
    if isinstance(value, dict):
        return any(_has_launch_config(item, name) for item in value.values())
    if hasattr(value, "_PythonExpression__expression"):
        return _has_launch_config(getattr(value, "_PythonExpression__expression"), name)
    if isinstance(value, (list, tuple)):
        return any(_has_launch_config(item, name) for item in value)
    return False


def _param_key_text(key):
    if isinstance(key, tuple):
        return "".join(_text_of(item) for item in key)
    return _text_of(key)


def _parameter_dicts(node):
    parameters = getattr(node, "parameters", None)
    if parameters is None:
        parameters = getattr(node, "_Node__parameters", [])
    return [
        {_param_key_text(key): value for key, value in item.items()}
        for item in parameters
        if isinstance(item, dict)
    ]


def test_route_a_declares_adapter_defaults_for_isaac_xyz_cloud():
    args = _declared_arguments(_launch_description())

    assert args["enable_adapter"] == "true"
    assert args["input_topic"] == "/points_raw"
    assert args["output_topic"] == "/points_fast_lio"
    assert args["timestamp_unit"] == "0"
    assert args["lidar_type"] == "2"
    assert args["scan_rate_hz"] in {"10", "10.0"}
    assert args["scan_line"] == "32"
    assert args["frame_id"] == "lidar_link"
    assert args["derive_time_if_missing"] == "true"
    assert args["derive_ring_if_missing"] == "true"
    assert args["derive_intensity_if_missing"] == "true"
    assert args["publish_sensor_static_tf"] == "false"
    assert "python_executable" in args


def test_route_a_does_not_publish_sensor_static_tf_by_default():
    nodes = _nodes(_launch_description())
    sensor_tf_nodes = [
        node
        for node in nodes
        if _node_name(node) in {"base_link_to_imu_tf", "base_link_to_lidar_tf"}
    ]

    assert len(sensor_tf_nodes) == 2
    assert all(_has_launch_config(getattr(node, "condition", None), "publish_sensor_static_tf") for node in sensor_tf_nodes)


def test_route_a_wraps_adapter_in_enable_adapter_condition():
    groups = [action for action in _walk_actions(_launch_description().entities) if isinstance(action, GroupAction)]

    assert groups, "adapter should be conditionally wrapped in a GroupAction"
    assert any(getattr(group, "condition", None) is not None for group in groups)


def test_route_a_passes_shared_lidar_params_to_fast_lio_with_typed_overrides():
    nodes = _nodes(_launch_description())
    fast_lio_nodes = [node for node in nodes if getattr(node, "node_executable", None) == "fastlio_mapping"]
    assert len(fast_lio_nodes) == 1

    param_dict = {}
    for item in _parameter_dicts(fast_lio_nodes[0]):
        param_dict.update(item)

    assert _has_launch_config(param_dict["common.lid_topic"], "output_topic")
    for param_name, launch_arg in {
        "preprocess.lidar_type": "lidar_type",
        "preprocess.timestamp_unit": "timestamp_unit",
        "preprocess.scan_rate": "scan_rate_hz",
        "preprocess.scan_line": "scan_line",
    }.items():
        value = param_dict[param_name]
        assert isinstance(value, ParameterValue)
        assert _has_launch_config(value, launch_arg)
