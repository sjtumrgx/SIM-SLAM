import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

import go2w_controller
import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer
from sensor_msgs.msg import Imu, JointState


class DummyFilter:
    def registerCallback(self, callback, queue, index):  # noqa: N802 - ROS filter API name
        self.callback = callback
        self.queue = queue
        self.index = index
        return (callback, queue, index)


def test_go2w_sensor_sync_accepts_isaac_timestamp_jitter():
    sync = go2w_controller.make_sensor_synchronizer(
        [DummyFilter(), DummyFilter()],
        queue_size=30,
        slop=0.05,
    )

    assert isinstance(sync, ApproximateTimeSynchronizer)
    assert sync.slop.nanoseconds >= 50_000_000
    assert not sync.allow_headerless


class DummyClock:
    class Time:
        nanoseconds = 1_000_000_000

        def to_msg(self):
            return Time(sec=1, nanosec=0)

    def now(self):
        return self.Time()


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _controller_without_ros_node(action):
    controller = object.__new__(go2w_controller.GO2WController)
    controller._logger = types.SimpleNamespace(error=lambda *_args, **_kwargs: None)
    clock = DummyClock()
    controller.get_clock = lambda: clock
    controller.default_pos = np.array([
        0.1, 0.1, -0.1, -0.1,
        0.8, 0.8, 1.0, 1.0,
        -1.5, -1.5, -1.5, -1.5,
        0.0, 0.0, 0.0, 0.0,
    ])
    controller.joint_names = [
        "FL_hip_joint",
        "RL_hip_joint",
        "FR_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
        "FL_foot_joint",
        "FR_foot_joint",
        "RL_foot_joint",
        "RR_foot_joint",
    ]
    controller.action_length = len(controller.default_pos)
    controller._action_scale = np.array([
        0.125, 0.125, 0.125, 0.125,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        5.0, 5.0, 5.0, 5.0,
    ])
    controller._previous_action = np.zeros(controller.action_length)
    controller._filtered_action = np.zeros(controller.action_length)
    controller._filter_alpha = 1.0
    controller.hold_without_cmd_vel = False
    controller.cmd_vel_timeout_sec = 0.75
    controller._cmd_vel_active = True
    controller._last_cmd_vel_time = 1.0
    controller._joint_command = JointState()
    controller._joint_publisher = RecordingPublisher()
    controller._policy_counter = 0
    controller._decimation = 1
    controller._last_tick_time = 0.0
    controller._compute_observation = lambda *_args: np.zeros(53)
    controller._compute_action = lambda *_args: np.array(action, dtype=float)
    return controller


def _joint_state_for_controller(controller):
    msg = JointState()
    msg.name = list(controller.joint_names)
    msg.position = [float(index) * 0.01 for index, _ in enumerate(msg.name)]
    msg.velocity = [0.0 for _ in msg.name]
    return msg


def test_go2w_joint_command_contains_only_finite_values():
    controller = _controller_without_ros_node([0.5] * 16)

    go2w_controller.GO2WController.synchronized_callback(
        controller,
        _joint_state_for_controller(controller),
        Imu(),
    )

    assert controller._joint_publisher.messages
    command = controller._joint_publisher.messages[-1]
    assert np.isfinite(command.position).all()
    assert np.isfinite(command.velocity).all()
    assert np.isfinite(command.effort).all()


def test_go2w_cmd_vel_is_clipped_to_debug_safe_limits():
    controller = object.__new__(go2w_controller.GO2WController)
    controller.max_cmd_vel_x = 0.20
    controller.max_cmd_vel_y = 0.10
    controller.max_cmd_vel_yaw = 0.30
    clock = DummyClock()
    controller.get_clock = lambda: clock
    msg = Twist()
    msg.linear.x = 2.0
    msg.linear.y = -2.0
    msg.angular.z = 3.0

    go2w_controller.GO2WController.cmd_vel_callback(controller, msg)

    assert controller._cmd_vel.linear.x == 0.20
    assert controller._cmd_vel.linear.y == -0.10
    assert controller._cmd_vel.angular.z == 0.30


def test_go2w_holds_default_pose_until_first_cmd_vel():
    controller = _controller_without_ros_node([1.0] * 16)
    controller.hold_without_cmd_vel = True
    controller._cmd_vel_active = False

    go2w_controller.GO2WController.synchronized_callback(
        controller,
        _joint_state_for_controller(controller),
        Imu(),
    )

    command = controller._joint_publisher.messages[-1]
    assert np.allclose(command.position[:12], controller.default_pos[:12])
    assert np.allclose(command.velocity[-4:], [0.0, 0.0, 0.0, 0.0])
