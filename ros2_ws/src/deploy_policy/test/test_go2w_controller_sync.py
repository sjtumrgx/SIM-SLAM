import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

import go2w_controller
from message_filters import ApproximateTimeSynchronizer


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
