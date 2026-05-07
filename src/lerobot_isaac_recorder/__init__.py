"""
lerobot-isaac-recorder
======================

D435 camera + SO-101 teleoperation dual-write recorder.

Records episodes simultaneously to LeRobot Parquet (for policy training)
and stable-worldmodel HDF5 (for world-model training via LeWM).

Public API
----------
- ``RecordingConfig``  — session configuration dataclass
- ``RecordingSession`` — main session orchestrator
- ``DualWriter``       — parallel Parquet + HDF5 writer
- ``D435Stream``       — Intel RealSense D435 wrapper (pyrealsense2 soft-import)
- ``EpisodeSchema``    — canonical superset schema declaration

Quick start (dry-run, no hardware)::

    from lerobot_isaac_recorder import RecordingConfig, RecordingSession
    from lerobot_isaac_recorder.d435 import make_d435
    from lerobot_isaac_recorder.so101_teleop import MockSO101Teleop

    cfg = RecordingConfig(repo_id="test/demo", dry_run=True)
    cam = make_d435(mock=True)
    teleop = MockSO101Teleop()

    session = RecordingSession(cfg, camera=cam, teleop=teleop, writer=None)
    with session:
        buf = session.record_episode(0)   # returns 5-step synthetic episode
        print(buf.pixels[0].shape)        # (480, 640, 3)
"""

from lerobot_isaac_recorder.config import RecordingConfig
from lerobot_isaac_recorder.d435 import D435Stream
from lerobot_isaac_recorder.dual_writer import DualWriter
from lerobot_isaac_recorder.recorder import RecordingSession
from lerobot_isaac_recorder.schema import EpisodeSchema

__all__ = [
    "RecordingConfig",
    "RecordingSession",
    "DualWriter",
    "D435Stream",
    "EpisodeSchema",
]
__version__ = "0.1.0"
