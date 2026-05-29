"""
test_training_completeness.py — end-to-end checks that recordings carry
everything the downstream LeRobot policy / world-model trainers need.

These lock in the contract reconciled against ``lerobot-isaac-training``:
- Parquet frames carry ``next.reward`` + ``next.done`` (the world-model bridge
  reads those exact columns) under a configurable camera key.
- ``observation.state`` is 12-dim (joint_pos[6] + joint_vel[6]).
- HDF5 output is self-describing (root attrs) and persists captured depth.
- Per-episode timestamps are rebased to start at zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_data_recorder.config import RecordingConfig
from robot_data_recorder.d435 import MockD435Stream
from robot_data_recorder.dual_writer import DualWriter
from robot_data_recorder.recorder import RecordingSession
from robot_data_recorder.so101_teleop import MockSO101Teleop


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class _FakeKeyListener:
    """Replays one scripted key; behaves like a tty."""

    is_tty = True

    def __init__(self, after: int, key: str | None) -> None:
        self._after = after
        self._key = key
        self._calls = 0

    def __enter__(self) -> "_FakeKeyListener":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def poll(self, timeout: float = 0.0) -> str | None:
        self._calls += 1
        return self._key if self._calls == self._after else None


class _CapturingDataset:
    """Stand-in LeRobotDataset that records every add_frame call."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.saved = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved += 1

    def finalize(self) -> None:
        pass


def _episode(n: int = 4, success: bool = True) -> dict[str, np.ndarray]:
    reward = np.zeros(n, dtype=np.float32)
    if success:
        reward[-1] = 1.0
    done = np.array([False] * (n - 1) + [True], dtype=bool)
    return {
        "pixels": np.zeros((n, 480, 640, 3), dtype=np.uint8),
        "action": np.zeros((n, 6), dtype=np.float32),
        "state": np.zeros((n, 12), dtype=np.float32),
        "proprio": np.zeros((n, 12), dtype=np.float32),
        "done": done,
        "timestamp": np.arange(n, dtype=np.float32) / 30.0,
        "episode_idx": np.zeros(n, dtype=np.int64),
        "step_idx": np.arange(n, dtype=np.int64),
        "reward": reward,
    }


# ------------------------------------------------------------------ #
# Parquet line — reward/done/camera key reach the frames
# ------------------------------------------------------------------ #


def _writer_with_dataset(cfg: RecordingConfig, ds: _CapturingDataset) -> DualWriter:
    w = DualWriter.__new__(DualWriter)
    w._config = cfg
    w._lerobot_dataset = ds
    w._hdf5_writer = None
    w._episode_count = 0
    w._output_paths = {}
    w._finalized = False
    return w


def test_parquet_frames_carry_reward_done_and_camera_key(tmp_path: Path) -> None:
    import robot_data_recorder.dual_writer as dw_mod
    from unittest.mock import patch

    cfg = RecordingConfig(
        repo_id="t/cam", format="parquet", output_dir=str(tmp_path),
        camera_name="overhead",
    )
    ds = _CapturingDataset()
    with patch.object(dw_mod, "_HAS_LEROBOT", True):
        writer = _writer_with_dataset(cfg, ds)
        writer._write_lerobot(_episode(n=4, success=True))

    assert len(ds.frames) == 4
    f0, fl = ds.frames[0], ds.frames[-1]
    # Configurable camera key is honoured.
    assert "observation.images.overhead" in f0
    assert "observation.images.d435_rgb" not in f0
    # Reward + done present on every frame.
    for fr in ds.frames:
        assert "next.reward" in fr
        assert "next.done" in fr
    # Sparse terminal reward + terminal done (stored as 1-element arrays).
    assert np.asarray(f0["next.reward"]).reshape(-1)[0] == 0.0
    assert np.asarray(fl["next.reward"]).reshape(-1)[0] == 1.0
    assert bool(np.asarray(f0["next.done"]).reshape(-1)[0]) is False
    assert bool(np.asarray(fl["next.done"]).reshape(-1)[0]) is True
    # State is the 12-dim observation.
    assert np.asarray(f0["observation.state"]).shape == (12,)


def test_parquet_failure_episode_has_zero_reward(tmp_path: Path) -> None:
    import robot_data_recorder.dual_writer as dw_mod
    from unittest.mock import patch

    cfg = RecordingConfig(repo_id="t/fail", format="parquet", output_dir=str(tmp_path))
    ds = _CapturingDataset()
    with patch.object(dw_mod, "_HAS_LEROBOT", True):
        writer = _writer_with_dataset(cfg, ds)
        writer._write_lerobot(_episode(n=3, success=False))

    assert all(
        np.asarray(fr["next.reward"]).reshape(-1)[0] == 0.0 for fr in ds.frames
    )
    # Only the last frame is a terminal.
    assert bool(np.asarray(ds.frames[-1]["next.done"])) is True


# ------------------------------------------------------------------ #
# HDF5 line — self-describing metadata + depth persistence
# ------------------------------------------------------------------ #


def test_hdf5_writes_training_metadata(tmp_path: Path) -> None:
    import h5py

    cfg = RecordingConfig(
        repo_id="t/meta", format="hdf5", output_dir=str(tmp_path),
        task="pick cube", camera_name="overhead", fps=30,
    )
    writer = DualWriter(cfg)
    writer.write_episode(_episode(n=4))
    paths = writer.finalize()

    with h5py.File(paths["hdf5"], "r") as f:
        a = f.attrs
        assert int(a["fps"]) == 30
        assert str(a["task"]) == "pick cube"
        assert str(a["camera_name"]) == "overhead"
        assert int(a["state_dim"]) == 12
        assert int(a["action_dim"]) == 6
        assert a["image_layout"] == "HWC" or a["image_layout"] == b"HWC"
        motors = [m.decode() if isinstance(m, bytes) else str(m)
                  for m in a["motor_names"]]
        assert "gripper" in motors
        # Core training arrays + episode index present.
        for key in ("pixels", "action", "state", "reward", "ep_offset", "ep_len"):
            assert key in f


def test_hdf5_persists_depth_when_present(tmp_path: Path) -> None:
    import h5py

    cfg = RecordingConfig(repo_id="t/depth", format="hdf5", output_dir=str(tmp_path))
    ep = _episode(n=4)
    ep["depth"] = np.zeros((4, 480, 640), dtype=np.uint16)

    writer = DualWriter(cfg)
    writer.write_episode(ep)
    paths = writer.finalize()

    with h5py.File(paths["hdf5"], "r") as f:
        assert "depth" in f
        assert f["depth"].shape == (4, 480, 640)
        assert f["depth"].dtype == np.uint16


def test_hdf5_omits_depth_when_absent(tmp_path: Path) -> None:
    import h5py

    cfg = RecordingConfig(repo_id="t/nodepth", format="hdf5", output_dir=str(tmp_path))
    writer = DualWriter(cfg)
    writer.write_episode(_episode(n=4))
    paths = writer.finalize()

    with h5py.File(paths["hdf5"], "r") as f:
        assert "depth" not in f


# ------------------------------------------------------------------ #
# Recorder — 12-dim state, depth capture, timestamp rebase
# ------------------------------------------------------------------ #


def _record_one(cfg: RecordingConfig, enable_depth: bool):
    cam = MockD435Stream(resolution=cfg.resolution, fps=cfg.fps, enable_depth=enable_depth)
    teleop = MockSO101Teleop()
    listener = _FakeKeyListener(after=4, key=" ")
    session = RecordingSession(
        config=cfg, camera=cam, teleop=teleop, writer=None, key_listener=listener
    )
    with session:
        return session.record_episode(0)


def test_recorder_emits_12dim_state() -> None:
    cfg = RecordingConfig(repo_id="t/st", format="hdf5", max_steps=100, fps=30)
    buf = _record_one(cfg, enable_depth=False)
    ep = buf.to_dict()
    assert ep["state"].shape[1] == 12
    assert ep["proprio"].shape[1] == 12
    assert ep["action"].shape[1] == 6


def test_recorder_captures_depth_when_enabled() -> None:
    cfg = RecordingConfig(
        repo_id="t/d", format="hdf5", max_steps=100, fps=30, enable_depth=True
    )
    buf = _record_one(cfg, enable_depth=True)
    ep = buf.to_dict()
    assert "depth" in ep
    assert ep["depth"].shape == (len(buf.pixels), 480, 640)
    assert ep["depth"].dtype == np.uint16


def test_recorder_omits_depth_when_disabled() -> None:
    cfg = RecordingConfig(repo_id="t/nd", format="hdf5", max_steps=100, fps=30)
    buf = _record_one(cfg, enable_depth=False)
    assert "depth" not in buf.to_dict()


def test_timestamp_rebased_to_zero() -> None:
    cfg = RecordingConfig(repo_id="t/ts", format="hdf5", max_steps=100, fps=30)
    buf = _record_one(cfg, enable_depth=False)
    ts = buf.to_dict()["timestamp"]
    assert ts[0] == 0.0
    assert (np.diff(ts) >= 0).all()
