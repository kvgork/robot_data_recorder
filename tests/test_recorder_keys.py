"""
test_recorder_keys.py — episode-end key handling in RecordingSession.

Uses an injected fake KeyListener to drive the loop without needing a
real tty or hardware. Camera + teleop are MockD435Stream / MockSO101Teleop.
"""

from __future__ import annotations

from typing import Optional

from robot_data_recorder.config import RecordingConfig
from robot_data_recorder.d435 import MockD435Stream
from robot_data_recorder.recorder import RecordingSession
from robot_data_recorder.so101_teleop import MockSO101Teleop


class _FakeKeyListener:
    """Replays a scripted key sequence and acts like a tty for messaging."""

    is_tty = True

    def __init__(self, after: int, key: Optional[str]) -> None:
        self._after = after
        self._key = key
        self._calls = 0

    def __enter__(self) -> "_FakeKeyListener":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def poll(self, timeout: float = 0.0) -> Optional[str]:
        self._calls += 1
        if self._calls == self._after:
            return self._key
        return None


def _make_session(cfg: RecordingConfig, listener: _FakeKeyListener) -> RecordingSession:
    cam = MockD435Stream(resolution=cfg.resolution, fps=cfg.fps)
    teleop = MockSO101Teleop()
    return RecordingSession(
        config=cfg, camera=cam, teleop=teleop, writer=None, key_listener=listener
    )


def test_space_ends_episode_early() -> None:
    cfg = RecordingConfig(repo_id="t/sp", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=5, key=" ")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)

    assert len(buf.pixels) == 5
    assert buf.done[-1] is True
    assert session.abort_requested is False


def test_enter_ends_episode_early() -> None:
    cfg = RecordingConfig(repo_id="t/en", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=3, key="\n")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert len(buf.pixels) == 3


def test_q_aborts_session() -> None:
    cfg = RecordingConfig(repo_id="t/q", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=2, key="q")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
        assert session.abort_requested is True
    assert len(buf.pixels) == 2
    assert buf.done[-1] is True


def test_no_key_press_runs_to_max_steps() -> None:
    cfg = RecordingConfig(repo_id="t/cap", format="hdf5", max_steps=4, fps=120)
    listener = _FakeKeyListener(after=999, key=None)
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert len(buf.pixels) == 4
    assert buf.done[-1] is True
    assert session.abort_requested is False
