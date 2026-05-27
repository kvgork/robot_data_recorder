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


# ------------------------------------------------------------------ #
# Success / failure labelling -> terminal reward
# ------------------------------------------------------------------ #


def test_space_marks_success_with_terminal_reward() -> None:
    cfg = RecordingConfig(repo_id="t/succ", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=4, key=" ")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert buf.success is True
    assert buf.reward[-1] == 1.0
    assert all(r == 0.0 for r in buf.reward[:-1])
    assert buf.done[-1] is True


def test_s_key_marks_success() -> None:
    cfg = RecordingConfig(repo_id="t/succ-s", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=2, key="s")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert buf.success is True
    assert buf.reward[-1] == 1.0


def test_f_marks_failure_zero_reward() -> None:
    cfg = RecordingConfig(repo_id="t/fail", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=3, key="f")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert len(buf.pixels) == 3
    assert buf.success is False
    assert buf.done[-1] is True
    assert all(r == 0.0 for r in buf.reward)
    assert session.abort_requested is False


def test_max_steps_defaults_to_failure() -> None:
    cfg = RecordingConfig(repo_id="t/capfail", format="hdf5", max_steps=4, fps=120)
    listener = _FakeKeyListener(after=999, key=None)
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert buf.success is False
    assert all(r == 0.0 for r in buf.reward)


def test_q_abort_marks_failure() -> None:
    cfg = RecordingConfig(repo_id="t/qfail", format="hdf5", max_steps=100, fps=30)
    listener = _FakeKeyListener(after=2, key="q")
    with _make_session(cfg, listener) as session:
        buf = session.record_episode(0)
    assert session.abort_requested is True
    assert buf.success is False
    assert all(r == 0.0 for r in buf.reward)


# ------------------------------------------------------------------ #
# Inter-episode start gate
# ------------------------------------------------------------------ #


def test_wait_for_start_returns_true_on_space() -> None:
    cfg = RecordingConfig(repo_id="t/start", format="hdf5", max_steps=4, fps=30)
    listener = _FakeKeyListener(after=2, key=" ")
    with _make_session(cfg, listener) as session:
        assert session.wait_for_start(0, 5) is True
        assert session.abort_requested is False


def test_wait_for_start_returns_false_on_q() -> None:
    cfg = RecordingConfig(repo_id="t/abort", format="hdf5", max_steps=4, fps=30)
    listener = _FakeKeyListener(after=1, key="q")
    with _make_session(cfg, listener) as session:
        assert session.wait_for_start(0, 5) is False
        assert session.abort_requested is True


def test_wait_for_start_skips_in_dry_run() -> None:
    cfg = RecordingConfig(
        repo_id="t/dry", format="hdf5", max_steps=4, fps=30, dry_run=True
    )
    listener = _FakeKeyListener(after=999, key=None)
    with _make_session(cfg, listener) as session:
        # Should return immediately even though no key will ever come
        assert session.wait_for_start(0, 5) is True


def test_wait_for_start_skips_when_listener_not_tty() -> None:
    """Non-tty listeners must fall through so scripted runs are unchanged."""
    cfg = RecordingConfig(repo_id="t/nontty", format="hdf5", max_steps=4, fps=30)

    class _NonTty:
        is_tty = False

        def __enter__(self) -> "_NonTty":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def poll(self, timeout: float = 0.0):
            return None

    with _make_session(cfg, _NonTty()) as session:
        assert session.wait_for_start(0, 5) is True
        assert session.abort_requested is False
