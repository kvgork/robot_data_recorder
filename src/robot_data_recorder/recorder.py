"""
recorder
========

Main recording session class.

``RecordingSession`` orchestrates a single multi-episode recording run:
1. Start camera + teleop hardware.
2. For each episode: tick at 1/fps, read camera + arm, accumulate in buffer.
3. Write buffer via DualWriter.
4. Finalize outputs.

The session is designed to run headless (no GUI). Episode end is signalled
by ``max_steps`` (scaffolding) or a keyboard interrupt (real deployment).

Usage::

    from robot_data_recorder.recorder import RecordingSession
    from robot_data_recorder.d435 import make_d435
    from robot_data_recorder.so101_teleop import MockSO101Teleop
    from robot_data_recorder.dual_writer import DualWriter
    from robot_data_recorder.config import RecordingConfig

    cfg = RecordingConfig(repo_id="test", format="hdf5", dry_run=True)
    cam = make_d435(mock=True)
    teleop = MockSO101Teleop()
    # writer omitted in dry_run mode
    session = RecordingSession(cfg, camera=cam, teleop=teleop, writer=None)
    with session:
        buf = session.record_episode(0)
        print(len(buf.pixels), "frames")
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from robot_data_recorder.config import RecordingConfig
from robot_data_recorder.keylistener import KeyListener, NullKeyListener

if TYPE_CHECKING:
    from robot_data_recorder.d435 import D435Stream, MockD435Stream
    from robot_data_recorder.dual_writer import DualWriter
    from robot_data_recorder.so101_teleop import MockSO101Teleop, SO101Teleop


# Keys that mark "this episode is done"
_END_EPISODE_KEYS = frozenset({" ", "\n", "\r", "s", "S"})
# Keys that mark "start the next episode"
_START_EPISODE_KEYS = _END_EPISODE_KEYS
# Keys that mark "abort the whole session"
_ABORT_KEYS = frozenset({"q", "Q", "\x03"})  # Ctrl-C as a printable byte


@dataclass
class EpisodeBuffer:
    """Accumulates raw per-step data during a single episode rollout.

    All lists are appended in lock-step; after the episode ends they are
    stacked into numpy arrays and passed to ``DualWriter.write_episode()``.

    Attributes
    ----------
    episode_idx:
        Global episode index in the recording session.
    pixels:
        List of RGB frames, each (H, W, 3) uint8.
    action:
        List of action vectors, each (A,) float32.
    state:
        List of state vectors, each (S,) float32.
    proprio:
        List of proprio vectors, each (P,) float32.
    done:
        List of done flags.
    timestamp:
        List of hardware timestamps (float, seconds).
    reward:
        List of reward scalars (0.0 for teleop).
    """

    episode_idx: int = 0
    pixels: list = field(default_factory=list)
    action: list = field(default_factory=list)
    state: list = field(default_factory=list)
    proprio: list = field(default_factory=list)
    done: list = field(default_factory=list)
    timestamp: list = field(default_factory=list)
    reward: list = field(default_factory=list)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert lists to numpy arrays in the canonical schema format."""
        n = len(self.pixels)
        return {
            "pixels": np.stack(self.pixels, axis=0).astype(np.uint8),
            "action": np.stack(self.action, axis=0).astype(np.float32),
            "state": np.stack(self.state, axis=0).astype(np.float32),
            "proprio": np.stack(self.proprio, axis=0).astype(np.float32),
            "done": np.array(self.done, dtype=bool),
            "timestamp": np.array(self.timestamp, dtype=np.float32),
            "episode_idx": np.full(n, self.episode_idx, dtype=np.int64),
            "step_idx": np.arange(n, dtype=np.int64),
            "reward": np.array(self.reward, dtype=np.float32),
        }


class RecordingSession:
    """Orchestrates a multi-episode recording session.

    Parameters
    ----------
    config:
        Recording configuration.
    camera:
        D435Stream or MockD435Stream instance.
    teleop:
        SO101Teleop or MockSO101Teleop instance.
    writer:
        DualWriter instance. May be ``None`` in dry-run mode.
    """

    def __init__(
        self,
        config: RecordingConfig,
        camera: Any,
        teleop: Any,
        writer: Optional["DualWriter"],
        key_listener: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._camera = camera
        self._teleop = teleop
        self._writer = writer
        # Allow injection for tests; default behaviour picks the right
        # backend based on whether stdin is a tty.
        if key_listener is not None:
            self._key_listener = key_listener
        else:
            kl = KeyListener()
            self._key_listener = kl if kl.is_tty else NullKeyListener()
        self._abort_requested = False

    @property
    def abort_requested(self) -> bool:
        """True after the operator pressed an abort key (``q``)."""
        return self._abort_requested

    # ------------------------------------------------------------------ #
    # Inter-episode gate
    # ------------------------------------------------------------------ #

    def wait_for_start(self, episode_idx: int, total: int) -> bool:
        """Block until the operator confirms the next episode should begin.

        Used between episodes so the operator can reset the workspace
        before recording starts. SPACE / ENTER / ``s`` starts the next
        episode; ``q`` aborts the whole session. Non-tty environments
        fall through immediately so scripted runs are unchanged.

        Returns
        -------
        bool
            ``True`` if recording should proceed, ``False`` if the
            operator asked to abort.
        """
        if self._config.dry_run:
            return True
        if not getattr(self._key_listener, "is_tty", False):
            return True

        prompt = (
            f"[robot-data-record] Reset workspace, then press SPACE/ENTER to "
            f"start episode {episode_idx + 1}/{total} (q to abort)... "
        )
        # Use sys.stdout for an inline prompt without an extra newline
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            while True:
                key = self._key_listener.poll(timeout=0.05)
                if key is None:
                    continue
                if key in _ABORT_KEYS:
                    self._abort_requested = True
                    sys.stdout.write("aborted.\n")
                    sys.stdout.flush()
                    return False
                if key in _START_EPISODE_KEYS:
                    sys.stdout.write("go.\n")
                    sys.stdout.flush()
                    return True
                # Ignore everything else — keep waiting.
        except KeyboardInterrupt:
            self._abort_requested = True
            sys.stdout.write("aborted.\n")
            sys.stdout.flush()
            return False

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "RecordingSession":
        self._camera.start()
        self._teleop.start()
        self._key_listener.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        try:
            self._key_listener.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._camera.stop()
        except Exception:
            pass
        try:
            self._teleop.stop()
        except Exception:
            pass
        if self._writer is not None:
            try:
                self._writer.finalize()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_episode(self, episode_idx: int) -> EpisodeBuffer:
        """Record one episode.

        Ticks at ``1/fps``, reads camera + arm, appends to buffer until
        the operator presses SPACE / Enter (end episode) or ``q`` (abort
        session). ``max_steps`` is still honoured as a hard safety
        ceiling so a forgotten terminal cannot record forever. When stdin
        is not a tty the key listener is a no-op and the loop falls back
        to ``max_steps``.

        Parameters
        ----------
        episode_idx:
            Global episode index (written into ``EpisodeBuffer.episode_idx``).

        Returns
        -------
        EpisodeBuffer
            Accumulated step data for this episode.
        """
        if self._config.dry_run:
            return self._synthetic_episode(episode_idx)

        buf = EpisodeBuffer(episode_idx=episode_idx)
        period = 1.0 / self._config.fps
        max_steps = self._config.max_steps

        if getattr(self._key_listener, "is_tty", False):
            print(
                f"[robot-data-record] Episode {episode_idx + 1}: "
                "press SPACE/ENTER to save, 'q' to abort session "
                f"(safety cap: {max_steps} steps)."
            )

        ended_by_key = False
        last_step = 0
        for step in range(max_steps):
            last_step = step
            t0 = time.monotonic()

            frame = self._camera.read_frame()
            arm_state = self._teleop.read_state()
            action = self._teleop.read_action()

            # State vector = full motor positions (5 joints + gripper).
            # lerobot 0.4.4 already includes the gripper in joint_pos, so
            # we copy the array directly without an extra concat.
            state_vec = np.asarray(arm_state["joint_pos"], dtype=np.float32)

            buf.pixels.append(frame["rgb"])
            buf.action.append(action)
            buf.state.append(state_vec)
            buf.proprio.append(state_vec.copy())  # proprio = state for SO-101
            # Will fix the last entry to True after the loop.
            buf.done.append(False)
            buf.timestamp.append(float(arm_state["timestamp"]))
            buf.reward.append(0.0)

            key = self._key_listener.poll()
            if key in _END_EPISODE_KEYS:
                ended_by_key = True
                break
            if key in _ABORT_KEYS:
                self._abort_requested = True
                ended_by_key = True
                break

            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if buf.done:
            buf.done[-1] = True
        if not ended_by_key and last_step == max_steps - 1:
            print(
                f"[robot-data-record] Episode {episode_idx + 1}: hit max_steps "
                f"({max_steps}); auto-saving. Increase --max-steps for longer episodes."
            )
        return buf

    @staticmethod
    def _synthetic_episode(episode_idx: int, n_steps: int = 5) -> EpisodeBuffer:
        """Return a deterministic 5-step synthetic episode for dry-run mode."""
        rng = np.random.default_rng(episode_idx)
        buf = EpisodeBuffer(episode_idx=episode_idx)

        for t in range(n_steps):
            buf.pixels.append(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
            buf.action.append(rng.standard_normal(6).astype(np.float32))
            state = rng.standard_normal(6).astype(np.float32)
            buf.state.append(state)
            buf.proprio.append(state.copy())
            buf.done.append(t == n_steps - 1)
            buf.timestamp.append(float(t) / 30.0)
            buf.reward.append(0.0)

        return buf

    def save_episode(self, buffer: EpisodeBuffer) -> None:
        """Dispatch episode buffer to the DualWriter.

        Parameters
        ----------
        buffer:
            Completed episode buffer from ``record_episode()``.
        """
        if self._writer is None:
            return
        ep_dict = buffer.to_dict()
        self._writer.write_episode(ep_dict)


class MockRecordingSession(RecordingSession):
    """Recording session that never touches hardware.

    Useful in tests when you want to exercise the save path without
    instantiating real camera/teleop objects.
    """

    def __init__(self, config: RecordingConfig, writer: Optional[Any] = None) -> None:
        from robot_data_recorder.d435 import MockD435Stream
        from robot_data_recorder.so101_teleop import MockSO101Teleop

        cam = MockD435Stream(resolution=config.resolution, fps=config.fps)
        teleop = MockSO101Teleop()
        super().__init__(config=config, camera=cam, teleop=teleop, writer=writer)
