"""
keylistener
===========

Non-blocking single-key listener for Unix terminals.

Used by :class:`robot_data_recorder.recorder.RecordingSession` to let the
operator end an episode (or abort the whole session) with a single key
press instead of relying on a fixed ``max_steps`` ceiling.

Falls back to a no-op listener when stdin is not a tty (pipes, test
harnesses, daemonised launchers). The recorder then relies on
``max_steps`` as before.
"""

from __future__ import annotations

import os
import select
import sys
from typing import Optional


class KeyListener:
    """Context manager that puts the controlling tty in cbreak mode and
    polls for single keystrokes without blocking the recorder loop.

    Usage::

        with KeyListener() as kl:
            while True:
                key = kl.poll()
                if key == ' ':
                    break
    """

    def __init__(self, stream: Optional[object] = None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._fd: Optional[int]
        try:
            self._fd = self._stream.fileno()  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError):
            self._fd = None
        self._is_tty = bool(self._fd is not None and os.isatty(self._fd))
        self._old_settings: Optional[list] = None

    @property
    def is_tty(self) -> bool:
        """True when the stream is a real terminal we can put in cbreak mode."""
        return self._is_tty

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "KeyListener":
        if not self._is_tty:
            return self
        import termios  # noqa: PLC0415
        import tty  # noqa: PLC0415

        self._old_settings = termios.tcgetattr(self._fd)
        try:
            tty.setcbreak(self._fd)
        except Exception:
            # Best effort — restore and disable polling if cbreak failed.
            self._restore()
            self._is_tty = False
        return self

    def __exit__(self, *_: object) -> None:
        self._restore()

    def _restore(self) -> None:
        if self._old_settings is not None and self._fd is not None:
            try:
                import termios  # noqa: PLC0415

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #

    def poll(self, timeout: float = 0.0) -> Optional[str]:
        """Return the next pending key, or ``None`` if nothing is buffered.

        Parameters
        ----------
        timeout:
            Seconds to wait for a key. Default ``0`` is fully non-blocking.
        """
        if not self._is_tty or self._fd is None:
            return None
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        try:
            ch = os.read(self._fd, 1)
        except OSError:
            return None
        if not ch:
            return None
        try:
            return ch.decode("utf-8", errors="replace")
        except Exception:
            return None


class NullKeyListener:
    """No-op stand-in for tests and non-tty environments."""

    is_tty = False

    def __enter__(self) -> "NullKeyListener":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def poll(self, timeout: float = 0.0) -> Optional[str]:
        return None
