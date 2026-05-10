"""
test_keylistener.py — KeyListener / NullKeyListener unit tests.

The real KeyListener requires a tty; tests therefore exercise the
NullKeyListener fallback and the auto-fallback inside KeyListener
when stdin is not a terminal (the case under pytest).
"""

from __future__ import annotations

import io

from robot_data_recorder.keylistener import KeyListener, NullKeyListener


def test_null_listener_poll_returns_none() -> None:
    kl = NullKeyListener()
    with kl as kl_inner:
        assert kl_inner.poll() is None
        assert kl_inner.is_tty is False


def test_keylistener_falls_back_when_not_a_tty() -> None:
    # Pytest captures stdout/stderr/stdin so the underlying stream is not a tty
    kl = KeyListener()
    with kl as kl_inner:
        assert kl_inner.is_tty is False
        assert kl_inner.poll() is None


def test_keylistener_handles_stream_without_fileno() -> None:
    fake = io.StringIO("hello")  # raises on fileno
    kl = KeyListener(stream=fake)
    with kl as kl_inner:
        assert kl_inner.is_tty is False
        assert kl_inner.poll() is None
