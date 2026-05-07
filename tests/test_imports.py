"""
test_imports.py — Verify clean importability without heavy optional deps.

These tests MUST pass on a plain Python environment with only h5py, numpy,
pyyaml installed.  pyrealsense2, lerobot, and stable_worldmodel must NOT
be present for these to be meaningful.
"""

from __future__ import annotations


def test_package_imports_cleanly() -> None:
    """Top-level import succeeds without optional heavy deps."""
    import lerobot_isaac_recorder  # noqa: F401


def test_version_string() -> None:
    from lerobot_isaac_recorder import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_public_exports_available() -> None:
    from lerobot_isaac_recorder import (
        D435Stream,
        DualWriter,
        EpisodeSchema,
        RecordingConfig,
        RecordingSession,
    )

    assert D435Stream is not None
    assert DualWriter is not None
    assert EpisodeSchema is not None
    assert RecordingConfig is not None
    assert RecordingSession is not None


def test_realsense_flag_accessible() -> None:
    """_HAS_REALSENSE flag is always importable regardless of pyrealsense2."""
    from lerobot_isaac_recorder.d435 import _HAS_REALSENSE

    assert isinstance(_HAS_REALSENSE, bool)


def test_lerobot_flag_accessible() -> None:
    """_HAS_LEROBOT flag is always importable regardless of lerobot."""
    from lerobot_isaac_recorder.so101_teleop import _HAS_LEROBOT

    assert isinstance(_HAS_LEROBOT, bool)


def test_dual_writer_lerobot_flag_accessible() -> None:
    from lerobot_isaac_recorder.dual_writer import _HAS_LEROBOT

    assert isinstance(_HAS_LEROBOT, bool)


def test_stable_worldmodel_flag_accessible() -> None:
    from lerobot_isaac_recorder.dual_writer import _HAS_STABLE_WORLDMODEL

    assert isinstance(_HAS_STABLE_WORLDMODEL, bool)


def test_schema_module_imports_cleanly() -> None:
    from lerobot_isaac_recorder import schema  # noqa: F401


def test_config_module_imports_cleanly() -> None:
    from lerobot_isaac_recorder import config  # noqa: F401


def test_recorder_module_imports_cleanly() -> None:
    from lerobot_isaac_recorder import recorder  # noqa: F401


def test_cli_module_imports_cleanly() -> None:
    from lerobot_isaac_recorder import cli  # noqa: F401
