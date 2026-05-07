"""
conftest.py — package-level pytest configuration for lerobot-isaac-recorder.

Registers markers so tests can be skipped selectively when heavy deps
(pyrealsense2, lerobot, stable_worldmodel) are not installed.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_realsense: test requires pyrealsense2 and a connected D435 camera",
    )
    config.addinivalue_line(
        "markers",
        "requires_lerobot: test requires the lerobot package and its dependencies",
    )
    config.addinivalue_line(
        "markers",
        "requires_stable_worldmodel: test requires the stable-worldmodel package",
    )
