"""
d435
====

Intel RealSense D435 camera stream wrapper.

``pyrealsense2`` is soft-imported: the module loads without it, but
``D435Stream.start()`` raises a clear ``ImportError`` if it is absent.

Usage (real hardware)::

    from lerobot_isaac_recorder.d435 import make_d435

    with make_d435(fps=30) as cam:
        frame = cam.read_frame()   # {'rgb': ..., 'depth': None, 'timestamp': ...}

Usage (tests / dry-run)::

    with make_d435(mock=True) as cam:
        frame = cam.read_frame()
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

# Soft-import pyrealsense2 -------------------------------------------------- #
try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    _HAS_REALSENSE = True
except ImportError:
    rs = None  # type: ignore[assignment]
    _HAS_REALSENSE = False
# --------------------------------------------------------------------------- #


class D435Stream:
    """Thin wrapper around the Intel RealSense D435 pipeline.

    Parameters
    ----------
    serial:
        Camera serial number. ``None`` selects the first available device.
    resolution:
        (width, height) in pixels. Default: (640, 480).
    fps:
        Frames per second. Default: 30.
    enable_depth:
        Capture depth stream alongside RGB. Default: ``False``.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        enable_depth: bool = False,
    ) -> None:
        self._serial = serial
        self._resolution = resolution
        self._fps = fps
        self._enable_depth = enable_depth
        self._pipeline = None
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the RealSense pipeline.

        Raises
        ------
        ImportError
            If ``pyrealsense2`` is not installed.
        RuntimeError
            If no RealSense device is found.
        """
        if not _HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 is required to use D435Stream. "
                "Install with: pip install pyrealsense2"
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()

        if self._serial:
            cfg.enable_device(self._serial)

        w, h = self._resolution
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self._fps)

        if self._enable_depth:
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, self._fps)

        self._pipeline.start(cfg)
        self._started = True

    def read_frame(self) -> dict:
        """Read one frame from the camera.

        Returns
        -------
        dict with keys:
            ``rgb``       : np.ndarray uint8  (H, W, 3)
            ``depth``     : np.ndarray uint16 (H, W) or ``None``
            ``timestamp`` : float — hardware timestamp in seconds
        """
        if not self._started:
            raise RuntimeError("Call start() before read_frame()")

        frames = self._pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        rgb = np.asarray(color_frame.get_data(), dtype=np.uint8)
        # RealSense returns BGR; convert to RGB for consistency with LeRobot
        rgb = rgb[:, :, ::-1].copy()

        depth = None
        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            depth = np.asarray(depth_frame.get_data(), dtype=np.uint16)

        ts = color_frame.get_timestamp() / 1000.0  # ms -> seconds

        return {"rgb": rgb, "depth": depth, "timestamp": ts}

    def stop(self) -> None:
        """Stop the pipeline and release hardware resources."""
        if self._started and self._pipeline is not None:
            self._pipeline.stop()
            self._started = False

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "D435Stream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class MockD435Stream:
    """Synthetic camera stream for tests and dry-run mode.

    Generates deterministic numpy frames with the correct shapes and dtypes.
    No hardware required.

    Parameters
    ----------
    serial:
        Ignored; present for API parity with D435Stream.
    resolution:
        (width, height). Default: (640, 480).
    fps:
        Ignored; present for API parity.
    enable_depth:
        Whether to include a synthetic depth array.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        enable_depth: bool = False,
    ) -> None:
        self._serial = serial
        self._resolution = resolution
        self._fps = fps
        self._enable_depth = enable_depth
        self._frame_idx = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def read_frame(self) -> dict:
        """Return a synthetic frame.

        Returns
        -------
        dict with keys:
            ``rgb``       : np.ndarray uint8  (H, W, 3) — incrementing value
            ``depth``     : np.ndarray uint16 (H, W) or ``None``
            ``timestamp`` : float
        """
        if not self._started:
            raise RuntimeError("Call start() before read_frame()")

        w, h = self._resolution
        rng = np.random.default_rng(self._frame_idx)
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

        depth = None
        if self._enable_depth:
            depth = rng.integers(500, 2000, (h, w), dtype=np.uint16)

        ts = time.monotonic()
        self._frame_idx += 1

        return {"rgb": rgb, "depth": depth, "timestamp": ts}

    def stop(self) -> None:
        self._started = False

    def __enter__(self) -> "MockD435Stream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def make_d435(
    serial: Optional[str] = None,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    enable_depth: bool = False,
    mock: bool = False,
) -> D435Stream | MockD435Stream:
    """Factory for camera streams.

    Parameters
    ----------
    serial:
        Camera serial number. ``None`` picks first device.
    resolution:
        (width, height) in pixels.
    fps:
        Target frame rate.
    enable_depth:
        Whether to enable the depth stream.
    mock:
        If ``True``, return a ``MockD435Stream`` regardless of hardware.

    Returns
    -------
    D435Stream | MockD435Stream
    """
    if mock:
        return MockD435Stream(
            serial=serial,
            resolution=resolution,
            fps=fps,
            enable_depth=enable_depth,
        )
    return D435Stream(
        serial=serial,
        resolution=resolution,
        fps=fps,
        enable_depth=enable_depth,
    )
