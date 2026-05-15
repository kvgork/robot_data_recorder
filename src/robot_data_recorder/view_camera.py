"""
view_camera
===========

Lightweight live viewer for the RealSense D435 using Tkinter + Pillow.

Opens a Tk window streaming the colour frame (and optionally depth).
Press ``q`` or ESC to quit.

CLI::

    pixi run view-cam
    pixi run view-cam -- --depth
    pixi run view-cam -- --serial 123456789 --fps 60
    pixi run view-cam -- --mock          # synthetic frames, no hardware
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from typing import Optional

import numpy as np

from robot_data_recorder.d435 import make_d435


def _parse_resolution(s: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in s.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must be WIDTHxHEIGHT (e.g. 640x480), got {s!r}"
        ) from exc
    return w, h


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a uint16 depth array to an RGB uint8 visualisation (JET-like)."""
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    dmin = int(depth[valid].min())
    dmax = int(depth.max())
    scale = 255.0 / max(dmax - dmin, 1)
    norm = np.clip((depth.astype(np.int32) - dmin) * scale, 0, 255).astype(np.uint8)
    norm[~valid] = 0

    # JET-like colormap without OpenCV
    r = np.clip(1.5 - np.abs(norm / 255.0 * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(norm / 255.0 * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(norm / 255.0 * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


class Viewer:
    def __init__(
        self,
        cam,
        resolution: tuple[int, int],
        enable_depth: bool,
        title_extra: str = "",
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont, ImageTk  # noqa: PLC0415

        self._cam = cam
        self._enable_depth = enable_depth
        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageFont = ImageFont
        self._ImageTk = ImageTk

        self._root = tk.Tk()
        self._root.title(f"RealSense D435{title_extra}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Key>", self._on_key)
        self._root.geometry(
            f"+{100}+{100}"
            if not enable_depth
            else f"{resolution[0] * 2 + 20}x{resolution[1] + 20}+100+100"
        )

        self._label_rgb = tk.Label(self._root)
        self._label_rgb.pack(side=tk.LEFT, padx=4, pady=4)

        self._label_depth: Optional[tk.Label] = None
        if enable_depth:
            self._label_depth = tk.Label(self._root)
            self._label_depth.pack(side=tk.LEFT, padx=4, pady=4)

        try:
            self._font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except OSError:
            self._font = ImageFont.load_default()

        self._stopping = False
        self._photo_rgb = None
        self._photo_depth = None
        self._frame_count = 0
        self._t_last = time.monotonic()
        self._fps_display = 0.0

    def _on_close(self) -> None:
        self._stopping = True

    def _on_key(self, event: "tk.Event") -> None:
        if event.keysym in ("q", "Escape"):
            self._stopping = True

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        if now - self._t_last >= 1.0:
            self._fps_display = self._frame_count / (now - self._t_last)
            self._t_last = now
            self._frame_count = 0

    def _tick(self) -> None:
        if self._stopping:
            self._root.destroy()
            return
        try:
            frame = self._cam.read_frame()
        except Exception as exc:
            print(f"read_frame error: {exc}", file=sys.stderr)
            self._stopping = True
            self._root.after(10, self._tick)
            return

        rgb = frame["rgb"]
        self._update_fps()

        img = self._Image.fromarray(rgb)
        draw = self._ImageDraw.Draw(img)
        draw.text(
            (10, 6),
            f"{self._fps_display:5.1f} fps  (q/ESC quit)",
            font=self._font,
            fill=(0, 255, 0),
        )
        self._photo_rgb = self._ImageTk.PhotoImage(img)
        self._label_rgb.configure(image=self._photo_rgb)

        if self._enable_depth and frame.get("depth") is not None and self._label_depth:
            depth_rgb = _colorize_depth(frame["depth"])
            dimg = self._Image.fromarray(depth_rgb)
            self._photo_depth = self._ImageTk.PhotoImage(dimg)
            self._label_depth.configure(image=self._photo_depth)

        self._root.after(1, self._tick)

    def run(self) -> None:
        self._cam.start()
        self._root.after(1, self._tick)
        self._root.mainloop()


def run(
    serial: Optional[str],
    resolution: tuple[int, int],
    fps: int,
    enable_depth: bool,
    mock: bool,
) -> int:
    try:
        from PIL import Image  # noqa: F401, PLC0415
    except ImportError as exc:
        print(f"Pillow required for tkinter viewer: {exc}", file=sys.stderr)
        return 1

    cam = make_d435(
        serial=serial,
        resolution=resolution,
        fps=fps,
        enable_depth=enable_depth,
        mock=mock,
    )

    title_extra = f"  {resolution[0]}x{resolution[1]}@{fps}fps"
    if mock:
        title_extra += "  [mock]"
    print(
        f"streaming {resolution[0]}x{resolution[1]} @ {fps}fps"
        f"{' + depth' if enable_depth else ''}"
        f"{' [mock]' if mock else ''} — close window or press q/ESC to quit",
        flush=True,
    )

    viewer = Viewer(cam, resolution, enable_depth, title_extra=title_extra)
    try:
        viewer.run()
    finally:
        cam.stop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="robot-data-view",
        description="Live RealSense D435 viewer (Tkinter window).",
    )
    p.add_argument(
        "--serial",
        default=os.environ.get("LERO_CAM_SERIAL"),
        help="Camera serial number (default: $LERO_CAM_SERIAL or first device).",
    )
    p.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(640, 480),
        help="Frame resolution WIDTHxHEIGHT (default: 640x480).",
    )
    p.add_argument("--fps", type=int, default=30, help="Target frame rate (default: 30).")
    p.add_argument("--depth", action="store_true", help="Also stream + display depth.")
    p.add_argument("--mock", action="store_true", help="Use MockD435Stream (no hardware).")
    args = p.parse_args(argv)

    return run(
        serial=args.serial,
        resolution=args.resolution,
        fps=args.fps,
        enable_depth=args.depth,
        mock=args.mock,
    )


if __name__ == "__main__":
    sys.exit(main())
