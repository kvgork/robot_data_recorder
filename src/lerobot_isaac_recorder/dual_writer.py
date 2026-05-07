"""
dual_writer
===========

Parallel LeRobot Parquet + LeWM HDF5 writer.

Both ``lerobot`` and ``stable_worldmodel`` are soft-imported.  Missing
a backend raises a clear ``ImportError`` only when ``write_episode()``
is called with a format that needs it — not at import time.

Architecture (Path B from research synthesis):
- After each episode, call ``lerobot_dataset.save_episode()`` AND
  ``HDF5Writer.write_episode(ep)``.
- Both writers operate on the same in-memory episode dict.
- ``finalize()`` closes both writers and returns output paths.

Usage::

    cfg = RecordingConfig(repo_id="koen/pickplace", format="dual")
    writer = DualWriter(cfg)
    writer.write_episode(ep_dict)
    paths = writer.finalize()
    # paths == {"parquet": Path(...), "hdf5": Path(...)}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from lerobot_isaac_recorder.config import RecordingConfig
from lerobot_isaac_recorder.schema import (
    SCHEMA,
    lerobot_features_dict,
    validate_episode_buffer,
)

# Soft-import lerobot -------------------------------------------------------- #
try:
    import lerobot  # type: ignore[import-untyped]  # noqa: F401

    _HAS_LEROBOT = True
except ImportError:
    lerobot = None  # type: ignore[assignment]
    _HAS_LEROBOT = False

# Soft-import stable_worldmodel ---------------------------------------------- #
try:
    import stable_worldmodel  # type: ignore[import-untyped]  # noqa: F401

    _HAS_STABLE_WORLDMODEL = True
except ImportError:
    stable_worldmodel = None  # type: ignore[assignment]
    _HAS_STABLE_WORLDMODEL = False
# --------------------------------------------------------------------------- #

_FORMATS = {"parquet", "hdf5", "dual"}


class DualWriter:
    """Writes episode data to LeRobot (Parquet) and/or LeWM (HDF5).

    Parameters
    ----------
    config:
        ``RecordingConfig`` instance. Uses ``config.format``,
        ``config.repo_id``, and ``config.output_dir``.
    """

    def __init__(self, config: RecordingConfig) -> None:
        if config.format not in _FORMATS:
            raise ValueError(
                f"Unknown format {config.format!r}. Expected one of {_FORMATS}"
            )
        self._config = config
        self._lerobot_dataset: Any = None
        self._hdf5_writer: Any = None
        self._episode_count = 0
        self._output_paths: dict[str, Optional[Path]] = {"parquet": None, "hdf5": None}

        self._init_writers()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _init_writers(self) -> None:
        fmt = self._config.format
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if fmt in ("parquet", "dual"):
            self._init_lerobot(out_dir)

        if fmt in ("hdf5", "dual"):
            self._init_hdf5(out_dir)

    def _init_lerobot(self, out_dir: Path) -> None:
        if not _HAS_LEROBOT:
            raise ImportError(
                "lerobot is required for parquet/dual format. "
                "Install with: bash scripts/install_lerobot.sh"
            )

        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

            w, h = self._config.resolution
            features = lerobot_features_dict(
                action_dim=7,  # SO-101: 6 joints + gripper
                state_dim=7,
                image_shape=(3, h, w),  # channels-first for LeRobot
            )
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=self._config.repo_id,
                fps=self._config.fps,
                root=str(out_dir / self._config.repo_id),
                features=features,
            )
            self._output_paths["parquet"] = out_dir / self._config.repo_id
        except ImportError as exc:
            raise ImportError(
                "lerobot LeRobotDataset not found. "
                "Install with: bash scripts/install_lerobot.sh"
            ) from exc

    def _init_hdf5(self, out_dir: Path) -> None:
        if not _HAS_STABLE_WORLDMODEL:
            raise ImportError(
                "stable_worldmodel is required for hdf5/dual format. "
                "Install with: pip install stable-worldmodel"
            )

        try:
            from stable_worldmodel.data import HDF5Writer  # type: ignore[import-untyped]  # noqa: PLC0415

            safe_name = self._config.repo_id.replace("/", "__")
            h5_path = out_dir / f"{safe_name}.h5"
            self._hdf5_writer = HDF5Writer(str(h5_path))
            self._output_paths["hdf5"] = h5_path
        except ImportError as exc:
            raise ImportError(
                "stable_worldmodel.data.HDF5Writer not found. "
                "Install with: pip install stable-worldmodel"
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def write_episode(self, ep: dict[str, Any]) -> None:
        """Validate and write one episode buffer to the configured backends.

        Parameters
        ----------
        ep:
            Episode dict with per-step numpy arrays. See ``schema.py`` for the
            expected fields and dtypes.

        Raises
        ------
        ValueError
            If the episode buffer fails schema validation.
        ImportError
            If a required backend library is not installed.
        """
        validate_episode_buffer(ep)

        fmt = self._config.format
        if fmt in ("parquet", "dual"):
            self._write_lerobot(ep)
        if fmt in ("hdf5", "dual"):
            self._write_hdf5(ep)

        self._episode_count += 1

    def _write_lerobot(self, ep: dict[str, Any]) -> None:
        """Write one episode to the LeRobot dataset."""
        if not _HAS_LEROBOT:
            raise ImportError(
                "lerobot is required for parquet/dual format. "
                "Install with: bash scripts/install_lerobot.sh"
            )

        dataset = self._lerobot_dataset
        n_steps = ep["pixels"].shape[0]

        for t in range(n_steps):
            frame: dict[str, Any] = {
                "observation.images.d435_rgb": ep["pixels"][t],
                "observation.state": ep["state"][t],
                "action": ep["action"][t],
                "timestamp": float(ep["timestamp"][t]),
            }
            dataset.add_frame(frame)

        dataset.save_episode(task=self._config.task)

    def _write_hdf5(self, ep: dict[str, Any]) -> None:
        """Write one episode to the stable-worldmodel HDF5 file."""
        if not _HAS_STABLE_WORLDMODEL:
            raise ImportError(
                "stable_worldmodel is required for hdf5/dual format. "
                "Install with: pip install stable-worldmodel"
            )

        # HDF5Writer expects ep as {col: list_of_step_arrays} or a contiguous dict
        self._hdf5_writer.write_episode(ep)

    def finalize(self) -> dict[str, Path]:
        """Close both writers and return output paths.

        Returns
        -------
        dict
            ``{"parquet": Path | None, "hdf5": Path | None}``
            depending on format.
        """
        fmt = self._config.format

        if fmt in ("parquet", "dual") and self._lerobot_dataset is not None:
            try:
                self._lerobot_dataset.finalize()
            except Exception:
                pass  # best effort

        if fmt in ("hdf5", "dual") and self._hdf5_writer is not None:
            try:
                self._hdf5_writer.__exit__(None, None, None)
            except Exception:
                try:
                    self._hdf5_writer.close()
                except Exception:
                    pass

        return {k: v for k, v in self._output_paths.items() if v is not None}
