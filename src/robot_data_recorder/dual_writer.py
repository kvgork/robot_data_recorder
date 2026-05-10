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

    cfg = RecordingConfig(repo_id="myuser/pickplace", format="dual")
    writer = DualWriter(cfg)
    writer.write_episode(ep_dict)
    paths = writer.finalize()
    # paths == {"parquet": Path(...), "hdf5": Path(...)}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from robot_data_recorder.config import RecordingConfig
from robot_data_recorder.schema import (
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
            # lerobot >=0.4.4 dropped the `lerobot.common.*` namespace
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

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
        # stable_worldmodel ships HDF5Dataset (read-only). The recorder uses a
        # local writer that produces files compatible with that loader.
        safe_name = self._config.repo_id.replace("/", "__")
        h5_path = out_dir / f"{safe_name}.h5"
        self._hdf5_writer = _HDF5EpisodeWriter(h5_path)
        self._output_paths["hdf5"] = h5_path

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
                "task": self._config.task,
            }
            dataset.add_frame(frame)

        # lerobot >=0.4 takes task per frame; save_episode no longer accepts it
        dataset.save_episode()

    def _write_hdf5(self, ep: dict[str, Any]) -> None:
        """Write one episode to the LeWM-compatible HDF5 file."""
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
                self._hdf5_writer.close()
            except Exception:
                pass

        return {k: v for k, v in self._output_paths.items() if v is not None}


# --------------------------------------------------------------------------- #
# Local HDF5 writer (LeWM-compatible)
# --------------------------------------------------------------------------- #


class _HDF5EpisodeWriter:
    """Append-mode HDF5 writer producing files readable by ``stable_worldmodel.data.HDF5Dataset``.

    File layout (matches the ``SyncWorld.record_dataset`` schema):
    - One resizable dataset per per-step field in :data:`SCHEMA.STEP_FIELDS`.
    - ``ep_offset`` (int64) and ``ep_len`` (int32) index arrays.
    - Datasets are created lazily on the first :meth:`write_episode` call so
      shapes can be derived from real data.
    """

    def __init__(self, h5_path: Path) -> None:
        import h5py  # noqa: PLC0415

        self._h5py = h5py
        self._path = Path(h5_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(
            self._path,
            mode="w",
            libver="latest",
            fs_strategy="page",
            fs_page_size=4 * 1024 * 1024,
        )
        self._file.swmr_mode = True
        self._initialised = False
        self._global_ptr = 0

    def write_episode(self, ep: dict[str, Any]) -> None:
        n_steps = int(ep["pixels"].shape[0])
        if n_steps == 0:
            return

        if not self._initialised:
            self._init_datasets(ep)
            self._initialised = True

        for key in SCHEMA.STEP_FIELDS:
            ds = self._file[key]
            curr = ds.shape[0]
            ds.resize(curr + n_steps, axis=0)
            ds[curr:] = np.asarray(ep[key])

        ep_offset = self._file["ep_offset"]
        ep_len = self._file["ep_len"]
        idx = ep_offset.shape[0]
        ep_offset.resize(idx + 1, axis=0)
        ep_len.resize(idx + 1, axis=0)
        ep_offset[idx] = self._global_ptr
        ep_len[idx] = n_steps

        self._global_ptr += n_steps
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None  # type: ignore[assignment]

    def _init_datasets(self, ep: dict[str, Any]) -> None:
        h5py = self._h5py
        for key, (dtype_str, _) in SCHEMA.STEP_FIELDS.items():
            sample = np.asarray(ep[key][0])
            shape = (0,) + sample.shape
            maxshape = (None,) + sample.shape
            chunks: tuple[int, ...]
            if sample.ndim >= 2:
                chunks = (32,) + sample.shape
            else:
                chunks = (1024,) + sample.shape
            self._file.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=np.dtype(dtype_str),
                chunks=chunks,
            )

        self._file.create_dataset(
            "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._file.create_dataset(
            "ep_len", shape=(0,), maxshape=(None,), dtype=np.int32
        )
