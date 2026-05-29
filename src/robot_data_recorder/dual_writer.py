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
        self._finalized = False

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
                "Install with: pip install lerobot"
            )

        try:
            # lerobot >=0.4.4 dropped the `lerobot.common.*` namespace
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

            w, h = self._config.resolution
            features = lerobot_features_dict(
                action_dim=6,  # SO-101: 5 joints + gripper (lerobot motor count)
                state_dim=12,  # joint_pos[6] + joint_vel[6] — matches trainer
                # Channels-last (HWC) to match the uint8 frames the recorder
                # feeds to add_frame; declaring CHW here would make the declared
                # feature shape disagree with the data and trip add_frame's
                # shape validation.
                image_shape=(h, w, 3),
                camera_key=self._config.camera_name,
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
                "Install with: pip install lerobot"
            ) from exc

    def _init_hdf5(self, out_dir: Path) -> None:
        # stable_worldmodel ships HDF5Dataset (read-only). The recorder uses a
        # local writer that produces files compatible with that loader.
        safe_name = self._config.repo_id.replace("/", "__")
        h5_path = out_dir / f"{safe_name}.h5"
        self._hdf5_writer = _HDF5EpisodeWriter(
            h5_path, metadata=self._hdf5_metadata()
        )
        self._output_paths["hdf5"] = h5_path

    def _hdf5_metadata(self) -> dict[str, Any]:
        """Build file-level metadata for the HDF5 output.

        A world-model trainer needs frame rate and action/observation
        semantics to consume the file standalone; the raw datasets carry none
        of this, so it is written into the file's root ``attrs``.
        """
        from robot_data_recorder.so101_teleop import _SO101_MOTORS  # noqa: PLC0415

        w, h = self._config.resolution
        n_motors = len(_SO101_MOTORS)
        return {
            "fps": int(self._config.fps),
            "task": str(self._config.task),
            "repo_id": str(self._config.repo_id),
            "robot": "so101",
            "camera_name": str(self._config.camera_name),
            "motor_names": list(_SO101_MOTORS),
            "image_layout": "HWC",
            "image_height": int(h),
            "image_width": int(w),
            "image_channels": 3,
            "action_dim": n_motors,                 # joint position targets
            "state_dim": 2 * n_motors,              # joint_pos + joint_vel
            "state_layout": "joint_pos[6]+joint_vel[6]",
            "reward_convention": "sparse_terminal_success",
            "schema_version": 1,
        }

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
                "Install with: pip install lerobot"
            )

        dataset = self._lerobot_dataset
        n_steps = ep["pixels"].shape[0]
        reward = ep["reward"]
        done = ep["done"]
        cam_key = f"observation.images.{self._config.camera_name}"

        for t in range(n_steps):
            # lerobot >=0.4 validates frames against `features` only. The
            # default features (timestamp, frame_index, episode_index, ...)
            # are auto-populated by add_frame from `frame_index / fps`, so
            # passing `timestamp` here triggers an "Extra features" error.
            #
            # next.reward / next.done carry the operator's success label and
            # episode termination into the Parquet dataset. These are the exact
            # columns the world-model bridge reads to fill its `rewards` and
            # `dones` arrays; the recorder writes a sparse terminal reward (1.0
            # on the last frame of a successful episode, 0.0 otherwise), so a
            # trainer can also filter successes by reward > 0. 1-element arrays
            # match the layout LeRobot expects for scalar `next.*` features.
            frame: dict[str, Any] = {
                cam_key: ep["pixels"][t],
                "observation.state": ep["state"][t],
                "action": ep["action"][t],
                "next.reward": np.array([reward[t]], dtype=np.float32),
                "next.done": np.array([bool(done[t])], dtype=np.bool_),
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
        out = {k: v for k, v in self._output_paths.items() if v is not None}

        # finalize() is called from both RecordingSession.__exit__ and the CLI;
        # closing the writers twice can make the backends raise. Make the second
        # call a no-op that still returns the output paths.
        if getattr(self, "_finalized", False):
            return out
        self._finalized = True

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

        return out


# --------------------------------------------------------------------------- #
# Local HDF5 writer (LeWM-compatible)
# --------------------------------------------------------------------------- #


class _HDF5EpisodeWriter:
    """Append-mode HDF5 writer producing files readable by ``stable_worldmodel.data.HDF5Dataset``.

    File layout (matches the ``SyncWorld.record_dataset`` schema):
    - One resizable dataset per per-step field in :data:`SCHEMA.STEP_FIELDS`,
      plus any optional fields (e.g. ``depth``) present in the first episode.
    - ``ep_offset`` (int64) and ``ep_len`` (int32) index arrays.
    - File-level training metadata (fps, task, motor names, image layout, ...)
      written into the root group ``attrs`` so the file is self-describing.
    - Datasets are created lazily on the first :meth:`write_episode` call so
      shapes can be derived from real data.

    Parameters
    ----------
    h5_path:
        Output ``.h5`` path.
    metadata:
        Optional dict written to the file's root ``attrs``. Carries the frame
        rate and action/observation semantics a world-model trainer needs to
        consume the file standalone.
    """

    def __init__(
        self, h5_path: Path, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        import h5py  # noqa: PLC0415

        self._h5py = h5py
        self._path = Path(h5_path)
        self._metadata = metadata or {}
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
        # Per-step fields actually written; fixed on first episode.
        self._step_field_keys: list[str] = []

    def write_episode(self, ep: dict[str, Any]) -> None:
        n_steps = int(ep["pixels"].shape[0])
        if n_steps == 0:
            return

        if not self._initialised:
            self._init_datasets(ep)
            self._initialised = True

        for key in self._step_field_keys:
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
        # SWMR forbids adding attributes after the first flush, so write
        # file-level metadata before any data lands.
        self._write_attrs()

        # Required step fields, plus optional fields present in this episode.
        fields: dict[str, tuple[str, int]] = dict(SCHEMA.STEP_FIELDS)
        fields.update(
            {k: v for k, v in SCHEMA.OPTIONAL_STEP_FIELDS.items() if k in ep}
        )
        self._step_field_keys = list(fields)

        for key, (dtype_str, _) in fields.items():
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

    def _write_attrs(self) -> None:
        for key, value in self._metadata.items():
            if isinstance(value, (list, tuple)):
                if any(isinstance(v, str) for v in value):
                    # h5py cannot store fixed-width unicode (e.g. "<U13"); use a
                    # variable-length UTF-8 string array instead.
                    arr = np.array(
                        list(value), dtype=self._h5py.string_dtype()
                    )
                else:
                    arr = np.asarray(value)
                self._file.attrs[key] = arr
            else:
                self._file.attrs[key] = value
