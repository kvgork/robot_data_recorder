"""
test_dual_writer.py — DualWriter tests using monkeypatched backends.

All lerobot and stable_worldmodel calls are replaced with lightweight
in-memory fakes.  No optional deps required.

Tests:
- format=parquet only writes Parquet, skips HDF5
- format=hdf5 only writes HDF5, skips Parquet
- format=dual writes both
- ImportError raised cleanly when lerobot missing in parquet/dual mode
- ImportError raised cleanly when stable_worldmodel missing in hdf5/dual mode
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_data_recorder.config import RecordingConfig
from robot_data_recorder.schema import validate_episode_buffer


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_episode(n: int = 5) -> dict:
    return {
        "pixels": np.zeros((n, 480, 640, 3), dtype=np.uint8),
        "action": np.zeros((n, 7), dtype=np.float32),
        "state": np.zeros((n, 7), dtype=np.float32),
        "proprio": np.zeros((n, 7), dtype=np.float32),
        "done": np.array([False] * (n - 1) + [True], dtype=bool),
        "timestamp": np.arange(n, dtype=np.float32) / 30.0,
        "episode_idx": np.zeros(n, dtype=np.int64),
        "step_idx": np.arange(n, dtype=np.int64),
        "reward": np.zeros(n, dtype=np.float32),
    }


def _make_mock_lerobot_dataset() -> MagicMock:
    """Return a mock LeRobotDataset that records calls."""
    ds = MagicMock()
    ds.add_frame = MagicMock()
    ds.save_episode = MagicMock()
    ds.finalize = MagicMock()
    return ds


def _make_mock_hdf5_writer() -> MagicMock:
    """Return a mock HDF5Writer."""
    w = MagicMock()
    w.write_episode = MagicMock()
    w.__enter__ = MagicMock(return_value=w)
    w.__exit__ = MagicMock(return_value=False)
    return w


def _patch_lerobot(mock_ds: MagicMock):
    """Context manager that patches lerobot.LeRobotDataset.create."""
    lerobot_module = MagicMock()
    dataset_module = MagicMock()
    dataset_module.LeRobotDataset = MagicMock()
    dataset_module.LeRobotDataset.create = MagicMock(return_value=mock_ds)
    return patch.dict(
        "sys.modules",
        {
            "lerobot": lerobot_module,
            "lerobot.common": MagicMock(),
            "lerobot.common.datasets": MagicMock(),
            "lerobot.common.datasets.lerobot_dataset": dataset_module,
        },
    )


def _patch_stable_worldmodel(mock_writer: MagicMock):
    """Context manager that patches stable_worldmodel.data.HDF5Writer."""
    swm_module = MagicMock()
    data_module = MagicMock()
    data_module.HDF5Writer = MagicMock(return_value=mock_writer)
    return patch.dict(
        "sys.modules",
        {
            "stable_worldmodel": swm_module,
            "stable_worldmodel.data": data_module,
        },
    )


# ------------------------------------------------------------------ #
# format=parquet — writes LeRobot, skips HDF5
# ------------------------------------------------------------------ #

def test_parquet_format_only_calls_lerobot(tmp_path: Path) -> None:
    mock_ds = _make_mock_lerobot_dataset()
    mock_writer = _make_mock_hdf5_writer()

    cfg = RecordingConfig(
        repo_id="test/parquet",
        format="parquet",
        output_dir=str(tmp_path),
    )

    import robot_data_recorder.dual_writer as dw_mod

    with _patch_lerobot(mock_ds), _patch_stable_worldmodel(mock_writer):
        with (
            patch.object(dw_mod, "_HAS_LEROBOT", True),
            patch.object(dw_mod, "_HAS_STABLE_WORLDMODEL", False),
        ):
            from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415

            writer = DualWriter.__new__(DualWriter)
            writer._config = cfg
            writer._lerobot_dataset = mock_ds
            writer._hdf5_writer = None
            writer._episode_count = 0
            writer._output_paths = {"parquet": tmp_path / "test__parquet", "hdf5": None}

            ep = _make_episode()
            writer._write_lerobot(ep)

    mock_ds.add_frame.assert_called()
    mock_ds.save_episode.assert_called_once()
    mock_writer.write_episode.assert_not_called()


# ------------------------------------------------------------------ #
# format=hdf5 — writes HDF5, skips LeRobot
# ------------------------------------------------------------------ #

def test_hdf5_format_only_calls_hdf5writer(tmp_path: Path) -> None:
    mock_ds = _make_mock_lerobot_dataset()
    mock_writer = _make_mock_hdf5_writer()

    import robot_data_recorder.dual_writer as dw_mod

    with _patch_stable_worldmodel(mock_writer):
        with (
            patch.object(dw_mod, "_HAS_LEROBOT", False),
            patch.object(dw_mod, "_HAS_STABLE_WORLDMODEL", True),
        ):
            from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415

            writer = DualWriter.__new__(DualWriter)
            writer._config = RecordingConfig(
                repo_id="test/hdf5",
                format="hdf5",
                output_dir=str(tmp_path),
            )
            writer._lerobot_dataset = None
            writer._hdf5_writer = mock_writer
            writer._episode_count = 0
            writer._output_paths = {"parquet": None, "hdf5": tmp_path / "test__hdf5.h5"}

            ep = _make_episode()
            writer._write_hdf5(ep)

    mock_writer.write_episode.assert_called_once_with(ep)
    mock_ds.add_frame.assert_not_called()


# ------------------------------------------------------------------ #
# format=dual — writes both
# ------------------------------------------------------------------ #

def test_dual_format_calls_both(tmp_path: Path) -> None:
    mock_ds = _make_mock_lerobot_dataset()
    mock_writer = _make_mock_hdf5_writer()

    import robot_data_recorder.dual_writer as dw_mod

    with _patch_lerobot(mock_ds), _patch_stable_worldmodel(mock_writer):
        with (
            patch.object(dw_mod, "_HAS_LEROBOT", True),
            patch.object(dw_mod, "_HAS_STABLE_WORLDMODEL", True),
        ):
            from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415

            writer = DualWriter.__new__(DualWriter)
            writer._config = RecordingConfig(
                repo_id="test/dual",
                format="dual",
                output_dir=str(tmp_path),
            )
            writer._lerobot_dataset = mock_ds
            writer._hdf5_writer = mock_writer
            writer._episode_count = 0
            writer._output_paths = {
                "parquet": tmp_path / "test__dual",
                "hdf5": tmp_path / "test__dual.h5",
            }

            ep = _make_episode()
            writer.write_episode(ep)

    mock_ds.add_frame.assert_called()
    mock_ds.save_episode.assert_called_once()
    mock_writer.write_episode.assert_called_once()


# ------------------------------------------------------------------ #
# ImportError for lerobot missing in parquet/dual mode
# ------------------------------------------------------------------ #

def test_write_lerobot_raises_importerror_when_lerobot_missing(tmp_path: Path) -> None:
    import robot_data_recorder.dual_writer as dw_mod

    with patch.object(dw_mod, "_HAS_LEROBOT", False):
        from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415

        writer = DualWriter.__new__(DualWriter)
        writer._config = RecordingConfig(
            repo_id="test/lerobot-missing",
            format="parquet",
            output_dir=str(tmp_path),
        )
        writer._lerobot_dataset = None
        writer._hdf5_writer = None
        writer._episode_count = 0
        writer._output_paths = {}

        ep = _make_episode()
        with pytest.raises(ImportError, match="lerobot is required"):
            writer._write_lerobot(ep)


# ------------------------------------------------------------------ #
# HDF5 writer is now local — does NOT require stable_worldmodel
# ------------------------------------------------------------------ #

def test_hdf5_writer_works_without_stable_worldmodel(tmp_path: Path) -> None:
    """The local _HDF5EpisodeWriter only needs h5py; stable_worldmodel is read-only."""
    import robot_data_recorder.dual_writer as dw_mod

    cfg = RecordingConfig(
        repo_id="test/swm-missing",
        format="hdf5",
        output_dir=str(tmp_path),
    )
    with patch.object(dw_mod, "_HAS_STABLE_WORLDMODEL", False):
        writer = dw_mod.DualWriter(cfg)
        ep = _make_episode()
        writer.write_episode(ep)
        paths = writer.finalize()

    assert "hdf5" in paths
    assert paths["hdf5"].exists()


# ------------------------------------------------------------------ #
# Schema validation is called inside write_episode
# ------------------------------------------------------------------ #

def test_write_episode_validates_schema(tmp_path: Path) -> None:
    mock_ds = _make_mock_lerobot_dataset()
    mock_writer = _make_mock_hdf5_writer()

    import robot_data_recorder.dual_writer as dw_mod

    with _patch_lerobot(mock_ds), _patch_stable_worldmodel(mock_writer):
        with (
            patch.object(dw_mod, "_HAS_LEROBOT", True),
            patch.object(dw_mod, "_HAS_STABLE_WORLDMODEL", True),
        ):
            from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415

            writer = DualWriter.__new__(DualWriter)
            writer._config = RecordingConfig(
                repo_id="test/validation",
                format="dual",
                output_dir=str(tmp_path),
            )
            writer._lerobot_dataset = mock_ds
            writer._hdf5_writer = mock_writer
            writer._episode_count = 0
            writer._output_paths = {}

            bad_ep = {"pixels": np.zeros((5, 480, 640, 3), dtype=np.uint8)}
            with pytest.raises(ValueError, match="missing required fields"):
                writer.write_episode(bad_ep)
