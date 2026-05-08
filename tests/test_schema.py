"""
test_schema.py — Unit tests for schema.py.

Covers:
- EpisodeSchema dataclass structure
- validate_episode_buffer happy path
- validate_episode_buffer failure cases (4 types)
- compute_ep_offset correctness
- lerobot_features_dict shape output
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_data_recorder.schema import (
    EpisodeSchema,
    compute_ep_offset,
    lerobot_features_dict,
    validate_episode_buffer,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_valid_episode(n: int = 10) -> dict:
    """Return a minimal valid episode buffer with n steps."""
    return {
        "pixels": np.zeros((n, 480, 640, 3), dtype=np.uint8),
        "action": np.zeros((n, 7), dtype=np.float32),
        "state": np.zeros((n, 7), dtype=np.float32),
        "proprio": np.zeros((n, 7), dtype=np.float32),
        "done": np.zeros(n, dtype=bool),
        "timestamp": np.zeros(n, dtype=np.float32),
        "episode_idx": np.zeros(n, dtype=np.int64),
        "step_idx": np.arange(n, dtype=np.int64),
        "reward": np.zeros(n, dtype=np.float32),
    }


# ------------------------------------------------------------------ #
# EpisodeSchema dataclass
# ------------------------------------------------------------------ #

def test_schema_step_fields_declared() -> None:
    schema = EpisodeSchema()
    assert "pixels" in schema.STEP_FIELDS
    assert "action" in schema.STEP_FIELDS
    assert "state" in schema.STEP_FIELDS
    assert "proprio" in schema.STEP_FIELDS
    assert "done" in schema.STEP_FIELDS
    assert "timestamp" in schema.STEP_FIELDS
    assert "episode_idx" in schema.STEP_FIELDS
    assert "step_idx" in schema.STEP_FIELDS
    assert "reward" in schema.STEP_FIELDS


def test_schema_ep_fields_declared() -> None:
    schema = EpisodeSchema()
    assert "ep_len" in schema.EP_FIELDS
    assert "ep_offset" in schema.EP_FIELDS


def test_schema_pixels_dtype() -> None:
    schema = EpisodeSchema()
    dtype, ndim = schema.STEP_FIELDS["pixels"]
    assert dtype == "uint8"
    assert ndim == 4  # T, H, W, C


def test_schema_action_dtype() -> None:
    schema = EpisodeSchema()
    dtype, ndim = schema.STEP_FIELDS["action"]
    assert dtype == "float32"
    assert ndim == 2


# ------------------------------------------------------------------ #
# validate_episode_buffer — happy path
# ------------------------------------------------------------------ #

def test_validate_happy_path() -> None:
    ep = _make_valid_episode(5)
    validate_episode_buffer(ep)  # must not raise


def test_validate_minimum_one_step() -> None:
    ep = _make_valid_episode(1)
    validate_episode_buffer(ep)


# ------------------------------------------------------------------ #
# validate_episode_buffer — failure cases
# ------------------------------------------------------------------ #

def test_validate_missing_key() -> None:
    ep = _make_valid_episode(5)
    del ep["action"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_episode_buffer(ep)


def test_validate_wrong_shape_ndim() -> None:
    ep = _make_valid_episode(5)
    # pixels must be 4-D; give it 3-D
    ep["pixels"] = np.zeros((5, 480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        validate_episode_buffer(ep)


def test_validate_wrong_dtype() -> None:
    ep = _make_valid_episode(5)
    # action must be float32; give it int32
    ep["action"] = np.zeros((5, 7), dtype=np.int32)
    with pytest.raises(ValueError, match="dtype"):
        validate_episode_buffer(ep)


def test_validate_mismatched_lengths() -> None:
    ep = _make_valid_episode(5)
    # reward has 7 steps instead of 5
    ep["reward"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="inconsistent step counts"):
        validate_episode_buffer(ep)


def test_validate_not_dict_raises() -> None:
    with pytest.raises(ValueError, match="dict"):
        validate_episode_buffer([1, 2, 3])  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# compute_ep_offset
# ------------------------------------------------------------------ #

def test_compute_ep_offset_basic() -> None:
    offsets = compute_ep_offset([10, 20, 15])
    expected = np.array([0, 10, 30], dtype=np.int64)
    np.testing.assert_array_equal(offsets, expected)


def test_compute_ep_offset_single() -> None:
    offsets = compute_ep_offset([42])
    np.testing.assert_array_equal(offsets, np.array([0], dtype=np.int64))


def test_compute_ep_offset_empty() -> None:
    offsets = compute_ep_offset([])
    assert offsets.shape == (0,)
    assert offsets.dtype == np.int64


def test_compute_ep_offset_dtype() -> None:
    offsets = compute_ep_offset([5, 10])
    assert offsets.dtype == np.int64


# ------------------------------------------------------------------ #
# lerobot_features_dict
# ------------------------------------------------------------------ #

def test_lerobot_features_dict_keys() -> None:
    feats = lerobot_features_dict(action_dim=7, state_dim=7, image_shape=(3, 480, 640))
    assert "observation.images.d435_rgb" in feats
    assert "observation.state" in feats
    assert "action" in feats


def test_lerobot_features_dict_image_shape() -> None:
    feats = lerobot_features_dict(action_dim=7, state_dim=7, image_shape=(3, 480, 640))
    assert feats["observation.images.d435_rgb"]["shape"] == (3, 480, 640)


def test_lerobot_features_dict_action_dim() -> None:
    feats = lerobot_features_dict(action_dim=7, state_dim=6, image_shape=(3, 480, 640))
    assert feats["action"]["shape"] == (7,)
    assert feats["observation.state"]["shape"] == (6,)


def test_lerobot_features_dict_image_dtype() -> None:
    feats = lerobot_features_dict(action_dim=7, state_dim=7, image_shape=(3, 480, 640))
    assert feats["observation.images.d435_rgb"]["dtype"] == "image"
