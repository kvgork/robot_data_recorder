"""
schema
======

Canonical superset schema for robot-data-recorder episodes.

The schema is a superset of:
- stable-worldmodel HDF5Dataset (pixels, action, reward, ep_len, ep_offset)
- tanguy-pauwels/lerobot-dataset-to-HDF5 output (adds done, timestamp, episode_idx,
  step_idx, state, proprio)

All field shapes are relative to T (total steps across all episodes) or N_ep
(number of episodes). Concrete dims depend on the hardware config:
- H, W, C: image height, width, channels (default 480, 640, 3)
- A: action dimension (default 6 for SO-101: 5 joints + gripper)
- S: state dimension (same as action for SO-101)
- P: proprio dimension (same as state for SO-101)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeSchema:
    """Declares the canonical field set for one recorded episode.

    Attributes
    ----------
    Per-step arrays (shape relative to T = number of steps in episode):
        pixels      : uint8,   (T, H, W, C)
        action      : float32, (T, A)
        state       : float32, (T, S)
        proprio     : float32, (T, P)
        done        : bool,    (T,)
        timestamp   : float32, (T,)
        episode_idx : int64,   (T,)
        step_idx    : int64,   (T,)
        reward      : float32, (T,)

    Optional per-step arrays (written only when captured):
        depth       : uint16,  (T, H, W)  — D435 depth, HDF5 only

    Per-episode arrays (shape relative to N_ep = number of episodes so far):
        ep_len      : int64,  (N_ep,)
        ep_offset   : int64,  (N_ep,)
    """

    # Per-step field metadata: (dtype_str, ndim_min)
    # ndim_min is the minimum number of axes (1 = 1-D vector, 3 = image T×H×W×C, etc.)
    STEP_FIELDS: dict[str, tuple[str, int]] = field(
        default_factory=lambda: {
            "pixels": ("uint8", 4),       # T, H, W, C
            "action": ("float32", 2),     # T, A
            "state": ("float32", 2),      # T, S
            "proprio": ("float32", 2),    # T, P
            "done": ("bool", 1),          # T
            "timestamp": ("float32", 1),  # T
            "episode_idx": ("int64", 1),  # T
            "step_idx": ("int64", 1),     # T
            "reward": ("float32", 1),     # T
        }
    )

    # Optional per-step fields: validated only when present in the buffer.
    # `depth` is HDF5-only (no LeRobot depth feature declared yet).
    OPTIONAL_STEP_FIELDS: dict[str, tuple[str, int]] = field(
        default_factory=lambda: {
            "depth": ("uint16", 3),       # T, H, W
        }
    )

    EP_FIELDS: dict[str, str] = field(
        default_factory=lambda: {
            "ep_len": "int64",
            "ep_offset": "int64",
        }
    )


# Module-level singleton
SCHEMA = EpisodeSchema()


def validate_episode_buffer(ep: dict[str, Any]) -> None:
    """Validate an episode buffer dict against the canonical schema.

    The buffer is expected to contain per-step numpy arrays for a single episode.
    Per-episode fields (ep_len, ep_offset) are NOT expected here — they are
    computed by DualWriter after all episodes are collected.

    Parameters
    ----------
    ep:
        Dict mapping field name -> numpy array.

    Raises
    ------
    ValueError
        If a required field is missing, has wrong dtype, wrong ndim, or if
        step arrays have inconsistent first-axis length.
    """
    if not isinstance(ep, dict):
        raise ValueError(f"Episode buffer must be a dict, got {type(ep)}")

    # Check required step fields
    required = list(SCHEMA.STEP_FIELDS.keys())
    missing = [k for k in required if k not in ep]
    if missing:
        raise ValueError(f"Episode buffer missing required fields: {missing}")

    # Required fields plus any optional fields actually present in the buffer.
    to_check = dict(SCHEMA.STEP_FIELDS)
    to_check.update(
        {k: v for k, v in SCHEMA.OPTIONAL_STEP_FIELDS.items() if k in ep}
    )

    lengths: dict[str, int] = {}
    for field_name, (expected_dtype, expected_ndim) in to_check.items():
        arr = ep[field_name]
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"Field '{field_name}' must be a numpy ndarray, got {type(arr)}"
            )

        # Check dtype
        actual_dtype = arr.dtype
        if not np.issubdtype(actual_dtype, np.dtype(expected_dtype).type):
            raise ValueError(
                f"Field '{field_name}' has dtype {actual_dtype}, "
                f"expected {expected_dtype}"
            )

        # Check ndim
        if arr.ndim < expected_ndim:
            raise ValueError(
                f"Field '{field_name}' has ndim {arr.ndim}, "
                f"expected at least {expected_ndim}"
            )

        lengths[field_name] = arr.shape[0]

    # Check consistent first-axis length
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        bad = {k: v for k, v in lengths.items() if v != min(unique_lengths)}
        raise ValueError(
            f"Episode buffer fields have inconsistent step counts: "
            f"{lengths}. Mismatched fields: {bad}"
        )


def compute_ep_offset(ep_lens: list[int]) -> np.ndarray:
    """Compute cumulative episode offsets from a list of episode lengths.

    Episode i occupies rows [offset[i], offset[i] + ep_lens[i]) in all
    per-step arrays.

    Parameters
    ----------
    ep_lens:
        List of episode lengths in order.

    Returns
    -------
    np.ndarray
        1-D int64 array of length len(ep_lens).

    Examples
    --------
    >>> compute_ep_offset([10, 20, 15])
    array([ 0, 10, 30])
    """
    if not ep_lens:
        return np.array([], dtype=np.int64)
    offsets = np.zeros(len(ep_lens), dtype=np.int64)
    for i in range(1, len(ep_lens)):
        offsets[i] = offsets[i - 1] + ep_lens[i - 1]
    return offsets


def _state_names(state_dim: int) -> list[str]:
    """Column names for ``observation.state``.

    For the canonical SO-101 layout (12 = joint_pos[6] + joint_vel[6]) the names
    match the Isaac/synthetic recorder exactly so real and simulated datasets
    share one feature schema and can be merged. Other dims fall back to generic
    ``motor_i`` names.
    """
    if state_dim == 12:
        return [f"joint_pos_{i}" for i in range(6)] + [
            f"joint_vel_{i}" for i in range(6)
        ]
    return [f"motor_{i}" for i in range(state_dim)]


def lerobot_features_dict(
    action_dim: int,
    state_dim: int,
    image_shape: tuple[int, int, int],
    camera_key: str = "overhead",
) -> dict[str, dict]:
    """Build a LeRobot v3 features dict for dataset creation.

    Used with ``LeRobotDataset.create(features=lerobot_features_dict(...))``.

    The schema is aligned with the canonical SO-101 dataset the
    ``lerobot-isaac-training`` workspace consumes (its Isaac recorder, the
    world-model bridge, and the policy/world-model trainers), so recordings
    drop straight into that pipeline:

    - ``observation.state`` is ``(12,)`` = joint_pos[6] + joint_vel[6].
    - the camera is ``observation.images.<camera_key>`` (default ``wrist`` — the
      view the bridge selects by default for DreamerV3 / LeWorldModel).
    - ``next.reward`` and ``next.done`` are present because the world-model
      bridge reads those exact columns to populate the ``rewards`` and ``dones``
      arrays of its HDF5 output. Without them success/termination would be lost
      from world-model training, and success and failure episodes would be
      indistinguishable in the Parquet dataset.

    Parameters
    ----------
    action_dim:
        Number of action dimensions (6 for SO-101: 5 joints + gripper).
    state_dim:
        Number of state dimensions (12 for SO-101: joint_pos[6] + joint_vel[6]).
    image_shape:
        ``(H, W, C)`` — height, width, channels. For ``dtype="image"`` LeRobot
        stores frames channels-last (HWC), the same layout the recorder feeds to
        ``add_frame``. Declaring the feature shape channels-last keeps the
        declared shape and the per-frame data in agreement so ``add_frame`` does
        not raise a shape-mismatch error.
    camera_key:
        Short camera name; the feature is exposed as
        ``observation.images.<camera_key>``.

    Returns
    -------
    dict
        Features dict compatible with LeRobot v3 ``create()`` API.
    """
    h, w, c = image_shape
    return {
        f"observation.images.{camera_key}": {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": _state_names(state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }
