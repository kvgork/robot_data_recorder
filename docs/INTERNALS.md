# lerobot-isaac-recorder — Internals

---

## File Walk-Through

### `schema.py` — Canonical Schema

The schema declares the **superset** of:

1. `stable-worldmodel` HDF5 schema (pixels, action, reward, ep_len, ep_offset)
2. `tanguy-pauwels/lerobot-dataset-to-HDF5` output (adds done, timestamp, episode_idx, step_idx, state, proprio)

This superset means:
- The HDF5 output satisfies `stable-worldmodel.data.HDF5Dataset` requirements
- The LeRobot output satisfies `LeRobotDataset` requirements
- Both consumers get all fields they need from the same recording

**`EpisodeSchema`** is a frozen dataclass with two dicts:
- `STEP_FIELDS`: `{name: (dtype_str, min_ndim)}` — validated per step
- `EP_FIELDS`: `{name: dtype}` — computed after all episodes (ep_len, ep_offset)

**`validate_episode_buffer`** runs before every `DualWriter.write_episode()` call.
It checks dtype compatibility using `np.issubdtype` and ndim lower bounds (not exact
shape) because T, H, W, A, S dimensions vary by config.

**`compute_ep_offset`** produces the cumulative sum of episode lengths. This is the
standard HDF5 indexing scheme:
```
episode 0: rows [0,          0+ep_len[0])
episode 1: rows [ep_offset[1], ep_offset[1]+ep_len[1])
```

**`lerobot_features_dict`** generates the features dict passed to
`LeRobotDataset.create()`. Camera key is always `observation.images.d435_rgb`.
LeRobot expects channels-first `(C, H, W)` in features but channels-last `(H, W, C)`
in actual frame data — this is handled in `DualWriter._write_lerobot()`.

---

### `config.py` — Flat Config

`RecordingConfig` is intentionally flat (no nested sub-configs). Reasons:

1. `to_dict()` produces a clean single-level JSON for `--dry-run` readability
2. Simpler YAML files (no indentation levels needed)
3. `from_yaml()` can use a simple `{k: v for k in known_fields}` filter

The `from_yaml()` method resolves in two ways:
1. If `path` is an existing file, load it directly with `yaml.safe_load`
2. Otherwise delegate to `lerobot_isaac_configs.load_config()` for named configs

---

### `d435.py` — Camera Stream

**Soft-import guard:**
```python
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    rs = None
    _HAS_REALSENSE = False
```

`D435Stream.start()` checks `_HAS_REALSENSE` and raises `ImportError` with an
actionable message. This defers the error to the first `start()` call, not import time.

**RGB conversion:** RealSense returns BGR8. `d435.py` converts to RGB at frame capture
to match LeRobot's convention. Conversion uses `[:, :, ::-1].copy()` (copy avoids
negative-stride numpy arrays that can cause issues with h5py).

**MockD435Stream:** Uses `np.random.default_rng(frame_idx)` for deterministic frame
generation. Same shapes as real camera. Depth array is uint16 in range 500–2000 mm
(typical D435 operating range).

---

### `so101_teleop.py` — Arm Interface

Wraps `lerobot.common.robot_devices.robots.factory.make_robot`. In Phase 0 this
import is inside `start()` so the module loads cleanly without lerobot installed.

**State format:** `read_state()` returns a dict with:
- `joint_pos` (6,) — revolute joint angles in radians
- `joint_vel` (6,) — joint angular velocities
- `gripper` float — normalized 0.0 (closed) to 1.0 (open)
- `timestamp` float — `time.monotonic()` in seconds

**Action format:** `read_action()` returns a `(7,)` float32 array:
`[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper]`

This matches the SO-101 convention used in LeRobot teleoperation scripts.

---

### `dual_writer.py` — Parallel Write

**Initialization:** `DualWriter.__init__()` calls `_init_writers()` which calls
`_init_lerobot()` and/or `_init_hdf5()` depending on `config.format`. Both
initializations require the respective libraries to be present.

**Write path (per episode):**
```
write_episode(ep)
  │
  ├─ validate_episode_buffer(ep)      # fail fast on bad data
  ├─ _write_lerobot(ep)               # if format in (parquet, dual)
  │   ├─ for t in range(T): dataset.add_frame(...)
  │   └─ dataset.save_episode(task=...)
  └─ _write_hdf5(ep)                  # if format in (hdf5, dual)
      └─ hdf5_writer.write_episode(ep)
```

**HDF5 path:** stable-worldmodel's `HDF5Writer` operates in SWMR (Single-Writer
Multiple-Reader) mode. This is HDF5's safe concurrent access mechanism. The writer
is opened once at init and closed in `finalize()`.

**Output path naming:** HDF5 file uses `{repo_id.replace("/", "__")}.h5` so that
`koen/pickplace` becomes `koen__pickplace.h5`. Slashes are not valid in filenames.

**Finalize:** Calls `LeRobotDataset.finalize()` (required before HF push) and then
closes the HDF5 writer. Both calls are wrapped in `try/except` for best-effort close.

---

### `recorder.py` — Session Orchestrator

**EpisodeBuffer:** A dataclass of lists. All per-step data is appended during the
rollout, then `to_dict()` stacks them into numpy arrays. This avoids pre-allocating
arrays of unknown length.

**Tick timing:** The rollout loop uses `time.sleep(max(0, period - elapsed))` where
`period = 1/fps`. This gives approximate rate control. For sub-millisecond jitter
accuracy, a hardware trigger would be needed (future work).

**Dry-run:** `RecordingSession._synthetic_episode()` generates 5 deterministic steps
using `np.random.default_rng(episode_idx)`. This ensures reproducible test data and
allows `--dry-run` to exercise the full buffer-to-dict path without hardware.

**State vector:** For SO-101, both `state` and `proprio` are `[joint_pos(6), gripper(1)]`
— a 7-element float32 vector. The distinction between state and proprio is inherited
from the LeWM schema where tasks may have different state (global) and proprio (local)
observations. For SO-101 they are identical.

---

### `cli.py` — Command-Line Interface

**Argument naming:** Uses kebab-case (`--repo-id`, `--arm-port`) as is standard for
POSIX CLIs. Argparse converts these to underscore attributes (`args.repo_id`,
`args.arm_port`) internally.

**Resolution parsing:** `_parse_resolution("640x480")` is a standalone function,
not an `argparse.type=`, to allow it to be called directly in tests.

**Dry-run path:** Resolves config, prints JSON, then prints the equivalent CLI command
for copy-paste. Does NOT import `DualWriter` or hardware classes — those are imported
only on the real-hardware path.

**Real-hardware path:** All heavy imports (`make_d435`, `DualWriter`, `SO101Teleop`,
`RecordingSession`) happen inside the `if not args.dry_run` branch so the module
is importable without optional deps.

---

## Schema Rationale

The superset schema was chosen to maximize compatibility with both consumers:

| Field | stable-worldmodel needs | LeRobot needs | Source |
|-------|------------------------|---------------|--------|
| `pixels` | Yes | Yes (as image) | Both |
| `action` | Yes | Yes | Both |
| `state` | Optional | Yes (`observation.state`) | Both |
| `proprio` | Optional | No (embedded in state) | LeWM only |
| `reward` | Yes (zeros for teleop) | No | LeWM |
| `done` | No | Yes (terminal signal) | LeRobot |
| `timestamp` | No | Yes | LeRobot |
| `episode_idx` | No | Yes | LeRobot |
| `step_idx` | No | Yes | LeRobot |
| `ep_len` | Yes | No (computed from Parquet) | LeWM |
| `ep_offset` | Yes | No | LeWM |

Using a superset means `validate_episode_buffer` can check all fields in one pass,
and both writers receive exactly what they need without field selection logic.

---

## Soft-Import Strategy

Per ADR-0003. Three module-level flags:

| Flag | Module | Dep |
|------|--------|-----|
| `_HAS_REALSENSE` | `d435.py` | `pyrealsense2` |
| `_HAS_LEROBOT` | `so101_teleop.py` | `lerobot` |
| `_HAS_LEROBOT` | `dual_writer.py` | `lerobot` |
| `_HAS_STABLE_WORLDMODEL` | `dual_writer.py` | `stable_worldmodel` |

Test suite patches these flags via `unittest.mock.patch.object` to simulate missing
deps without actually uninstalling packages:

```python
with patch.object(dw_mod, "_HAS_LEROBOT", False):
    with pytest.raises(ImportError, match="lerobot is required"):
        writer._write_lerobot(ep)
```

This pattern is safe because `_HAS_LEROBOT` is read at function call time, not at
module import time.

---

## Future Work

| Feature | Scope | Notes |
|---------|-------|-------|
| Real teleop loop | Phase 1 | Requires SO-101 on bench; lerobot robot factory |
| Camera calibration | Phase 2 | Intrinsics JSON persisted alongside HDF5 |
| Isaac sim mirror writer | Path C | `lerobot-isaac-env.make_env()` writes HDF5 |
| Multi-camera | Phase 2 | Dict of `D435Stream` keyed by camera name |
| Episode quality preview | Phase 2 | Real-time SAL/TED score during recording |
| Push to HuggingFace Hub | Phase 2 | `LeRobotDataset.push_to_hub()` after finalize |
| Depth key in LeRobot | Phase 2 | Add `observation.images.d435_depth` feature |
