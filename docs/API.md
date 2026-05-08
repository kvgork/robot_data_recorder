# robot-data-recorder — Public API Reference

---

## Module: `robot_data_recorder`

Top-level public exports.

```python
from robot_data_recorder import (
    RecordingConfig,
    RecordingSession,
    DualWriter,
    D435Stream,
    EpisodeSchema,
)
```

| Export | Module | Description |
|--------|--------|-------------|
| `RecordingConfig` | `config` | Session configuration dataclass |
| `RecordingSession` | `recorder` | Main recording orchestrator |
| `DualWriter` | `dual_writer` | Parallel Parquet + HDF5 writer |
| `D435Stream` | `d435` | RealSense D435 wrapper |
| `EpisodeSchema` | `schema` | Canonical schema declaration |

---

## Module: `robot_data_recorder.config`

### `class RecordingConfig`

Flat dataclass capturing all recording session parameters.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_id` | `str` | `"local/recording"` | HF repo id or local dataset name |
| `num_episodes` | `int` | `1` | Episodes to record |
| `format` | `str` | `"dual"` | `parquet` / `hdf5` / `dual` |
| `output_dir` | `str` | `"./datasets"` | Base output directory |
| `task` | `str` | `"unspecified"` | Task description for metadata |
| `fps` | `int` | `30` | Recording frame rate (Hz) |
| `arm_port` | `str` | `"/dev/ttyUSB0"` | SO-101 follower serial port |
| `leader_port` | `str \| None` | `None` | SO-101 leader serial port |
| `camera_serial` | `str \| None` | `None` | D435 serial (`None` = AUTO) |
| `resolution` | `tuple[int,int]` | `(640, 480)` | Camera (width, height) |
| `enable_depth` | `bool` | `False` | Enable depth stream |
| `max_steps` | `int` | `200` | Max steps per episode |
| `dry_run` | `bool` | `False` | Skip hardware, print config |

**Class methods:**

#### `RecordingConfig.from_yaml(path) -> RecordingConfig`

Load from YAML file or named config.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | File path or config name for `load_config()` |

**Raises:** `FileNotFoundError` if path cannot be resolved.

#### `RecordingConfig.to_dict() -> dict`

Return JSON-serialisable dict for `--dry-run` printout.

---

## Module: `robot_data_recorder.schema`

### `class EpisodeSchema` (frozen dataclass)

Declares canonical field set. Module-level singleton: `SCHEMA = EpisodeSchema()`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `STEP_FIELDS` | `dict[str, tuple[str, int]]` | `{name: (dtype, min_ndim)}` |
| `EP_FIELDS` | `dict[str, str]` | `{name: dtype}` for per-episode fields |

### `validate_episode_buffer(ep: dict) -> None`

Validate per-step episode dict against `SCHEMA.STEP_FIELDS`.

**Raises:**
- `ValueError` — missing field, wrong dtype, wrong ndim, or inconsistent step count
- `ValueError` — input is not a dict

### `compute_ep_offset(ep_lens: list[int]) -> np.ndarray`

Compute cumulative start offsets from list of episode lengths.

**Returns:** `np.ndarray` int64, shape `(len(ep_lens),)`.

**Example:**
```python
compute_ep_offset([10, 20, 15])
# array([ 0, 10, 30])
```

### `lerobot_features_dict(action_dim, state_dim, image_shape) -> dict`

Build LeRobot v3 features dict for `LeRobotDataset.create()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `action_dim` | `int` | Action space dimension |
| `state_dim` | `int` | State/proprio dimension |
| `image_shape` | `tuple[int,int,int]` | `(C, H, W)` — channels-first |

**Returns:** `dict` with keys `observation.images.d435_rgb`, `observation.state`, `action`.

---

## Module: `robot_data_recorder.d435`

### `class D435Stream`

RealSense D435 camera wrapper. `pyrealsense2` soft-imported.

**Constructor:**
```python
D435Stream(
    serial: str | None = None,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    enable_depth: bool = False,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Open pipeline. Raises `ImportError` if pyrealsense2 missing. |
| `read_frame()` | `dict` | `{"rgb": uint8(H,W,3), "depth": uint16(H,W)\|None, "timestamp": float}` |
| `stop()` | `None` | Close pipeline |
| `__enter__` / `__exit__` | — | Context manager |

### `class MockD435Stream`

API-identical synthetic stream for tests. Same parameters as `D435Stream`.

### `make_d435(serial, resolution, fps, enable_depth, mock) -> D435Stream | MockD435Stream`

Factory function. Set `mock=True` for tests.

---

## Module: `robot_data_recorder.so101_teleop`

### `class SO101Teleop`

SO-101 leader/follower teleoperation. `lerobot` soft-imported.

**Constructor:**
```python
SO101Teleop(arm_port: str, leader_port: str | None = None)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Connect to hardware. Raises `ImportError` if lerobot missing. |
| `read_state()` | `dict` | `{"joint_pos": float32(6,), "joint_vel": float32(6,), "gripper": float, "timestamp": float}` |
| `read_action()` | `np.ndarray float32(7,)` | Current leader action `[joints×6, gripper]` |
| `stop()` | `None` | Disconnect |

### `class MockSO101Teleop`

API-identical synthetic arm for tests.

---

## Module: `robot_data_recorder.dual_writer`

### `class DualWriter`

Parallel LeRobot Parquet + stable-worldmodel HDF5 writer.

**Constructor:**
```python
DualWriter(config: RecordingConfig)
```

Raises `ValueError` if `config.format` is not one of `parquet / hdf5 / dual`.
Raises `ImportError` if required backend library is missing for the chosen format.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `write_episode(ep: dict)` | `None` | Validate + write to configured backends |
| `finalize()` | `dict[str, Path]` | Close writers, return `{format: path}` |

**Format dispatch:**

| `config.format` | LeRobot called | HDF5Writer called |
|-----------------|----------------|-------------------|
| `parquet` | Yes | No |
| `hdf5` | No | Yes |
| `dual` | Yes | Yes |

---

## Module: `robot_data_recorder.recorder`

### `class EpisodeBuffer` (dataclass)

Accumulates per-step data during one episode.

**Attributes:** `episode_idx`, `pixels`, `action`, `state`, `proprio`, `done`, `timestamp`, `reward` (all lists).

**Method:** `to_dict() -> dict[str, np.ndarray]` — stack lists into numpy arrays.

### `class RecordingSession`

**Constructor:**
```python
RecordingSession(
    config: RecordingConfig,
    camera: D435Stream | MockD435Stream,
    teleop: SO101Teleop | MockSO101Teleop,
    writer: DualWriter | None,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__enter__()` | `self` | Start camera + teleop |
| `__exit__(...)` | `None` | Stop + finalize writer |
| `record_episode(episode_idx)` | `EpisodeBuffer` | Rollout loop; 5-step synthetic in dry-run mode |
| `save_episode(buffer)` | `None` | Dispatch to `DualWriter.write_episode()` |

### `class MockRecordingSession`

Subclass of `RecordingSession` that auto-creates `MockD435Stream` + `MockSO101Teleop`.

---

## Module: `robot_data_recorder.cli`

### `main(argv: list[str] | None = None) -> int`

CLI entrypoint. Returns `0` on success.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | required | Dataset name |
| `--num-episodes` | `1` | Episodes to record |
| `--format` | `dual` | `parquet` / `hdf5` / `dual` |
| `--resolution` | `640x480` | Camera resolution |
| `--fps` | `30` | Frame rate |
| `--arm-port` | `/dev/ttyUSB0` | SO-101 follower port |
| `--leader-port` | `None` | SO-101 leader port |
| `--camera-serial` | `None` | D435 serial (AUTO = first) |
| `--output-dir` | `./datasets` | Output directory |
| `--task` | `unspecified` | Task description |
| `--max-steps` | `200` | Episode timeout |
| `--depth` | `False` | Enable depth stream |
| `--dry-run` | `False` | Print config, exit 0 |
| `--config` | `None` | YAML base config |
