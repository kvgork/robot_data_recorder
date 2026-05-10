# robot-data-recorder

D435 camera + SO-101 teleoperation **dual-write recorder** for the
`lerobot-isaac-training` workspace.

Records demonstration episodes simultaneously to:
- **LeRobot v3 Parquet** — for policy training with ACT / SmolVLA / DiffPolicy
- **stable-worldmodel HDF5** — for world-model training with LeWM

Architecture: **Path B (in-session dual-write)**. After each episode the recorder
calls `LeRobotDataset.save_episode()` and `HDF5Writer.write_episode()` on the same
in-memory buffer. Both writes happen before the next episode starts.

---

## Status

**Phase 0 — scaffold complete.**
All classes implemented and tested with mock hardware.
Real hardware loop (D435 + SO-101) requires `pyrealsense2` + `lerobot` at runtime.

| Component | Status |
|-----------|--------|
| `RecordingConfig` | Implemented |
| `EpisodeSchema` + validation | Implemented |
| `D435Stream` + `MockD435Stream` | Implemented |
| `SO101Teleop` + `MockSO101Teleop` | Implemented |
| `DualWriter` (parquet / hdf5 / dual) | Implemented |
| `RecordingSession` | Implemented |
| CLI (`robot-data-record`) | Implemented |
| Real hardware loop | Phase 1 (requires bench) |
| Camera calibration persistence | Phase 2 |
| Isaac sim mirror writer (Path C) | Future |

---

## Installation

### Monorepo mode (pixi)

```bash
cd ~/workspaces/lerobot-isaac-training
pixi install         # installs all workspace packages
```

### Standalone mode

```bash
cd packages/robot-data-recorder
pixi install         # standalone pixi environment (dormant in monorepo mode)
```

### Direct pip install

```bash
pip install -e packages/robot-data-recorder/
```

### Optional heavy deps

```bash
# For real D435 camera recording:
pip install pyrealsense2

# For LeRobot Parquet output (parquet / dual format):
bash scripts/install_lerobot.sh    # or: pip install lerobot

# For LeWM HDF5 output (hdf5 / dual format):
pip install stable-worldmodel
```

---

## Quick Example

> **Requires pixi env.** The `robot-data-record` console script lives in the
> workspace pixi environment. Either drop into a shell first:
> ```bash
> pixi shell
> robot-data-record ...
> ```
> or prefix each call with `pixi run`:
> ```bash
> pixi run robot-data-record ...
> ```
> Plain `robot-data-record` outside pixi gives `command not found`. All bash
> examples below assume `pixi shell` is already active.

### Dry-run (no hardware, no deps)

```bash
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --num-episodes=10 \
  --format=dual \
  --task="pick and place cube" \
  --dry-run
```

Expected output:
```
[robot-data-record] DRY RUN — resolved config:
{
  "repo_id": "myuser/so101-pickplace",
  "num_episodes": 10,
  "format": "dual",
  ...
}
```

### Python API (mock hardware)

```python
from robot_data_recorder import RecordingConfig, RecordingSession
from robot_data_recorder.d435 import make_d435
from robot_data_recorder.so101_teleop import MockSO101Teleop

cfg = RecordingConfig(
    repo_id="myuser/so101-pickplace",
    num_episodes=3,
    format="hdf5",    # no lerobot needed
    output_dir="./datasets",
    dry_run=True,
)
cam = make_d435(mock=True)
teleop = MockSO101Teleop()

with RecordingSession(cfg, camera=cam, teleop=teleop, writer=None) as session:
    for ep_idx in range(cfg.num_episodes):
        buf = session.record_episode(ep_idx)
        print(f"Episode {ep_idx}: {len(buf.pixels)} steps")
```

### Pre-flight hardware check

Before the first real-hardware recording session, verify env vars, serial
ports, dialout group membership, and the RealSense device:

```bash
pixi run robot-data-check              # quick checks, no hardware connect
pixi run robot-data-check --connect    # also try lerobot SO-101 connect()
pixi run robot-data-check --json       # machine-readable output
```

Exits 0 when everything required is reachable, 1 when a blocker is found.

### Episode control keys

While recording, press a single key in the launching terminal:

| Key            | Effect                                                           |
|----------------|------------------------------------------------------------------|
| `SPACE` / `ENTER` / `s` | End current episode and save it. Move to next episode.   |
| `q`            | End current episode, save it, then abort the rest of the session. |

`max_steps` (default `18000` ≈ 10 min @ 30 Hz) is still honoured as a hard
safety ceiling; it only fires when no key is pressed. When stdin is not a
tty (e.g. wrapped by a launcher), the listener is disabled and `max_steps`
behaves like the old fixed-length cutoff.

### Real hardware (D435 + SO-101)

Hardware ports/serial pulled from env vars (set via `pixi run setup-env` or
exported in `~/.bashrc`):

| Env var | Default | Purpose |
|---------|---------|---------|
| `LERO_FOLLOWER_PORT` | `/dev/ttyUSB0` | SO-101 follower arm serial port |
| `LERO_LEADER_PORT` | unset | SO-101 leader arm serial port |
| `LERO_CAM_SERIAL` | unset (= AUTO) | RealSense D435 serial number |

When set, CLI flags `--arm-port`, `--leader-port`, `--camera-serial` and
`RecordingConfig` defaults pick them up automatically:

```bash
# Uses $LERO_FOLLOWER_PORT, $LERO_LEADER_PORT, $LERO_CAM_SERIAL
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --num-episodes=50 \
  --format=dual \
  --task="pick and place cube"
```

Override on command line if needed:

```bash
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --arm-port=<follower-tty> \
  --leader-port=<leader-tty> \
  --camera-serial=<d435-serial>
```

---

## Public API

### `RecordingConfig`

```python
@dataclass
class RecordingConfig:
    repo_id: str = "local/recording"
    num_episodes: int = 1
    format: str = "dual"           # parquet | hdf5 | dual
    output_dir: str = "./datasets"
    task: str = "unspecified"
    fps: int = 30
    arm_port: str = field(default_factory=_env_follower_port)   # $LERO_FOLLOWER_PORT
    leader_port: str | None = field(default_factory=_env_leader_port)  # $LERO_LEADER_PORT
    camera_serial: str | None = field(default_factory=_env_camera_serial)  # $LERO_CAM_SERIAL
    resolution: tuple[int, int] = (640, 480)
    enable_depth: bool = False
    max_steps: int = 200
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path) -> RecordingConfig: ...
    def to_dict(self) -> dict: ...
```

### `RecordingSession`

```python
class RecordingSession:
    def __init__(self, config, camera, teleop, writer): ...
    def __enter__(self) -> RecordingSession: ...  # starts camera + teleop
    def __exit__(self, *_): ...                    # stops + finalizes
    def record_episode(self, episode_idx) -> EpisodeBuffer: ...
    def save_episode(self, buffer) -> None: ...
```

### `DualWriter`

```python
class DualWriter:
    def __init__(self, config: RecordingConfig): ...
    def write_episode(self, ep: dict) -> None: ...
    def finalize(self) -> dict[str, Path]: ...
```

### `make_d435` factory

```python
def make_d435(
    serial=None,
    resolution=(640, 480),
    fps=30,
    enable_depth=False,
    mock=False,          # True -> MockD435Stream for tests
) -> D435Stream | MockD435Stream: ...
```

### Schema helpers

```python
from robot_data_recorder.schema import (
    validate_episode_buffer,    # raises ValueError on bad buffer
    compute_ep_offset,           # [10, 20] -> [0, 10]
    lerobot_features_dict,       # features dict for LeRobotDataset.create()
)
```

---

## Episode Schema

The canonical superset schema (from `schema.EpisodeSchema`):

| Field | Dtype | Shape | Description |
|-------|-------|-------|-------------|
| `pixels` | uint8 | (T, H, W, C) | RGB frames |
| `action` | float32 | (T, A) | Commanded joint angles + gripper |
| `state` | float32 | (T, S) | Observed joint positions + gripper |
| `proprio` | float32 | (T, P) | Same as state for SO-101 |
| `done` | bool | (T,) | Episode terminal flag |
| `timestamp` | float32 | (T,) | Hardware timestamp (seconds) |
| `episode_idx` | int64 | (T,) | Global episode index |
| `step_idx` | int64 | (T,) | Within-episode step index |
| `reward` | float32 | (T,) | Reward (0.0 for teleop) |
| `ep_len` | int64 | (N_ep,) | Length of each episode |
| `ep_offset` | int64 | (N_ep,) | Cumulative start offset |

Episode i occupies rows `[ep_offset[i], ep_offset[i] + ep_len[i])`.

---

## Dependencies

### Hard (always required)

| Package | Purpose |
|---------|---------|
| `h5py` | HDF5 read/write |
| `numpy` | Array operations |
| `pyyaml` | YAML config loading |
| `lerobot-isaac-configs` | Workspace config loader |

### Soft (required only for specific features)

| Package | Required for | Install |
|---------|-------------|---------|
| `pyrealsense2` | Real D435 camera | `pip install pyrealsense2` |
| `lerobot` | Parquet output | `bash scripts/install_lerobot.sh` |
| `stable-worldmodel` | HDF5 output | `pip install stable-worldmodel` |

---

## Configuration

### YAML config file

```yaml
# recording_default.yaml
repo_id: myuser/so101-pickplace
num_episodes: 50
format: dual
fps: 30
arm_port: /dev/ttyUSB0
leader_port: /dev/ttyUSB1
resolution: [640, 480]
task: "pick and place cube"
max_steps: 300
```

Load via:

```bash
robot-data-record --config=recording_default --dry-run
```

Or in Python:

```python
cfg = RecordingConfig.from_yaml("configs/recording_default.yaml")
```

---

## Running Tests

```bash
cd packages/robot-data-recorder
python3 -m pytest tests/ -v
```

No hardware or optional deps required. All tests use mocks.

To run only unit tests (no markers):

```bash
pytest tests/ -m 'not requires_realsense and not requires_lerobot'
```

---

## Spinout

This package can be extracted as a standalone PyPI package:

```bash
git subtree split -P packages/robot-data-recorder -b spinout-recorder
```

After spinout, update sibling `pyproject.toml` files to reference the PyPI version.
See `../../docs/ARCHITECTURE.md` (spinout section).

---

## Source-of-Truth Pointers

- ADR-0003 (soft-import discipline): `../../docs/adr/0003-soft-import-discipline.md`
- Workspace architecture: `../../docs/ARCHITECTURE.md`
