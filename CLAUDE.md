# lerobot-isaac-recorder — Package Orientation

**Role:** Hardware-facing recorder. D435 camera + SO-101 teleop → dual-write episodes to
LeRobot Parquet (policy training) and stable-worldmodel HDF5 (world-model training).
**Phase:** 0 (scaffold) — all classes implemented and tested; real hardware loop requires bench.
**Status:** 54 tests passing. pyrealsense2 / lerobot / stable-worldmodel soft-imported.

---

## What This Package Does

Records teleoperation demonstration episodes from a physical setup:
- Intel RealSense D435 RGB(-D) camera
- SO-101 leader + follower arm

Writes simultaneously to both target consumers:
1. `LeRobotDataset` (Parquet + MP4) → policy training (ACT, SmolVLA, DiffPolicy)
2. `HDF5Writer` (stable-worldmodel schema) → world-model training (LeWM)

Both outputs use the same in-memory episode buffer (Path B dual-write).

---

## Internal Structure

| File | Role |
|------|------|
| `src/lerobot_isaac_recorder/__init__.py` | Public exports + `__version__` |
| `src/lerobot_isaac_recorder/schema.py` | `EpisodeSchema`, `validate_episode_buffer`, `compute_ep_offset`, `lerobot_features_dict` |
| `src/lerobot_isaac_recorder/config.py` | `RecordingConfig` dataclass with `from_yaml()` / `to_dict()` |
| `src/lerobot_isaac_recorder/d435.py` | `D435Stream`, `MockD435Stream`, `make_d435()` factory |
| `src/lerobot_isaac_recorder/so101_teleop.py` | `SO101Teleop`, `MockSO101Teleop` |
| `src/lerobot_isaac_recorder/dual_writer.py` | `DualWriter` — parallel Parquet + HDF5 write |
| `src/lerobot_isaac_recorder/recorder.py` | `EpisodeBuffer`, `RecordingSession`, `MockRecordingSession` |
| `src/lerobot_isaac_recorder/cli.py` | argparse CLI, `main()` entrypoint |
| `tests/conftest.py` | Pytest markers for hardware deps |
| `tests/test_imports.py` | Clean importability without heavy deps |
| `tests/test_schema.py` | Schema dataclass + validation + helpers |
| `tests/test_dual_writer.py` | Format-mode tests with monkeypatched backends |
| `tests/test_cli.py` | Argparse smoke + dry-run |

---

## Soft-Import Pattern

Per ADR-0003. Heavy deps NEVER at module top-level:

```python
# d435.py
try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    rs = None
    _HAS_REALSENSE = False

class D435Stream:
    def start(self):
        if not _HAS_REALSENSE:
            raise ImportError("pyrealsense2 required: pip install pyrealsense2")
```

Same pattern in `so101_teleop.py` (`_HAS_LEROBOT`) and `dual_writer.py`
(`_HAS_LEROBOT`, `_HAS_STABLE_WORLDMODEL`).

---

## Public API

```python
from lerobot_isaac_recorder import (
    RecordingConfig,   # session config
    RecordingSession,  # orchestrator
    DualWriter,        # parallel writer
    D435Stream,        # camera (soft-dep)
    EpisodeSchema,     # schema declaration
)
from lerobot_isaac_recorder.d435 import make_d435           # factory
from lerobot_isaac_recorder.so101_teleop import MockSO101Teleop
from lerobot_isaac_recorder.schema import validate_episode_buffer
```

CLI entrypoint: `lerobot-isaac-record` (declared in `pyproject.toml`).

---

## Coupling

- **Depends on:** `lerobot-isaac-configs` (workspace sibling, loads recording YAML)
- **Soft deps:** `pyrealsense2`, `lerobot`, `stable-worldmodel`
- **Does NOT depend on:** `lerobot-isaac-env`, `lerobot-isaac-adapters`, `lerobot-isaac-synthetic`
- **Future:** `lerobot-isaac-env` integration for Isaac sim mirror writer (Path C)

Dependency graph position: `recorder → configs` (leaf + hardware interface).

---

## Episode Schema (canonical superset)

See `schema.py` → `EpisodeSchema`. Key fields:

| Field | Shape | Dtype |
|-------|-------|-------|
| `pixels` | (T, H, W, C) | uint8 |
| `action` | (T, 7) | float32 |
| `state` | (T, 7) | float32 |
| `proprio` | (T, 7) | float32 |
| `done` | (T,) | bool |
| `ep_len` | (N_ep,) | int64 |
| `ep_offset` | (N_ep,) | int64 |

---

## Testing

```bash
cd packages/lerobot-isaac-recorder
python3 -m pytest tests/ -q       # all tests, no hardware required
python3 -m pytest tests/ -v       # verbose
```

No hardware or optional deps needed. All tests use mocks.

---

## How to Extend

### Add a new camera type

1. Create `src/lerobot_isaac_recorder/my_camera.py` with `start()`, `read_frame()`, `stop()`.
2. Add `MyCameraStream` and `MockMyCameraStream` following `d435.py` pattern.
3. Update `RecordingConfig` with new camera params.
4. Wire into `cli.py` behind a `--camera-type` flag.

### Add a new output format

1. Add `my_format` to `_FORMATS` in `dual_writer.py`.
2. Add `_init_my_format()` and `_write_my_format()` methods.
3. Soft-import the backend library.
4. Add test cases in `tests/test_dual_writer.py`.

### Add a real hardware test

Mark with `@pytest.mark.requires_realsense` or `@pytest.mark.requires_lerobot`
so CI skips it automatically.

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` §14
- Research: `05-Wiki/project-context/research/2026-05-07-dual-recording-lerobot-leworldmodel/details.md`
- ADR-0003 (soft-import): `../../docs/adr/0003-soft-import-discipline.md`
- Workspace ARCHITECTURE.md: `../../docs/ARCHITECTURE.md`
