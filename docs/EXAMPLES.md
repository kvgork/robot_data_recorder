# robot-data-recorder — Examples

---

## 1. Minimal Dry-Run (no hardware, no deps)

```bash
robot-data-record \
  --repo-id=myuser/so101-test \
  --num-episodes=1 \
  --dry-run
```

Expected output:
```
[robot-data-record] DRY RUN — resolved config:
{
  "repo_id": "myuser/so101-test",
  "num_episodes": 1,
  "format": "dual",
  "output_dir": "./datasets",
  ...
}
[robot-data-record] Would run:
  robot-data-record --repo-id=myuser/so101-test --num-episodes=1 --format=dual --output-dir=./datasets
```

Equivalent in Python:
```python
from robot_data_recorder.cli import main

rc = main(["--repo-id=myuser/so101-test", "--num-episodes=1", "--dry-run"])
assert rc == 0
```

---

## 2. Real D435 + SO-101 Setup (dual format)

Full hardware recording session writing both LeRobot Parquet and LeWM HDF5.

**Prerequisites:**
```bash
pip install pyrealsense2
bash scripts/install_lerobot.sh
pip install stable-worldmodel
```

**Wired through the workspace meta CLI (preferred):**
```bash
lerobot-isaac record \
  --repo-id=myuser/so101-pickplace \
  --num-episodes=50 \
  --format=dual \
  --arm-port=/dev/ttyUSB0 \
  --leader-port=/dev/ttyUSB1 \
  --camera-serial=AUTO \
  --fps=30 \
  --task="pick and place cube" \
  --output-dir=./datasets
```

**Direct package CLI:**
```bash
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --num-episodes=50 \
  --format=dual \
  --arm-port=/dev/ttyUSB0 \
  --leader-port=/dev/ttyUSB1 \
  --output-dir=./datasets \
  --task="pick and place cube"
```

**Python API equivalent:**
```python
from robot_data_recorder import RecordingConfig, RecordingSession
from robot_data_recorder.d435 import make_d435
from robot_data_recorder.dual_writer import DualWriter
from robot_data_recorder.so101_teleop import SO101Teleop

cfg = RecordingConfig(
    repo_id="myuser/so101-pickplace",
    num_episodes=50,
    format="dual",
    arm_port="/dev/ttyUSB0",
    leader_port="/dev/ttyUSB1",
    output_dir="./datasets",
    task="pick and place cube",
)
camera = make_d435(fps=cfg.fps, resolution=cfg.resolution)
teleop = SO101Teleop(arm_port=cfg.arm_port, leader_port=cfg.leader_port)
writer = DualWriter(cfg)

with RecordingSession(cfg, camera=camera, teleop=teleop, writer=writer) as session:
    for ep_idx in range(cfg.num_episodes):
        buf = session.record_episode(ep_idx)
        session.save_episode(buf)
        print(f"Saved episode {ep_idx + 1}/{cfg.num_episodes}")

paths = writer.finalize()
print(f"Parquet: {paths.get('parquet')}")
print(f"HDF5:    {paths.get('hdf5')}")
```

---

## 3. Parquet-Only (LeRobot policy training only)

When you only want to train a policy and skip world-model data collection:

```bash
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --format=parquet \
  --num-episodes=50 \
  --arm-port=/dev/ttyUSB0 \
  --leader-port=/dev/ttyUSB1
```

Requires: `pyrealsense2` + `lerobot`.
Does NOT require: `stable-worldmodel`.

---

## 4. HDF5-Only (LeWM world-model training only)

When you only want to train a world model and skip LeRobot policy data:

```bash
robot-data-record \
  --repo-id=myuser/so101-pickplace \
  --format=hdf5 \
  --num-episodes=50 \
  --arm-port=/dev/ttyUSB0 \
  --leader-port=/dev/ttyUSB1
```

Requires: `pyrealsense2` + `stable-worldmodel`.
Does NOT require: `lerobot`.

**Reading the HDF5 file back:**
```python
import h5py
import numpy as np

with h5py.File("datasets/myuser__so101-pickplace.h5", "r") as f:
    n_ep = len(f["ep_len"])
    for ep_idx in range(n_ep):
        start = int(f["ep_offset"][ep_idx])
        length = int(f["ep_len"][ep_idx])
        frames = f["pixels"][start:start + length]   # (T, H, W, C) uint8
        actions = f["action"][start:start + length]  # (T, 7) float32
        print(f"Episode {ep_idx}: {length} steps, frames shape: {frames.shape}")
```

---

## 5. Multi-Camera (Placeholder — Future Work)

> **Note:** Multi-camera support is not yet implemented. This example shows the
> intended future API. See §14.7 of the build plan for scope.

```python
# Future: multiple D435 cameras (e.g. front + wrist)
from robot_data_recorder.d435 import make_d435

front_cam = make_d435(serial="123456789", resolution=(640, 480), fps=30)
wrist_cam = make_d435(serial="987654321", resolution=(640, 480), fps=30)

# Future RecordingConfig will accept a list of cameras and map each to a LeRobot
# observation.images.<camera_name> key and a separate HDF5 dataset column.
cfg = RecordingConfig(
    cameras={
        "d435_front": front_cam,
        "d435_wrist": wrist_cam,
    },
    ...
)
```

Until multi-camera is implemented, record each camera separately and merge HDF5 files:

```bash
# Record with front camera
robot-data-record --repo-id=myuser/front --camera-serial=123456789 --format=hdf5

# Record with wrist camera (separate session)
robot-data-record --repo-id=myuser/wrist --camera-serial=987654321 --format=hdf5
```

---

## 6. YAML Config File

Save a config file and reference it by name:

```yaml
# configs/recording_default.yaml
repo_id: myuser/so101-pickplace
num_episodes: 50
format: dual
fps: 30
arm_port: /dev/ttyUSB0
leader_port: /dev/ttyUSB1
resolution: [640, 480]
task: "pick and place cube"
max_steps: 300
output_dir: ./datasets
```

```bash
robot-data-record --config=recording_default --dry-run
# or override individual fields:
robot-data-record --config=recording_default --num-episodes=10 --dry-run
```

In Python:
```python
cfg = RecordingConfig.from_yaml("configs/recording_default.yaml")
# or by name (via lerobot_isaac_configs.load_config):
cfg = RecordingConfig.from_yaml("recording_default")
```
