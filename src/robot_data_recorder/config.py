"""
config
======

RecordingConfig dataclass for robot-data-recorder.

All recording session parameters are captured here. The dataclass is
intentionally flat (no nested sub-configs) to keep YAML files simple
and to make ``to_dict()`` produce a clean printout for ``--dry-run``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_follower_port() -> str:
    """Default follower-arm serial port. Reads ``LERO_FOLLOWER_PORT`` env var.

    Set via ``pixi run setup-env`` (writes ``.env``) or exported in ``~/.bashrc``.
    """
    return os.environ.get("LERO_FOLLOWER_PORT", "/dev/ttyUSB0")


def _env_leader_port() -> str | None:
    """Default leader-arm serial port. Reads ``LERO_LEADER_PORT`` env var."""
    val = os.environ.get("LERO_LEADER_PORT")
    return val or None


def _env_camera_serial() -> str | None:
    """Default D435 camera serial. Reads ``LERO_CAM_SERIAL`` env var."""
    val = os.environ.get("LERO_CAM_SERIAL")
    return val or None


@dataclass
class RecordingConfig:
    """Full configuration for a recording session.

    Attributes
    ----------
    repo_id:
        HuggingFace repo id or local dataset name (e.g. ``myuser/so101-pickplace``).
    num_episodes:
        Number of episodes to record per session.
    format:
        Output format: ``parquet`` (LeRobot only), ``hdf5`` (LeWM only), or
        ``dual`` (both simultaneously).
    output_dir:
        Base directory for output files. Default: ``./datasets``.
    task:
        Human-readable task description written into dataset metadata.
    fps:
        Recording frame rate (Hz). Default: 30.
    arm_port:
        Serial port for the SO-101 follower arm (e.g. ``/dev/ttyUSB0``).
        Default reads ``LERO_FOLLOWER_PORT`` env var, falling back to ``/dev/ttyUSB0``.
    leader_port:
        Serial port for the SO-101 leader arm. ``None`` disables leader control.
        Default reads ``LERO_LEADER_PORT`` env var.
    camera_serial:
        RealSense D435 serial number. ``None`` / ``"AUTO"`` selects first device.
        Default reads ``LERO_CAM_SERIAL`` env var.
    resolution:
        Camera resolution as ``(width, height)`` tuple. Default: ``(640, 480)``.
    enable_depth:
        Whether to capture the D435 depth stream. Default: ``False``.
    max_steps:
        Hard safety ceiling on episode length. The recorder normally ends an
        episode when the operator presses SPACE/ENTER; ``max_steps`` only
        kicks in if no key is pressed. Default: ``18000`` (= 10 minutes
        @ 30 Hz).
    dry_run:
        If ``True``, bypass all hardware and write nothing. Default: ``False``.
    """

    repo_id: str = "local/recording"
    num_episodes: int = 1
    format: str = "dual"
    output_dir: str = "./datasets"
    task: str = "unspecified"
    fps: int = 30
    arm_port: str = field(default_factory=_env_follower_port)
    leader_port: str | None = field(default_factory=_env_leader_port)
    camera_serial: str | None = field(default_factory=_env_camera_serial)
    resolution: tuple[int, int] = field(default_factory=lambda: (640, 480))
    enable_depth: bool = False
    max_steps: int = 18000
    dry_run: bool = False

    # ------------------------------------------------------------------ #
    # Class-method constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RecordingConfig":
        """Load config from a YAML file via lerobot_isaac_configs.load_config.

        The YAML file is resolved through the configs package loader so that
        both named configs (``recording_default``) and absolute paths work.

        Parameters
        ----------
        path:
            Path to a YAML file OR a config name understood by
            ``lerobot_isaac_configs.load_config``.

        Returns
        -------
        RecordingConfig
            Populated from the YAML content; missing keys use defaults.
        """
        import yaml  # pyyaml is a hard dep

        resolved = Path(path)
        if resolved.is_file():
            with open(resolved) as fh:
                data = yaml.safe_load(fh) or {}
        else:
            # Delegate to configs package loader
            try:
                from lerobot_isaac_configs import load_config  # noqa: PLC0415

                data = load_config(str(path))
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not resolve config '{path}' as file or named config: {exc}"
                ) from exc

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        # resolution is stored as list in YAML, convert to tuple
        if "resolution" in filtered and isinstance(filtered["resolution"], list):
            filtered["resolution"] = tuple(filtered["resolution"])

        return cls(**filtered)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for ``--dry-run`` printout."""
        d = dataclasses.asdict(self)
        d["resolution"] = list(d["resolution"])  # tuple -> list for JSON compat
        return d
