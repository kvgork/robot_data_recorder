"""
test_config_env.py — RecordingConfig env-var defaults.

LERO_FOLLOWER_PORT / LERO_LEADER_PORT / LERO_CAM_SERIAL feed the
arm_port / leader_port / camera_serial defaults.
"""

from __future__ import annotations

import pytest

from robot_data_recorder.config import RecordingConfig


def test_arm_port_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/from-env-follower")
    cfg = RecordingConfig()
    assert cfg.arm_port == "/dev/from-env-follower"


def test_arm_port_fallback_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_FOLLOWER_PORT", raising=False)
    cfg = RecordingConfig()
    assert cfg.arm_port == "/dev/ttyUSB0"


def test_leader_port_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_LEADER_PORT", "/dev/from-env-leader")
    cfg = RecordingConfig()
    assert cfg.leader_port == "/dev/from-env-leader"


def test_leader_port_none_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_LEADER_PORT", raising=False)
    cfg = RecordingConfig()
    assert cfg.leader_port is None


def test_leader_port_none_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_LEADER_PORT", "")
    cfg = RecordingConfig()
    assert cfg.leader_port is None


def test_camera_serial_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_CAM_SERIAL", "ENVSERIAL999")
    cfg = RecordingConfig()
    assert cfg.camera_serial == "ENVSERIAL999"


def test_camera_serial_none_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_CAM_SERIAL", raising=False)
    cfg = RecordingConfig()
    assert cfg.camera_serial is None


def test_camera_serial_none_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_CAM_SERIAL", "")
    cfg = RecordingConfig()
    assert cfg.camera_serial is None


def test_explicit_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/from-env")
    monkeypatch.setenv("LERO_LEADER_PORT", "/dev/from-env-leader")
    monkeypatch.setenv("LERO_CAM_SERIAL", "from-env-serial")
    cfg = RecordingConfig(
        arm_port="/dev/explicit",
        leader_port="/dev/explicit-leader",
        camera_serial="explicit-serial",
    )
    assert cfg.arm_port == "/dev/explicit"
    assert cfg.leader_port == "/dev/explicit-leader"
    assert cfg.camera_serial == "explicit-serial"
