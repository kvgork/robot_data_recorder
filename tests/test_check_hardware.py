"""
test_check_hardware.py — unit tests for the hardware pre-flight check.

All tests run without real hardware: env vars are monkeypatched and the
RealSense / lerobot probes are mocked or skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_data_recorder import check_hardware as ch


# ------------------------------------------------------------------ #
# Env-var checks
# ------------------------------------------------------------------ #


def test_env_var_set_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/null")
    r = ch._check_env_var("LERO_FOLLOWER_PORT", required=True)
    assert r.status == "ok"
    assert r.detail == "/dev/null"


def test_env_var_unset_required_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_FOLLOWER_PORT", raising=False)
    r = ch._check_env_var("LERO_FOLLOWER_PORT", required=True)
    assert r.status == "fail"
    assert r.is_blocker() is True


def test_env_var_unset_optional_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_LEADER_PORT", raising=False)
    r = ch._check_env_var("LERO_LEADER_PORT", required=False)
    assert r.status == "warn"
    assert r.is_blocker() is False


# ------------------------------------------------------------------ #
# tty checks
# ------------------------------------------------------------------ #


def test_tty_check_passes_for_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # /dev/null exists and is rw on every Linux system
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/null")
    r = ch._check_tty("follower", "LERO_FOLLOWER_PORT")
    assert r.status == "ok"


def test_tty_check_fails_for_missing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/definitely-not-a-real-port")
    r = ch._check_tty("follower", "LERO_FOLLOWER_PORT")
    assert r.status == "fail"
    assert "does not exist" in r.detail


def test_tty_check_skips_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LERO_LEADER_PORT", raising=False)
    r = ch._check_tty("leader", "LERO_LEADER_PORT")
    assert r.status == "skip"
    assert r.is_blocker() is False


# ------------------------------------------------------------------ #
# Top-level driver
# ------------------------------------------------------------------ #


def test_run_checks_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/null")
    monkeypatch.delenv("LERO_LEADER_PORT", raising=False)
    monkeypatch.delenv("LERO_CAM_SERIAL", raising=False)
    report = ch.run_checks(connect=False)
    assert isinstance(report, ch.CheckReport)
    assert len(report.results) > 0
    # Every result must have a known status
    for r in report.results:
        assert r.status in {"ok", "fail", "warn", "skip"}


def test_main_json_output_is_parseable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/null")
    rc = ch.main(["--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "results" in payload
    assert isinstance(payload["results"], list)
    assert isinstance(rc, int)


def test_main_returns_nonzero_on_blocker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("LERO_FOLLOWER_PORT", raising=False)
    rc = ch.main(["--json"])
    assert rc == 1


def test_main_returns_zero_when_required_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With all hardware-touching probes disabled via env, a present
    follower env var + dialout membership should not block."""
    monkeypatch.setenv("LERO_FOLLOWER_PORT", "/dev/null")
    # Force the realsense probe to a non-blocking failure mode by stubbing
    monkeypatch.setattr(
        ch,
        "_check_realsense",
        lambda _serial: [
            ch.CheckResult(
                name="realsense:import",
                status="warn",
                detail="stubbed",
                required=False,
            )
        ],
    )
    # Stub dialout group check to non-blocking
    monkeypatch.setattr(
        ch,
        "_check_dialout_group",
        lambda: ch.CheckResult(
            name="group:dialout", status="ok", detail="stubbed"
        ),
    )
    rc = ch.main([])
    assert rc == 0
