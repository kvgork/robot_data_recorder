"""
check_hardware
==============

Pre-flight hardware connectivity check for robot-data-recorder.

Verifies:
- Required env vars (LERO_FOLLOWER_PORT, LERO_LEADER_PORT, LERO_CAM_SERIAL)
- Serial tty paths exist + are accessible (read/write perm via dialout group)
- pyrealsense2 importable and the D435 (or any RealSense) camera enumerated
- Optional: try connect()/disconnect() on SO101 follower + leader to verify
  the full lerobot path works (use ``--connect``; takes a few seconds).

Exit code: 0 if all required checks pass, 1 otherwise. Optional/info checks
do not change the exit code.

CLI::

    pixi run robot-data-check
    pixi run robot-data-check --connect       # also try lerobot connect()
    pixi run robot-data-check --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ANSI colours (auto-disabled if stdout is not a tty)
def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_USE_COLOUR = _supports_colour()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOUR else s


def _ok(s: str) -> str:
    return _c("32", s)


def _fail(s: str) -> str:
    return _c("31", s)


def _warn(s: str) -> str:
    return _c("33", s)


def _dim(s: str) -> str:
    return _c("2", s)


@dataclass
class CheckResult:
    name: str
    status: str   # "ok" | "fail" | "warn" | "skip"
    detail: str = ""
    required: bool = True

    def is_blocker(self) -> bool:
        return self.required and self.status == "fail"


@dataclass
class CheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    def has_blockers(self) -> bool:
        return any(r.is_blocker() for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"results": [asdict(r) for r in self.results]}


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _check_env_var(name: str, *, required: bool) -> CheckResult:
    val = os.environ.get(name)
    if val:
        return CheckResult(name=f"env:{name}", status="ok", detail=val, required=required)
    if required:
        return CheckResult(
            name=f"env:{name}",
            status="fail",
            detail="unset (set via `pixi run setup-env` or ~/.bashrc)",
            required=True,
        )
    return CheckResult(
        name=f"env:{name}", status="warn", detail="unset (optional)", required=False
    )


def _check_tty(label: str, env_name: str) -> CheckResult:
    port = os.environ.get(env_name)
    if not port:
        return CheckResult(
            name=f"tty:{label}",
            status="skip",
            detail=f"{env_name} unset",
            required=False,
        )
    p = Path(port)
    if not p.exists():
        return CheckResult(
            name=f"tty:{label}",
            status="fail",
            detail=f"{port} does not exist (USB unplugged?)",
        )
    if not os.access(p, os.R_OK | os.W_OK):
        return CheckResult(
            name=f"tty:{label}",
            status="fail",
            detail=f"{port} not readable/writable (run: sudo usermod -aG dialout $USER, then re-login)",
        )
    return CheckResult(name=f"tty:{label}", status="ok", detail=str(port))


def _check_dialout_group() -> CheckResult:
    try:
        import grp  # noqa: PLC0415

        user = os.environ.get("USER") or os.getlogin()
        members = {
            g.gr_name for g in grp.getgrall() if user in g.gr_mem
        }
        if "dialout" in members:
            return CheckResult(name="group:dialout", status="ok", detail=user)
        return CheckResult(
            name="group:dialout",
            status="fail",
            detail=(
                f"user {user!r} not in dialout group "
                "(run: sudo usermod -aG dialout $USER, then re-login)"
            ),
        )
    except Exception as exc:
        return CheckResult(
            name="group:dialout", status="warn", detail=str(exc), required=False
        )


def _check_realsense(expected_serial: Optional[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        out.append(
            CheckResult(
                name="realsense:import",
                status="fail",
                detail=f"pyrealsense2 not installed: {exc}",
            )
        )
        return out

    out.append(CheckResult(name="realsense:import", status="ok"))

    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
    except Exception as exc:
        out.append(
            CheckResult(
                name="realsense:enumerate",
                status="fail",
                detail=f"rs.context().query_devices() failed: {exc}",
            )
        )
        return out

    if not devices:
        out.append(
            CheckResult(
                name="realsense:enumerate",
                status="fail",
                detail="no RealSense devices found (USB3 cable connected?)",
            )
        )
        return out

    found_serials: list[str] = []
    for d in devices:
        try:
            name = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
        except Exception:
            name, serial = "<unknown>", "<unknown>"
        found_serials.append(serial)
        out.append(
            CheckResult(
                name="realsense:device",
                status="ok",
                detail=f"{name} (serial {serial})",
                required=False,
            )
        )

    if expected_serial:
        if expected_serial in found_serials:
            out.append(
                CheckResult(
                    name="realsense:serial-match",
                    status="ok",
                    detail=f"LERO_CAM_SERIAL={expected_serial} present",
                )
            )
        else:
            out.append(
                CheckResult(
                    name="realsense:serial-match",
                    status="fail",
                    detail=(
                        f"LERO_CAM_SERIAL={expected_serial} not among "
                        f"connected devices: {found_serials}"
                    ),
                )
            )
    return out


def _try_connect_so101_follower(port: str) -> CheckResult:
    try:
        from lerobot.robots.so_follower import (  # noqa: PLC0415
            SO101Follower,
            SO101FollowerConfig,
        )

        robot = SO101Follower(SO101FollowerConfig(port=port, id="hwcheck_follower"))
        robot.connect()
        try:
            obs = robot.get_observation()
            n_keys = len(obs)
        finally:
            robot.disconnect()
        return CheckResult(
            name="so101:follower",
            status="ok",
            detail=f"connected on {port} ({n_keys} motor keys)",
        )
    except Exception as exc:
        return CheckResult(
            name="so101:follower",
            status="fail",
            detail=f"connect failed on {port}: {exc}",
        )


def _try_connect_so101_leader(port: str) -> CheckResult:
    try:
        from lerobot.teleoperators.so_leader import (  # noqa: PLC0415
            SO101Leader,
            SO101LeaderConfig,
        )

        leader = SO101Leader(SO101LeaderConfig(port=port, id="hwcheck_leader"))
        leader.connect()
        try:
            action = leader.get_action()
            n_keys = len(action)
        finally:
            leader.disconnect()
        return CheckResult(
            name="so101:leader",
            status="ok",
            detail=f"connected on {port} ({n_keys} motor keys)",
        )
    except Exception as exc:
        return CheckResult(
            name="so101:leader",
            status="fail",
            detail=f"connect failed on {port}: {exc}",
        )


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #


def run_checks(connect: bool = False) -> CheckReport:
    """Run every check and return a :class:`CheckReport`."""
    report = CheckReport()

    # Env vars
    report.add(_check_env_var("LERO_FOLLOWER_PORT", required=True))
    report.add(_check_env_var("LERO_LEADER_PORT", required=False))
    report.add(_check_env_var("LERO_CAM_SERIAL", required=False))

    # User group
    report.add(_check_dialout_group())

    # Serial ports
    report.add(_check_tty("follower", "LERO_FOLLOWER_PORT"))
    report.add(_check_tty("leader", "LERO_LEADER_PORT"))

    # RealSense
    for r in _check_realsense(os.environ.get("LERO_CAM_SERIAL")):
        report.add(r)

    # Optional: full lerobot connect
    if connect:
        follower_port = os.environ.get("LERO_FOLLOWER_PORT")
        leader_port = os.environ.get("LERO_LEADER_PORT")
        if follower_port:
            report.add(_try_connect_so101_follower(follower_port))
        if leader_port:
            report.add(_try_connect_so101_leader(leader_port))

    return report


_GLYPH = {
    "ok": "[ OK ]",
    "fail": "[FAIL]",
    "warn": "[WARN]",
    "skip": "[SKIP]",
}


def _format_human(report: CheckReport) -> str:
    lines: list[str] = []
    for r in report.results:
        glyph = _GLYPH[r.status]
        if r.status == "ok":
            glyph = _ok(glyph)
        elif r.status == "fail":
            glyph = _fail(glyph)
        elif r.status == "warn":
            glyph = _warn(glyph)
        else:
            glyph = _dim(glyph)
        line = f"{glyph} {r.name:<28} {r.detail}".rstrip()
        lines.append(line)

    blockers = [r for r in report.results if r.is_blocker()]
    lines.append("")
    if blockers:
        lines.append(_fail(f"FAIL: {len(blockers)} blocking check(s)"))
        for r in blockers:
            lines.append(f"  - {r.name}: {r.detail}")
    else:
        lines.append(_ok("OK: hardware ready for recording"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robot-data-check",
        description="Verify D435 + SO-101 hardware is wired up for robot-data-record.",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Also try a real lerobot connect() on the SO-101 arms (slower).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human output.",
    )
    args = parser.parse_args(argv)

    report = run_checks(connect=args.connect)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_human(report))

    return 1 if report.has_blockers() else 0


if __name__ == "__main__":
    sys.exit(main())
