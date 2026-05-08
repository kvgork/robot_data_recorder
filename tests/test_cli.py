"""
test_cli.py — CLI argument parsing and dry-run tests.

All tests run without hardware or optional heavy deps.
"""

from __future__ import annotations

import pytest

from robot_data_recorder.cli import _build_parser, _parse_resolution, main


# ------------------------------------------------------------------ #
# _parse_resolution
# ------------------------------------------------------------------ #

def test_parse_resolution_standard() -> None:
    assert _parse_resolution("640x480") == (640, 480)


def test_parse_resolution_uppercase() -> None:
    assert _parse_resolution("1280X720") == (1280, 720)


def test_parse_resolution_invalid_raises() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_resolution("not-a-resolution")


# ------------------------------------------------------------------ #
# Argparse smoke tests
# ------------------------------------------------------------------ #

def test_parser_requires_repo_id() -> None:
    """Omitting --repo-id must cause a parse error."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-episodes=1"])


def test_parser_default_format_is_dual() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--repo-id=test/run"])
    assert args.format == "dual"


def test_parser_format_parquet() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--repo-id=test/run", "--format=parquet"])
    assert args.format == "parquet"


def test_parser_format_hdf5() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--repo-id=test/run", "--format=hdf5"])
    assert args.format == "hdf5"


def test_parser_invalid_format_rejected() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--repo-id=test/run", "--format=npy"])


def test_parser_all_flags_parseable() -> None:
    """Smoke test: all documented flags parse without error."""
    parser = _build_parser()
    args = parser.parse_args([
        "--repo-id=myuser/pickplace",
        "--num-episodes=5",
        "--format=dual",
        "--output-dir=/tmp/datasets",
        "--task=pick and place",
        "--resolution=640x480",
        "--fps=30",
        "--arm-port=/dev/ttyUSB0",
        "--leader-port=/dev/ttyUSB1",
        "--camera-serial=AUTO",
        "--max-steps=100",
        "--dry-run",
    ])
    assert args.repo_id == "myuser/pickplace"
    assert args.num_episodes == 5
    assert args.dry_run is True


def test_parser_depth_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--repo-id=test/run", "--depth"])
    assert args.depth is True


def test_parser_depth_flag_off_by_default() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--repo-id=test/run"])
    assert args.depth is False


# ------------------------------------------------------------------ #
# --dry-run returns 0
# ------------------------------------------------------------------ #

def test_dry_run_returns_zero() -> None:
    rc = main(["--repo-id=test/demo", "--num-episodes=1", "--dry-run"])
    assert rc == 0


def test_dry_run_parquet_returns_zero() -> None:
    rc = main(["--repo-id=test/demo", "--format=parquet", "--dry-run"])
    assert rc == 0


def test_dry_run_hdf5_returns_zero() -> None:
    rc = main(["--repo-id=test/demo", "--format=hdf5", "--dry-run"])
    assert rc == 0


def test_dry_run_prints_config(capsys: pytest.CaptureFixture) -> None:
    main(["--repo-id=test/demo", "--dry-run"])
    captured = capsys.readouterr()
    assert "repo_id" in captured.out
    assert "test/demo" in captured.out
    assert "DRY RUN" in captured.out


# ------------------------------------------------------------------ #
# Help text mentions all formats
# ------------------------------------------------------------------ #

def test_help_mentions_parquet(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "parquet" in captured.out


def test_help_mentions_hdf5(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "hdf5" in captured.out


def test_help_mentions_dual(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "dual" in captured.out
