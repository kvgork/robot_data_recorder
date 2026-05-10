"""
cli
===

Command-line entrypoint for robot-data-recorder.

    robot-data-record --repo-id=myuser/pickplace --num-episodes=10 \\
        --format=dual --resolution=640x480 --fps=30 \\
        --arm-port=/dev/ttyUSB0 --camera-serial=AUTO \\
        --output-dir=./datasets --task="pick and place cube"

Use ``--dry-run`` to print the resolved config and exit without touching hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    env_follower = os.environ.get("LERO_FOLLOWER_PORT", "/dev/ttyUSB0")
    env_leader = os.environ.get("LERO_LEADER_PORT") or None
    env_cam = os.environ.get("LERO_CAM_SERIAL") or None
    p = argparse.ArgumentParser(
        prog="robot-data-record",
        description=(
            "Record teleoperation episodes from D435 camera + SO-101 arm "
            "and write to LeRobot Parquet and/or LeWM HDF5 format."
        ),
    )

    # Dataset / output
    p.add_argument(
        "--repo-id",
        required=True,
        metavar="REPO_ID",
        help="HuggingFace repo id or local name (e.g. myuser/so101-pickplace)",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes to record (default: 1)",
    )
    p.add_argument(
        "--format",
        choices=["parquet", "hdf5", "dual"],
        default="dual",
        help="Output format: parquet (LeRobot only), hdf5 (LeWM only), dual (both). "
             "Default: dual",
    )
    p.add_argument(
        "--output-dir",
        default="./datasets",
        metavar="DIR",
        help="Base directory for output files (default: ./datasets)",
    )
    p.add_argument(
        "--task",
        default="unspecified",
        metavar="TEXT",
        help="Human-readable task description written to dataset metadata",
    )

    # Camera
    p.add_argument(
        "--resolution",
        default="640x480",
        metavar="WxH",
        help="Camera resolution as WxH (default: 640x480)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="FPS",
        help="Recording frame rate in Hz (default: 30)",
    )
    p.add_argument(
        "--camera-serial",
        default=env_cam,
        metavar="SERIAL",
        help=(
            "RealSense D435 serial number. 'AUTO' or omit to use first device. "
            "Default reads $LERO_CAM_SERIAL."
        ),
    )
    p.add_argument(
        "--depth",
        action="store_true",
        default=False,
        help="Enable D435 depth stream (written as 'depth' uint16 key in HDF5)",
    )

    # Arm
    p.add_argument(
        "--arm-port",
        default=env_follower,
        metavar="PORT",
        help=(
            "Serial port for SO-101 follower arm. "
            "Default reads $LERO_FOLLOWER_PORT, falling back to /dev/ttyUSB0."
        ),
    )
    p.add_argument(
        "--leader-port",
        default=env_leader,
        metavar="PORT",
        help=(
            "Serial port for SO-101 leader arm. Omit for scripted/replay mode. "
            "Default reads $LERO_LEADER_PORT."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        metavar="N",
        help="Maximum steps per episode (default: 200). Used as episode timeout.",
    )

    # Session control
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print resolved config and exit 0 without touching hardware.",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH_OR_NAME",
        help="Optional YAML config file or named config to load defaults from.",
    )

    return p


def _parse_resolution(s: str) -> tuple[int, int]:
    """Parse 'WxH' string into (width, height) tuple."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Resolution must be WxH (e.g. 640x480), got {s!r}"
        ) from exc


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv:
        Argument list. If ``None``, reads ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    from robot_data_recorder.config import RecordingConfig  # noqa: PLC0415

    # Start from YAML base if --config provided
    if args.config:
        cfg = RecordingConfig.from_yaml(args.config)
    else:
        cfg = RecordingConfig()

    # CLI overrides always win
    resolution = _parse_resolution(args.resolution)
    camera_serial = (
        None if args.camera_serial in (None, "AUTO") else args.camera_serial
    )

    cfg = RecordingConfig(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        format=args.format,
        output_dir=args.output_dir,
        task=args.task,
        fps=args.fps,
        arm_port=args.arm_port,
        leader_port=args.leader_port,
        camera_serial=camera_serial,
        resolution=resolution,
        enable_depth=args.depth,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[robot-data-record] DRY RUN — resolved config:")
        print(json.dumps(cfg.to_dict(), indent=2))
        print()
        print("[robot-data-record] Would run:")
        print(
            f"  robot-data-record "
            f"--repo-id={cfg.repo_id} "
            f"--num-episodes={cfg.num_episodes} "
            f"--format={cfg.format} "
            f"--output-dir={cfg.output_dir}"
        )
        return 0

    # Real recording path
    from robot_data_recorder.d435 import make_d435  # noqa: PLC0415
    from robot_data_recorder.dual_writer import DualWriter  # noqa: PLC0415
    from robot_data_recorder.recorder import RecordingSession  # noqa: PLC0415
    from robot_data_recorder.so101_teleop import SO101Teleop  # noqa: PLC0415

    camera = make_d435(
        serial=cfg.camera_serial,
        resolution=cfg.resolution,
        fps=cfg.fps,
        enable_depth=cfg.enable_depth,
    )
    teleop = SO101Teleop(arm_port=cfg.arm_port, leader_port=cfg.leader_port)
    writer = DualWriter(cfg)

    with RecordingSession(cfg, camera=camera, teleop=teleop, writer=writer) as session:
        for ep_idx in range(cfg.num_episodes):
            print(
                f"[robot-data-record] Recording episode {ep_idx + 1}/{cfg.num_episodes} ..."
            )
            buf = session.record_episode(ep_idx)
            session.save_episode(buf)
            print(
                f"[robot-data-record] Episode {ep_idx + 1} saved ({len(buf.pixels)} steps)"
            )

    paths = writer.finalize()
    print("[robot-data-record] Done. Output paths:")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
