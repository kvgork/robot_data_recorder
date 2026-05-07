"""
so101_teleop
============

SO-101 robotic arm teleoperation wrapper.

``lerobot`` is soft-imported: the module loads without it, but
``SO101Teleop.start()`` raises a clear ``ImportError`` if it is absent.

The class wraps leader/follower arm communication for teleoperation data
collection. In leader mode, the operator moves the leader arm and the
follower mirrors it while joint positions and actions are recorded.

Usage (real hardware)::

    from lerobot_isaac_recorder.so101_teleop import SO101Teleop

    teleop = SO101Teleop(arm_port="/dev/ttyUSB0", leader_port="/dev/ttyUSB1")
    teleop.start()
    state = teleop.read_state()
    action = teleop.read_action()
    teleop.stop()

Usage (tests)::

    from lerobot_isaac_recorder.so101_teleop import MockSO101Teleop

    teleop = MockSO101Teleop()
    teleop.start()
    state = teleop.read_state()   # synthetic values
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

# Soft-import lerobot -------------------------------------------------------- #
try:
    import lerobot  # type: ignore[import-untyped]  # noqa: F401

    _HAS_LEROBOT = True
except ImportError:
    lerobot = None  # type: ignore[assignment]
    _HAS_LEROBOT = False
# --------------------------------------------------------------------------- #

_SO101_DOF = 6  # 6 revolute joints
_SO101_ACTION_DIM = 7  # 6 joints + gripper


class SO101Teleop:
    """SO-101 leader/follower teleoperation interface.

    Parameters
    ----------
    arm_port:
        Serial port for the SO-101 follower arm (e.g. ``/dev/ttyUSB0``).
    leader_port:
        Serial port for the SO-101 leader arm. If ``None``, the follower arm
        receives zero commands and the caller must supply actions externally.
    """

    def __init__(
        self,
        arm_port: str,
        leader_port: Optional[str] = None,
    ) -> None:
        self._arm_port = arm_port
        self._leader_port = leader_port
        self._robot = None
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Connect to the arm hardware via lerobot.

        Raises
        ------
        ImportError
            If ``lerobot`` is not installed.
        """
        if not _HAS_LEROBOT:
            raise ImportError(
                "lerobot is required to use SO101Teleop. "
                "Install with: bash scripts/install_lerobot.sh"
            )

        # Real implementation would call lerobot's robot factory here.
        # Scaffolding: wrap in try/except so tests with mocked lerobot pass.
        try:
            from lerobot.common.robot_devices.robots.factory import make_robot  # noqa: PLC0415

            cfg = {
                "robot_type": "so101",
                "port": self._arm_port,
            }
            if self._leader_port:
                cfg["leader_port"] = self._leader_port

            self._robot = make_robot(cfg)
            self._robot.connect()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to SO-101 on {self._arm_port}: {exc}"
            ) from exc

        self._started = True

    def read_state(self) -> dict:
        """Read current arm state.

        Returns
        -------
        dict with keys:
            ``joint_pos`` : np.ndarray float32 (6,) — joint positions in radians
            ``joint_vel`` : np.ndarray float32 (6,) — joint velocities
            ``gripper``   : float — gripper opening (0.0 = closed, 1.0 = open)
            ``timestamp`` : float — monotonic clock in seconds
        """
        if not self._started:
            raise RuntimeError("Call start() before read_state()")

        obs = self._robot.get_observation()
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(obs.get("joint_vel", np.zeros(_SO101_DOF)), dtype=np.float32)
        gripper = float(obs.get("gripper", 0.0))

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper": gripper,
            "timestamp": time.monotonic(),
        }

    def read_action(self) -> np.ndarray:
        """Read the current commanded action from the leader arm.

        Returns
        -------
        np.ndarray float32 (7,) — [joint_0 .. joint_5, gripper]
        """
        if not self._started:
            raise RuntimeError("Call start() before read_action()")

        action = self._robot.get_action()
        return np.asarray(action, dtype=np.float32).reshape(_SO101_ACTION_DIM)

    def stop(self) -> None:
        """Disconnect from the arm hardware."""
        if self._started and self._robot is not None:
            self._robot.disconnect()
            self._started = False


class MockSO101Teleop:
    """Synthetic SO-101 for tests and dry-run mode.

    Returns deterministic random state and action arrays with correct shapes.
    """

    def __init__(
        self,
        arm_port: str = "/dev/null",
        leader_port: Optional[str] = None,
    ) -> None:
        self._arm_port = arm_port
        self._leader_port = leader_port
        self._step = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def read_state(self) -> dict:
        rng = np.random.default_rng(self._step)
        self._step += 1
        return {
            "joint_pos": rng.standard_normal(_SO101_DOF).astype(np.float32),
            "joint_vel": rng.standard_normal(_SO101_DOF).astype(np.float32),
            "gripper": float(rng.uniform(0.0, 1.0)),
            "timestamp": time.monotonic(),
        }

    def read_action(self) -> np.ndarray:
        rng = np.random.default_rng(self._step * 1000)
        return rng.standard_normal(_SO101_ACTION_DIM).astype(np.float32)

    def stop(self) -> None:
        self._started = False
