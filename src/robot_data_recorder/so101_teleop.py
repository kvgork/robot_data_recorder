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

    from robot_data_recorder.so101_teleop import SO101Teleop

    teleop = SO101Teleop(arm_port="/dev/ttyUSB0", leader_port="/dev/ttyUSB1")
    teleop.start()
    state = teleop.read_state()
    action = teleop.read_action()
    teleop.stop()

Usage (tests)::

    from robot_data_recorder.so101_teleop import MockSO101Teleop

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

# Motor order matches lerobot's SO101 follower/leader observation dicts.
# 5 revolute joints + gripper; gripper is treated as the 6th "motor" by lerobot.
_SO101_MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_SO101_KEYS: tuple[str, ...] = tuple(f"{m}.pos" for m in _SO101_MOTORS)
_SO101_DOF = len(_SO101_MOTORS)        # 6 — full motor vector
_SO101_ACTION_DIM = len(_SO101_MOTORS)  # 6 — joint+gripper positions


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
        self._robot = None       # SO101Follower
        self._leader = None      # SO101Leader (or None)
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

        # lerobot >=0.4.4 dropped the legacy `make_robot` factory.
        # Use the per-robot config + class pair directly.
        try:
            from lerobot.robots.so_follower import (  # noqa: PLC0415
                SO101Follower,
                SO101FollowerConfig,
            )

            self._robot = SO101Follower(
                SO101FollowerConfig(port=self._arm_port, id="so101_follower")
            )
            self._robot.connect()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to SO-101 follower on {self._arm_port}: {exc}"
            ) from exc

        if self._leader_port:
            try:
                from lerobot.teleoperators.so_leader import (  # noqa: PLC0415
                    SO101Leader,
                    SO101LeaderConfig,
                )

                self._leader = SO101Leader(
                    SO101LeaderConfig(port=self._leader_port, id="so101_leader")
                )
                self._leader.connect()
            except Exception as exc:
                # Roll back follower so the next attempt starts clean.
                try:
                    self._robot.disconnect()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Failed to connect to SO-101 leader on {self._leader_port}: {exc}"
                ) from exc

        self._started = True

    def read_state(self) -> dict:
        """Read current follower-arm state.

        Returns
        -------
        dict with keys:
            ``joint_pos`` : np.ndarray float32 (6,) — full motor positions
                (5 joints + gripper) in the order :data:`_SO101_MOTORS`.
            ``joint_vel`` : np.ndarray float32 (6,) — zeros (lerobot SO101
                follower does not expose velocity through ``get_observation``).
            ``gripper``   : float — gripper position (== ``joint_pos[-1]``),
                kept for back-compat with downstream code.
            ``timestamp`` : float — monotonic clock in seconds.
        """
        if not self._started:
            raise RuntimeError("Call start() before read_state()")

        obs = self._robot.get_observation()
        joint_pos = np.array(
            [float(obs[k]) for k in _SO101_KEYS], dtype=np.float32
        )

        return {
            "joint_pos": joint_pos,
            "joint_vel": np.zeros(_SO101_DOF, dtype=np.float32),
            "gripper": float(joint_pos[-1]),
            "timestamp": time.monotonic(),
        }

    def read_action(self) -> np.ndarray:
        """Read the leader action and command the follower to mirror it.

        With a leader connected this is the actual teleoperation step:
        the leader's current joint positions are read, forwarded to the
        follower via ``send_action``, and returned as a numpy array for
        the recording buffer. Without a leader the follower's own state
        is returned as a no-op so downstream buffers stay aligned.

        Returns
        -------
        np.ndarray float32 (6,) — [shoulder_pan, shoulder_lift, elbow_flex,
            wrist_flex, wrist_roll, gripper]
        """
        if not self._started:
            raise RuntimeError("Call start() before read_action()")

        if self._leader is not None:
            action_dict = self._leader.get_action()
            # Mirror leader → follower; this is what makes the arms move.
            # send_action returns the (potentially clipped) action that was
            # actually written to the motors — log/record that one.
            sent = self._robot.send_action(action_dict)
            raw = sent if sent is not None else action_dict
        else:
            raw = self._robot.get_observation()

        return np.array(
            [float(raw[k]) for k in _SO101_KEYS], dtype=np.float32
        )

    def stop(self) -> None:
        """Disconnect from the arm hardware (leader + follower)."""
        if not self._started:
            return
        if self._leader is not None:
            try:
                self._leader.disconnect()
            except Exception:
                pass
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
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
        joint_pos = rng.standard_normal(_SO101_DOF).astype(np.float32)
        return {
            "joint_pos": joint_pos,
            "joint_vel": np.zeros(_SO101_DOF, dtype=np.float32),
            "gripper": float(joint_pos[-1]),
            "timestamp": time.monotonic(),
        }

    def read_action(self) -> np.ndarray:
        rng = np.random.default_rng(self._step * 1000)
        return rng.standard_normal(_SO101_ACTION_DIM).astype(np.float32)

    def stop(self) -> None:
        self._started = False
