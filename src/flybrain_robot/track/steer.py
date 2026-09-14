"""Steering from the brain's path estimate, for wheels or for drones.

The research found the motion pathway follows the line one step behind and never
leads. So steering combines the brain's lateral estimate with the brain's own
lateral-motion signal (a derivative term), which counters that lag the way an
optomotor reflex damps drift.
"""

from dataclasses import dataclass

import numpy as np

from ..brain.base import MotorActivity
from ..protocol import validate


@dataclass(frozen=True)
class BrainEstimate:
    lateral: float = 0.0  # brain cursor, path right of vehicle = +
    velocity: float = 0.0  # brain lateral-motion signal this frame


@dataclass(frozen=True)
class SteerCommand:
    forward: float = 0.0  # 0..1
    turn: float = 0.0  # -1..1, + = turn right
    lateral: float = 0.0
    confidence: float = 0.0
    lost: bool = False


class TrackController:
    def __init__(
        self,
        kp=0.9,
        kd=0.3,
        base_speed=0.5,
        turn_gain=0.5,
        min_confidence=0.25,
        lost_frames=15,
    ):
        self.kp, self.kd = kp, kd
        self.base_speed, self.turn_gain = base_speed, turn_gain
        self.min_confidence, self.lost_frames = min_confidence, lost_frames
        self.reset()

    def reset(self):
        self.missing = 0
        self.command = SteerCommand()

    def update(self, estimate, observation):
        """Return MotorActivity for the differential decoder; keeps `command`."""
        if observation.confidence < self.min_confidence:
            self.missing += 1
        else:
            self.missing = 0
        if self.missing >= self.lost_frames:
            self.command = SteerCommand(confidence=observation.confidence, lost=True)
            return MotorActivity()
        turn = float(np.clip(self.kp * estimate.lateral + self.kd * estimate.velocity, -1, 1))
        forward = self.base_speed * (1 - 0.5 * abs(turn))
        self.command = SteerCommand(
            forward=forward,
            turn=turn,
            lateral=float(np.clip(estimate.lateral, -1, 1)),
            confidence=observation.confidence,
        )
        return MotorActivity(
            left=forward + self.turn_gain * turn,
            right=forward - self.turn_gain * turn,
        )


def track_command(sequence, forward=0.0, yaw_rate=0.0, lateral=0.0, confidence=0.0,
                  emergency_stop=False):
    """Drone / generic vehicle steering datagram (see docs/TRACK_FOLLOWING.md)."""
    return validate(
        dict(
            type="track_command",
            sequence=sequence,
            forward=forward,
            yaw_rate=yaw_rate,
            lateral=lateral,
            confidence=confidence,
            emergency_stop=emergency_stop,
        )
    )


def command_from_steer(sequence, steer, max_speed, emergency_stop=False):
    stop = emergency_stop or steer.lost
    return track_command(
        sequence,
        forward=0.0 if stop else round(steer.forward * max_speed, 2),
        yaw_rate=0.0 if stop else round(steer.turn * max_speed, 2),
        lateral=round(steer.lateral, 4),
        confidence=round(steer.confidence, 4),
        emergency_stop=stop,
    )
