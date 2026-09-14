"""Deterministic closed-loop track world: no camera, no network.

A path is drawn into a top-down raster. A differential-drive vehicle carries a
virtual down-and-forward camera that samples that raster, so the encoder sees a
view of the path in front of it, just as a real line-following robot or a
low-flying drone would.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

RESOLUTION = 0.004  # metres per world pixel


@dataclass
class Track:
    points: np.ndarray  # (N, 2) metres, evenly spaced centreline
    width: float = 0.02
    polarity: str = "light"
    spacing: float = 0.005

    @property
    def length(self):
        return self.spacing * (len(self.points) - 1)


def make_track(kind="curvy", seed=0, length=8.0, width=0.02, polarity="light"):
    """Build an evenly sampled centreline. `curvy` has a smooth random curvature."""
    spacing = 0.005
    s = np.arange(0, length + spacing, spacing)
    if kind == "sine":
        xy = np.column_stack([s, 0.25 * np.sin(2 * np.pi * s / 2.0)])
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate(([0], np.cumsum(steps)))
        grid = np.arange(0, arc[-1], spacing)
        xy = np.column_stack([np.interp(grid, arc, xy[:, 0]), np.interp(grid, arc, xy[:, 1])])
    elif kind == "curvy":
        rng = np.random.default_rng(seed)
        curvature = np.zeros_like(s)
        for _ in range(4):
            wavelength = rng.uniform(1.0, 3.0)
            curvature += rng.uniform(-0.7, 0.7) * np.sin(
                2 * np.pi * s / wavelength + rng.uniform(0, 2 * np.pi)
            )
        curvature = np.clip(curvature, -2.5, 2.5)  # tightest radius 0.4 m
        heading = np.cumsum(curvature) * spacing
        xy = np.column_stack(
            [np.cumsum(np.cos(heading)) * spacing, np.cumsum(np.sin(heading)) * spacing]
        )
    elif kind == "straight":
        xy = np.column_stack([s, np.zeros_like(s)])
    else:
        raise ValueError(f"Unknown track kind: {kind}")
    return Track(xy, width=width, polarity=polarity, spacing=spacing)


@dataclass
class Pose:
    x: float
    y: float
    heading: float


class TrackWorld:
    """Top-down raster plus a vehicle-mounted camera."""

    def __init__(
        self,
        track,
        seed=0,
        frame_size=(240, 320),
        near=0.05,
        far=0.45,
        half_width=0.2,
        wheelbase=0.12,
        metres_per_unit=0.01,
    ):
        self.track = track
        self.frame_h, self.frame_w = frame_size
        self.near, self.far, self.half_width = near, far, half_width
        self.wheelbase = wheelbase
        self.metres_per_unit = metres_per_unit  # wheel speed per command unit
        margin = far + 0.5
        self.xmin, self.ymin = track.points.min(axis=0) - margin
        xmax, self.ymax = track.points.max(axis=0) + margin
        width = int(math.ceil((xmax - self.xmin) / RESOLUTION))
        height = int(math.ceil((self.ymax - self.ymin) / RESOLUTION))
        rng = np.random.default_rng(seed)
        ground = rng.normal(60, 12, (height // 8 + 1, width // 8 + 1)).clip(0, 255)
        world = cv2.resize(ground.astype(np.uint8), (width, height), cv2.INTER_LINEAR)
        pixels = np.round(self._to_pixels(track.points) * 16).astype(np.int32)
        thickness = max(1, int(round(track.width / RESOLUTION)))
        cv2.polylines(world, [pixels], False, 235, thickness, cv2.LINE_AA, shift=4)
        self.world = 255 - world if track.polarity == "dark" else world
        start = track.points[0]
        direction = track.points[5] - track.points[0]
        self.pose = Pose(float(start[0]), float(start[1]), math.atan2(direction[1], direction[0]))
        self.index = 0

    def _to_pixels(self, xy):
        return np.column_stack([(xy[:, 0] - self.xmin), (self.ymax - xy[:, 1])]) / RESOLUTION

    def camera(self):
        """BGR frame; top rows look furthest ahead, image right is vehicle right."""
        p = self.pose
        c, s = math.cos(p.heading), math.sin(p.heading)
        df_dv = -(self.far - self.near) / (self.frame_h - 1)
        dl_du = 2 * self.half_width / (self.frame_w - 1)
        # world = pos + forward * (c, s) + lateral * (s, -c); then to pixels.
        wx_u, wx_v = dl_du * s, df_dv * c
        wy_u, wy_v = -dl_du * c, df_dv * s
        wx_0 = p.x + self.far * c - self.half_width * s
        wy_0 = p.y + self.far * s + self.half_width * c
        matrix = np.array(
            [
                [wx_u / RESOLUTION, wx_v / RESOLUTION, (wx_0 - self.xmin) / RESOLUTION],
                [-wy_u / RESOLUTION, -wy_v / RESOLUTION, (self.ymax - wy_0) / RESOLUTION],
            ]
        )
        gray = cv2.warpAffine(
            self.world,
            matrix,
            (self.frame_w, self.frame_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=0 if self.track.polarity == "light" else 255,
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def drive(self, left, right, dt):
        """Advance the unicycle model from wheel commands (-100..100)."""
        vl, vr = left * self.metres_per_unit, right * self.metres_per_unit
        v, omega = (vl + vr) / 2, (vr - vl) / self.wheelbase
        p = self.pose
        heading = p.heading + omega * dt
        mid = p.heading + omega * dt / 2
        self.pose = Pose(p.x + v * math.cos(mid) * dt, p.y + v * math.sin(mid) * dt, heading)
        return v, omega

    def lateral_error(self):
        """Signed distance to the path in metres (path right of vehicle = +)."""
        pts = self.track.points
        lo, hi = max(0, self.index - 100), min(len(pts), self.index + 400)
        window = pts[lo:hi]
        position = np.array([self.pose.x, self.pose.y])
        k = int(np.argmin(np.linalg.norm(window - position, axis=1)))
        self.index = lo + k
        offset = pts[self.index] - position
        c, s = math.cos(self.pose.heading), math.sin(self.pose.heading)
        return float(offset[0] * s - offset[1] * c)

    @property
    def progress(self):
        return self.index / (len(self.track.points) - 1)


@dataclass
class Trace:
    true_lateral: list = field(default_factory=list)
    measured_lateral: list = field(default_factory=list)
    brain_lateral: list = field(default_factory=list)
    brain_velocity: list = field(default_factory=list)
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    lost: int = 0


def run_closed_loop(brain, world, decoder, steps, dt=1 / 30, on_step=None):
    """Drive `world` with `brain` through the upstream MotorDecoder."""
    from ..telemetry import Telemetry

    trace, omega = Trace(), 0.0
    for index in range(steps):
        frame = world.camera()
        sensory = brain.encoder.encode(frame)
        activity = brain.step(sensory, Telemetry(gyro=(0, 0, math.degrees(omega))), dt)
        left, right = decoder.update(activity, now=index * dt)
        _, omega = world.drive(left, right, dt)
        observation = brain.encoder.observation
        trace.true_lateral.append(world.lateral_error())
        trace.measured_lateral.append(observation.lateral)
        trace.brain_lateral.append(brain.estimate.lateral)
        trace.brain_velocity.append(brain.estimate.velocity)
        trace.left.append(left)
        trace.right.append(right)
        trace.progress.append(world.progress)
        trace.lost += int(brain.controller.command.lost)
        if on_step:
            on_step(index, frame, trace)
        if world.progress >= 0.99:
            break
    return trace
