import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

BACKENDS = ("mock", "malecns", "optomotor-track", "connectome-track")
TRACK_BACKENDS = ("optomotor-track", "connectome-track")


@dataclass
class Config:
    backend: str = "mock"
    camera: int = 0
    video: str | None = None
    pc_host: str = "0.0.0.0"
    pc_port: int = 9001
    robot_ip: str = "192.168.1.50"
    robot_port: int = 9000
    max_speed: float = 35
    smoothing: float = 0.25
    dead_zone: float = 5
    looming_threshold: float = 0.3
    watchdog_timeout: float = 0.5
    invert_left: bool = False
    invert_right: bool = False
    dataset_path: str | None = None
    logging_level: str = "INFO"
    output: str = "differential"
    track_polarity: str = "light"
    track_roi: float = 0.6
    track_substeps: int = 4
    track_anchor: float = 1.0
    track_kp: float = 0.9
    track_kd: float = 0.3
    track_base_speed: float = 0.5
    track_turn_gain: float = 0.5
    track_min_confidence: float = 0.25
    track_lost_frames: int = 15
    track_connectome_path: str | None = None
    track_wiring: str = "real"

    def __post_init__(self):
        bounds = {
            "max_speed": (0, 100),
            "smoothing": (0.001, 1),
            "dead_zone": (0, 100),
            "looming_threshold": (0.001, 1),
            "watchdog_timeout": (0.01, 0.5),
            "track_roi": (0.1, 1),
            "track_substeps": (1, 32),
            "track_anchor": (0, 1),
            "track_kp": (0, 10),
            "track_kd": (0, 10),
            "track_base_speed": (0, 1),
            "track_turn_gain": (0, 1),
            "track_min_confidence": (0, 1),
            "track_lost_frames": (1, 300),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be between {low} and {high}")
        for name in ("pc_port", "robot_port"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 65535:
                raise ValueError(f"Invalid {name}")
        for name in ("invert_left", "invert_right"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("track_substeps", "track_lost_frames"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.backend not in BACKENDS:
            raise ValueError("Unknown backend")
        if self.output not in ("differential", "track"):
            raise ValueError("output must be differential or track")
        if self.track_polarity not in ("light", "dark"):
            raise ValueError("track_polarity must be light or dark")
        if self.track_wiring not in ("real", "shuffled", "random"):
            raise ValueError("track_wiring must be real, shuffled or random")


def load_config(path=None):
    file = Path(path or "config.yaml")
    if not file.exists():
        if path:
            raise ValueError(f"Config does not exist: {path}")
        return Config()
    data = yaml.safe_load(file.read_text()) or {}
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Config)}:
        raise ValueError("Unknown config fields or invalid mapping")
    return Config(**data)
