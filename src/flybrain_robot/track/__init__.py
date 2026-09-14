"""Path and track following: camera path -> ribbon -> insect motion model -> steering."""

from .brains import ConnectomeTrackBrain, OptomotorTrackBrain
from .ribbon import PathRibbonEncoder
from .steer import TrackController


def build_track_brain(cfg, backend):
    """Construct encoder, controller and brain for a track backend from Config."""
    encoder = PathRibbonEncoder(
        polarity=cfg.track_polarity, roi=cfg.track_roi, substeps=cfg.track_substeps
    )
    controller = TrackController(
        kp=cfg.track_kp,
        kd=cfg.track_kd,
        base_speed=cfg.track_base_speed,
        turn_gain=cfg.track_turn_gain,
        min_confidence=cfg.track_min_confidence,
        lost_frames=cfg.track_lost_frames,
    )
    if backend == "optomotor-track":
        return OptomotorTrackBrain(encoder, controller, anchor=cfg.track_anchor)
    if backend == "connectome-track":
        return ConnectomeTrackBrain(
            encoder,
            cfg.track_connectome_path,
            controller,
            anchor=cfg.track_anchor,
            wiring=cfg.track_wiring,
        )
    raise ValueError(f"Not a track backend: {backend}")
