import json
import os
import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from flybrain_robot import main as main_module
from flybrain_robot.config import Config
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.protocol import PacketError, decode, encode
from flybrain_robot.track import build_track_brain
from flybrain_robot.track.brains import OptomotorTrackBrain
from flybrain_robot.track.metrics import lead_lag, spearman, summarise
from flybrain_robot.track.ribbon import PathObservation, PathRibbonEncoder
from flybrain_robot.track.sim import Pose, TrackWorld, make_track, run_closed_loop
from flybrain_robot.track.steer import (
    BrainEstimate,
    SteerCommand,
    TrackController,
    command_from_steer,
    track_command,
)


def line_frame(x, polarity="light"):
    frame = np.full((240, 320, 3), 50 if polarity == "light" else 200, np.uint8)
    cv2.line(frame, (x, 0), (x, 239), (230,) * 3 if polarity == "light" else (20,) * 3, 12)
    return frame


@pytest.mark.parametrize("polarity", ["light", "dark"])
def test_ribbon_lateral_sign(polarity):
    encoder = PathRibbonEncoder(polarity=polarity)
    encoder.encode(line_frame(60, polarity))
    left = encoder.observation
    encoder.reset()
    encoder.encode(line_frame(260, polarity))
    right = encoder.observation
    assert left.lateral < -0.4 and right.lateral > 0.4
    assert left.confidence == right.confidence == 1.0


def test_no_path_has_no_confidence_and_blank_ribbon():
    encoder = PathRibbonEncoder()
    encoder.encode(np.full((240, 320, 3), 80, np.uint8))
    assert encoder.observation.confidence == 0
    assert len(encoder.ribbon_frames) == encoder.substeps
    assert encoder.ribbon_frames[-1].max() == 0


def test_ribbon_puts_path_right_up():
    encoder = PathRibbonEncoder()
    encoder.encode(line_frame(260))
    rows = np.flatnonzero(encoder.ribbon_frames[-1].max(axis=1) > 0.5)
    assert rows.mean() < encoder.height / 2


def test_sim_camera_matches_lateral_error():
    world = TrackWorld(make_track("straight"))
    encoder = PathRibbonEncoder()
    for offset in (-0.1, 0.1):
        world.pose = Pose(0.5, offset, 0.0)
        encoder.reset()
        encoder.encode(world.camera())
        assert np.sign(encoder.observation.lateral) == np.sign(world.lateral_error())
        # Heading +x, left is +y: a vehicle below the line sees it on its left.
        assert np.sign(world.lateral_error()) == np.sign(offset)


def test_controller_turns_toward_path_and_stops_when_lost():
    controller = TrackController(lost_frames=3)
    seen = PathObservation(lateral=0.5, confidence=1.0)
    right = controller.update(BrainEstimate(lateral=0.5), seen)
    assert right.left > right.right
    left = controller.update(BrainEstimate(lateral=-0.5), seen)
    assert left.left < left.right
    for _ in range(3):
        activity = controller.update(BrainEstimate(lateral=0.5), PathObservation())
    assert controller.command.lost and activity.left == activity.right == 0


def reichardt_velocity(direction):
    encoder = PathRibbonEncoder()
    brain = OptomotorTrackBrain(encoder, anchor=0.0)
    yy = np.arange(encoder.height)[:, None] * np.ones((1, encoder.width))
    total = 0.0
    for step in range(120):
        frames = [
            ((yy + direction * (4 * step + s)) % 16 < 8).astype(np.float32) for s in range(4)
        ]
        before = brain.y
        brain.y = brain.perceive(frames, brain.y)
        if step > 20:
            total += brain.y - before
    return total


def test_reichardt_detector_is_direction_selective():
    # direction=+1 moves the pattern up the image (rows decrease).
    assert reichardt_velocity(+1) > 0 > reichardt_velocity(-1)


def test_closed_loop_optomotor_follows_track():
    cfg = Config()
    brain = build_track_brain(cfg, "optomotor-track")
    world = TrackWorld(make_track("curvy", seed=1, length=4.0))
    decoder = MotorDecoder(**{k: getattr(cfg, k) for k in ("max_speed", "smoothing", "dead_zone")})
    trace = run_closed_loop(brain, world, decoder, steps=900)
    summary = summarise(trace)
    assert summary["completion"] >= 0.95
    assert summary["rms_error_m"] < 0.03
    assert summary["lost_frames"] == 0


def test_closed_loop_is_deterministic():
    def run():
        brain = build_track_brain(Config(), "optomotor-track")
        world = TrackWorld(make_track("sine", length=1.5))
        return run_closed_loop(brain, world, MotorDecoder(), steps=60).true_lateral

    assert run() == run()


def test_track_command_roundtrip_and_validation():
    packet = track_command(7, forward=17.5, yaw_rate=-8, lateral=0.2, confidence=0.9)
    assert decode(encode(packet)) == packet
    for key, value in (("lateral", 1.5), ("confidence", -0.1), ("forward", 101), ("yaw_rate", None)):
        with pytest.raises(PacketError):
            decode(json.dumps({**packet, key: value}).encode())
    with pytest.raises(PacketError):
        decode(json.dumps({**packet, "emergency_stop": 0}).encode())


def test_lost_path_track_command_is_emergency_stop():
    packet = command_from_steer(1, SteerCommand(forward=0.5, turn=0.3, lost=True), 35)
    assert packet["emergency_stop"] and packet["forward"] == packet["yaw_rate"] == 0


def test_metrics_lag_sign():
    rng = np.random.default_rng(0)
    truth = np.convolve(rng.normal(size=400), np.ones(9) / 9, mode="same")
    follower = np.roll(truth, 2)
    lag, rho = max(lead_lag(follower, truth, 4), key=lambda row: row[1])
    assert lag == 2 and rho > 0.99
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_track_config_validation():
    with pytest.raises(ValueError):
        Config(output="mavlink")
    with pytest.raises(ValueError):
        Config(track_polarity="blue")
    with pytest.raises(ValueError):
        Config(track_lost_frames=2.5)
    assert Config(backend="connectome-track").track_wiring == "real"


def test_synthetic_track_cli_uses_no_camera_or_socket(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("hardware or network access")

    monkeypatch.setattr(main_module.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    main_module.main(["--backend", "optomotor-track", "--synthetic-track", "--steps", "40",
                      "--output", "track"])
    output = capsys.readouterr().out
    assert "error " in output and "summary steps=40" in output


def test_track_output_needs_track_backend():
    with pytest.raises(SystemExit) as exc:
        main_module.main(["--synthetic", "--output", "track", "--steps", "1"])
    assert exc.value.code == 2


CONNECTOME = os.environ.get(
    "FLYBRAIN_CONNECTOME",
    str(Path(__file__).resolve().parents[2] / "bee/data/processed/female_connectome.npz"),
)


@pytest.mark.skipif(not Path(CONNECTOME).exists(), reason="connectome .npz not available")
def test_connectome_backend_runs_all_wiring_arms():
    pytest.importorskip("numba")
    for wiring in ("real", "shuffled"):
        cfg = Config(backend="connectome-track", track_connectome_path=CONNECTOME,
                     track_wiring=wiring)
        brain = build_track_brain(cfg, "connectome-track")
        world = TrackWorld(make_track("straight", length=1.0))
        trace = run_closed_loop(brain, world, MotorDecoder(), steps=8)
        assert len(trace.true_lateral) == 8
        assert brain.traversals > 0 and np.isfinite(brain.y)
