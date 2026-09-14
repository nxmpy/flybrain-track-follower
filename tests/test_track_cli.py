import cv2
import numpy as np
import pytest

from flybrain_robot.config import load_config
from flybrain_robot.track.sim import Pose, TrackWorld, make_track
from flybrain_robot.track_cli import main


def write_video(path, polarity="light", frames=20):
    world = TrackWorld(make_track("straight", length=3, polarity=polarity))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(frames):
        world.pose = Pose(0.5 + i * 0.006, 0.03, 0.0)
        writer.write(world.camera())
    writer.release()
    return path


@pytest.mark.parametrize("preset", ["robot", "drone"])
def test_init_writes_valid_config_and_refuses_overwrite(tmp_path, preset, capsys):
    path = tmp_path / "config.yaml"
    assert main(["init", "--preset", preset, "--path", str(path)]) == 0
    cfg = load_config(path)
    assert cfg.backend == "optomotor-track"
    assert cfg.output == ("track" if preset == "drone" else "differential")
    with pytest.raises(SystemExit) as exc:
        main(["init", "--path", str(path)])
    assert exc.value.code == 2
    assert "--force" in capsys.readouterr().err
    assert main(["init", "--path", str(path), "--force"]) == 0


@pytest.mark.parametrize("polarity", ["light", "dark"])
def test_check_recommends_polarity_and_writes_snapshot(tmp_path, polarity, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = write_video(tmp_path / "clip.mp4", polarity)
    snapshot = tmp_path / "snap.png"
    assert main(["check", "--video", str(video), "--snapshot", str(snapshot)]) == 0
    out = capsys.readouterr().out
    assert f"Recommended setting: track_polarity: {polarity}" in out
    assert cv2.imread(str(snapshot)).shape == (240, 320, 3)


def test_check_reports_missing_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(10):
        writer.write(np.full((240, 320, 3), 90, np.uint8))
    writer.release()
    assert main(["check", "--video", str(video), "--snapshot", str(tmp_path / "s.png")]) == 1
    assert "No reliable path found" in capsys.readouterr().out


def test_sim_prints_plain_result_and_records_preview(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = tmp_path / "preview.mp4"
    main(["sim", "--steps", "30", "--kind", "sine", "--record", str(record)])
    out = capsys.readouterr().out
    assert "Result: still on track when --steps ran out" in out
    capture = cv2.VideoCapture(str(record))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (240, 640, 3)


def test_run_video_dry_run_steers_toward_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["run", "--video", str(write_video(tmp_path / "clip.mp4")), "--steps", "20"])
    out = capsys.readouterr().out
    assert "UDP=off (dry-run)" in out and "path right" in out


def test_config_typo_names_the_closest_setting(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("track_polarty: dark\n")
    with pytest.raises(SystemExit):
        main(["sim", "--steps", "1"])
    assert "did you mean track_polarity?" in capsys.readouterr().err


def test_mock_backend_rejects_preview_with_hint(capsys):
    from flybrain_robot import main as bridge_main

    with pytest.raises(SystemExit):
        bridge_main.main(["--synthetic", "--record", "x.mp4", "--steps", "1"])
    assert "--backend optomotor-track" in capsys.readouterr().err
