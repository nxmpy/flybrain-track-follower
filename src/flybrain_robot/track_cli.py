"""flybrain-track: set up, check and run camera path following.

  flybrain-track init --preset robot      write a starter config.yaml
  flybrain-track check --camera 0         is the path visible? which settings?
  flybrain-track sim                      closed-loop simulated track
  flybrain-track run --camera 0           follow a real path (dry-run until --send)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import main as bridge_main
from .config import TRACK_BACKENDS, load_config

PRESETS = {
    "robot": {
        "about": "Differential-drive ground robot following a line on the floor",
        "values": {
            "backend": "optomotor-track",
            "output": "differential",
            "robot_ip": "192.168.1.50",
            "max_speed": 35,
            "dead_zone": 2,
            "track_polarity": "light",
            "track_base_speed": 0.5,
        },
    },
    "drone": {
        "about": "Drone or other vehicle whose companion computer reads track_command",
        "values": {
            "backend": "optomotor-track",
            "output": "track",
            "robot_ip": "192.168.1.60",
            "max_speed": 30,
            "dead_zone": 0,
            "track_polarity": "light",
            "track_base_speed": 0.3,
            "track_lost_frames": 10,
        },
    },
}

COMMENTS = {
    "backend": "optomotor-track (fast, default) or connectome-track (experimental)",
    "output": "differential = left/right wheels, track = forward/yaw_rate datagram",
    "robot_ip": "vehicle address; commands are only sent with --send",
    "max_speed": "command scale 0..100; start low",
    "dead_zone": "commands below this are zeroed; keep small for line following",
    "track_polarity": "light = bright line on dark floor, dark = black tape on light floor",
    "track_base_speed": "cruising speed as a fraction of max_speed",
    "track_lost_frames": "frames without a visible path before emergency stop",
}


def cmd_init(args):
    path = Path(args.path)
    if path.exists() and not args.force:
        raise ValueError(f"{path} already exists; use --force to overwrite")
    preset = PRESETS[args.preset]
    lines = [f"# flybrain-track config: {preset['about']}.",
             "# All options: config.example.yaml. Check your camera: flybrain-track check", ""]
    for key, value in preset["values"].items():
        lines.append(f"{key}: {value}  # {COMMENTS[key]}")
    path.write_text("\n".join(lines) + "\n")
    load_config(path)  # validate what was written
    print(f"Wrote {path} ({args.preset} preset).")
    print("Next: flybrain-track check --camera 0   then   flybrain-track run --camera 0")


def open_source(args):
    origin = args.video if args.video else args.camera
    capture = cv2.VideoCapture(origin)
    if not capture.isOpened():
        hint = "check the file path" if args.video else "try another --camera index"
        raise ValueError(f"Cannot open image source {origin!r}: {hint}")
    return capture


def cmd_check(args):
    from .track.overlay import camera_panel
    from .track.ribbon import PathRibbonEncoder

    cfg = load_config(args.config)
    capture = open_source(args)
    frames = []
    try:
        while len(frames) < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError("No frames could be read from the image source")

    results = {}
    for polarity in ("light", "dark"):
        encoder = PathRibbonEncoder(polarity=polarity, roi=cfg.track_roi)
        seen, lateral = [], []
        for frame in frames:
            encoder.encode(frame)
            seen.append(encoder.observation.confidence)
            if encoder.observation.confidence >= cfg.track_min_confidence:
                lateral.append(encoder.observation.lateral)
        results[polarity] = (encoder, float(np.mean(seen)), lateral)
    best = max(results, key=lambda p: results[p][1])
    encoder, seen, lateral = results[best]

    print(f"Checked {len(frames)} frames from {args.video or f'camera {args.camera}'}.")
    for polarity, (_, score, _) in results.items():
        label = "bright line on dark floor" if polarity == "light" else "dark line on light floor"
        print(f"  {polarity:5} ({label}): path seen in {score:.0%} of look-ahead rows")
    snapshot = Path(args.snapshot)
    cv2.imwrite(str(snapshot), camera_panel(frames[-1], encoder))
    print(f"Snapshot with detected path: {snapshot}")

    if seen < 0.25:
        print("\nNo reliable path found. Try: more contrast between path and floor, even "
              "lighting, pointing the camera further down, or a larger track_roi.")
        return 1
    if lateral:
        jitter = float(np.median(np.abs(np.diff(lateral)))) if len(lateral) > 2 else 0.0
        print(f"Path position: mean {np.mean(lateral):+.2f} (-1 left .. +1 right), "
              f"frame-to-frame jitter {jitter:.3f}")
        if abs(np.mean(lateral)) > 0.5:
            print("  The path is near the image edge; centre the camera over it.")
        if jitter > 0.05:
            print("  Detection is jumpy; check for glare, shadows or a second line in view.")
    quality = "good" if seen >= 0.7 else "usable"
    print(f"\nDetection is {quality}. Recommended setting: track_polarity: {best}")
    if best != cfg.track_polarity:
        print(f"  Your config uses '{cfg.track_polarity}'; change it to '{best}'.")
    return 0


def resolve_backend(args):
    if args.backend:
        return args.backend
    cfg = load_config(args.config)
    return cfg.backend if cfg.backend in TRACK_BACKENDS else "optomotor-track"


def common_argv(args):
    argv = ["--backend", resolve_backend(args), "--steps", str(args.steps)]
    for flag in ("config", "record", "output"):
        if getattr(args, flag):
            argv += [f"--{flag}", str(getattr(args, flag))]
    if args.show:
        argv.append("--show")
    return argv


def cmd_sim(args):
    argv = common_argv(args) + ["--synthetic-track", "--track-kind", args.kind,
                                "--seed", str(args.seed)]
    return bridge_main.main(argv)


def cmd_run(args):
    argv = common_argv(args)
    argv += ["--video", args.video] if args.video else ["--camera", str(args.camera)]
    if args.send:
        argv.append("--send")
    return bridge_main.main(argv)


def add_run_options(parser, steps):
    parser.add_argument("--config", help="config file (default: ./config.yaml if present)")
    parser.add_argument("--backend", choices=TRACK_BACKENDS)
    parser.add_argument("--output", choices=["differential", "track"])
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--record", metavar="VIDEO.mp4", help="save an annotated preview video")
    parser.add_argument("--show", action="store_true", help="live preview window (q to quit)")


def add_source(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    source.add_argument("--video", help="video file instead of a camera")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="flybrain-track", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    init = commands.add_parser("init", help="write a starter config.yaml")
    init.add_argument("--preset", choices=PRESETS, default="robot")
    init.add_argument("--path", default="config.yaml")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=cmd_init)

    check = commands.add_parser("check", help="test path detection on a camera or video")
    add_source(check)
    check.add_argument("--config")
    check.add_argument("--frames", type=int, default=60)
    check.add_argument("--snapshot", default="track-check.png")
    check.set_defaults(func=cmd_check)

    sim = commands.add_parser("sim", help="follow a simulated track (no hardware)")
    add_run_options(sim, steps=2000)
    sim.add_argument("--kind", choices=["curvy", "sine", "straight"], default="curvy")
    sim.add_argument("--seed", type=int, default=1)
    sim.set_defaults(func=cmd_sim)

    run = commands.add_parser("run", help="follow a path from a camera or video")
    add_source(run)
    add_run_options(run, steps=10**9)
    run.add_argument("--send", action="store_true", help="really send commands to the vehicle")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, OSError, cv2.error, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
