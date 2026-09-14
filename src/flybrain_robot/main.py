"""Headless CLI. Network transmission requires --send; dry-run is the default."""

import argparse
import logging
import math
import time

import cv2
import yaml

from .brain.malecns import MaleCNSBackend
from .brain.mock import MockBrain
from .config import BACKENDS, TRACK_BACKENDS, load_config
from .controller import UDPBridge
from .motor_decoder import MotorDecoder
from .protocol import command
from .telemetry import Telemetry, synthetic_telemetry
from .vision import VisionEncoder, synthetic_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int)
    source.add_argument("--video")
    source.add_argument("--synthetic", action="store_true")
    source.add_argument(
        "--synthetic-track",
        action="store_true",
        help="Closed-loop simulated vehicle on a drawn track (track backends only)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--send", action="store_true", help="Enable physical UDP motor commands")
    parser.add_argument("--output", choices=["differential", "track"])
    parser.add_argument("--track-kind", choices=["curvy", "sine", "straight"], default="curvy")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wiring", choices=["real", "shuffled", "random"])
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int, default=300, help="Finite frame count (default: 300)")
    parser.add_argument("--record", help="Track backends: write an annotated preview video (.mp4)")
    parser.add_argument("--show", action="store_true", help="Track backends: live preview window")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    bridge, capture, decoder, sequence, output, preview = None, None, None, 0, "differential", None
    try:
        cfg = load_config(args.config)
        logging.basicConfig(level=cfg.logging_level)
        backend = args.backend or cfg.backend
        output = args.output or cfg.output
        if args.wiring:
            cfg.track_wiring = args.wiring
        is_track = backend in TRACK_BACKENDS
        if (args.synthetic_track or output == "track" or args.record or args.show) and not is_track:
            raise ValueError(
                "--synthetic-track, --output track, --record and --show need a track backend: "
                "add --backend optomotor-track (or use the flybrain-track command)"
            )
        if is_track:
            from .track import build_track_brain

            brain = build_track_brain(cfg, backend)
            encoder = brain.encoder
        else:
            brain = MockBrain() if backend == "mock" else MaleCNSBackend(cfg.dataset_path)
            encoder = VisionEncoder()
        decoder = MotorDecoder(
            **{
                name: getattr(cfg, name)
                for name in (
                    "max_speed",
                    "smoothing",
                    "dead_zone",
                    "looming_threshold",
                    "watchdog_timeout",
                    "invert_left",
                    "invert_right",
                )
            }
        )
        world, trace, omega = None, None, 0.0
        simulated = args.synthetic or args.synthetic_track
        video = args.video or (cfg.video if args.camera is None else None)
        if args.synthetic_track:
            from .track.sim import Trace, TrackWorld, make_track

            origin = f"synthetic-track:{args.track_kind}:{args.seed}"
            world = TrackWorld(
                make_track(args.track_kind, seed=args.seed, polarity=cfg.track_polarity)
            )
            trace = Trace()
        elif args.synthetic:
            origin = "synthetic"
        else:
            origin = video if video else (args.camera if args.camera is not None else cfg.camera)
            capture = cv2.VideoCapture(origin)
            if not capture.isOpened():
                hint = (
                    "check the file path"
                    if video
                    else "check it is connected, not used by another program, or try --camera 1"
                )
                raise ValueError(f"Cannot open image source {origin!r}: {hint}")
        if args.record or args.show:
            from .track.overlay import Preview

            preview = Preview(args.record, args.show)
        if args.send:
            bridge = UDPBridge(cfg)
        print(
            f"source={origin} backend={backend} output={output} "
            f"UDP={'enabled' if bridge else 'off (dry-run)'}"
        )
        previous = time.monotonic()
        for index in range(args.steps):
            started = time.monotonic()
            if world is not None:
                frame = world.camera()
            elif capture is None:
                frame = synthetic_frame(index)
            else:
                ok, frame = capture.read()
                if not ok:
                    break
            sensory = encoder.encode(frame)
            if bridge:
                telemetry = bridge.receive()
            elif world is not None:
                telemetry = Telemetry(gyro=(0.0, 0.0, math.degrees(omega)))
            else:
                telemetry = synthetic_telemetry(index / 30)
            now = time.monotonic()
            dt = 1 / 30 if simulated else max(0.001, min(now - previous, 0.5))
            previous = now
            activity = brain.step(sensory, telemetry, dt)
            lost = is_track and brain.controller.command.lost
            stopped = (bridge is not None and not bridge.fresh()) or lost
            left, right = decoder.update(activity, emergency_stop=stopped)
            sequence += 1
            if bridge:
                if output == "track":
                    from .track.steer import command_from_steer

                    packet = command_from_steer(
                        sequence, brain.controller.command, cfg.max_speed, stopped
                    )
                else:
                    packet = command(sequence, left, right, stopped)
                bridge.send(packet)
            if world is not None:
                _, omega = world.drive(left, right, dt)
                trace.true_lateral.append(world.lateral_error())
                trace.measured_lateral.append(encoder.observation.lateral)
                trace.brain_lateral.append(brain.estimate.lateral)
                trace.brain_velocity.append(brain.estimate.velocity)
                trace.left.append(left)
                trace.right.append(right)
                trace.progress.append(world.progress)
                trace.lost += int(lost)
            if index % 10 == 0:
                status = (
                    "fresh" if bridge and bridge.fresh() else ("stale" if bridge else "synthetic")
                )
                if is_track:
                    print(track_status(index, dt, encoder, brain, left, right, lost, stopped,
                                       world, trace))
                else:
                    print(
                        f"frame={index:04d} Hz={1 / dt:.1f} motion={sensory.left_motion:.2f}/"
                        f"{sensory.right_motion:.2f} looming={sensory.looming:.2f} IMU={status} "
                        f"motor={activity.left:.2f}/{activity.right:.2f} "
                        f"command={left:+d}/{right:+d} watchdog={'STOP' if stopped else 'ready'}"
                    )
            if preview is not None and not preview(
                render_preview(frame, encoder, brain, left, right, lost)
            ):
                break
            if world is not None:
                if world.progress >= 0.99:
                    break
            else:
                time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
        if trace is not None:
            from .track.metrics import summarise

            summary = summarise(trace)
            print("summary " + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()
            ))
            if "rms_error_m" in summary:
                if summary["completion"] >= 0.95:
                    verdict = "followed the whole track"
                elif summary["lost_frames"]:
                    verdict = "lost the path"
                else:
                    verdict = "still on track when --steps ran out"
                print(
                    f"Result: {verdict} - {summary['completion']:.0%} of the course, "
                    f"typical error {summary['rms_error_m'] * 1000:.0f} mm, "
                    f"worst {summary['max_error_m'] * 1000:.0f} mm."
                )
        if args.record:
            print(f"Preview video written to {args.record}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    except (ValueError, OSError, NotImplementedError, cv2.error, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if preview is not None:
            preview.close()
        if decoder:
            decoder.stop()
        if bridge:
            try:
                if output == "track":
                    from .track.steer import track_command

                    bridge.send(track_command(sequence + 1, emergency_stop=True))
                else:
                    bridge.send(command(sequence + 1, emergency_stop=True))
            except OSError:
                logging.warning("Final stop datagram failed; robot watchdog must stop motors")
            finally:
                bridge.close()
        if capture:
            capture.release()


def render_preview(frame, encoder, brain, left, right, lost):
    from .track.overlay import render

    return render(frame, encoder, brain, left, right, lost)


def track_status(index, dt, encoder, brain, left, right, lost, stopped, world, trace):
    obs, steer = encoder.observation, brain.controller.command
    side = "right" if obs.lateral > 0.05 else ("left" if obs.lateral < -0.05 else "ahead")
    turn = "right" if steer.turn > 0.05 else ("left" if steer.turn < -0.05 else "straight")
    parts = [
        f"frame {index:5d}",
        f"{1 / dt:5.1f} Hz",
        f"path {side:>5} {obs.lateral:+.2f}",
        f"seen {obs.confidence:4.0%}",
        f"steer {turn:>8} {steer.turn:+.2f}",
        f"wheels L{left:+4d} R{right:+4d}",
    ]
    if world is not None:
        parts.append(f"error {trace.true_lateral[-1] * 1000:+4.0f} mm")
        parts.append(f"course {world.progress:4.0%}")
    if lost:
        parts.append("PATH LOST - stopped")
    elif stopped:
        parts.append("STOP (no fresh telemetry)")
    return " | ".join(parts)


if __name__ == "__main__":
    main()
