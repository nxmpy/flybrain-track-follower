"""Render the closed-loop track demo for the README. Install the [demo] extra first."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from flybrain_robot.config import Config
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.track import build_track_brain
from flybrain_robot.track.sim import TrackWorld, make_track, run_closed_loop

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TEXT, MUTED, GREEN, BLUE, AMBER = (
    "#08121d",
    "#122331",
    "#eaf5fc",
    "#8fa9ba",
    "#65e8ba",
    "#72cce8",
    "#f2c46d",
)


def font(size):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    cfg = Config()
    brain = build_track_brain(cfg, "optomotor-track")
    world = TrackWorld(make_track("curvy", seed=1, length=5.0))
    decoder = MotorDecoder(max_speed=cfg.max_speed, smoothing=cfg.smoothing, dead_zone=cfg.dead_zone)
    images = []

    def draw(index, frame, trace):
        if index % 8:
            return
        image = Image.new("RGB", (1080, 470), BG)
        d = ImageDraw.Draw(image)
        d.text((32, 20), "SIMULATED TRACK / ACTUAL MODEL OUTPUT", font=font(16), fill=GREEN)
        d.text((32, 52), "Camera path → ribbon → insect motion model → wheels", font=font(26),
               fill=TEXT)
        for box in ((24, 100, 364, 400), (380, 100, 720, 400), (736, 100, 1056, 400)):
            d.rounded_rectangle(box, 12, fill=PANEL)
        d.text((40, 114), "01  VEHICLE CAMERA", font=font(14), fill=BLUE)
        image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (34, 146))
        d.text((396, 114), "02  RIBBON STIMULUS (path right = up)", font=font(14), fill=BLUE)
        ribbon = (brain.encoder.ribbon_frames[-1] * 255).astype(np.uint8)
        image.paste(Image.fromarray(ribbon).convert("RGB"), (390, 190))
        d.text((752, 114), "03  LATERAL ERROR (last 4 s)", font=font(14), fill=BLUE)
        errors = np.asarray(trace.true_lateral[-120:]) * 1000
        top, height, left = 160, 150, 756
        d.line((left, top + height / 2, left + 280, top + height / 2), fill="#203b4d")
        points = [
            (left + 280 * i / 119, top + height / 2 - np.clip(e, -50, 50) * height / 100)
            for i, e in enumerate(errors)
        ]
        if len(points) > 1:
            d.line(points, fill=AMBER, width=2)
        d.text((756, 316), "±50 mm", font=font(12), fill=MUTED)
        d.text((752, 340), f"LEFT {trace.left[-1]:+03d}   RIGHT {trace.right[-1]:+03d}",
               font=font(20), fill=GREEN)
        d.text((32, 420), f"t = {index / 30:0.1f} s   error = {errors[-1]:+.0f} mm   "
               f"progress = {world.progress:.0%}", font=font(14), fill=MUTED)
        d.text((650, 420), "Software simulation. Not a physical robot recording.", font=font(12),
               fill=MUTED)
        small = image.resize((810, 352), Image.Resampling.LANCZOS)
        images.append(small.quantize(colors=32, dither=Image.Dither.NONE))

    run_closed_loop(brain, world, decoder, steps=480, on_step=draw)
    out = ROOT / "assets" / "track-demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=267, loop=0,
                   optimize=True)
    images[len(images) // 2].convert("RGB").save(ROOT / "assets" / "track-demo.png")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
