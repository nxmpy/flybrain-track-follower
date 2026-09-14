"""Does a brain's own motion signal track lateral path motion? (open loop)

The vehicle is moved along a straight path with a smooth random lateral offset
and heading, independent of any steering. Each backend runs with anchor=0 so its
per-frame velocity is purely the brain's. Reports Spearman rho between that
velocity and the measured lateral change of the path, and the best lag
(positive = follows). This is the test the research follower claim needed.

    python scripts/open_loop_test.py --backend optomotor-track
    python scripts/open_loop_test.py --backend connectome-track \
        --connectome ../bee/data/processed/male_connectome.npz --wiring random
"""

import argparse
import time

import numpy as np

from flybrain_robot.track.brains import ConnectomeTrackBrain, OptomotorTrackBrain
from flybrain_robot.track.metrics import best_lag, spearman
from flybrain_robot.track.ribbon import PathRibbonEncoder
from flybrain_robot.track.sim import Pose, TrackWorld, make_track


def smooth_noise(rng, n, scale, width=25):
    kernel = np.hanning(width * 2 + 1)
    kernel /= kernel.sum()
    series = np.convolve(rng.normal(0, 1, n + 2 * width), kernel, mode="same")[width:-width]
    return scale * series / np.abs(series).max()


def run(brain, encoder, steps, seed, warmup=60):
    rng = np.random.default_rng(seed)
    world = TrackWorld(make_track("straight", length=0.2 * steps / 30 + 2))
    offsets = smooth_noise(rng, steps, 0.1)
    headings = smooth_noise(rng, steps, 0.3)
    lateral, velocity = [], []
    started = time.monotonic()
    for i in range(steps):
        world.pose = Pose(0.5 + 0.2 * i / 30, float(offsets[i]), float(headings[i]))
        encoder.encode(world.camera())
        brain.step(None, None, 1 / 30)
        lateral.append(encoder.observation.lateral)
        velocity.append(brain.estimate.velocity)
    hz = steps / (time.monotonic() - started)
    lateral = np.asarray(lateral)
    change = np.diff(lateral, prepend=lateral[0])[warmup:]
    velocity = np.asarray(velocity)[warmup:]
    lag, lag_rho = best_lag(velocity, change)
    return {"rho": spearman(velocity, change), "best_lag": lag, "best_lag_rho": lag_rho, "hz": hz}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["optomotor-track", "connectome-track"],
                        default="optomotor-track")
    parser.add_argument("--connectome")
    parser.add_argument("--wiring", choices=["real", "shuffled", "random"], default="real")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    encoder = PathRibbonEncoder()
    if args.backend == "optomotor-track":
        brain = OptomotorTrackBrain(encoder, anchor=0.0)
    else:
        brain = ConnectomeTrackBrain(encoder, args.connectome, anchor=0.0, wiring=args.wiring)
    result = run(brain, encoder, args.steps, args.seed)
    print(f"backend={args.backend} wiring={args.wiring} " + " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items()
    ))


if __name__ == "__main__":
    main()
