"""Are the connectome's T4/T5 populations direction-selective in this engine?

Presents full-field square-wave gratings drifting up, down, left and right to
the research simulator and prints each subtype's firing fraction. Real T4/T5
subtypes each prefer one direction; similar rates across directions mean the
up-minus-down cursor carries no direction signal.

    python scripts/grating_test.py ../bee/data/processed/male_connectome.npz [real|random]
"""

import sys

import numpy as np

from flybrain_robot.track.connectome import csr, retinotopy
from flybrain_robot.track.connectome.defaults import research_config
from flybrain_robot.track.connectome.engine import Brain
from flybrain_robot.track.connectome.retina import Encoder, build_map

path = sys.argv[1]
wiring = sys.argv[2] if len(sys.argv) > 2 else "real"
cfg = research_config()
c = csr.load(path)
if wiring == "random":
    c = csr.random_graph(c, 1)
retinotopy.assign(c, cfg)
types = [f"T{k}{d}" for k in "45" for d in "abcd"]
idx = {t: c.type_index([t]) for t in types}
H, W = 160, 320
yy, xx = np.mgrid[0:H, 0:W]


def grating(direction, phase, period=24):
    coord = {"up": yy + phase, "down": yy - phase, "left": xx + phase, "right": xx - phase}[
        direction
    ]
    return ((coord % period) < period / 2).astype(np.float32)


res = {}
for direction in ("up", "down", "left", "right"):
    brain = Brain(c, cfg)
    enc = Encoder(build_map(c, cfg), cfg, c.n_neurons)
    counts = {t: 0 for t in types}
    for f in range(120):
        drive = enc.encode(grating(direction, 2 * f))
        for _ in range(2):
            fired, _ = brain.step(drive)
            drive = np.zeros_like(drive)
            if f >= 20:
                m = np.zeros(c.n_neurons, bool)
                m[fired] = True
                for t in types:
                    counts[t] += m[idx[t]].sum()
    res[direction] = {t: counts[t] / (idx[t].size * 200) for t in types}
print(c.name, wiring, "firing fraction per cycle")
print("type   " + "  ".join(f"{d:>6}" for d in res))
for t in types:
    print(f"{t:6} " + "  ".join(f"{res[d][t]:.4f}" for d in res))
