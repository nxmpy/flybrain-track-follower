# Vendored from the beedictor research project (bee/src/simulator/cursor.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""Prediction cursor driven by the fly's own direction-selective motion cells.

T4 and T5 are the direction-selective neurons of the Drosophila optic lobe. Their
four subtypes each prefer one cardinal direction; the c and d subtypes are the
vertical pair. The cursor velocity is therefore read directly off the animal's
own upward- and downward-motion channels:

    v = mean(fire: T4c, T5c) - mean(fire: T4d, T5d)

This requires no training, assigns no market meaning to either brain, and makes
no arbitrary choice about which neurons mean "up".

Why MEANS and not sums: the up and down populations are not the same size. In
our prepared data the up pool exceeds the down pool by 6.8% (female) and 5.1%
(male). Summing spike counts would bake a permanent upward drift into every
observation and make both brains look mildly bullish. Means remove that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import Connectome


@dataclass
class RunningStats:
    """Causal running mean/std. Only ever sees values already observed."""
    window: int
    _buf: list[float] = field(default_factory=list)

    def push(self, x: float) -> None:
        self._buf.append(float(x))
        if len(self._buf) > self.window:
            del self._buf[: len(self._buf) - self.window]

    def z(self, x: float) -> float:
        """Standardise using history only. Returns 0 until enough history."""
        if len(self._buf) < 20:
            return 0.0
        arr = np.asarray(self._buf, dtype=np.float64)
        sd = float(arr.std())
        if sd < 1e-12:
            return 0.0
        return float((x - arr.mean()) / sd)


class Cursor:
    """Tracks one brain's vertical prediction cursor across an observation."""

    def __init__(self, c: Connectome, cfg: dict[str, Any]):
        cur = cfg["cursor"]
        self.up_idx = c.type_index(list(cur["up_types"]))
        self.down_idx = c.type_index(list(cur["down_types"]))
        if self.up_idx.size == 0 or self.down_idx.size == 0:
            raise ValueError(
                f"{c.name}: cursor readout populations are empty "
                f"(up={self.up_idx.size}, down={self.down_idx.size}). "
                "Check cell-type annotations survived pruning."
            )
        self.gain = float(cur["gain"])
        self.y_max = float(cur["y_max"])
        self.aggregate = cur.get("aggregate", "mean")
        self.stats = RunningStats(int(cur["zscore_window"]))

        n = c.n_neurons
        self._up_mask = np.zeros(n, dtype=bool)
        self._down_mask = np.zeros(n, dtype=bool)
        self._up_mask[self.up_idx] = True
        self._down_mask[self.down_idx] = True

        self.y = 0.0
        self.start_y = 0.0
        self.trajectory: list[float] = []
        self._velocities: list[float] = []

        # Causal running BASELINE and SCALE for the raw velocity.
        #
        # Both are necessary, and the baseline is the one that is easy to miss.
        # The up and down populations have different resting firing rates in each
        # network, purely because they sit in different connectivity. Measured on
        # LITUSDT, the uncorrected readout drifted -0.67 per observation in the
        # male brain (99.5% of observations negative) and +0.34 in the female,
        # with the stimulus-driven variation riding on top as a much smaller
        # term. Left uncorrected, each brain's "prediction" would have been a
        # fixed anatomical constant wearing the costume of a market response -
        # and it would have produced entirely plausible-looking output.
        #
        # So the cursor integrates the deviation of the velocity from that
        # brain's OWN running mean, divided by its OWN running spread. Both
        # statistics are updated only from samples already consumed, so this
        # stays strictly causal and never looks at the other brain.
        self._mu = 0.0
        self._mad = 0.0
        self._n = 0
        self.baseline_min_samples = int(cur.get("baseline_min_samples", 50))
        self.scale_floor = float(cur.get("scale_floor", 1e-9))

    def begin(self, start_y: float) -> None:
        """Place the cursor at its reference position for a new observation."""
        self.y = float(start_y)
        self.start_y = float(start_y)
        self.trajectory = [float(start_y)]
        self._velocities = []

    def _raw_velocity(self, fired: np.ndarray) -> float:
        """Up-channel minus down-channel activity, normalised per population."""
        if fired.size == 0:
            return 0.0
        up = int(self._up_mask[fired].sum())
        down = int(self._down_mask[fired].sum())
        if self.aggregate == "sum":
            return float(up - down)
        return float(up) / self.up_idx.size - float(down) / self.down_idx.size

    def update(self, fired: np.ndarray) -> float:
        """Advance the cursor by one cycle. Returns the raw velocity."""
        v = self._raw_velocity(fired)
        self._velocities.append(v)

        # Use the statistics as they stood BEFORE this sample, then fold it in.
        if self._n >= self.baseline_min_samples and self._mad > self.scale_floor:
            v_norm = (v - self._mu) / self._mad
        else:
            v_norm = 0.0          # still establishing the baseline

        self._n += 1
        # Exponential forgetting once enough history exists, so a slow drift in
        # network state cannot permanently bias the readout.
        rate = 1.0 / self._n if self._n <= 2000 else 5e-4
        dev = abs(v - self._mu)
        self._mu += rate * (v - self._mu)
        self._mad += rate * (dev - self._mad)

        self.y = float(np.clip(self.y + self.gain * v_norm, -self.y_max, self.y_max))
        self.trajectory.append(self.y)
        return v

    def finish(self) -> dict[str, float]:
        """Freeze the observation and standardise it against causal history.

        The z-score uses only velocities from PREVIOUS observations, so this
        remains blind: nothing about the future enters the readout.
        """
        mean_v = float(np.mean(self._velocities)) if self._velocities else 0.0
        z = self.stats.z(mean_v)
        self.stats.push(mean_v)

        traj = np.asarray(self.trajectory, dtype=np.float64)
        delta = float(self.y - self.start_y)
        # Stability: how monotone the cursor path was. A cursor that drifts
        # steadily is a different claim from one that oscillates to the same end.
        steps = np.diff(traj)
        if steps.size and np.abs(steps).sum() > 1e-12:
            stability = float(abs(steps.sum()) / np.abs(steps).sum())
        else:
            stability = 0.0
        return {
            "velocity_baseline": float(self._mu),
            "velocity_scale": float(self._mad),
            "start_y": self.start_y,
            "final_y": float(self.y),
            "delta": delta,
            "velocity_mean": mean_v,
            "velocity_z": z,
            "stability": stability,
            "path_length": float(np.abs(steps).sum()) if steps.size else 0.0,
        }


def classify(delta: float, epsilon: float) -> str:
    """UP / DOWN / FLAT with a mandatory dead zone."""
    if delta > epsilon:
        return "UP"
    if delta < -epsilon:
        return "DOWN"
    return "FLAT"


def pair_state(male_dir: str, female_dir: str) -> str:
    """Compact joint-state identifier, e.g. M_UP__F_DOWN."""
    return f"M_{male_dir}__F_{female_dir}"
