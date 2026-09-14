# Vendored from the beedictor research project (bee/src/simulator/engine.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""V0 neural engine: persistent-activation threshold units on a fixed connectome.

Design constraints this satisfies:
  * no Python object per neuron or per synapse - only contiguous arrays
  * no NetworkX at runtime
  * propagation is event-driven: only neurons that actually fired are walked,
    which at a few percent activity is far cheaper than a full sparse matvec
  * state persists across frames AND across market candles
  * activity is clamped and refractory-gated so it cannot run away
  * bit-identical results for a fixed seed

The biological weights are FIXED. Nothing here learns. Only global parameters
(decay, threshold, gain, cycles) are tunable, which preserves the experimental
question: does the biological topology itself produce useful dynamics?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from numba import njit

    NUMBA = True
except ImportError:  # optional extra: pip install -e ".[connectome]"
    NUMBA = False

    def njit(*args, **kwargs):
        return lambda function: function

from .schema import Connectome


@njit(cache=True, fastmath=False)
def _propagate(edge_offsets, edge_targets, edge_weights, nt_sign,
               fired_idx, accum):
    """Scatter each firing neuron's weighted output onto its postsynaptic targets.

    Returns the number of edges traversed, which is the honest unit of work for
    benchmarking (candles/second alone hides how much the network actually did).
    """
    traversals = 0
    for k in range(fired_idx.size):
        i = fired_idx[k]
        sign = nt_sign[i]
        if sign == 0.0:
            # Still count the traversal cost we skipped deliberately.
            traversals += edge_offsets[i + 1] - edge_offsets[i]
            continue
        for e in range(edge_offsets[i], edge_offsets[i + 1]):
            accum[edge_targets[e]] += sign * edge_weights[e]
        traversals += edge_offsets[i + 1] - edge_offsets[i]
    return traversals


@njit(cache=True, fastmath=False)
def _step(state, refrac, accum, external,
          edge_offsets, edge_targets, edge_weights, nt_sign,
          decay, threshold, clamp, refractory_steps, fired_out):
    """One neural cycle. Returns (n_fired, edge_traversals)."""
    n = state.size
    # Leak, then integrate external drive and last cycle's network input.
    for i in range(n):
        if refrac[i] > 0:
            refrac[i] -= 1
            state[i] = 0.0
        else:
            s = state[i] * decay + external[i] + accum[i]
            if s > clamp:
                s = clamp
            elif s < -clamp:
                s = -clamp
            state[i] = s
        accum[i] = 0.0

    n_fired = 0
    for i in range(n):
        if state[i] >= threshold:
            fired_out[n_fired] = i
            n_fired += 1
            state[i] = 0.0                 # reset after spike
            refrac[i] = refractory_steps

    trav = _propagate(edge_offsets, edge_targets, edge_weights, nt_sign,
                      fired_out[:n_fired], accum)
    return n_fired, trav


@dataclass
class Metrics:
    """Per-observation activity summary. Recorded for every brain, every step."""
    active_fraction: float
    participation: float
    stability: float
    mean_fired: float
    peak_fired: int
    edge_traversals: int


class Brain:
    """One connectome plus its mutable simulation state."""

    def __init__(self, c: Connectome, cfg: dict[str, Any]):
        s = cfg["simulator"]
        self.c = c
        self.n = c.n_neurons
        self.decay = np.float32(s["decay"])
        self.threshold = np.float32(s["threshold"])
        self.clamp = np.float32(s["state_clamp"])
        self.refractory_steps = np.int32(s["refractory_steps"])

        self.state = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int32)
        self.accum = np.zeros(self.n, dtype=np.float32)
        self._fired_buf = np.zeros(self.n, dtype=np.int64)
        # Union of neurons that fired at any point in this observation.
        self._ever_fired = np.zeros(self.n, dtype=bool)

    def reset(self) -> None:
        """Full reset. Used between symbols, never between candles."""
        self.state[:] = 0.0
        self.refrac[:] = 0
        self.accum[:] = 0.0
        self._ever_fired[:] = False

    def begin_observation(self) -> None:
        """Clear per-observation accounting WITHOUT touching neural state."""
        self._ever_fired[:] = False

    def step(self, external: np.ndarray) -> tuple[np.ndarray, int]:
        """Advance one cycle. Returns (fired indices view, edge traversals)."""
        n_fired, trav = _step(
            self.state, self.refrac, self.accum, external,
            self.c.edge_offsets, self.c.edge_targets, self.c.edge_weights,
            self.c.nt_sign,
            self.decay, self.threshold, self.clamp, self.refractory_steps,
            self._fired_buf,
        )
        fired = self._fired_buf[:n_fired]
        self._ever_fired[fired] = True
        return fired, int(trav)

    def health_check(self) -> None:
        """Fail loudly on numerical breakdown rather than silently producing
        meaningless records."""
        if not np.isfinite(self.state).all():
            raise FloatingPointError(
                f"{self.c.name}: non-finite neural state - simulation diverged"
            )

    @property
    def participation(self) -> float:
        """Fraction of the whole network that took part in this observation."""
        return float(self._ever_fired.mean())
