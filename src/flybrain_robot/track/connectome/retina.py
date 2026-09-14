# Vendored from the beedictor research project (bee/src/encoder/retina.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""Map a rendered chart frame onto each brain's visual input layer.

Both brains are shown the SAME rendered image. Each samples it through its own
retinotopic geometry, because they are different animals with different eyes.
That is the operative definition of "identical stimulus" in this study and it is
stated explicitly in the final report.

Biological note on the drive mode: lamina monopolar cells are not luminance
followers - L1/L2 are temporally high-pass and respond to contrast *changes*.
The default `temporal_contrast` mode reproduces that by subtracting an adapting
baseline, which is also what generates the transients the T4/T5 correlators need.
`luminance` is available for comparison and is the simpler, less faithful choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import Connectome


@dataclass
class RetinaMap:
    """Precomputed, immutable sampling plan from image pixels to input neurons."""

    neuron_idx: np.ndarray      # uint32, indices into the connectome
    px_row: np.ndarray          # int32, pixel row per input neuron
    px_col: np.ndarray          # int32, pixel column per input neuron
    gain: float                 # opsin-derived colour gain
    color_name: str
    n_inputs: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_inputs = int(self.neuron_idx.size)


def build_map(c: Connectome, cfg: dict[str, Any]) -> RetinaMap:
    """Assign each input-layer neuron the pixel its receptive field looks at."""
    from . import retinotopy

    idx = retinotopy.input_indices(c, cfg)
    v = cfg["visual"]
    h, w = int(v["viewport_h"]), int(v["viewport_w"])
    chart_w = w - int(v["prediction_column_px"])

    x = np.nan_to_num(c.pos_x[idx], nan=0.5)
    y = np.nan_to_num(c.pos_y[idx], nan=0.5)
    # Receptive fields tile the chart area only; the prediction column is blank
    # by construction and no input neuron looks at it.
    col = np.clip((x * (chart_w - 1)).astype(np.int32), 0, chart_w - 1)
    # Retinotopic y is bottom-up; image rows are top-down.
    row = np.clip(((1.0 - y) * (h - 1)).astype(np.int32), 0, h - 1)

    # Colour gain comes from the research opsin model (Govardovskii template, Rh1):
    # white 1.398, green 0.443, red 0.0002. It is stored as a constant here.
    name = str(v.get("stimulus_color", "white"))
    gate = float(v.get("stimulus_gain", 1.398))
    return RetinaMap(neuron_idx=idx.astype(np.uint32),
                     px_row=row, px_col=col, gain=gate, color_name=name)


class Encoder:
    """Turns frames into per-neuron external drive, with L1-style adaptation."""

    def __init__(self, rmap: RetinaMap, cfg: dict[str, Any], n_neurons: int):
        self.map = rmap
        self.n_neurons = n_neurons
        e = cfg.get("encoder", {})
        self.drive_mode: str = e.get("drive_mode", "temporal_contrast")
        self.adapt_tau: float = float(e.get("adapt_tau", 0.35))
        self.input_gain: float = float(cfg["simulator"]["input_gain"])
        self.clip: float = float(e.get("clip", 4.0))
        self._baseline = np.zeros(rmap.n_inputs, dtype=np.float32)
        self._primed = False

    def reset(self) -> None:
        """Clear adaptation state (between symbols, never between candles)."""
        self._baseline[:] = 0.0
        self._primed = False

    def sample(self, frame: np.ndarray) -> np.ndarray:
        """Sample one frame at every input neuron's receptive field."""
        if frame.ndim == 3:            # two-channel mode: sum the planes
            frame = frame.sum(axis=0)
        return frame[self.map.px_row, self.map.px_col].astype(np.float32)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Return a length-n_neurons external drive vector for this frame."""
        lum = self.sample(frame)

        if self.drive_mode == "luminance":
            signal = lum
        elif self.drive_mode == "temporal_contrast":
            if not self._primed:
                self._baseline[:] = lum
                self._primed = True
            signal = lum - self._baseline
            self._baseline += self.adapt_tau * (lum - self._baseline)
        else:
            raise ValueError(f"unknown drive_mode: {self.drive_mode!r}")

        # Colour gain from the opsin model. This is where a red stimulus becomes
        # a near-silent one (Rh1 drive at ~0.04% of green).
        signal = signal * self.map.gain * self.input_gain
        np.clip(signal, -self.clip, self.clip, out=signal)

        drive = np.zeros(self.n_neurons, dtype=np.float32)
        drive[self.map.neuron_idx] = signal
        return drive


def frame_hash(frame: np.ndarray) -> str:
    """Stable digest of a rendered frame.

    Used to *prove*, per observation, that both brains received byte-identical
    stimuli rather than merely asserting it.
    """
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(frame, dtype=np.float32).tobytes()
                          ).hexdigest()[:16]
