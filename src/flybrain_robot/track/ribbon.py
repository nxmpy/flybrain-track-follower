"""Camera frame -> path centreline -> white ribbon on black.

The beedictor study showed the fly's T4/T5 motion pathway follows a white ribbon
on black (a price chart) closely, one step behind. This encoder turns a camera
view of a path into exactly that stimulus:

* rows of the near-ground region of interest become ribbon columns
  (nearest row on the left, the look-ahead on the right, blank column at the far
  right where the chart's future used to be);
* the path's lateral position in each row becomes the ribbon's height
  (path to the right of the vehicle = up).

Lateral drift of the path therefore moves the ribbon up or down, which is the
motion the research's vertical T4c/T5c versus T4d/T5d readout responds to.
Between two camera frames the centreline is interpolated over a few sub-steps
so the motion detectors see continuous movement, as the chart sweep did.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..vision import SensoryInput


@dataclass(frozen=True)
class PathObservation:
    """Classical measurement of the path in one frame (vehicle frame, right +)."""

    lateral: float = 0.0
    heading: float = 0.0
    confidence: float = 0.0
    centres: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)


def _runs(mask):
    """Start/stop indices of consecutive True runs in a 1-D mask."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return edges[0::2], edges[1::2]


class PathRibbonEncoder:
    """Extracts the path and renders ribbon frames for the track brains."""

    def __init__(
        self,
        polarity="light",
        roi=0.6,
        samples=48,
        substeps=4,
        min_contrast=40,
        max_width=0.5,
        viewport=(160, 320),
        prediction_column_px=16,
        thickness=3,
    ):
        if polarity not in ("light", "dark"):
            raise ValueError("polarity must be 'light' or 'dark'")
        if not 0.1 <= roi <= 1 or samples < 4 or substeps < 1:
            raise ValueError("Invalid ribbon encoder settings")
        self.polarity = polarity
        self.roi = roi
        self.samples = samples
        self.substeps = substeps
        self.min_contrast = min_contrast
        self.max_width = max_width
        self.height, self.width = viewport
        self.prediction_column_px = prediction_column_px
        self.thickness = thickness
        self.reset()

    def reset(self):
        self.previous = None
        self.observation = PathObservation()
        self.ribbon_frames = []

    def measure(self, frame):
        """Per-row path centres in [-1, 1] (NaN where no path is visible)."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if self.polarity == "dark":
            gray = 255 - gray
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        top = int(round(gray.shape[0] * (1 - self.roi)))
        rows = np.linspace(gray.shape[0] - 1, top, self.samples).round().astype(int)
        half = (gray.shape[1] - 1) / 2
        prior = self.previous
        centres = np.full(self.samples, np.nan)
        for i, r in enumerate(rows):
            line = gray[r]
            low, high = float(line.min()), float(line.max())
            if high - low < self.min_contrast:
                continue
            mask = line >= low + 0.5 * (high - low)
            if mask.mean() > self.max_width:
                continue
            starts, stops = _runs(mask)
            mids = (starts + stops - 1) / 2
            guess = half if prior is None or np.isnan(prior[i]) else prior[i] * half + half
            k = int(np.argmin(np.abs(mids - guess)))
            span = np.arange(starts[k], stops[k])
            weight = line[span] - low
            centres[i] = (float((span * weight).sum() / weight.sum()) - half) / half
        return centres

    def observe(self, centres):
        valid = np.flatnonzero(~np.isnan(centres))
        confidence = valid.size / centres.size
        if valid.size < 2:
            return PathObservation(confidence=confidence, centres=centres)
        near = valid[: max(2, valid.size // 3)]
        depth = valid / (centres.size - 1)
        slope = float(np.polyfit(depth, centres[valid], 1)[0])
        return PathObservation(
            lateral=float(np.clip(centres[near].mean(), -1, 1)),
            heading=float(np.clip(slope, -2, 2)),
            confidence=confidence,
            centres=centres,
        )

    def render(self, centres):
        """Draw one ribbon frame (float32, 0..1) from per-row centres."""
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        valid = ~np.isnan(centres)
        if valid.sum() >= 2:
            chart_w = self.width - self.prediction_column_px
            xs = np.linspace(0, chart_w - 1, centres.size)[valid]
            ys = (0.5 - 0.5 * np.clip(centres[valid], -1, 1)) * (self.height - 1)
            points = np.round(np.column_stack([xs, ys]) * 16).astype(np.int32)
            cv2.polylines(image, [points], False, 255, self.thickness, cv2.LINE_AA, shift=4)
        return image.astype(np.float32) / 255

    def encode(self, frame):
        centres = self.measure(frame)
        self.observation = self.observe(centres)
        start = centres if self.previous is None else self.previous
        # Interpolate where both frames saw the path; otherwise show the new view.
        both = ~np.isnan(start) & ~np.isnan(centres)
        self.ribbon_frames = []
        for s in range(1, self.substeps + 1):
            blend = centres.copy()
            t = s / self.substeps
            blend[both] = start[both] + (centres[both] - start[both]) * t
            self.ribbon_frames.append(self.render(blend))
        self.previous = centres
        return SensoryInput()
