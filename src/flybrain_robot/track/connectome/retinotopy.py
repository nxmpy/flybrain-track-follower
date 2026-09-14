# Vendored from the beedictor research project (bee/src/connectome/retinotopy.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""Assign retinotopic receptive-field coordinates to the visual input layer.

The two datasets give us different raw geometry, so each is handled natively and
both are reduced to the same normalized [0,1]^2 receptive-field space:

  female (FlyWire) : 3D soma coordinates -> seeded PCA to 2D, per optic lobe
  male   (MaleCNS) : assignedOlHex1/2, the optic lobe's own hexagonal column
                     lattice, which is already a retinotopic coordinate

Rank-normalisation is then applied per axis so that uneven sampling in either
coordinate system maps onto the same uniform grid. This is what makes the two
retinae comparable without pretending they are identical.

IMPORTANT - the definition of "identical stimulus": both brains are shown the
*same rendered image*. Each samples it through its *own* retina geometry, because
they are different animals with different eyes. The image is identical; the
sampling is brain-specific. This is documented in the final report.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .schema import Connectome

log = logging.getLogger(__name__)


def _rank_normalize(v: np.ndarray) -> np.ndarray:
    """Map values to [0,1] by rank, so spacing irregularities do not distort the
    receptive-field grid. Ties get the same position."""
    out = np.full(v.size, np.nan, dtype=np.float32)
    ok = np.isfinite(v)
    if ok.sum() <= 1:
        out[ok] = 0.5
        return out
    vals = v[ok]
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, dtype=np.float64)
    ranks[order] = np.arange(vals.size)
    out[ok] = (ranks / max(vals.size - 1, 1)).astype(np.float32)
    return out


def _pca2(xyz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic projection of 3D positions onto their two principal axes."""
    x = xyz - xyz.mean(axis=0, keepdims=True)
    # SVD is deterministic; sign is fixed below so runs are reproducible.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2]
    for i in range(comps.shape[0]):
        # Fix sign by the largest-magnitude loading, removing SVD sign ambiguity.
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    return x @ comps.T


def assign(c: Connectome, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill c.pos_x / c.pos_y in [0,1] for the configured input cell types.

    Mutates the connectome in place and returns a report.
    """
    rcfg = cfg["retinotopy"]
    seed = int(cfg["project"]["seed"])
    input_types = list(rcfg["input_cell_types"])

    sel = np.flatnonzero(np.isin(c.cell_type, input_types))
    report: dict[str, Any] = {
        "input_cell_types": input_types,
        "n_input_cells": int(sel.size),
        "method": "hex_lattice" if c.meta.get("hex_lattice") else "pca3d",
        "sides": {},
    }
    if sel.size == 0:
        log.warning("%s: no input cells of types %s", c.name, input_types)
        return report

    new_x = np.full(c.n_neurons, np.nan, dtype=np.float32)
    new_y = np.full(c.n_neurons, np.nan, dtype=np.float32)

    # Group by side so the two optic lobes are mapped independently; neurons with
    # no side label are treated as one additional group rather than discarded.
    sides = c.side[sel]
    for side_val in sorted(set(sides.tolist())):
        grp = sel[sides == side_val]
        if c.meta.get("hex_lattice"):
            raw = np.column_stack([c.pos_x[grp], c.pos_y[grp]]).astype(np.float64)
            usable = np.isfinite(raw).all(axis=1)
            xy = raw                       # already a retinotopic lattice
        else:
            # Female: project the 3D soma cloud of this optic lobe onto its two
            # principal axes. The lamina is a curved sheet in the EM volume, so
            # the full 3D geometry gives a better map than any two raw axes.
            raw3 = np.column_stack(
                [c.pos_x[grp], c.pos_y[grp], c.pos_z[grp]]
            ).astype(np.float64)
            usable = np.isfinite(raw3).all(axis=1)
            xy = np.full((grp.size, 2), np.nan)
            if usable.sum() >= 3:
                xy[usable] = _pca2(raw3[usable], seed)

        gx = _rank_normalize(np.where(usable, xy[:, 0], np.nan))
        gy = _rank_normalize(np.where(usable, xy[:, 1], np.nan))
        new_x[grp] = gx
        new_y[grp] = gy
        report["sides"][side_val or "unlabelled"] = {
            "n": int(grp.size),
            "with_coordinates": int(usable.sum()),
        }

    # Any input cell without usable geometry is placed on a deterministic
    # pseudo-lattice rather than dropped, so both brains keep their full input
    # layer. Flagged in the report so this is never mistaken for real geometry.
    missing = np.flatnonzero(np.isnan(new_x[sel]))
    if missing.size:
        rng = np.random.default_rng(seed)
        idx = sel[missing]
        k = int(np.ceil(np.sqrt(idx.size)))
        order = rng.permutation(idx.size)
        new_x[idx] = ((order % k) / max(k - 1, 1)).astype(np.float32)
        new_y[idx] = ((order // k) / max(k - 1, 1)).astype(np.float32)
    report["synthetic_lattice_cells"] = int(missing.size)
    report["synthetic_lattice_fraction"] = float(missing.size / sel.size)

    c.pos_x, c.pos_y = new_x, new_y
    return report


def input_indices(c: Connectome, cfg: dict[str, Any]) -> np.ndarray:
    """Indices of the visual input layer, ordered for stable stimulus mapping."""
    sel = np.flatnonzero(np.isin(c.cell_type, list(cfg["retinotopy"]["input_cell_types"])))
    return sel.astype(np.uint32)
