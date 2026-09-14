# Vendored from the beedictor research project (bee/src/connectome/csr.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""CSR construction, persistence and topology-preserving controls.

No Python objects per neuron or per synapse anywhere: everything is a contiguous
NumPy array so the Numba engine can walk it without boxing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from .schema import Connectome


def transform_weights(syn: np.ndarray, how: str) -> np.ndarray:
    """Synapse count -> connection strength. Raw counts are never used directly."""
    s = syn.astype(np.float32)
    if how == "linear":
        out = s
    elif how == "sqrt":
        out = np.sqrt(s)
    elif how == "log1p":
        out = np.log1p(s)
    else:
        raise ValueError(f"unknown weight_transform: {how!r}")
    return out.astype(np.float32)


def build_csr(
    n_neurons: int,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build outgoing-edge CSR. Sorting by `pre` gives contiguous rows."""
    order = np.argsort(pre, kind="stable")
    pre_s = pre[order]
    targets = post[order].astype(np.uint32)
    weights = weight[order].astype(np.float32)

    counts = np.bincount(pre_s, minlength=n_neurons).astype(np.int64)
    offsets = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, targets, weights


def normalize_weights(
    offsets: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    n_neurons: int,
    how: str,
) -> np.ndarray:
    """Scale weights so the two brains are comparable despite different densities.

    `per_target_sum` divides each edge by the total incoming weight of its target,
    so every neuron receives unit total drive regardless of how many afferents the
    reconstruction pipeline happened to resolve. This is important here: the male
    dataset resolves ~6.6x more edges than the female one, and without this the
    male network would simply be driven harder for reasons of EM methodology
    rather than biology.
    """
    if how == "none":
        return weights
    if how == "per_target_sum":
        incoming = np.zeros(n_neurons, dtype=np.float64)
        np.add.at(incoming, targets, weights)
        incoming[incoming == 0.0] = 1.0
        return (weights / incoming[targets]).astype(np.float32)
    if how == "global_max":
        m = weights.max() or 1.0
        return (weights / m).astype(np.float32)
    raise ValueError(f"unknown weight_normalize: {how!r}")


def save(path: Path, c: Connectome) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=c.name, sex=c.sex, version=c.version,
        orig_id=c.orig_id,
        cell_type=c.cell_type.astype("U24"),
        superclass=c.superclass.astype("U24"),
        side=c.side.astype("U4"),
        nt_sign=c.nt_sign,
        pos_x=c.pos_x, pos_y=c.pos_y, pos_z=c.pos_z,
        edge_offsets=c.edge_offsets,
        edge_targets=c.edge_targets,
        edge_weights=c.edge_weights,
        meta=np.array([repr(c.meta)], dtype=object),
    )


def load(path: Path) -> Connectome:
    z = np.load(path, allow_pickle=True)
    return Connectome(
        name=str(z["name"]), sex=str(z["sex"]), version=str(z["version"]),
        orig_id=z["orig_id"],
        cell_type=z["cell_type"].astype(object),
        superclass=z["superclass"].astype(object),
        side=z["side"].astype(object),
        nt_sign=z["nt_sign"],
        pos_x=z["pos_x"], pos_y=z["pos_y"], pos_z=z["pos_z"],
        edge_offsets=z["edge_offsets"],
        edge_targets=z["edge_targets"],
        edge_weights=z["edge_weights"],
        meta=ast.literal_eval(str(z["meta"][0])),
    )


# ------------------------------------------------------------------ controls

def shuffle_preserving_degree(c: Connectome, seed: int) -> Connectome:
    """Control arm C: destroy topology, keep degree and weight distributions.

    Out-degree per neuron is preserved exactly (CSR offsets are untouched) and the
    multiset of weights is preserved exactly (weights are permuted, not resampled).
    Only *who connects to whom* is randomized.
    """
    rng = np.random.default_rng(seed)
    n = c.n_neurons
    targets = rng.integers(0, n, size=c.n_edges, dtype=np.uint32)
    weights = c.edge_weights.copy()
    rng.shuffle(weights)
    meta = dict(c.meta)
    meta.update({"control": "shuffled_degree_preserving", "control_seed": seed})
    return Connectome(
        name=f"{c.name}_shuffled", sex=c.sex, version=c.version,
        orig_id=c.orig_id, cell_type=c.cell_type, superclass=c.superclass,
        side=c.side, nt_sign=c.nt_sign, pos_x=c.pos_x, pos_y=c.pos_y, pos_z=c.pos_z,
        edge_offsets=c.edge_offsets.copy(), edge_targets=targets,
        edge_weights=weights, meta=meta,
    )


def random_graph(c: Connectome, seed: int) -> Connectome:
    """Control arm D: Erdos-Renyi-like graph matched on size, edge count and
    weight distribution, but with a flat (non-biological) degree distribution."""
    rng = np.random.default_rng(seed)
    n, m = c.n_neurons, c.n_edges
    pre = rng.integers(0, n, size=m, dtype=np.int64)
    post = rng.integers(0, n, size=m, dtype=np.int64)
    weights = c.edge_weights.copy()
    rng.shuffle(weights)
    offsets, targets, w = build_csr(n, pre, post, weights)
    meta = dict(c.meta)
    meta.update({"control": "random_graph", "control_seed": seed})
    return Connectome(
        name=f"{c.name}_random", sex=c.sex, version=c.version,
        orig_id=c.orig_id, cell_type=c.cell_type, superclass=c.superclass,
        side=c.side, nt_sign=c.nt_sign, pos_x=c.pos_x, pos_y=c.pos_y, pos_z=c.pos_z,
        edge_offsets=offsets, edge_targets=targets, edge_weights=w, meta=meta,
    )


def split_half(c: Connectome, seed: int, which: int) -> Connectome:
    """Control arm E: split one brain into two disjoint halves.

    Each half keeps only edges whose endpoints both fall inside it. The readout
    cell types (T4/T5) are split proportionally, so each half has its own motion
    detectors. If two halves of the *same* brain produce as much apparent signal
    as male-vs-female, the male/female difference explains nothing.
    """
    rng = np.random.default_rng(seed)
    n = c.n_neurons
    assign = rng.integers(0, 2, size=n)
    keep = np.flatnonzero(assign == which)
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size)

    # Expand CSR to COO, filter to edges internal to this half, rebuild.
    deg = np.diff(c.edge_offsets)
    pre = np.repeat(np.arange(n, dtype=np.int64), deg)
    post = c.edge_targets.astype(np.int64)
    mask = (remap[pre] >= 0) & (remap[post] >= 0)
    offsets, targets, weights = build_csr(
        keep.size, remap[pre[mask]], remap[post[mask]], c.edge_weights[mask]
    )
    meta = dict(c.meta)
    meta.update({"control": f"same_brain_half_{which}", "control_seed": seed,
                 "parent_neurons": n})
    return Connectome(
        name=f"{c.name}_half{which}", sex=c.sex, version=c.version,
        orig_id=c.orig_id[keep], cell_type=c.cell_type[keep],
        superclass=c.superclass[keep], side=c.side[keep],
        nt_sign=c.nt_sign[keep], pos_x=c.pos_x[keep], pos_y=c.pos_y[keep],
        pos_z=c.pos_z[keep],
        edge_offsets=offsets, edge_targets=targets, edge_weights=weights, meta=meta,
    )
