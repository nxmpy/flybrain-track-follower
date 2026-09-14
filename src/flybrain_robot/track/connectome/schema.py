# Vendored from the beedictor research project (bee/src/connectome/schema.py).
# Adapted for flybrain-track-follower; see NOTICE.
"""One internal schema both connectomes are normalized into.

FlyWire (female) and MaleCNS (male) use different identifiers, different cell
type spellings and different coordinate systems. Everything downstream sees only
this schema, so no simulator or analysis code needs to know which source a brain
came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Canonical cell-type spellings. The two datasets disagree on naming
# (FlyWire "R1-6" vs MaleCNS "R1-R6"), so both are mapped onto one vocabulary.
CANONICAL_TYPE_ALIASES: dict[str, str] = {
    "R1-R6": "R1-6",
    "R1-6": "R1-6",
    "R7p": "R7p", "R7y": "R7y", "R7d": "R7", "R7": "R7",
    "R8p": "R8p", "R8y": "R8y", "R8d": "R8", "R8": "R8",
}

# Neurotransmitter vocabulary. FlyWire uses ACH/GABA/GLUT/SER/DA/OCT,
# MaleCNS uses lowercase full names.
NT_ALIASES: dict[str, str] = {
    "ach": "ACH", "acetylcholine": "ACH", "ACH": "ACH",
    "gaba": "GABA", "GABA": "GABA",
    "glut": "GLUT", "glutamate": "GLUT", "GLUT": "GLUT",
    "ser": "SER", "serotonin": "SER", "5ht": "SER", "SER": "SER",
    "da": "DA", "dopamine": "DA", "DA": "DA",
    "oct": "OCT", "octopamine": "OCT", "OCT": "OCT",
    "his": "HIS", "histamine": "HIS", "HIS": "HIS",
    "unknown": "UNK", "unclear": "UNK", "": "UNK",
}


def canon_type(t: str | None) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    t = str(t).strip()
    return CANONICAL_TYPE_ALIASES.get(t, t)


def canon_nt(nt: str | None) -> str:
    if nt is None or (isinstance(nt, float) and np.isnan(nt)):
        return "UNK"
    return NT_ALIASES.get(str(nt).strip().lower(), str(nt).strip().upper() or "UNK")


@dataclass
class Connectome:
    """A normalized connectome in compact contiguous arrays.

    Neurons are indexed 0..n_neurons-1. `orig_id` maps back to the source
    identifier (FlyWire root_id / MaleCNS bodyId).

    Connectivity is CSR over *outgoing* edges:
        edges of neuron i == edge_targets[edge_offsets[i]:edge_offsets[i+1]]
    """

    name: str
    sex: str
    version: str

    # --- node arrays (length n_neurons) ---
    orig_id: np.ndarray          # int64
    cell_type: np.ndarray        # object/str array of canonical types
    superclass: np.ndarray       # object/str
    side: np.ndarray             # object/str  L | R | ''
    nt_sign: np.ndarray          # float32, presynaptic sign from neurotransmitter
    pos_x: np.ndarray            # float32, retinotopic x in [0,1] (NaN if unknown)
    pos_y: np.ndarray            # float32, retinotopic y in [0,1] (NaN if unknown)
    pos_z: np.ndarray            # float32, raw third spatial axis (NaN if unknown)

    # --- CSR edge arrays ---
    edge_offsets: np.ndarray     # int64, length n_neurons+1
    edge_targets: np.ndarray     # uint32, length n_edges
    edge_weights: np.ndarray     # float32, length n_edges (already transformed)

    meta: dict[str, Any]

    @property
    def n_neurons(self) -> int:
        return int(self.orig_id.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_targets.size)

    def type_index(self, types: list[str]) -> np.ndarray:
        """Indices of every neuron whose canonical type is in `types`."""
        want = set(types)
        return np.flatnonzero(np.isin(self.cell_type, list(want))).astype(np.uint32)

    def nbytes(self) -> int:
        return int(sum(
            a.nbytes for a in (
                self.orig_id, self.nt_sign, self.pos_x, self.pos_y, self.pos_z,
                self.edge_offsets, self.edge_targets, self.edge_weights,
            )
        ))

    def summary(self) -> dict[str, Any]:
        deg = np.diff(self.edge_offsets)
        return {
            "name": self.name,
            "sex": self.sex,
            "version": self.version,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "mean_out_degree": float(deg.mean()) if deg.size else 0.0,
            "max_out_degree": int(deg.max()) if deg.size else 0,
            "isolated_neurons": int((deg == 0).sum()),
            "weight_mean": float(self.edge_weights.mean()) if self.n_edges else 0.0,
            "weight_max": float(self.edge_weights.max()) if self.n_edges else 0.0,
            "array_mb": round(self.nbytes() / 2**20, 1),
            **self.meta,
        }
