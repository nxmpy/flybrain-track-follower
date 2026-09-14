"""Research parameters carried over from bee/config/default.yaml.

The simulator, encoder and cursor values are the calibrated ones from the
beedictor study, where the T4/T5 cursor tracked a white ribbon at rho 0.84-0.91.
Only the global knobs are here; no synaptic weight is ever tuned.
"""

import copy

DEFAULTS = {
    "project": {"seed": 20260912},
    "visual": {
        "viewport_h": 160,
        "viewport_w": 320,
        "prediction_column_px": 16,
        "ribbon_thickness_px": 3,
        "stimulus_color": "white",
        "stimulus_gain": 1.398,
    },
    "retinotopy": {"input_cell_types": ["L1", "L2"]},
    "encoder": {"drive_mode": "temporal_contrast", "adapt_tau": 0.35, "clip": 4.0},
    "simulator": {
        "cycles_per_step": 2,
        "decay": 0.85,
        "threshold": 0.02,
        "refractory_steps": 2,
        "state_clamp": 10.0,
        "input_gain": 1.58,
    },
    "cursor": {
        # The ribbon is drawn so lateral path offset is the vertical axis, which
        # makes the research's vertical motion pair the lateral readout.
        "up_types": ["T4c", "T5c"],
        "down_types": ["T4d", "T5d"],
        "aggregate": "mean",
        "gain": 0.02,
        "scale_floor": 1.0e-9,
        "baseline_min_samples": 50,
        "y_max": 2.0,
        "zscore_window": 500,
    },
}


def research_config():
    return copy.deepcopy(DEFAULTS)
