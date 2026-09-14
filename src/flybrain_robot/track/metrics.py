"""Follow-quality metrics, including the research's lead/lag test."""

import numpy as np


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def lead_lag(estimate, truth, max_lag=12):
    """Spearman rho at each lag. Positive lag = the estimate follows the truth.

    Mirrors bee/src/observe/smell.py:lead_lag without the pandas dependency.
    """
    b, p = np.asarray(estimate, float), np.asarray(truth, float)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = b[:lag], p[-lag:]
        elif lag > 0:
            x, y = b[lag:], p[:-lag]
        else:
            x, y = b, p
        if len(x) >= 20:
            rows.append((lag, spearman(x, y)))
    return rows


def summarise(trace, off_track=0.05, warmup=15):
    """Closed-loop follow quality. Correlations belong in open-loop tests, because
    a good controller keeps the true error near zero and noise then dominates."""
    true = np.asarray(trace.true_lateral[warmup:], float)
    if true.size < 3:
        return {"steps": len(trace.true_lateral)}
    outside = np.abs(true) > off_track
    events = int(np.count_nonzero(outside[1:] & ~outside[:-1]) + outside[0])
    return {
        "steps": len(trace.true_lateral),
        "rms_error_m": float(np.sqrt(np.mean(true**2))),
        "max_error_m": float(np.max(np.abs(true))),
        "completion": float(trace.progress[-1]) if trace.progress else 0.0,
        "off_track_events": events,
        "lost_frames": int(trace.lost),
    }


def best_lag(estimate, truth, max_lag=4):
    """(lag, rho) with the highest correlation; positive lag = estimate follows."""
    rows = [row for row in lead_lag(estimate, truth, max_lag) if np.isfinite(row[1])]
    return max(rows, key=lambda row: row[1]) if rows else (0, float("nan"))
