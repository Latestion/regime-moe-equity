from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_hist = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_hist = np.clip(ref_hist, 1e-6, None)
    cur_hist = np.clip(cur_hist, 1e-6, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def drift_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    recent_days: int = 60,
) -> dict:
    if panel.empty:
        return {"ok": False, "reason": "empty_panel", "features": []}
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    cutoff = data["date"].max() - pd.offsets.BDay(recent_days)
    reference = data[data["date"] < cutoff]
    recent = data[data["date"] >= cutoff]
    rows = []
    for col in feature_cols:
        if col not in data or not pd.api.types.is_numeric_dtype(data[col]):
            continue
        psi = _psi(reference[col], recent[col])
        ref_mean = pd.to_numeric(reference[col], errors="coerce").mean()
        cur_mean = pd.to_numeric(recent[col], errors="coerce").mean()
        ref_std = pd.to_numeric(reference[col], errors="coerce").std()
        mean_shift = (
            float((cur_mean - ref_mean) / ref_std)
            if pd.notna(ref_std) and ref_std > 0 else 0.0
        )
        severity = "high" if psi >= 0.25 else "medium" if psi >= 0.10 else "low"
        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "mean_shift_std": round(mean_shift, 4),
            "severity": severity,
        })
    rows.sort(key=lambda item: item["psi"], reverse=True)
    high = sum(item["severity"] == "high" for item in rows)
    return {
        "ok": high == 0,
        "asof": str(data["date"].max().date()),
        "recent_days": recent_days,
        "high_drift_features": high,
        "features": rows,
    }
