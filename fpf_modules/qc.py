
"""
FPF - Data Quality (QC)

Separa:
PASS
WARN
FAIL
NA (fase não jogada)
"""

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .metrics import time_to_seconds


@dataclass
class QCThresholds:
    min_points: int = 200
    min_pct_time_valid: float = 70.0

    vmax_hard_mps: float = 11.0
    jump_hard_m: float = 12.0
    gap_hard_s: float = 2.0


def qc_gps_df(
    df: pd.DataFrame,
    phase_present: bool = True,
    thr: QCThresholds | None = None
) -> Dict[str, Any]:

    thr = thr or QCThresholds()

    out = {
        "qc_grade": "FAIL",
        "qc_flags": "",
        "n_points": 0,
        "pct_time_valid": np.nan,
        "vmax_mps_qc": np.nan,
        "n_jumps_gt15m": 0,
        "n_gaps_gt2s": 0,
    }

    # fase não jogada
    if not phase_present:
        out["qc_grade"] = "NA"
        out["qc_flags"] = "fase_nao_jogada"
        return out

    if df is None or df.empty:
        out["qc_flags"] = "empty_df"
        return out

    for col in ["Time", "X_UTM", "Y_UTM"]:
        if col not in df.columns:
            out["qc_flags"] = f"missing_{col}"
            return out

    t = pd.to_numeric(time_to_seconds(df["Time"]), errors="coerce")
    x = pd.to_numeric(df["X_UTM"], errors="coerce")
    y = pd.to_numeric(df["Y_UTM"], errors="coerce")

    valid = t.notna() & x.notna() & y.notna()
    out["pct_time_valid"] = float(valid.mean() * 100.0)

    dfv = pd.DataFrame({
        "t": t[valid],
        "x": x[valid],
        "y": y[valid]
    }).sort_values("t")

    out["n_points"] = int(len(dfv))

    if len(dfv) < 2:
        out["qc_flags"] = "too_few_points"
        return out

    tt = dfv["t"].values
    dx = np.diff(dfv["x"].values)
    dy = np.diff(dfv["y"].values)
    dt = np.diff(tt)

    step = np.hypot(dx, dy)

    out["n_jumps_gt15m"] = int((step > thr.jump_hard_m).sum())
    out["n_gaps_gt2s"] = int((dt > thr.gap_hard_s).sum())

    v = step / dt
    out["vmax_mps_qc"] = float(np.nanmax(v))

    flags: List[str] = []

    if out["n_points"] < thr.min_points:
        flags.append("low_points")

    if out["pct_time_valid"] < thr.min_pct_time_valid:
        flags.append("low_valid_pct")

    if out["vmax_mps_qc"] > thr.vmax_hard_mps:
        flags.append("vmax_implausible")

    if out["n_jumps_gt15m"] > 0:
        flags.append("teleport_jumps")

    if out["n_gaps_gt2s"] > 0:
        flags.append("gaps_gt2s")

    if "low_points" in flags or "low_valid_pct" in flags:
        grade = "FAIL"
    elif flags:
        grade = "WARN"
    else:
        grade = "PASS"

    out["qc_grade"] = grade
    out["qc_flags"] = ";".join(flags)

    return out
