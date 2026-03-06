"""FPF - Data Quality (QC) utilities for GPS positional data.

v2:
- Separa 'fase não jogada' (NA) de falha de dados (FAIL)
- Flags + score simples (PASS/WARN/FAIL/NA)
- Rápido, sem dependências externas

Uso recomendado:
  qc = qc_gps_df(df_f, phase_present=True/False)

Onde:
- phase_present=False significa que a fase não existia para aquele atleta (não entrou / não jogou)
- phase_present=True significa que havia dados para a fase (mesmo que depois fiquem inválidos)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .metrics import time_to_seconds


@dataclass
class QCThresholds:
    # mínimos de “dados úteis”
    min_points: int = 200
    min_pct_time_valid: float = 70.0  # %
    # plausibilidade (futebol)
    vmax_hard_mps: float = 11.0        # hard cap de plausibilidade
    jump_hard_m: float = 12.0          # salto espacial demasiado alto
    gap_hard_s: float = 2.0            # gap temporal relevante


def qc_gps_df(
    df: pd.DataFrame,
    phase_present: bool = True,
    thr: QCThresholds | None = None
) -> Dict[str, Any]:
    """QC por DataFrame (já filtrado por atleta/fase).

    Espera colunas: Time, X_UTM, Y_UTM

    Retorna:
      - qc_grade: PASS/WARN/FAIL/NA
      - qc_flags: flags separadas por ';'
      - n_points, pct_time_valid, vmax_mps_qc, n_jumps_gt15m, n_gaps_gt2s
    """
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

    # Fase não existente (não jogou / não foi submetida): NA, não é erro de qualidade
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
    out["pct_time_valid"] = float(valid.mean() * 100.0) if len(valid) else np.nan

    dfv = pd.DataFrame({"t": t[valid], "x": x[valid], "y": y[valid]}).sort_values("t")
    out["n_points"] = int(len(dfv))
    if len(dfv) < 2:
        out["qc_flags"] = "too_few_points"
        return out

    tt = dfv["t"].to_numpy(dtype=float)
    dx = np.diff(dfv["x"].to_numpy(dtype=float))
    dy = np.diff(dfv["y"].to_numpy(dtype=float))
    dt = np.diff(tt)

    good = dt > 0
    if not np.any(good):
        out["qc_flags"] = "non_increasing_time"
        return out

    dx, dy, dt = dx[good], dy[good], dt[good]

    out["n_gaps_gt2s"] = int(np.sum(dt > thr.gap_hard_s))

    step = np.hypot(dx, dy)
    out["n_jumps_gt15m"] = int(np.sum(step > thr.jump_hard_m))

    v = np.where(dt > 0, step / dt, np.nan)
    out["vmax_mps_qc"] = float(np.nanmax(v)) if v.size else np.nan

    flags: List[str] = []

    # FAIL (qualidade insuficiente para interpretar)
    if out["n_points"] < thr.min_points:
        flags.append("low_points")
    if np.isfinite(out["pct_time_valid"]) and out["pct_time_valid"] < thr.min_pct_time_valid:
        flags.append("low_valid_pct")

    # WARN (utilizável com reservas / investigar)
    if np.isfinite(out["vmax_mps_qc"]) and out["vmax_mps_qc"] > thr.vmax_hard_mps:
        flags.append("vmax_implausible")
    if out["n_jumps_gt15m"] > 0:
        flags.append("teleport_jumps")
    if out["n_gaps_gt2s"] > 0:
        flags.append("gaps_gt2s")

    if ("low_points" in flags) or ("low_valid_pct" in flags):
        grade = "FAIL"
    elif flags:
        grade = "WARN"
    else:
        grade = "PASS"

    out["qc_grade"] = grade
    out["qc_flags"] = ";".join(flags) if flags else ""
    return out
