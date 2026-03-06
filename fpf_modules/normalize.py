"""
FPF - Pitch normalization utilities.

Objetivo:
- Criar um referencial canónico do campo para comparabilidade
- Garantir X em [0, dist_x] e Y em [0, dist_y]
- Permitir flip opcional em X (direção de ataque)
"""

from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def normalize_pitch_xy(
    df: pd.DataFrame,
    dist_x: float,
    dist_y: float,
    flip_x: bool = False,
    clip: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normaliza X_UTM/Y_UTM para um campo canónico."""
    if df is None or df.empty or "X_UTM" not in df.columns or "Y_UTM" not in df.columns:
        return df, {"normalized": False, "reason": "missing_cols_or_empty"}

    out = df.copy()

    x = pd.to_numeric(out["X_UTM"], errors="coerce")
    y = pd.to_numeric(out["Y_UTM"], errors="coerce")

    x_min = float(np.nanmin(x.to_numpy(dtype=float))) if x.notna().any() else 0.0
    y_min = float(np.nanmin(y.to_numpy(dtype=float))) if y.notna().any() else 0.0

    out["X_UTM"] = x - x_min
    out["Y_UTM"] = y - y_min

    if flip_x and np.isfinite(dist_x):
        out["X_UTM"] = float(dist_x) - out["X_UTM"]

    if clip and np.isfinite(dist_x) and np.isfinite(dist_y):
        out["X_UTM"] = out["X_UTM"].clip(lower=0.0, upper=float(dist_x))
        out["Y_UTM"] = out["Y_UTM"].clip(lower=0.0, upper=float(dist_y))

    meta = {
        "normalized": True,
        "x_shift": x_min,
        "y_shift": y_min,
        "flip_x": bool(flip_x),
        "clip": bool(clip),
        "dist_x": float(dist_x) if np.isfinite(dist_x) else None,
        "dist_y": float(dist_y) if np.isfinite(dist_y) else None,
    }
    return out, meta
