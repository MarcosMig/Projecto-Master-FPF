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

###################################
# TRACKING DATA                   #
###################################

def obter_direcao_ataque(tracking_df, largura_campo_x=120):
    """Identifica a direção do ataque com base na posição média inicial (Kick-off).
        Assume campo de 120m (Statsbomb).

    Args:
        tracking_df (DataFrame): DataFrame com dados de posição.
        largura_x (int, optional): Largura do campo. Defaults to 120 (Dados da StatsBomb).

    Returns:
        tracking_df: DataFrame original + coluna de direção ataque
    """

    meio_campo = largura_campo_x / 2

    # 1. Pegar apenas o primeiro frame do 1P (Kick-off)
    df_1p = tracking_df[tracking_df['fase'] == '1P']
    if df_1p.empty:
        return tracking_df

    ko_time = df_1p['time'].min()
    ko_data = df_1p[df_1p['time'] == ko_time]

    # 2. Calcular a média X no início
    ko_avg_x = ko_data['x_tr'].mean()

    # 3. Atribuir direção ataque
    if ko_avg_x <= meio_campo:
        dir_1p, dir_2p = 'L to R', 'R to L'
    else:
        dir_1p, dir_2p = 'R to L', 'L to R'

    # Atribuir os valores
    tracking_df.loc[tracking_df['fase'] == '1P', 'direcao_ataque'] = dir_1p
    tracking_df.loc[tracking_df['fase'] == '2P', 'direcao_ataque'] = dir_2p
    tracking_df.loc[tracking_df['fase'] == 'Warm-Up', 'direcao_ataque'] = 'N/A'

    return tracking_df



def normaliza_direcao_ataque(df, largura_campo_x = 120, comprimento_campo_y =80):
    """Normaliza direção de ataque, para que ambas equipas ataquem Esquerda -> Direita.
        Utiliza por defeito tamanho da StatsBomb 120 por 80
    Args:
        df: DataFrame containing tracking data with Attack Direction info
        largura_campo_x (int, optional): Largura do campo, defaults to 120.
        comprimento_campo_y (int, optional): Largura do camp, defaults to 80.
    """

    dir_ataque_mask = (df['direcao_ataque'] == 'R to L')

    df.loc[dir_ataque_mask, 'x_tr'] = largura_campo_x - df['x_tr']
    df.loc[dir_ataque_mask, 'y_tr'] = comprimento_campo_y - df['y_tr']

    df = df.drop(columns='direcao_ataque')

    return df


def _interpolate_short_tracking_gaps(
    tracking: pd.DataFrame,
    max_gap_s: float,
    sample_hz: float,
) -> pd.DataFrame:
    """Interpolate short in-window gaps without bridging substitutions.

    Assumes the synchronized export already contains the full per-player time grid.
    Interpolation is restricted to NaNs between the first and last valid sample of
    each player/phase, so pre-entry and post-exit intervals remain missing.
    """
    if tracking is None or tracking.empty:
        return tracking

    required_cols = {"atleta_id", "fase", "x_utm", "y_utm"}
    if not required_cols.issubset(tracking.columns):
        return tracking

    limit_frames = max(1, int(round(float(max_gap_s) * float(sample_hz))))
    out = tracking.copy()

    sort_cols = [col for col in ["atleta_id", "fase", "time_evento_s", "time"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()

    for _, group_idx in out.groupby(["atleta_id", "fase"], sort=False).groups.items():
        idx = list(group_idx)
        group = out.loc[idx, ["x_utm", "y_utm"]].copy()

        for coord_col in ["x_utm", "y_utm"]:
            series = pd.to_numeric(group[coord_col], errors="coerce")
            if series.notna().sum() < 2:
                continue

            group[coord_col] = series.interpolate(
                method="linear",
                limit=limit_frames,
                limit_direction="both",
                limit_area="inside",
            )

        out.loc[idx, ["x_utm", "y_utm"]] = group[["x_utm", "y_utm"]].to_numpy()

    return out

def normalize_tracking_data(
    df,
    dist_x,
    dist_y,
    pitch_x = 120,
    pitch_y = 80,
    clip_tolerance_m=0.5,
    raw_tolerance_m=5.0,
    short_gap_max_s=0.5,
    sample_hz=10.0,
):
    """Takes normalized meters and converts them to the target coordinate system.
    By default uses Statsbomb coordinates:
        - X [0, 120]
        - Y [0, 80]

    Calls heleprs functions:
    - Obtain atack direction
    - Normalize atack direction. L -> R

    Args:
        df : DataFrame
        dist_x (_type_): Max Pitch Length
        dist_y (_type_): Max Pitch width
        pitch_x (_type_): Max Pitch X
        pitch_y (_type_): Max Pitch X

    Returns:
        DataFrame: DataFrame ready for tracking data usage
    """
    tracking = df.copy()

    x_utm_raw = pd.to_numeric(tracking['x_utm'], errors='coerce')
    y_utm_raw = pd.to_numeric(tracking['y_utm'], errors='coerce')

    x_under = (0.0 - x_utm_raw).clip(lower=0.0)
    x_over = (x_utm_raw - float(dist_x)).clip(lower=0.0)
    y_under = (0.0 - y_utm_raw).clip(lower=0.0)
    y_over = (y_utm_raw - float(dist_y)).clip(lower=0.0)
    max_oob_m = pd.concat([x_under, x_over, y_under, y_over], axis=1).max(axis=1)

    small_oob = (max_oob_m > 0.0) & (max_oob_m <= float(clip_tolerance_m))
    large_oob = (max_oob_m > float(clip_tolerance_m)) & (max_oob_m <= float(raw_tolerance_m))
    gross_oob = max_oob_m > float(raw_tolerance_m)

    tracking['flag_oob_small_clip'] = small_oob.fillna(False)
    tracking['flag_oob_drop'] = (large_oob | gross_oob).fillna(False)
    tracking['flag_interpolated_short_gap'] = False
    tracking['oob_distance_m'] = pd.to_numeric(max_oob_m, errors='coerce').fillna(0.0)

    x_utm = x_utm_raw.copy()
    y_utm = y_utm_raw.copy()

    # Large out-of-bounds points are treated as missing instead of being glued to the line.
    drop_mask = (large_oob | gross_oob).fillna(False)
    if drop_mask.any():
        x_utm = x_utm.mask(drop_mask)
        y_utm = y_utm.mask(drop_mask)

    # Minor GPS overshoot around the touchlines is clipped back into the field.
    x_utm = x_utm.clip(lower=0.0, upper=float(dist_x))
    y_utm = y_utm.clip(lower=0.0, upper=float(dist_y))

    pre_interp_missing = x_utm.isna() | y_utm.isna()

    tracking['x_utm'] = x_utm
    tracking['y_utm'] = y_utm
    tracking = _interpolate_short_tracking_gaps(
        tracking,
        max_gap_s=float(short_gap_max_s),
        sample_hz=float(sample_hz),
    )
    x_utm = pd.to_numeric(tracking['x_utm'], errors='coerce')
    y_utm = pd.to_numeric(tracking['y_utm'], errors='coerce')
    post_interp_filled = pre_interp_missing & x_utm.notna() & y_utm.notna()
    tracking['flag_interpolated_short_gap'] = post_interp_filled.fillna(False)

    tracking['x_tr'] = (x_utm / float(dist_x)) * float(pitch_x)
    tracking['y_tr'] = (y_utm / float(dist_y)) * float(pitch_y)

    # Obter Direção de Ataque e Normalizar de forma a que ataquem sempre L -> R
    tracking = obter_direcao_ataque(tracking, pitch_x)
    tracking = normaliza_direcao_ataque(tracking, pitch_x, pitch_y)

    tracking['x_tr'] = pd.to_numeric(tracking['x_tr'], errors='coerce').clip(lower=0.0, upper=float(pitch_x))
    tracking['y_tr'] = pd.to_numeric(tracking['y_tr'], errors='coerce').clip(lower=0.0, upper=float(pitch_y))

    return tracking
