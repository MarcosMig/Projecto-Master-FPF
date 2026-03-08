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

def normalize_tracking_data(
    df,
    dist_x,
    dist_y,
    pitch_x = 120,
    pitch_y = 80
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

    # Scale data using
    tracking['x_tr'] = (tracking['x_utm'] / dist_x) * pitch_x
    tracking['y_tr'] = (tracking['y_utm'] / dist_y) * pitch_y

    # Obter Direção de Ataque e Normalizar de forma a que ataquem sempre L -> R
    tracking = obter_direcao_ataque(tracking, pitch_x)
    tracking = normaliza_direcao_ataque(tracking, pitch_x, pitch_y)

    return tracking
