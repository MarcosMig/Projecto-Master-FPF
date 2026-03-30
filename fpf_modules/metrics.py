import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull, QhullError

from .constants import (
    ACC_THR,
    COL_TIME,
    DEC_THR,
    HSR_MPS,
    SPRINT_BOUT_MIN_S,
    SPRINT_MPS,
)

MAX_VALID_SPEED_MPS = 11.0
MAX_VALID_STEP_M = 12.0
MAX_VALID_GAP_S = 2.0


def time_to_seconds(series: pd.Series) -> pd.Series:
    """Converte série temporal para segundos desde o início.

    Tenta formatos específicos antes de usar parsing genérico para evitar warnings.
    """
    s = series.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        med = float(s_num.dropna().median()) if s_num.notna().any() else 0.0
        if med > 1e12:
            return s_num / 1000.0
        if med > 1e9:
            return s_num.astype(float)
        return s_num.astype(float)

    # Try specific datetime formats first to avoid warnings
    dt = None

    # Try full timestamp format first (YYYY-MM-DD HH:MM:SS.ffffff)
    try:
        dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce", utc=True)
        if dt.notna().mean() > 0.5:  # If more than 50% parsed successfully
            pass  # Use this result
        else:
            dt = None
    except:
        dt = None

    # If full timestamp didn't work well, try time-only format (HH:MM:SS.ffffff)
    if dt is None or dt.notna().mean() < 0.5:
        try:
            dt = pd.to_datetime(s, format="%H:%M:%S.%f", errors="coerce", utc=True)
            if dt.notna().mean() > 0.5:
                pass  # Use this result
            else:
                dt = None
        except:
            dt = None

    # If specific formats didn't work, fall back to generic parsing (with warning suppressed)
    if dt is None or dt.notna().mean() < 0.5:
        dt = pd.to_datetime(s, errors="coerce", utc=True)

    if dt.notna().any():
        t0 = dt.dropna().iloc[0]
        return (dt - t0).dt.total_seconds()
    return pd.Series([np.nan] * len(s), index=s.index, dtype="float64")


def count_bouts(t: np.ndarray, mask: np.ndarray, min_dur_s: float) -> int:
    """Conta episódios consecutivos onde mask==True com duração >= min_dur_s.

    Robusto a:
      - arrays vazios
      - comprimentos diferentes entre t e mask
    """
    if t is None or mask is None:
        return 0
    n = min(len(t), len(mask))
    if n <= 0:
        return 0
    t = t[:n]
    mask = mask[:n]

    bouts = 0
    in_bout = False
    t_start = None

    for i in range(n):
        if bool(mask[i]) and not in_bout:
            in_bout = True
            t_start = float(t[i])

        if (not bool(mask[i])) and in_bout:
            # bout termina em i-1
            dur = float(t[i - 1]) - float(t_start) if t_start is not None else 0.0
            if dur >= float(min_dur_s):
                bouts += 1
            in_bout = False
            t_start = None

    if in_bout and t_start is not None:
        dur = float(t[n - 1]) - float(t_start)
        if dur >= float(min_dur_s):
            bouts += 1

    return bouts


def mark_valid_bouts(t: np.ndarray, mask: np.ndarray, min_dur_s: float) -> np.ndarray:
    """Marca samples pertencentes a bouts válidos com duração mínima."""
    if t is None or mask is None:
        return np.array([], dtype=bool)
    n = min(len(t), len(mask))
    if n <= 0:
        return np.array([], dtype=bool)

    t = t[:n]
    mask = mask[:n]
    valid_bout_mask = np.zeros(n, dtype=bool)
    in_bout = False
    t_start = None
    start_idx = None

    for i in range(n):
        if bool(mask[i]) and not in_bout:
            in_bout = True
            t_start = float(t[i])
            start_idx = i

        if (not bool(mask[i])) and in_bout:
            end_idx = i - 1
            dur = float(t[end_idx]) - float(t_start) if t_start is not None else 0.0
            if dur >= float(min_dur_s) and start_idx is not None:
                valid_bout_mask[start_idx : end_idx + 1] = True
            in_bout = False
            t_start = None
            start_idx = None

    if in_bout and t_start is not None and start_idx is not None:
        dur = float(t[n - 1]) - float(t_start)
        if dur >= float(min_dur_s):
            valid_bout_mask[start_idx:n] = True

    return valid_bout_mask


def compute_step_plausibility_stats(
    dt: np.ndarray,
    dist_step: np.ndarray,
    vmax_hard_mps: float = MAX_VALID_SPEED_MPS,
    jump_hard_m: float = MAX_VALID_STEP_M,
    gap_hard_s: float = MAX_VALID_GAP_S,
) -> dict:
    """Resume quantos steps falham critérios de plausibilidade."""
    if dt is None or dist_step is None or len(dt) == 0 or len(dist_step) == 0:
        return {
            "n_steps_total": 0,
            "n_steps_filtered_speed": 0,
            "n_steps_filtered_jump": 0,
            "n_steps_filtered_gap": 0,
            "pct_steps_filtered": 0.0,
            "plausible_mask": np.array([], dtype=bool),
        }

    v_raw = dist_step / dt
    filtered_speed = np.isfinite(v_raw) & (v_raw > vmax_hard_mps)
    filtered_jump = dist_step > jump_hard_m
    filtered_gap = dt > gap_hard_s
    plausible_mask = (
        np.isfinite(v_raw)
        & (~filtered_speed)
        & (~filtered_jump)
        & (~filtered_gap)
    )
    n_steps_total = int(len(dt))
    n_steps_filtered = int((~plausible_mask).sum())
    pct_steps_filtered = (n_steps_filtered / n_steps_total * 100.0) if n_steps_total > 0 else 0.0

    return {
        "n_steps_total": n_steps_total,
        "n_steps_filtered_speed": int(filtered_speed.sum()),
        "n_steps_filtered_jump": int(filtered_jump.sum()),
        "n_steps_filtered_gap": int(filtered_gap.sum()),
        "pct_steps_filtered": float(pct_steps_filtered),
        "plausible_mask": plausible_mask,
    }


def audit_timebase(df: pd.DataFrame, col_time: str, expected_hz: float = 10.0) -> dict:
    """Audita base temporal (por atleta/fase) para diagnosticar duplicados, gaps e Hz."""
    if df is None or df.empty or col_time not in df.columns:
        return {
            "n_rows": 0,
            "n_valid_t": 0,
            "wallclock_s": np.nan,
            "dt_median_s": np.nan,
            "hz_est": np.nan,
            "n_dt_neg": 0,
            "n_dt_zero": 0,
            "n_gaps_gt_0_2s": 0,
            "n_gaps_gt_2s": 0,
        }

    t = time_to_seconds(df[col_time])
    t = pd.to_numeric(t, errors="coerce").dropna()
    if t.shape[0] < 2:
        return {
            "n_rows": int(len(df)),
            "n_valid_t": int(t.shape[0]),
            "wallclock_s": 0.0,
            "dt_median_s": np.nan,
            "hz_est": np.nan,
            "n_dt_neg": 0,
            "n_dt_zero": 0,
            "n_gaps_gt_0_2s": 0,
            "n_gaps_gt_2s": 0,
        }

    t_raw = t.to_numpy(dtype=float)
    dt_raw = np.diff(t_raw)
    n_dt_neg = int((dt_raw < 0).sum())
    n_dt_zero = int((dt_raw == 0).sum())
    dt_pos = dt_raw[dt_raw > 0]

    dt_median = float(np.median(dt_pos)) if dt_pos.size else np.nan
    hz_est = float(1.0 / dt_median) if (dt_median and dt_median > 0) else np.nan

    gap_relevant_s = (2.0 / float(expected_hz)) if expected_hz > 0 else 0.2
    n_gaps_0_2 = int((dt_pos > gap_relevant_s).sum())
    n_gaps_2 = int((dt_pos > 2.0).sum())

    wallclock_s = float(np.nanmax(t_raw) - np.nanmin(t_raw))
    return {
        "n_rows": int(len(df)),
        "n_valid_t": int(len(t)),
        "wallclock_s": wallclock_s,
        "dt_median_s": dt_median,
        "hz_est": hz_est,
        "n_dt_neg": n_dt_neg,
        "n_dt_zero": n_dt_zero,
        "n_gaps_gt_0_2s": n_gaps_0_2,
        "n_gaps_gt_2s": n_gaps_2,
    }


def compute_metrics_for_df(df: pd.DataFrame) -> dict:
    """Calcula métricas para um atleta numa fase (df filtrado)."""
    default = {
        "duracao_min": 0.0,
        "dist_m": 0.0,
        "m_min": np.nan,
        "hsr_dist_m": 0.0,
        "sprint_dist_m": 0.0,
        "n_sprints": 0,
        "n_acc_2_5": 0,
        "n_dec_3_0": 0,
        "vmax_mps": np.nan,
        "peak_1m_m_min": np.nan,
        "hsr_pct": np.nan,
        "pct_time_valid": np.nan,
        "n_gaps_gt2s": 0,
        "n_points": 0,
        "active_time_min": 0.0,
        "active_pct": np.nan,
    }

    if df.empty:
        return default.copy()

    t_sec = time_to_seconds(df[COL_TIME])
    x = pd.to_numeric(df["X_UTM"], errors="coerce")
    y = pd.to_numeric(df["Y_UTM"], errors="coerce")
    valid = t_sec.notna() & x.notna() & y.notna()
    dfv = pd.DataFrame({"t": t_sec[valid], "x": x[valid], "y": y[valid]}).sort_values(
        "t"
    )

    out = default.copy()
    out["n_points"] = int(len(dfv))
    out["pct_time_valid"] = float(valid.mean() * 100.0) if len(valid) else np.nan

    if len(dfv) < 2:
        return out

    t = dfv["t"].to_numpy(dtype=float)
    dx = np.diff(dfv["x"].to_numpy(dtype=float))
    dy = np.diff(dfv["y"].to_numpy(dtype=float))
    dt = np.diff(t)

    good = dt > 0
    n_gaps = int(np.sum(dt[good] > 2.0)) if np.any(good) else 0

    dx, dy, dt = dx[good], dy[good], dt[good]
    if len(dt) == 0:
        return out

    step_start_t = t[:-1][good]
    step_end_t = t[1:][good]
    dist_step_raw = np.hypot(dx, dy)
    plausibility = compute_step_plausibility_stats(dt, dist_step_raw)
    plausible = plausibility["plausible_mask"]

    dist_step = dist_step_raw[plausible]
    dt = dt[plausible]
    step_start_t = step_start_t[plausible]
    step_end_t = step_end_t[plausible]

    if len(dt) == 0:
        return out

    dist_total = float(np.nansum(dist_step))
    dur_s = float(np.nansum(dt))
    dur_min = dur_s / 60.0 if dur_s > 0 else 0.0
    m_min = (dist_total / dur_min) if dur_min > 0 else np.nan

    v = dist_step / dt
    vmax = float(np.nanmax(v)) if len(v) else np.nan

    # Active time (tempo em movimento)
    ACTIVE_V_THR = 0.5  # m/s
    active_time_s = float(np.nansum(dt[v >= ACTIVE_V_THR])) if len(v) else 0.0
    active_time_min = active_time_s / 60.0 if active_time_s > 0 else 0.0
    active_pct = (active_time_s / dur_s * 100.0) if dur_s > 0 else np.nan

    dv = np.diff(v)
    dt2 = dt[1:]
    acc = np.where(dt2 > 0, dv / dt2, np.nan)

    hsr_dist = float(np.nansum(dist_step[v >= HSR_MPS]))
    sprint_mask = v >= SPRINT_MPS
    sprint_bout_mask = mark_valid_bouts(step_end_t, sprint_mask, SPRINT_BOUT_MIN_S)
    sprint_dist = float(np.nansum(dist_step[sprint_bout_mask]))
    hsr_pct = (hsr_dist / dist_total * 100.0) if dist_total > 0 else np.nan

    n_acc = int(np.nansum(acc >= ACC_THR))
    n_dec = int(np.nansum(acc <= DEC_THR))

    dist_cum = np.concatenate([[0.0], np.cumsum(dist_step)])
    # t_cum alinhado com dist_cum (1+len(dist_step)); usamos o t original pós-filter
    # Nota: t[1:] tem comprimento igual a np.diff(t) (antes de filtrar); usamos a mesma máscara good
    t_cum = np.concatenate([[step_start_t[0]], step_end_t])

    peak_1m = np.nan
    if len(t_cum) == len(dist_cum) and len(t_cum) > 1:
        best = 0.0
        j = 0
        for i in range(len(t_cum)):
            while j < len(t_cum) and t_cum[j] - t_cum[i] <= 60.0:
                j += 1
            if j - 1 >= i:
                dj = dist_cum[j - 1] - dist_cum[i]
                if dj > best:
                    best = dj
        peak_1m = float(best)

    n_sprints = count_bouts(step_end_t, sprint_mask, SPRINT_BOUT_MIN_S)

    return {
        "duracao_min": dur_min,
        "dist_m": dist_total,
        "m_min": m_min,
        "peak_1m_m_min": peak_1m,
        "vmax_mps": vmax,
        "hsr_dist_m": hsr_dist,
        "hsr_pct": hsr_pct,
        "sprint_dist_m": sprint_dist,
        "n_sprints": n_sprints,
        "n_acc_2_5": n_acc,
        "n_dec_3_0": n_dec,
        "n_points": int(len(dfv)),
        "pct_time_valid": float(valid.mean() * 100.0),
        "n_gaps_gt2s": n_gaps,
        "active_time_min": active_time_min,
        "active_pct": active_pct,
    }


# ── Tracking Data ─────────────────────────────────────────────────────


@st.cache_data
def calcular_area_cache(
    tracking_df, dist_x, dist_y, pitch_x=120, pitch_y=80
) -> pd.DataFrame:
    """Calcula a area ocupada por frame, convertendo de campo (default StatsBomb) em m²,
    usando as distancias originais do campo.

    Args:
        tracking_df (DataFrame): DataFrame que contem tracking data.
        dist_x (Float): Comprimento original do campo
        dist_y (Float): Largura original do campo.
        pitch_x (Integer): Comprimento do campo, default 120 (Statsbomb)
        pitch_y (Integer): Largura do campo, default 80 (Statsbomb)

    Returns:
        DataFrame: Área Ocupada por frame
    """
    df = tracking_df.copy()

    fator_conversao = (dist_x * dist_y) / (pitch_x * pitch_y)
    areas = []

    for time, frame in df.groupby("time_evento_s"):
        pts = frame[["x_tr", "y_tr"]].dropna().values

        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                areas.append(
                    {
                        "time_evento_s": time,
                        "area": round(hull.volume * fator_conversao, 2),
                    }
                )

            except QhullError:
                continue

    areas_df = pd.DataFrame(areas)

    return areas_df if not areas_df.empty else pd.DataFrame([])


@st.cache_data
def calcular_compactacao_cache(tracking_df, dist_x, dist_y, pitch_x=120, pitch_y=80):
    """Calcula compactação Vertical e Horizontal por Frame, converte de
    campo (default StatsBomb) para metros.

    Args:
        tracking_df (DataFrame): DataFrame que contem tracking data.
        dist_x (Float): Comprimento original do campo
        dist_y (Float): Largura original do campo.
        pitch_x (Integer): Comprimento do campo, default 120 (Statsbomb)
        pitch_y (Integer): Largura do campo, default 80 (Statsbomb)
    Returns:
        pd.DataFrame: DataFrame com a compactação vertical e horizontal por frame
        em metros.
    """
    if "time_evento_s" in tracking_df.columns:
        # Agrupar por tempo
        frame_data = tracking_df.groupby("time_evento_s").agg(
            x_min=("x_tr", "min"),
            x_max=("x_tr", "max"),
            y_min=("y_tr", "min"),
            y_max=("y_tr", "max"),
        )

        frame_data["comp_vertical"] = round(
            (frame_data["x_max"] - frame_data["x_min"]) * (pitch_x / dist_x), 2
        )
        frame_data["comp_horizontal"] = round(
            (frame_data["y_max"] - frame_data["y_min"]) * (pitch_y / dist_y), 2
        )

        frame_data = frame_data.drop(
            columns={"x_min", "x_max", "y_min", "y_max"}
        ).reset_index()

        return frame_data


def calcular_linhas(
    tracking_df,
    dist_x,
    pitch_x=120,
    linhas_pitch=False,
    n_linhas=3,
) -> dict | list:
    """Utilização do algoritmo KMeans para calculo das linhas tendo em conta a
    posição dos jogadores. Os resultados são convertidos em metros
    usando a distancia original do campo.

    Args:
        tracking_df (DataFrame): DataFrame que contem tracking data.
        dist_x (Float): Comprimento original do campo
        pitch_x (Integer): Comprimento do campo, default 120 (Statsbomb)
        linhas_pitch (Bool): Retorna tambem as linhas em coordenaas statsbomb.
        n_linhas (Integer): Número de linhas (clusters), default 3.
    Returns:
        dict | list:
        Se `n_linhas` for 3, retorna um dicionário com a localização
        das linhas e as distâncias entre elas.
        Para outros valores de `n_linhas`, retorna apenas uma lista com a localização
        das linhas.
    """

    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "A funcionalidade de cálculo de linhas requer a dependência opcional 'scikit-learn'."
        ) from exc

    # Conversão em metros
    fator_conversao = dist_x / pitch_x

    # Obter coordenadas x
    pts = tracking_df[["x_tr"]].dropna().values.reshape(-1, 1)

    # Fit Kmeans
    kmeans_lines = KMeans(n_clusters=n_linhas, random_state=42, n_init=10).fit(pts)

    # Ordenar linhas
    linhas_raw = np.sort(kmeans_lines.cluster_centers_.ravel())

    # Converter em metros
    linhas = linhas_raw * fator_conversao

    if n_linhas == 3:
        linha_def, linha_med, linha_ata = linhas

        resultado = {
            "linha_def": round(linha_def, 2),
            "linha_med": round(linha_med, 2),
            "linha_ata": round(linha_ata, 2),
            "dist_def_med": round(linha_med - linha_def, 2),
            "dist_def_ata": round(linha_ata - linha_def, 2),
            "dist_mid_ata": round(linha_ata - linha_med, 2),
        }

        if linhas_pitch:
            linha_def_r, linha_med_r, linha_ata_r = linhas_raw
            # Adicionar novas chaves
            resultado.update(
                {
                    "linha_def_sb": round(linha_def_r, 2),
                    "linha_med_sb": round(linha_med_r, 2),
                    "linha_ata_sb": round(linha_ata_r, 2),
                }
            )

        return resultado

    else:
        if linhas_pitch:
            return linhas_raw

        return linhas
