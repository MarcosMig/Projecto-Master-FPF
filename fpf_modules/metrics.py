import numpy as np
import pandas as pd

from .constants import (
    ACC_THR,
    COL_TIME,
    DEC_THR,
    HSR_MPS,
    SPRINT_BOUT_MIN_S,
    SPRINT_MPS,
)


def time_to_seconds(series: pd.Series) -> pd.Series:
    s = series.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        med = float(s_num.dropna().median()) if s_num.notna().any() else 0.0
        if med > 1e12:
            return s_num / 1000.0
        if med > 1e9:
            return s_num.astype(float)
        return s_num.astype(float)

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        t0 = dt.dropna().iloc[0]
        return (dt - t0).dt.total_seconds()
    return pd.Series([np.nan] * len(s), index=s.index, dtype="float64")


def count_bouts(t: np.ndarray, mask: np.ndarray, min_dur_s: float) -> int:
    """Conta episódios consecutivos onde mask==True com duração >= min_dur_s."""
    if len(t) == 0:
        return 0
    bouts = 0
    in_bout = False
    t_start = None
    for i in range(len(t)):
        if mask[i] and not in_bout:
            in_bout = True
            t_start = t[i]
        if (not mask[i]) and in_bout:
            dur = t[i - 1] - t_start if t_start is not None else 0.0
            if dur >= min_dur_s:
                bouts += 1
            in_bout = False
            t_start = None
    if in_bout and t_start is not None:
        dur = t[-1] - t_start
        if dur >= min_dur_s:
            bouts += 1
    return bouts


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

    t = t.sort_values().to_numpy(dtype=float)
    dt = np.diff(t)
    n_dt_neg = int((dt < 0).sum())
    n_dt_zero = int((dt == 0).sum())
    dt_pos = dt[dt > 0]

    dt_median = float(np.median(dt_pos)) if dt_pos.size else np.nan
    hz_est = float(1.0 / dt_median) if (dt_median and dt_median > 0) else np.nan

    n_gaps_0_2 = int((dt_pos > 0.2).sum())  # para 10Hz: >0.2s é gap relevante
    n_gaps_2 = int((dt_pos > 2.0).sum())

    wallclock_s = float(t[-1] - t[0])
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
    dfv = pd.DataFrame({"t": t_sec[valid], "x": x[valid], "y": y[valid]}).sort_values("t")

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

    dist_step = np.hypot(dx, dy)
    dist_total = float(np.nansum(dist_step))
    dur_s = float(np.nansum(dt))
    dur_min = dur_s / 60.0 if dur_s > 0 else 0.0
    m_min = (dist_total / dur_min) if dur_min > 0 else np.nan

    v = dist_step / dt
    vmax = float(np.nanmax(v)) if len(v) else np.nan

    # Active time (tempo em movimento)
    ACTIVE_V_THR = 0.5 # m/s
    active_time_s = float(np.nansum(dt[v >= ACTIVE_V_THR])) if len(v) else 0.0
    active_time_min = active_time_s / 60.0 if active_time_s > 0 else 0.0
    active_pct = (active_time_s / dur_s * 100.0) if dur_s > 0 else np.nan

    dv = np.diff(v)
    dt2 = dt[1:]
    acc = np.where(dt2 > 0, dv / dt2, np.nan)

    hsr_dist = float(np.nansum(dist_step[v >= HSR_MPS]))
    sprint_dist = float(np.nansum(dist_step[v >= SPRINT_MPS]))
    hsr_pct = (hsr_dist / dist_total * 100.0) if dist_total > 0 else np.nan

    n_acc = int(np.nansum(acc >= ACC_THR))
    n_dec = int(np.nansum(acc <= DEC_THR))

    dist_cum = np.concatenate([[0.0], np.cumsum(dist_step)])
    # t_cum alinhado com dist_cum (1+len(dist_step)); usamos o t original pós-filter
    # Nota: t[1:] tem comprimento igual a np.diff(t) (antes de filtrar); usamos a mesma máscara good
    t_cum = np.concatenate([[t[0]], t[1:][good]])

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

    t_mid = t[1:][good]
    sprint_mask = v >= SPRINT_MPS
    n_sprints = count_bouts(t_mid, sprint_mask, SPRINT_BOUT_MIN_S)

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
