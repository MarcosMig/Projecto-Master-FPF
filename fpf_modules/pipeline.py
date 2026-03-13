from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import savgol_filter

from .constants import COL_FASE, COL_LAT, COL_LON, COL_TIME
from .io_utils import get_atleta_id, infer_fase, read_csv_upload
from .metrics import time_to_seconds, compute_metrics_for_df

# helpers for analytic schema
from .data_manager import (
    resolve_athlete_sk,
    resolve_session_sk,
    resolve_game_sk,
    write_metrics,
)


HR_CANDIDATE_COLS = ["HR_bpm", "HR", "HeartRate", "Heart Rate", "Heart_Rate", "BPM", "Pulse"]


def _detect_hr_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for col in HR_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


def _format_clock_mmss(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    minutes = total // 60
    secs = total % 60
    return f"{sign}00:{minutes:02d}:{secs:02d}"


def _build_event_clock(fases_dict_s: dict) -> dict:
    event_clock = {}

    if "1P" in fases_dict_s:
        s1, e1 = fases_dict_s["1P"]
        dur1 = max(0.0, float(e1 - s1))
        extra1 = max(0.0, dur1 - 45.0 * 60.0)
        event_clock["1P"] = {
            "start_s": 0.0,
            "reg_end_s": 45.0 * 60.0,
            "end_s": 45.0 * 60.0 + extra1,
            "duration_s": dur1,
            "extra_s": extra1,
        }

    if "2P" in fases_dict_s:
        s2, e2 = fases_dict_s["2P"]
        dur2 = max(0.0, float(e2 - s2))
        extra2 = max(0.0, dur2 - 45.0 * 60.0)
        event_clock["2P"] = {
            "start_s": 45.0 * 60.0,
            "reg_end_s": 90.0 * 60.0,
            "end_s": 90.0 * 60.0 + extra2,
            "duration_s": dur2,
            "extra_s": extra2,
        }

    if "Warm-Up" in fases_dict_s:
        sw, ew = fases_dict_s["Warm-Up"]
        durw = max(0.0, float(ew - sw))
        event_clock["Warm-Up"] = {
            "start_s": -durw,
            "reg_end_s": 0.0,
            "end_s": 0.0,
            "duration_s": durw,
            "extra_s": 0.0,
        }

    return event_clock


def processar_atletas_para_temp(
    f_atleta_files, epsg, origin, R, aplicar_suav, janela, poly, temp_dir: Path
):
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    groups = {}
    for f in f_atleta_files:
        aid = get_atleta_id(f.name)
        groups.setdefault(aid, []).append(f)

    temp_files = []
    audit = {}
    issues = []

    for aid, files in groups.items():
        atleta_data = []
        audit[aid] = []
        for uf in files:
            try:
                df = read_csv_upload(uf)
                fase_n = infer_fase(uf.name)
                audit[aid].append(fase_n)

                if df.empty:
                    continue
                if COL_TIME not in df.columns:
                    issues.append((aid, uf.name, "Sem coluna Time"))
                    continue
                if COL_LAT not in df.columns or COL_LON not in df.columns:
                    issues.append((aid, uf.name, "Sem colunas Lat/Lon"))
                    continue

                hr_col = _detect_hr_col(df)
                if hr_col:
                    df["HR_bpm"] = pd.to_numeric(df[hr_col], errors="coerce")

                lon = df[COL_LON].astype(float)
                lat = df[COL_LAT].astype(float)
                ux, uy = trans.transform(lon.values, lat.values)
                p_loc = (R @ (np.vstack([ux, uy]).T - origin).T).T

                df["X_UTM"] = p_loc[:, 0]
                df["Y_UTM"] = p_loc[:, 1]

                nan_before = int(df["X_UTM"].isna().sum() + df["Y_UTM"].isna().sum())
                df["X_UTM"] = df["X_UTM"].interpolate(limit=1, limit_direction="both")
                df["Y_UTM"] = df["Y_UTM"].interpolate(limit=1, limit_direction="both")
                nan_after = int(df["X_UTM"].isna().sum() + df["Y_UTM"].isna().sum())
                df["_micro_gaps_corrigidos"] = max(0, nan_before - nan_after)

                if aplicar_suav and len(df) >= int(janela) and int(janela) % 2 == 1:
                    x = pd.Series(df["X_UTM"]).interpolate()
                    y = pd.Series(df["Y_UTM"]).interpolate()
                    try:
                        df["X_UTM"] = savgol_filter(x, int(janela), int(poly))
                        df["Y_UTM"] = savgol_filter(y, int(janela), int(poly))
                    except Exception:
                        pass

                df[COL_FASE] = fase_n
                df["Atleta_ID"] = aid
                cols_keep = [COL_TIME, "Atleta_ID", COL_FASE, COL_LAT, COL_LON, "X_UTM", "Y_UTM"]
                if "HR_bpm" in df.columns:
                    cols_keep.append("HR_bpm")
                atleta_data.append(df[cols_keep])
            except Exception as e:
                issues.append((aid, uf.name, f"Erro a processar: {e}"))

        if atleta_data:
            out = pd.concat(atleta_data, ignore_index=True).sort_values(by=COL_TIME)
            out_path = temp_dir / f"T_{aid}.csv"
            out.to_csv(out_path, sep=";", index=False)
            temp_files.append(out_path)

    return temp_files, audit, issues


def sincronizar(temp_files, out_dir: Path):
    fases_dict = {}
    all_unique_times = set()

    for f in temp_files:
        df = pd.read_csv(f, sep=";", usecols=[COL_TIME, COL_FASE]).dropna(subset=[COL_TIME])
        all_unique_times.update(df[COL_TIME].tolist())
        for fs in df[COL_FASE].dropna().unique():
            t_fase = df.loc[df[COL_FASE] == fs, COL_TIME]
            t_s, t_e = t_fase.min(), t_fase.max()
            if fs not in fases_dict:
                fases_dict[fs] = [t_s, t_e]
            else:
                fases_dict[fs][0] = min(fases_dict[fs][0], t_s)
                fases_dict[fs][1] = max(fases_dict[fs][1], t_e)

    master_df = pd.DataFrame({COL_TIME: sorted(list(all_unique_times))})
    master_df["__time_s"] = pd.to_numeric(time_to_seconds(master_df[COL_TIME]), errors="coerce")
    if master_df["__time_s"].isna().all():
        master_df["__time_s"] = np.arange(len(master_df), dtype=float)

    time_map = dict(zip(master_df[COL_TIME], master_df["__time_s"]))
    fases_dict_s = {
        fase: (float(time_map.get(t_s, np.nan)), float(time_map.get(t_e, np.nan)))
        for fase, (t_s, t_e) in fases_dict.items()
    }
    event_clock = _build_event_clock(fases_dict_s)

    out_files = []
    for f in temp_files:
        df_atl = pd.read_csv(f, sep=";")
        df_sync = pd.merge(master_df[[COL_TIME, "__time_s"]], df_atl, on=COL_TIME, how="left")
        aid = str(df_atl["Atleta_ID"].iloc[0]) if "Atleta_ID" in df_atl.columns else f.stem.replace("T_", "")
        df_sync["Atleta_ID"] = aid

        for fase, (t_s, t_e) in fases_dict.items():
            mask = (df_sync[COL_TIME] >= t_s) & (df_sync[COL_TIME] <= t_e)
            df_sync.loc[mask, COL_FASE] = fase

        df_sync["Time_Evento_s"] = np.nan
        df_sync["Periodo_Jogo"] = pd.NA
        for fase, win in fases_dict_s.items():
            if fase not in event_clock:
                continue
            t_s, t_e = win
            if not np.isfinite(t_s) or not np.isfinite(t_e):
                continue

            # compute event time and period for rows in this phase
            mask = df_sync[COL_FASE] == fase
            rel_s = df_sync.loc[mask, "__time_s"] - t_s
            abs_s = event_clock[fase]["start_s"] + rel_s
            df_sync.loc[mask, "Time_Evento_s"] = abs_s

            if fase == "1P":
                df_sync.loc[mask & (df_sync["Time_Evento_s"] <= 45.0 * 60.0), "Periodo_Jogo"] = "1P"
                df_sync.loc[mask & (df_sync["Time_Evento_s"] > 45.0 * 60.0), "Periodo_Jogo"] = "ET_1P"
            elif fase == "2P":
                df_sync.loc[mask & (df_sync["Time_Evento_s"] <= 90.0 * 60.0), "Periodo_Jogo"] = "2P"
                df_sync.loc[mask & (df_sync["Time_Evento_s"] > 90.0 * 60.0), "Periodo_Jogo"] = "ET_2P"
            elif fase == "Warm-Up":
                df_sync.loc[mask, "Periodo_Jogo"] = "Warm-Up"

        df_sync["Time_Evento"] = df_sync["Time_Evento_s"].map(_format_clock_mmss)
        df_sync["Minuto_Jogo"] = np.floor(df_sync["Time_Evento_s"] / 60.0)
        df_sync.loc[df_sync["Time_Evento_s"].isna(), "Minuto_Jogo"] = np.nan
        df_sync["Minuto_Jogo"] = df_sync["Minuto_Jogo"].astype("Int64")
        df_sync = df_sync.drop(columns=["__time_s"])

        out_path = out_dir / f"Player_{aid}_SYNC.csv"
        df_sync.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
        out_files.append(out_path)

    ordem_fases = {"Warm-Up": 0, "1P": 1, "2P": 2}
    fases_ordenadas = sorted(fases_dict.items(), key=lambda x: ordem_fases.get(x[0], 99))
    return out_files, fases_ordenadas, fases_dict, len(master_df), event_clock


# ------------------------------------------------
# analytic pipeline helpers
# ------------------------------------------------

def compute_and_save_metrics(
    temp_files: list,
    game_payload: dict = None,
    session_payload: dict = None,
    db_file: str = None,
):
    """Compute metrics from the temporary csv files and persist them.

    Parameters
    ----------
    temp_files : list of pathlib.Path
        CSV files produced by :func:`processar_atletas_para_temp`.
    game_payload : dict, optional
        Identifiers for the current game (e.g. date, opponent).  Used to
        resolve a game_sk.
    session_payload : dict, optional
        Metadata about the session.  Used to resolve a session_sk.
    db_file : str, optional
        Path to the duckdb file; forwarded to :func:`write_metrics`.
    """
    if game_payload is None:
        game_payload = {}
    if session_payload is None:
        session_payload = {}

    # resolve dimensions
    game_sk = resolve_game_sk(game_payload) if game_payload else None
    session_sk = resolve_session_sk(
        session_payload.get("fingerprint", None), session_payload
    )

    all_metrics = []
    for f in temp_files:
        df = pd.read_csv(f, sep=";")
        if df.empty:
            continue
        athlete_id = str(df["Atleta_ID"].iloc[0])
        athlete_sk_map = resolve_athlete_sk(df)
        athlete_sk = athlete_sk_map.get(athlete_id)

        # compute metrics per phase
        for fase, grp in df.groupby(COL_FASE):
            m = compute_metrics_for_df(grp)
            # flatten into a single row
            row = {
                "athlete_sk": athlete_sk,
                "session_sk": session_sk,
                "game_sk": game_sk,
                "phase": fase,
                **m,
            }
            all_metrics.append(row)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        write_metrics(metrics_df, db_file=db_file)
    return all_metrics

