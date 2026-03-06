from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import savgol_filter

from .constants import COL_FASE, COL_LAT, COL_LON, COL_TIME
from .io_utils import get_atleta_id, infer_fase, read_csv_upload
from .metrics import time_to_seconds


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
                atleta_data.append(df[[COL_TIME, "Atleta_ID", COL_FASE, COL_LAT, COL_LON, "X_UTM", "Y_UTM"]])
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
