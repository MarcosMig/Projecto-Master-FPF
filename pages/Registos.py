import pandas as pd
import streamlit as st
import re
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fpf_modules.reference_data import load_selection_reference
from fpf_modules.supabase_manager import cleanup_session_upload, initialize_schema, read_table


PHASE_ORDER = ["Warm-Up", "1P", "2P"]
PERFORMANCE_GROUPS = {
    "Volume": ["duracao_min", "dist_m", "hsr_dist_m", "sprint_dist_m", "active_time_min"],
    "Intensidade": ["m_min", "hsr_pct", "active_pct"],
    "Eventos": ["n_sprints", "n_acc_2_5", "n_dec_3_0"],
    "Picos de Fase": ["vmax_mps", "peak_1m_m_min"],
    "Carga": ["hr_avg_bpm", "external_load_score", "total_load_score", "player_load", "rhie_bouts", "trimp_banister"],
}
INDIVIDUAL_PROFILE_METRIC_MAP = {
    "duracao_min": ("duracao_min", "duracao_min_media", "Duração Média (min)"),
    "dist_m": ("dist_m_90", "dist_m_90", "Distância / 90 (m)"),
    "hsr_dist_m": ("hsr_dist_m_90", "hsr_dist_m_90", "Distância HSR / 90 (m)"),
    "sprint_dist_m": ("sprint_dist_m_90", "sprint_dist_m_90", "Distância Sprint / 90 (m)"),
    "active_time_min": ("active_time_min_90", "active_time_min_90", "Tempo Ativo / 90 (min)"),
    "m_min": ("m_min", "m_min", "Intensidade (m/min)"),
    "hsr_pct": ("hsr_pct", "hsr_pct", "HSR (%)"),
    "active_pct": ("active_pct", "active_pct", "Tempo Ativo (%)"),
    "n_sprints": ("n_sprints_90", "n_sprints_90", "N.º Sprints / 90"),
    "n_acc_2_5": ("n_acc_2_5_90", "n_acc_2_5_90", "N.º Acelerações / 90"),
    "n_dec_3_0": ("n_dec_3_0_90", "n_dec_3_0_90", "N.º Desacelerações / 90"),
    "hr_avg_bpm": ("hr_avg_bpm", "hr_avg_bpm", "FC Média (bpm)"),
    "external_load_score": ("external_load_score_90", "external_load_score_90", "Carga Externa / 90"),
    "total_load_score": ("total_load_score_90", "total_load_score_90", "Carga Total / 90"),
    "player_load": ("player_load_90", "player_load_90", "Player Load / 90"),
    "rhie_bouts": ("rhie_bouts_90", "rhie_bouts_90", "RHIE / 90"),
    "trimp_banister": ("trimp_banister_90", "trimp_banister_90", "TRIMP / 90"),
    "vmax_mps": ("vmax_mps", "vmax_mps_peak", "Velocidade Máxima (m/s)"),
    "peak_1m_m_min": ("peak_1m_m_min", "peak_1m_m_min_peak", "Pico 1 min (m/min)"),
}
COLLECTIVE_PROFILE_METRIC_MAP = {
    "duracao_min": ("duracao_min", "duracao_min", "Duração (min)"),
    "dist_m": ("dist_m_90", "dist_m_90", "Distância / 90 (m)"),
    "hsr_dist_m": ("hsr_dist_m_90", "hsr_dist_m_90", "Distância HSR / 90 (m)"),
    "sprint_dist_m": ("sprint_dist_m_90", "sprint_dist_m_90", "Distância Sprint / 90 (m)"),
    "active_time_min": ("active_time_min_90", "active_time_min_90", "Tempo Ativo / 90 (min)"),
    "m_min": ("m_min", "m_min", "Intensidade (m/min)"),
    "hsr_pct": ("hsr_pct", "hsr_pct", "HSR (%)"),
    "active_pct": ("active_pct", "active_pct", "Tempo Ativo (%)"),
    "n_sprints": ("n_sprints_90", "n_sprints_90", "N.º Sprints / 90"),
    "n_acc_2_5": ("n_acc_2_5_90", "n_acc_2_5_90", "N.º Acelerações / 90"),
    "n_dec_3_0": ("n_dec_3_0_90", "n_dec_3_0_90", "N.º Desacelerações / 90"),
    "hr_avg_bpm": ("hr_avg_bpm", "hr_avg_bpm", "FC Média (bpm)"),
    "external_load_score": ("external_load_score_90", "external_load_score_90", "Carga Externa / 90"),
    "total_load_score": ("total_load_score_90", "total_load_score_90", "Carga Total / 90"),
    "player_load": ("player_load_90", "player_load_90", "Player Load / 90"),
    "rhie_bouts": ("rhie_bouts_90", "rhie_bouts_90", "RHIE / 90"),
    "trimp_banister": ("trimp_banister_90", "trimp_banister_90", "TRIMP / 90"),
    "vmax_mps": ("vmax_mps", "vmax_mps", "Velocidade Máxima (m/s)"),
    "peak_1m_m_min": ("peak_1m_m_min", "peak_1m_m_min", "Pico 1 min (m/min)"),
}
COLLECTIVE_REFERENCE_COLUMN_MAP = {
    "duracao_min": "duracao_min_total",
    "dist_m": "dist_m_total",
    "hsr_dist_m": "hsr_dist_m_total",
    "sprint_dist_m": "sprint_dist_m_total",
    "active_time_min": "active_time_min_total",
    "m_min": "m_min_avg",
    "hsr_pct": "hsr_pct_avg",
    "active_pct": "active_pct_avg",
    "n_sprints": "n_sprints_total",
    "n_acc_2_5": "n_acc_2_5_total",
    "n_dec_3_0": "n_dec_3_0_total",
    "vmax_mps": "vmax_mps_max",
    "peak_1m_m_min": "peak_1m_m_min_max",
    "hr_avg_bpm": "hr_avg_bpm_avg",
    "external_load_score": "external_load_score_total",
    "total_load_score": "total_load_score_total",
    "player_load": "player_load_total",
    "rhie_bouts": "rhie_bouts_total",
    "trimp_banister": "trimp_banister_total",
}


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _format_date(value) -> str:
    if pd.isna(value) or value in ("", None):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _sanitize_filename_part(value: str) -> str:
    text = _clean_text_value(value)
    if not text:
        return "sessao"
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sessao"


def _session_pdf_names(session_fingerprint: str, selecao: str, contexto: str, jogo: str) -> tuple[str, str]:
    selecao_part = _sanitize_filename_part(selecao)
    contexto_part = _sanitize_filename_part(contexto)
    jogo_part = _sanitize_filename_part(jogo)
    base_name = f"{selecao_part}_{contexto_part}_{jogo_part}_{_sanitize_filename_part(session_fingerprint)[:16]}"
    return (
        f"{base_name}_coletivo.pdf",
        f"{base_name}_individual.pdf",
    )


def _load_session_reports() -> pd.DataFrame:
    try:
        initialize_schema()
        df = read_table("session_reports")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    for col in ["session_fingerprint", "selecao", "genero", "contexto", "jogo", "report_title", "report_txt"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_clean_text_value)

    if "session_sk" not in df.columns:
        df["session_sk"] = pd.NA
    df["session_sk"] = pd.to_numeric(df["session_sk"], errors="coerce").astype("Int64")

    if "data" not in df.columns:
        df["data"] = pd.NaT
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    if "updated_at" not in df.columns:
        df["updated_at"] = pd.NaT
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    return df.sort_values(["data", "updated_at"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _session_pdf_cache() -> dict[str, dict[str, bytes]]:
    return st.session_state.setdefault("session_pdf_cache", {})


def _clear_session_pdf_cache(session_fingerprint: str) -> None:
    _session_pdf_cache().pop(session_fingerprint, None)


def _normalize_athlete_identifier(value) -> str:
    text = _clean_text_value(value)
    if not text:
        return ""
    return re.sub(r"^0+(?=\d+$)", "", text)


def _safe_mean(series) -> float:
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _format_metric_value(metric_col: str, value) -> str:
    if value is None or pd.isna(value):
        return "-"
    if metric_col in {"m_min", "vmax_mps", "peak_1m_m_min", "trimp_banister", "player_load"}:
        return f"{float(value):.1f}"
    if metric_col in {"hsr_pct", "active_pct"}:
        return f"{float(value):.1f}%"
    return f"{int(round(float(value)))}" if abs(float(value) - round(float(value))) < 0.05 else f"{float(value):.1f}"


def _load_athlete_name_map() -> dict[str, str]:
    try:
        athletes_df = read_table("athletes")
    except Exception:
        return {}
    if athletes_df is None or athletes_df.empty or not {"atleta_id", "nome"}.issubset(athletes_df.columns):
        return {}
    work_df = athletes_df[["atleta_id", "nome"]].copy()
    work_df["atleta_id"] = work_df["atleta_id"].map(_normalize_athlete_identifier)
    work_df["nome"] = work_df["nome"].astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("") & work_df["nome"].ne("")]
    return dict(zip(work_df["atleta_id"], work_df["nome"]))


def _load_session_performance_rows(session_sk: int) -> pd.DataFrame:
    df = read_table("performance_metrics", {"session_sk": session_sk})
    if df is None or df.empty:
        return pd.DataFrame()
    if "fase" in df.columns:
        df["fase"] = df["fase"].map(_clean_text_value)
    if "atleta_id" in df.columns:
        df["atleta_id"] = df["atleta_id"].map(_normalize_athlete_identifier)
    return df


def _build_totals_by_athlete_from_session_df(session_df: pd.DataFrame) -> pd.DataFrame:
    if session_df is None or session_df.empty or "atleta_id" not in session_df.columns:
        return pd.DataFrame()
    totals_df = session_df.copy()
    totals_df = totals_df[totals_df["fase"].eq("Total")].copy() if "fase" in totals_df.columns else totals_df
    if totals_df.empty:
        return pd.DataFrame()
    duration = pd.to_numeric(totals_df.get("duracao_min"), errors="coerce") if "duracao_min" in totals_df.columns else pd.Series(np.nan, index=totals_df.index)
    valid_duration = duration.notna() & (duration > 0)
    per90_map = {
        "dist_m": "dist_m_90",
        "hsr_dist_m": "hsr_dist_m_90",
        "sprint_dist_m": "sprint_dist_m_90",
        "active_time_min": "active_time_min_90",
        "n_sprints": "n_sprints_90",
        "n_acc_2_5": "n_acc_2_5_90",
        "n_dec_3_0": "n_dec_3_0_90",
        "external_load_score": "external_load_score_90",
        "total_load_score": "total_load_score_90",
        "player_load": "player_load_90",
        "rhie_bouts": "rhie_bouts_90",
        "trimp_banister": "trimp_banister_90",
    }
    for source_col, target_col in per90_map.items():
        if source_col in totals_df.columns:
            totals_df[target_col] = np.where(valid_duration, pd.to_numeric(totals_df[source_col], errors="coerce") / duration * 90.0, np.nan)
    return totals_df


def _build_collective_phase_totals_from_session_df(session_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if session_df is None or session_df.empty or "fase" not in session_df.columns:
        return pd.DataFrame()
    work_df = session_df.copy()
    work_df["fase"] = work_df["fase"].astype(str).str.strip()
    work_df = work_df[work_df["fase"].isin(PHASE_ORDER)].copy()
    if work_df.empty:
        return pd.DataFrame()
    cols_present = [col for col in metric_cols if col in work_df.columns]
    if "duracao_min" in work_df.columns and "duracao_min" not in cols_present:
        cols_present = ["duracao_min"] + cols_present
    if not cols_present:
        return pd.DataFrame()
    for col in cols_present:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    aggregation_rules = {
        "m_min": "mean", "hsr_pct": "mean", "active_pct": "mean", "hr_avg_bpm": "mean",
        "vmax_mps": "max", "peak_1m_m_min": "max",
    }
    rows = []
    for phase in PHASE_ORDER:
        phase_df = work_df[work_df["fase"].eq(phase)].copy()
        if phase_df.empty:
            continue
        row = {"fase": phase}
        for col in cols_present:
            series = pd.to_numeric(phase_df[col], errors="coerce")
            rule = aggregation_rules.get(col, "sum")
            row[col] = float(series.mean()) if rule == "mean" and series.notna().any() else (
                float(series.max()) if rule == "max" and series.notna().any() else float(series.fillna(0.0).sum())
            )
        rows.append(row)
    collective_df = pd.DataFrame(rows)
    if collective_df.empty:
        return pd.DataFrame()
    duration = pd.to_numeric(collective_df["duracao_min"], errors="coerce") if "duracao_min" in collective_df.columns else pd.Series(np.nan, index=collective_df.index)
    valid_duration = duration.notna() & (duration > 0)
    per90_map = {
        "dist_m": "dist_m_90", "hsr_dist_m": "hsr_dist_m_90", "sprint_dist_m": "sprint_dist_m_90",
        "active_time_min": "active_time_min_90", "n_sprints": "n_sprints_90", "n_acc_2_5": "n_acc_2_5_90",
        "n_dec_3_0": "n_dec_3_0_90", "external_load_score": "external_load_score_90",
        "total_load_score": "total_load_score_90", "player_load": "player_load_90",
        "rhie_bouts": "rhie_bouts_90", "trimp_banister": "trimp_banister_90",
    }
    for source_col, target_col in per90_map.items():
        if source_col in collective_df.columns:
            collective_df[target_col] = np.where(valid_duration, pd.to_numeric(collective_df[source_col], errors="coerce") / duration * 90.0, np.nan)
    return collective_df


def _load_historical_total_rows(selecao: str, contexto: str) -> pd.DataFrame:
    hist_df = read_table("performance_metrics", {"selecao": selecao, "contexto": contexto, "fase": "Total"})
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    if "atleta_id" in hist_df.columns:
        hist_df["atleta_id_norm"] = hist_df["atleta_id"].map(_normalize_athlete_identifier)
    return hist_df


def _load_historical_individual_metric_reference(selecao: str, contexto: str, metric_col: str) -> pd.DataFrame:
    metric_spec = INDIVIDUAL_PROFILE_METRIC_MAP.get(metric_col)
    if metric_spec is None:
        return pd.DataFrame()
    historical_df = _load_historical_total_rows(selecao, contexto)
    if historical_df.empty:
        return pd.DataFrame()
    for col in ["duracao_min", "dist_m", "hsr_dist_m", "sprint_dist_m", "active_time_min", "n_sprints", "n_acc_2_5", "n_dec_3_0", "hr_avg_bpm", "external_load_score", "total_load_score", "player_load", "rhie_bouts", "trimp_banister", "vmax_mps", "peak_1m_m_min"]:
        if col in historical_df.columns:
            historical_df[col] = pd.to_numeric(historical_df[col], errors="coerce")
    _, profile_metric_key, _ = metric_spec
    rows = []
    for athlete_id_norm, athlete_df in historical_df.groupby("atleta_id_norm", dropna=False):
        if not athlete_id_norm:
            continue
        dur = float(athlete_df["duracao_min"].sum()) if "duracao_min" in athlete_df.columns else np.nan
        dist = float(athlete_df["dist_m"].sum()) if "dist_m" in athlete_df.columns else np.nan
        hsr = float(athlete_df["hsr_dist_m"].sum()) if "hsr_dist_m" in athlete_df.columns else np.nan
        sprint = float(athlete_df["sprint_dist_m"].sum()) if "sprint_dist_m" in athlete_df.columns else np.nan
        active_time = float(athlete_df["active_time_min"].sum()) if "active_time_min" in athlete_df.columns else np.nan
        n_sprints = float(athlete_df["n_sprints"].sum()) if "n_sprints" in athlete_df.columns else np.nan
        n_acc = float(athlete_df["n_acc_2_5"].sum()) if "n_acc_2_5" in athlete_df.columns else np.nan
        n_dec = float(athlete_df["n_dec_3_0"].sum()) if "n_dec_3_0" in athlete_df.columns else np.nan
        hr_avg = _safe_mean(athlete_df["hr_avg_bpm"]) if "hr_avg_bpm" in athlete_df.columns else np.nan
        external_load = float(athlete_df["external_load_score"].sum()) if "external_load_score" in athlete_df.columns else np.nan
        total_load = float(athlete_df["total_load_score"].sum()) if "total_load_score" in athlete_df.columns else np.nan
        player_load = float(athlete_df["player_load"].sum()) if "player_load" in athlete_df.columns else np.nan
        rhie_bouts = float(athlete_df["rhie_bouts"].sum()) if "rhie_bouts" in athlete_df.columns else np.nan
        trimp = float(athlete_df["trimp_banister"].sum()) if "trimp_banister" in athlete_df.columns else np.nan
        profile_metrics = {
            "duracao_min_media": _safe_mean(athlete_df["duracao_min"]),
            "m_min": (dist / dur) if pd.notna(dist) and pd.notna(dur) and dur > 0 else np.nan,
            "hsr_pct": (hsr / dist * 100.0) if pd.notna(hsr) and pd.notna(dist) and dist > 0 else np.nan,
            "active_pct": (active_time / dur * 100.0) if pd.notna(active_time) and pd.notna(dur) and dur > 0 else np.nan,
            "dist_m_90": (dist / dur * 90.0) if pd.notna(dist) and pd.notna(dur) and dur > 0 else np.nan,
            "hsr_dist_m_90": (hsr / dur * 90.0) if pd.notna(hsr) and pd.notna(dur) and dur > 0 else np.nan,
            "sprint_dist_m_90": (sprint / dur * 90.0) if pd.notna(sprint) and pd.notna(dur) and dur > 0 else np.nan,
            "n_sprints_90": (n_sprints / dur * 90.0) if pd.notna(n_sprints) and pd.notna(dur) and dur > 0 else np.nan,
            "n_acc_2_5_90": (n_acc / dur * 90.0) if pd.notna(n_acc) and pd.notna(dur) and dur > 0 else np.nan,
            "n_dec_3_0_90": (n_dec / dur * 90.0) if pd.notna(n_dec) and pd.notna(dur) and dur > 0 else np.nan,
            "active_time_min_90": (active_time / dur * 90.0) if pd.notna(active_time) and pd.notna(dur) and dur > 0 else np.nan,
            "hr_avg_bpm": hr_avg,
            "external_load_score_90": (external_load / dur * 90.0) if pd.notna(external_load) and pd.notna(dur) and dur > 0 else np.nan,
            "total_load_score_90": (total_load / dur * 90.0) if pd.notna(total_load) and pd.notna(dur) and dur > 0 else np.nan,
            "player_load_90": (player_load / dur * 90.0) if pd.notna(player_load) and pd.notna(dur) and dur > 0 else np.nan,
            "rhie_bouts_90": (rhie_bouts / dur * 90.0) if pd.notna(rhie_bouts) and pd.notna(dur) and dur > 0 else np.nan,
            "trimp_banister_90": (trimp / dur * 90.0) if pd.notna(trimp) and pd.notna(dur) and dur > 0 else np.nan,
            "vmax_mps_peak": float(athlete_df["vmax_mps"].max()) if "vmax_mps" in athlete_df.columns and athlete_df["vmax_mps"].notna().any() else np.nan,
            "peak_1m_m_min_peak": float(athlete_df["peak_1m_m_min"].max()) if "peak_1m_m_min" in athlete_df.columns and athlete_df["peak_1m_m_min"].notna().any() else np.nan,
        }
        rows.append({"atleta_id_norm": athlete_id_norm, "reference_value": profile_metrics.get(profile_metric_key, np.nan)})
    return pd.DataFrame(rows)


def _load_historical_collective_phase_reference(selecao: str, contexto: str, metric_col: str) -> pd.DataFrame:
    metric_spec = COLLECTIVE_PROFILE_METRIC_MAP.get(metric_col)
    source_col = COLLECTIVE_REFERENCE_COLUMN_MAP.get(metric_col)
    if not metric_spec or not source_col:
        return pd.DataFrame()
    hist_df = read_table("collective_performance_metrics", {"selecao": selecao, "contexto": contexto})
    if hist_df is None or hist_df.empty or source_col not in hist_df.columns:
        return pd.DataFrame()
    hist_df["fase"] = hist_df["fase"].map(_clean_text_value)
    hist_df = hist_df[hist_df["fase"].isin(PHASE_ORDER)].copy()
    _, profile_metric_key, _ = metric_spec
    rows = []
    for fase, fase_df in hist_df.groupby("fase", dropna=False):
        duration = pd.to_numeric(fase_df.get("duracao_min_total"), errors="coerce")
        current_series = pd.to_numeric(fase_df[source_col], errors="coerce")
        valid_duration = duration.notna() & (duration > 0)
        if profile_metric_key == "duracao_min":
            reference_value = _safe_mean(current_series)
        elif profile_metric_key.endswith("_90"):
            reference_value = _safe_mean(np.where(valid_duration, current_series / duration * 90.0, np.nan))
        else:
            reference_value = _safe_mean(current_series)
        rows.append({"fase": fase, "reference_value": reference_value})
    return pd.DataFrame(rows)


def _plot_grouped_bars_on_axis(ax, *, title: str, categories: list[str], current_values: list[float], reference_values: list[float], current_label: str, reference_label: str, metric_col: str, rotate_xticks: bool = False) -> None:
    x = np.arange(len(categories))
    width = 0.38
    current_series = pd.to_numeric(pd.Series(current_values), errors="coerce").fillna(0.0)
    reference_series = pd.to_numeric(pd.Series(reference_values), errors="coerce")
    reference_available = reference_series.notna().any()
    if not reference_available:
        reference_series = pd.Series([0.0] * len(categories))
    else:
        reference_series = reference_series.fillna(0.0)
    bars_current = ax.bar(x - (width / 2 if reference_available else 0), current_series, width=width if reference_available else 0.6, color="#7fb24d", edgecolor="#2f3b1f", linewidth=1.0, label=current_label)
    bars_reference = None
    if reference_available:
        bars_reference = ax.bar(x + width / 2, reference_series, width=width, color="#d9dde5", edgecolor="#6b7280", linewidth=1.0, label=reference_label)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90 if rotate_xticks else 0)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ymax = max(float(current_series.max()) if not current_series.empty else 0.0, float(reference_series.max()) if not reference_series.empty else 0.0)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    for bars, values in [(bars_current, current_series), (bars_reference, reference_series if reference_available else None)]:
        if bars is None or values is None:
            continue
        labels = [_format_metric_value(metric_col, value) for value in values]
        for rect, label in zip(bars, labels):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + (ymax * 0.02 if ymax > 0 else 0.02), label, ha="center", va="bottom", fontsize=8, color="#64748b")
    if reference_available:
        ax.legend(loc="upper right", frameon=False)


def _generate_session_pdfs_bytes(row: pd.Series) -> tuple[bytes, bytes]:
    session_sk = pd.to_numeric(pd.Series([row.get("session_sk")]), errors="coerce").iloc[0]
    if pd.isna(session_sk):
        raise ValueError("Session sem session_sk válido.")
    session_df = _load_session_performance_rows(int(session_sk))
    if session_df.empty:
        raise ValueError("Sem performance_metrics para esta sessão.")
    athlete_name_map = _load_athlete_name_map()
    totals_by_athlete = _build_totals_by_athlete_from_session_df(session_df)
    collective_buffer = io.BytesIO()
    individual_buffer = io.BytesIO()

    with PdfPages(collective_buffer) as pdf:
        for family_name, metric_cols in PERFORMANCE_GROUPS.items():
            collective_df = _build_collective_phase_totals_from_session_df(session_df, metric_cols)
            if collective_df.empty:
                continue
            for metric_col in metric_cols:
                metric_spec = COLLECTIVE_PROFILE_METRIC_MAP.get(metric_col)
                if metric_spec is None:
                    continue
                current_metric_col, _, display_label = metric_spec
                if current_metric_col not in collective_df.columns:
                    continue
                current_df = collective_df[["fase", current_metric_col]].copy()
                current_df = current_df[current_df["fase"].isin(PHASE_ORDER)].copy()
                if current_df.empty:
                    continue
                reference_df = _load_historical_collective_phase_reference(_clean_text_value(row.get("selecao")), _clean_text_value(row.get("contexto")), metric_col)
                current_map = dict(zip(current_df["fase"], pd.to_numeric(current_df[current_metric_col], errors="coerce")))
                reference_map = dict(zip(reference_df.get("fase", []), pd.to_numeric(reference_df.get("reference_value", []), errors="coerce"))) if reference_df is not None and not reference_df.empty else {}
                categories = [phase for phase in PHASE_ORDER if phase in current_map]
                fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
                fig.suptitle(family_name, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.98)
                _plot_grouped_bars_on_axis(ax, title=display_label, categories=categories, current_values=[current_map.get(cat, np.nan) for cat in categories], reference_values=[reference_map.get(cat, np.nan) for cat in categories], current_label="Sessão Atual", reference_label="Média Equipa", metric_col=metric_col)
                fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    with PdfPages(individual_buffer) as pdf:
        for family_name, metric_cols in PERFORMANCE_GROUPS.items():
            for metric_col in metric_cols:
                metric_spec = INDIVIDUAL_PROFILE_METRIC_MAP.get(metric_col)
                if metric_spec is None:
                    continue
                current_metric_col, _, display_label = metric_spec
                if current_metric_col not in totals_by_athlete.columns or "atleta_id" not in totals_by_athlete.columns:
                    continue
                chart_df = totals_by_athlete[["atleta_id", current_metric_col]].copy()
                chart_df["atleta_id"] = chart_df["atleta_id"].map(_normalize_athlete_identifier)
                chart_df[current_metric_col] = pd.to_numeric(chart_df[current_metric_col], errors="coerce")
                chart_df = chart_df[chart_df["atleta_id"].ne("") & chart_df[current_metric_col].notna()].copy()
                if chart_df.empty:
                    continue
                ref_df = _load_historical_individual_metric_reference(_clean_text_value(row.get("selecao")), _clean_text_value(row.get("contexto")), metric_col)
                ref_map = dict(zip(ref_df.get("atleta_id_norm", []), pd.to_numeric(ref_df.get("reference_value", []), errors="coerce"))) if ref_df is not None and not ref_df.empty else {}
                chart_df["athlete_label"] = chart_df["atleta_id"].map(lambda athlete_id: athlete_name_map.get(athlete_id, athlete_id))
                chart_df = chart_df.sort_values(by=current_metric_col, ascending=False).reset_index(drop=True)
                fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
                fig.suptitle(family_name, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.98)
                _plot_grouped_bars_on_axis(ax, title=display_label, categories=chart_df["athlete_label"].tolist(), current_values=chart_df[current_metric_col].tolist(), reference_values=[ref_map.get(aid, np.nan) for aid in chart_df["atleta_id"].tolist()], current_label="Sessão Atual", reference_label="Média Individual", metric_col=metric_col, rotate_xticks=True)
                fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return collective_buffer.getvalue(), individual_buffer.getvalue()


def _render_delete_controls(row: pd.Series, selection_code: str, *, trigger_label: str = "Eliminar Jogo | Treino", use_container_width: bool = True) -> None:
    session_fingerprint = _clean_text_value(row.get("session_fingerprint"))
    session_sk = pd.to_numeric(pd.Series([row.get("session_sk")]), errors="coerce").iloc[0]
    contexto_txt = _clean_text_value(row.get("contexto")) or "Sessao"
    jogo_txt = _clean_text_value(row.get("jogo")) or "-"
    delete_state_key = f"show_delete_controls_{session_fingerprint}"
    confirm_key_1 = f"confirm_delete_1_{session_fingerprint}"
    confirm_key_2 = f"confirm_delete_2_{session_fingerprint}"
    button_key = f"delete_session_{session_fingerprint}"

    if st.button(trigger_label, key=f"toggle_{button_key}", use_container_width=use_container_width):
        st.session_state[delete_state_key] = not st.session_state.get(delete_state_key, False)
        st.rerun()

    if not st.session_state.get(delete_state_key, False):
        return

    st.warning(
        f"Vais eliminar definitivamente esta sessao: {selection_code} | {contexto_txt} | {jogo_txt}. "
        "Todos os dados associados serao removidos."
    )
    confirm_1 = st.checkbox("Confirmo que quero eliminar esta sessao.", key=confirm_key_1)
    confirm_2 = st.checkbox("Confirmo que entendo que esta acao e irreversivel.", key=confirm_key_2)

    if st.button("Confirmar Eliminacao Definitiva", key=button_key, type="primary", use_container_width=True):
        if not confirm_1 or not confirm_2:
            st.error("Tens de validar as duas confirmacoes antes de eliminar.")
            return
        if pd.isna(session_sk):
            st.error("Nao foi possivel identificar o session_sk desta sessao.")
            return

        cleanup_stats = cleanup_session_upload(int(session_sk), delete_session_row=True)
        _clear_session_pdf_cache(session_fingerprint)
        if cleanup_stats.get("success", False):
            st.cache_data.clear()
            st.success("Sessao eliminada com sucesso.")
            for key in [delete_state_key, confirm_key_1, confirm_key_2]:
                st.session_state.pop(key, None)
            st.rerun()
        else:
            st.error(f"Eliminacao com falhas: {cleanup_stats.get('errors')}")


def _render_selection_section(selection_code: str, selection_df: pd.DataFrame) -> None:
    expander_label = f"{selection_code} ({len(selection_df)})"
    with st.expander(expander_label, expanded=False):
        header_cols = st.columns([1.0, 0.9, 1.4, 1.1, 1.1, 1.0], gap="small")
        header_cols[0].markdown("**Data**")
        header_cols[1].markdown("**Tipo**")
        header_cols[2].markdown("**Jogo / Treino**")
        header_cols[3].markdown("**Rel. Coletivo**")
        header_cols[4].markdown("**Rel. Individual**")
        header_cols[5].markdown("**Eliminar**")
        st.divider()

        for idx, (_, row) in enumerate(selection_df.iterrows()):
            session_fingerprint = _clean_text_value(row.get("session_fingerprint"))
            date_txt = _format_date(row.get("data"))
            contexto_txt = _clean_text_value(row.get("contexto")) or "Sessao"
            jogo_txt = _clean_text_value(row.get("jogo"))
            detail_txt = jogo_txt if contexto_txt == "Jogo" and jogo_txt else ("Treino" if contexto_txt == "Treino" else (jogo_txt or "-"))
            collective_name, individual_name = _session_pdf_names(
                session_fingerprint,
                _clean_text_value(row.get("selecao")),
                contexto_txt,
                jogo_txt,
            )
            cached_pdfs = _session_pdf_cache().get(session_fingerprint, {})
            collective_bytes = cached_pdfs.get("collective")
            individual_bytes = cached_pdfs.get("individual")

            row_cols = st.columns([1.0, 0.9, 1.4, 1.1, 1.1, 1.0], gap="small")
            row_cols[0].write(date_txt)
            row_cols[1].write(contexto_txt)
            row_cols[2].write(detail_txt)

            with row_cols[3]:
                if collective_bytes:
                    st.download_button(
                        "Coletivo",
                        data=collective_bytes,
                        file_name=collective_name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"collective_pdf_{selection_code}_{session_fingerprint}",
                    )
                else:
                    if st.button(
                        "Gerar",
                        use_container_width=True,
                        key=f"regen_collective_pdf_{selection_code}_{session_fingerprint}",
                    ):
                        try:
                            collective_bytes, individual_bytes = _generate_session_pdfs_bytes(row)
                            _session_pdf_cache()[session_fingerprint] = {
                                "collective": collective_bytes,
                                "individual": individual_bytes,
                            }
                        except Exception as exc:
                            st.error(f"Nao foi possivel regenerar os PDFs: {exc}")
                        else:
                            st.success("PDFs regenerados com sucesso.")
                            st.rerun()

            with row_cols[4]:
                if individual_bytes:
                    st.download_button(
                        "Individual",
                        data=individual_bytes,
                        file_name=individual_name,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"individual_pdf_{selection_code}_{session_fingerprint}",
                    )
                else:
                    if st.button(
                        "Gerar",
                        use_container_width=True,
                        key=f"regen_individual_pdf_{selection_code}_{session_fingerprint}",
                    ):
                        try:
                            collective_bytes, individual_bytes = _generate_session_pdfs_bytes(row)
                            _session_pdf_cache()[session_fingerprint] = {
                                "collective": collective_bytes,
                                "individual": individual_bytes,
                            }
                        except Exception as exc:
                            st.error(f"Nao foi possivel regenerar os PDFs: {exc}")
                        else:
                            st.success("PDFs regenerados com sucesso.")
                            st.rerun()

            with row_cols[5]:
                _render_delete_controls(row, selection_code, trigger_label="Eliminar", use_container_width=True)

            if idx < len(selection_df) - 1:
                st.divider()


st.title("Jogos | Treinos")
st.caption("Consulta os jogos e treinos publicados por selecao e acede aos relatorios individual e coletivo de cada sessao.")
st.markdown(
    """
    <style>
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) {
        margin-bottom: 0.15rem;
    }
    div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] > button,
    div[data-testid="stExpander"] div[data-testid="stDownloadButton"] > button {
        min-height: 2.25rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    div[data-testid="stExpander"] hr {
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

reports_df = _load_session_reports()
selection_options = ["Todas"] + load_selection_reference(active_only=False)
context_options = ["Todos", "Jogo", "Treino"]

filter_col1, filter_col2 = st.columns(2)
selected_selection = filter_col1.selectbox("Selecao", options=selection_options)
selected_context = filter_col2.selectbox("Contexto", options=context_options)

view_df = reports_df.copy()
if not view_df.empty and selected_selection != "Todas":
    view_df = view_df[view_df["selecao"].eq(selected_selection)].copy()
if not view_df.empty and selected_context != "Todos":
    view_df = view_df[view_df["contexto"].eq(selected_context)].copy()

if view_df.empty:
    st.info("Sem registos disponiveis. Depois de publicares uma sessao, os relatorios ficarao disponiveis aqui.")
else:
    for selection_code in view_df["selecao"].drop_duplicates().tolist():
        selection_df = view_df[view_df["selecao"].eq(selection_code)].copy()
        _render_selection_section(selection_code, selection_df)
