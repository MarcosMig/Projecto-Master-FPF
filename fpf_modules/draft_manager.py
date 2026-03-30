from __future__ import annotations

from pathlib import Path

import streamlit as st

from .constants import CLEANDATA_DIR
from .supabase_manager import append_dedup_parquet


DRAFT_STATE_DEFAULTS = {
    "df_metrics": None,
    "report_txt": None,
    "manual_metricas_txt": None,
    "process_done": False,
    "df_perf": None,
    "df_qc": None,
    "df_samples": None,
    "df_athlete_session": None,
    "df_time_audit": None,
    "draft_session_payload": None,
    "draft_context": None,
}

DRAFT_FILE_SPECS = {
    "performance_metrics": ("performance_metrics.parquet", ["session_fingerprint", "atleta_id", "phase_id"]),
    "quality_metrics": ("quality_metrics.parquet", ["session_fingerprint", "atleta_id", "phase_id"]),
    "samples": ("samples.parquet", ["session_fingerprint", "atleta_id", "phase_id", "time"]),
    "athlete_session": ("athlete_session.parquet", ["session_fingerprint", "atleta_id"]),
    "tracking": ("tracking.parquet", ["session_fingerprint", "atleta_id", "phase_id", "time"]),
}


def ensure_draft_session_state() -> None:
    """Initialize Streamlit session state used by the draft processing flow."""
    for key, default in DRAFT_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def store_draft_results(
    *,
    df_metrics,
    report_txt: str,
    manual_metricas_txt: str | None,
    df_perf,
    df_qc,
    df_samples,
    df_athlete_session,
    df_time_audit,
    draft_session_payload,
    draft_context,
) -> None:
    """Persist the latest draft outputs in Streamlit session state."""
    st.session_state.df_metrics = df_metrics
    st.session_state.report_txt = report_txt
    st.session_state.manual_metricas_txt = manual_metricas_txt
    st.session_state.df_perf = df_perf
    st.session_state.df_qc = df_qc
    st.session_state.df_samples = df_samples
    st.session_state.df_athlete_session = df_athlete_session
    st.session_state.df_time_audit = df_time_audit
    st.session_state.draft_session_payload = draft_session_payload
    st.session_state.draft_context = draft_context
    st.session_state.process_done = True


def save_draft_outputs(
    *,
    df_perf,
    df_qc,
    df_samples,
    df_athlete_session,
    df_tracking,
    base_dir: str = CLEANDATA_DIR,
) -> dict[str, str]:
    """Write draft analytics outputs to local parquet storage."""
    saved_paths = {}
    for name, df in {
        "performance_metrics": df_perf,
        "quality_metrics": df_qc,
        "samples": df_samples,
        "athlete_session": df_athlete_session,
        "tracking": df_tracking,
    }.items():
        if df is None or df.empty:
            continue
        filename, subset_keys = DRAFT_FILE_SPECS[name]
        path = str(Path(base_dir) / filename)
        append_dedup_parquet(df, path, subset_keys)
        saved_paths[name] = path
    return saved_paths


def clear_draft_session_state() -> None:
    """Remove in-memory draft outputs from the current Streamlit session."""
    for key, default in DRAFT_STATE_DEFAULTS.items():
        st.session_state[key] = default
