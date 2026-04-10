from __future__ import annotations

import streamlit as st

DRAFT_STATE_DEFAULTS = {
    "df_metrics": None,
    "report_txt": None,
    "manual_metricas_txt": None,
    "process_done": False,
    "df_perf": None,
    "df_collective_perf": None,
    "df_qc": None,
    "df_samples": None,
    "df_athlete_session": None,
    "df_time_audit": None,
    "draft_session_payload": None,
    "draft_context": None,
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
    df_collective_perf,
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
    st.session_state.df_collective_perf = df_collective_perf
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
    df_collective_perf,
    df_qc,
    df_samples,
    df_athlete_session,
    df_tracking,
) -> dict[str, str]:
    """Keep draft results in memory only.

    The online workflow should not depend on local parquet persistence.
    """
    return {}


def clear_draft_session_state() -> None:
    """Remove in-memory draft outputs from the current Streamlit session."""
    for key, default in DRAFT_STATE_DEFAULTS.items():
        st.session_state[key] = default
