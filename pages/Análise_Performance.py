import numpy as np
import pandas as pd
import streamlit as st

from fpf_modules.supabase_manager import initialize_schema, read_table


st.set_page_config(page_title="Análises Individuais", layout="wide")


PROFILE_OPTIONS = [
    "Jogo | Global",
    "Jogo | Últimos 5",
    "Jogo | Últimos 10",
    "Treino | Global",
    "Treino | Últimos 5",
    "Treino | Últimos 10",
]

COMPARISON_MODES = [
    "Perfil vs Perfil",
    "Jogo vs Perfil",
    "Jogo vs Jogo",
]

DISPLAY_METRICS = [
    ("dist_m_90", "Distância / 90"),
    ("m_min", "m/min"),
    ("hsr_pct", "HSR %"),
    ("active_pct", "Ativo %"),
    ("active_time_min_90", "Tempo Ativo / 90"),
    ("hsr_dist_m_90", "Distância HSR / 90"),
    ("sprint_dist_m_90", "Distância Sprint / 90"),
    ("n_sprints_90", "Nº Sprints / 90"),
    ("n_acc_2_5_90", "Acc / 90"),
    ("n_dec_3_0_90", "Dec / 90"),
    ("vmax_mps_peak", "Vmax Pico"),
    ("peak_1m_m_min_peak", "Pico 1m"),
]

SUM_METRICS = [
    "duracao_min",
    "dist_m",
    "hsr_dist_m",
    "sprint_dist_m",
    "n_sprints",
    "n_acc_2_5",
    "n_dec_3_0",
    "active_time_min",
]

PHOTO_WIDTH = 120
PHOTO_HEIGHT = 120
TOP_BLOCK_HEIGHT = 160
LABEL_BLOCK_HEIGHT = 34
HEADER_GAP_HEIGHT = 20
DELTA_TOP_HEIGHT = 93


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


@st.cache_data(show_spinner=False, ttl=300)
def _load_athletes() -> pd.DataFrame:
    initialize_schema()
    df = read_table("athletes")
    if df is None or df.empty:
        return pd.DataFrame(columns=["atleta_id", "nome", "foto_url", "posicao", "genero", "ativo"])

    for col in ["atleta_id", "nome", "foto_url", "posicao", "genero"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(_clean_text)

    if "ativo" not in df.columns:
        df["ativo"] = True
    df["ativo"] = df["ativo"].fillna(True).astype(bool)

    return (
        df[df["atleta_id"].ne("") & df["ativo"]]
        .drop_duplicates(subset=["atleta_id"], keep="last")
        .sort_values(["nome", "atleta_id"], na_position="last")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False, ttl=300)
def _load_perf_sessions() -> pd.DataFrame:
    initialize_schema()
    df = read_table("vw_perf_total_session")
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "duracao_min",
        "dist_m",
        "m_min",
        "vmax_mps",
        "peak_1m_m_min",
        "hsr_dist_m",
        "hsr_pct",
        "sprint_dist_m",
        "n_sprints",
        "n_acc_2_5",
        "n_dec_3_0",
        "active_time_min",
        "active_pct",
    ]
    for col in ["atleta_id", "contexto", "jogo", "selecao", "genero"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(_clean_text)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _parse_profile_option(profile_option: str) -> tuple[str, int | None]:
    contexto_txt, janela_txt = [part.strip() for part in profile_option.split("|", 1)]
    if "Últimos 5" in janela_txt:
        return contexto_txt, 5
    if "Últimos 10" in janela_txt:
        return contexto_txt, 10
    return contexto_txt, None


def _aggregate_profile(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}

    agg = {}
    for metric in SUM_METRICS:
        agg[f"{metric}_total"] = float(df[metric].sum()) if metric in df.columns else np.nan

    dur = agg.get("duracao_min_total", np.nan)
    dist = agg.get("dist_m_total", np.nan)
    hsr = agg.get("hsr_dist_m_total", np.nan)
    sprint = agg.get("sprint_dist_m_total", np.nan)
    active_time = agg.get("active_time_min_total", np.nan)
    n_sprints = agg.get("n_sprints_total", np.nan)
    n_acc = agg.get("n_acc_2_5_total", np.nan)
    n_dec = agg.get("n_dec_3_0_total", np.nan)

    agg["n_sessoes"] = int(df["session_sk"].nunique()) if "session_sk" in df.columns else int(len(df))
    agg["primeira_data"] = df["data"].min() if "data" in df.columns else pd.NaT
    agg["ultima_data"] = df["data"].max() if "data" in df.columns else pd.NaT
    agg["m_min"] = (dist / dur) if pd.notna(dur) and dur > 0 else np.nan
    agg["hsr_pct"] = (hsr / dist * 100.0) if pd.notna(dist) and dist > 0 else np.nan
    agg["active_pct"] = (active_time / dur * 100.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["dist_m_90"] = (dist / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["hsr_dist_m_90"] = (hsr / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["sprint_dist_m_90"] = (sprint / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["n_sprints_90"] = (n_sprints / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["n_acc_2_5_90"] = (n_acc / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["n_dec_3_0_90"] = (n_dec / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["active_time_min_90"] = (active_time / dur * 90.0) if pd.notna(dur) and dur > 0 else np.nan
    agg["vmax_mps_peak"] = float(df["vmax_mps"].max()) if "vmax_mps" in df.columns else np.nan
    agg["peak_1m_m_min_peak"] = float(df["peak_1m_m_min"].max()) if "peak_1m_m_min" in df.columns else np.nan
    return agg


def _build_profile_snapshot(
    athletes_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    athlete_id: str,
    profile_option: str,
) -> dict:
    athlete_row = athletes_df[athletes_df["atleta_id"].eq(athlete_id)].head(1)
    if athlete_row.empty:
        return {}

    athlete_info = athlete_row.iloc[0].to_dict()
    contexto, window_size = _parse_profile_option(profile_option)
    df_athlete = perf_df[
        perf_df["atleta_id"].eq(athlete_id) &
        perf_df["contexto"].eq(contexto)
    ].copy()

    if not df_athlete.empty and window_size is not None and "data" in df_athlete.columns:
        df_athlete = df_athlete.sort_values(["data", "session_sk"], ascending=[False, False]).head(window_size)

    return {
        "kind": "profile",
        "athlete": athlete_info,
        "label": profile_option,
        "sessions_df": df_athlete,
        "metrics": _aggregate_profile(df_athlete),
    }


def _build_session_option_map(perf_df: pd.DataFrame, athlete_id: str, contexto: str = "Jogo") -> tuple[list[int], dict[int, str]]:
    df_athlete = perf_df[
        perf_df["atleta_id"].eq(athlete_id) &
        perf_df["contexto"].eq(contexto)
    ].copy()
    if df_athlete.empty:
        return [], {}

    df_athlete = df_athlete.sort_values(["data", "session_sk"], ascending=[False, False])
    options = []
    labels = {}
    for _, row in df_athlete.iterrows():
        session_sk = int(row["session_sk"])
        if session_sk in labels:
            continue
        data_txt = pd.to_datetime(row["data"]).strftime("%d/%m/%Y") if pd.notna(row.get("data")) else "-"
        jogo_txt = _clean_text(row.get("jogo")) or f"Sessão {session_sk}"
        labels[session_sk] = f"{data_txt} | {jogo_txt}"
        options.append(session_sk)
    return options, labels


def _build_session_snapshot(
    athletes_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    athlete_id: str,
    session_sk: int,
) -> dict:
    athlete_row = athletes_df[athletes_df["atleta_id"].eq(athlete_id)].head(1)
    if athlete_row.empty:
        return {}

    athlete_info = athlete_row.iloc[0].to_dict()
    df_session = perf_df[
        perf_df["atleta_id"].eq(athlete_id) &
        perf_df["session_sk"].eq(session_sk)
    ].copy()
    if df_session.empty:
        return {}

    contexto = _clean_text(df_session["contexto"].iloc[0]) if "contexto" in df_session.columns else "Jogo"
    jogo_txt = _clean_text(df_session["jogo"].iloc[0]) if "jogo" in df_session.columns else f"Sessão {session_sk}"
    data = df_session["data"].iloc[0] if "data" in df_session.columns else pd.NaT
    data_txt = pd.to_datetime(data).strftime("%d/%m/%Y") if pd.notna(data) else "-"

    return {
        "kind": "session",
        "athlete": athlete_info,
        "label": f"{contexto} | {data_txt} | {jogo_txt}",
        "sessions_df": df_session,
        "metrics": _aggregate_profile(df_session),
    }


def _format_number(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}{suffix}"


def _format_profile_value(metric_key: str, value) -> str:
    if metric_key.endswith("_pct"):
        return _format_number(value, "%")
    if metric_key in {"n_sprints_90", "n_acc_2_5_90", "n_dec_3_0_90"}:
        return _format_number(value)
    return _format_number(value)


def _build_metric_table(snapshot: dict) -> pd.DataFrame:
    metrics = snapshot.get("metrics", {}) if snapshot else {}
    rows = []
    for metric_key, metric_label in DISPLAY_METRICS:
        rows.append(
            {
                "Métrica": metric_label,
                "Valor": _format_profile_value(metric_key, metrics.get(metric_key, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _build_delta_table(left_snapshot: dict, right_snapshot: dict) -> pd.DataFrame:
    rows = []
    left_metrics = left_snapshot.get("metrics", {}) if left_snapshot else {}
    right_metrics = right_snapshot.get("metrics", {}) if right_snapshot else {}

    for metric_key, _metric_label in DISPLAY_METRICS:
        left_value = left_metrics.get(metric_key, np.nan)
        right_value = right_metrics.get(metric_key, np.nan)
        delta_abs = (
            float(left_value) - float(right_value)
            if pd.notna(left_value) and pd.notna(right_value)
            else np.nan
        )
        delta_pct = (
            (delta_abs / float(right_value) * 100.0)
            if pd.notna(delta_abs) and pd.notna(right_value) and float(right_value) != 0
            else np.nan
        )
        rows.append(
            {
                "Delta": _format_profile_value(metric_key, delta_abs),
                "Delta %": _format_number(delta_pct, "%"),
            }
        )

    return pd.DataFrame(rows)


def _render_static_table(df: pd.DataFrame) -> None:
    row_height = 35
    header_height = 38
    table_height = header_height + max(len(df), 1) * row_height + 2
    st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True, height=table_height)


def _style_delta_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace("%", "")
    if text in {"", "-"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    if number > 0:
        return "color: #22c55e; font-weight: 600;"
    if number < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #e5e7eb;"


def _render_delta_table(df: pd.DataFrame) -> None:
    row_height = 35
    header_height = 38
    table_height = header_height + max(len(df), 1) * row_height + 2
    styled = (
        df.reset_index(drop=True)
        .style
        .applymap(_style_delta_value, subset=["Delta", "Delta %"])
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, height=table_height)


def _render_header_block(snapshot: dict | None) -> None:
    athlete = snapshot.get("athlete", {}) if snapshot else {}

    top_left, top_right = st.columns([0.9, 1.6], gap="large")
    with top_left:
        foto_url = _clean_text(athlete.get("foto_url"))
        if foto_url:
            st.image(foto_url, width=PHOTO_WIDTH)
        else:
            st.markdown(f"<div style='height: {PHOTO_HEIGHT}px;'></div>", unsafe_allow_html=True)

    with top_right:
        if snapshot:
            st.markdown(f"**Atleta:** {_clean_text(athlete.get('nome')) or '-'}")
            st.markdown(f"**Posição:** {_clean_text(athlete.get('posicao')) or '-'}")
        else:
            st.markdown(f"<div style='height: {PHOTO_HEIGHT}px;'></div>", unsafe_allow_html=True)


def _render_label_block(snapshot: dict) -> None:
    label = snapshot.get("label", "") if snapshot else ""
    if snapshot and snapshot.get("kind") == "session":
        st.caption(label)
    else:
        st.markdown(f"<div style='height: {LABEL_BLOCK_HEIGHT}px;'></div>", unsafe_allow_html=True)


def _render_profile_panel(title: str, snapshot: dict) -> None:
    st.markdown(f"### {title}")
    if not snapshot:
        st.info("Seleciona um item para comparar.")
        return

    _render_header_block(snapshot)
    _render_label_block(snapshot)
    st.markdown(f"<div style='height: {HEADER_GAP_HEIGHT}px;'></div>", unsafe_allow_html=True)
    _render_static_table(_build_metric_table(snapshot))


def _render_delta_panel(left_snapshot: dict, right_snapshot: dict) -> None:
    st.markdown("### Deltas")
    st.markdown(f"<div style='height: {DELTA_TOP_HEIGHT + LABEL_BLOCK_HEIGHT}px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='height: {HEADER_GAP_HEIGHT}px;'></div>", unsafe_allow_html=True)
    _render_delta_table(_build_delta_table(left_snapshot, right_snapshot))


def _resolve_modes(comparison_mode: str) -> tuple[str, str]:
    if comparison_mode == "Jogo vs Perfil":
        return "Jogo", "Perfil"
    if comparison_mode == "Jogo vs Jogo":
        return "Jogo", "Jogo"
    return "Perfil", "Perfil"


st.title("Análises Individuais")
st.caption("Comparação lado a lado entre atletas, perfis e jogos.")

athletes_df = _load_athletes()
perf_df = _load_perf_sessions()

if athletes_df.empty:
    st.warning("Sem atletas ativos na base de dados.")
    st.stop()

if perf_df.empty:
    st.warning("Sem dados de performance disponíveis para construir perfis.")
    st.stop()

athlete_display_map = {
    row["atleta_id"]: f"{_clean_text(row['nome'])} ({_clean_text(row['atleta_id'])})"
    for _, row in athletes_df.iterrows()
}
athlete_options = list(athlete_display_map.keys())

comparison_mode = st.selectbox("Modo de comparação", options=COMPARISON_MODES)
left_mode, right_mode = _resolve_modes(comparison_mode)

selector_left, selector_right = st.columns(2, gap="large")

with selector_left:
    athlete_left = st.selectbox(
        "Atleta",
        options=athlete_options,
        format_func=lambda aid: athlete_display_map.get(aid, aid),
        key="athlete_compare_left",
    )
    if left_mode == "Perfil":
        profile_left = st.selectbox("Perfil", options=PROFILE_OPTIONS, key="profile_compare_left")
        left_snapshot = _build_profile_snapshot(athletes_df, perf_df, athlete_left, profile_left)
    else:
        left_game_options, left_game_map = _build_session_option_map(perf_df, athlete_left, contexto="Jogo")
        if left_game_options:
            left_game = st.selectbox(
                "Jogo",
                options=left_game_options,
                format_func=lambda sid: left_game_map.get(sid, str(sid)),
                key="session_compare_left",
            )
            left_snapshot = _build_session_snapshot(athletes_df, perf_df, athlete_left, left_game)
        else:
            st.info("Sem jogos disponíveis para este atleta.")
            left_snapshot = {}

with selector_right:
    default_right_index = 1 if len(athlete_options) > 1 else 0
    athlete_right = st.selectbox(
        "Atleta",
        options=athlete_options,
        index=default_right_index,
        format_func=lambda aid: athlete_display_map.get(aid, aid),
        key="athlete_compare_right",
    )
    if right_mode == "Perfil":
        profile_right = st.selectbox("Perfil", options=PROFILE_OPTIONS, key="profile_compare_right")
        right_snapshot = _build_profile_snapshot(athletes_df, perf_df, athlete_right, profile_right)
    else:
        right_game_options, right_game_map = _build_session_option_map(perf_df, athlete_right, contexto="Jogo")
        if right_game_options:
            right_game = st.selectbox(
                "Jogo",
                options=right_game_options,
                format_func=lambda sid: right_game_map.get(sid, str(sid)),
                key="session_compare_right",
            )
            right_snapshot = _build_session_snapshot(athletes_df, perf_df, athlete_right, right_game)
        else:
            st.info("Sem jogos disponíveis para este atleta.")
            right_snapshot = {}

panel_left, panel_delta, panel_right = st.columns([1.2, 1.0, 1.2], gap="large")
with panel_left:
    _render_profile_panel(f"{left_mode} Esquerdo", left_snapshot)
with panel_delta:
    _render_delta_panel(left_snapshot, right_snapshot)
with panel_right:
    _render_profile_panel(f"{right_mode} Direito", right_snapshot)

st.markdown("## Sessões Utilizadas")
sessions_left, sessions_right = st.columns(2, gap="large")

with sessions_left:
    st.caption(f"Sessões usadas no lado esquerdo ({left_mode.lower()})")
    left_sessions = left_snapshot.get("sessions_df", pd.DataFrame()) if left_snapshot else pd.DataFrame()
    if left_sessions.empty:
        st.info("Sem sessões para este lado.")
    else:
        cols = [c for c in ["data", "contexto", "jogo", "duracao_min", "dist_m", "m_min", "vmax_mps"] if c in left_sessions.columns]
        st.dataframe(left_sessions[cols].sort_values("data", ascending=False), use_container_width=True, hide_index=True)

with sessions_right:
    st.caption(f"Sessões usadas no lado direito ({right_mode.lower()})")
    right_sessions = right_snapshot.get("sessions_df", pd.DataFrame()) if right_snapshot else pd.DataFrame()
    if right_sessions.empty:
        st.info("Sem sessões para este lado.")
    else:
        cols = [c for c in ["data", "contexto", "jogo", "duracao_min", "dist_m", "m_min", "vmax_mps"] if c in right_sessions.columns]
        st.dataframe(right_sessions[cols].sort_values("data", ascending=False), use_container_width=True, hide_index=True)
