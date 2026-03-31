import html

import numpy as np
import pandas as pd
import streamlit as st

from fpf_modules.draft_manager import ensure_draft_session_state
from fpf_modules.supabase_manager import initialize_schema, read_table


st.set_page_config(page_title="Analise de Perfis", layout="wide")


PROFILE_OPTIONS = [
    "Jogo | Global",
    "Jogo | Últimos 5",
    "Jogo | Últimos 10",
    "Treino | Global",
    "Treino | Últimos 5",
    "Treino | Últimos 10",
]

COMPARISON_MODES = [
    "Jogo vs Jogo",
]

DISPLAY_METRICS = [
    ("duracao_min_media", "Minutos Medios"),
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

PROFILE_METRIC_GROUPS = {
    "Volume": [
        ("duracao_min_media", "Minutos Medios"),
        ("active_time_min_90", "Tempo Ativo / 90"),
        ("dist_m_90", "Distância / 90"),
        ("hsr_dist_m_90", "Distância HSR / 90"),
        ("sprint_dist_m_90", "Distância Sprint / 90"),
    ],
    "Intensidade": [
        ("m_min", "m/min"),
        ("hsr_pct", "HSR %"),
        ("active_pct", "Ativo %"),
    ],
    "Eventos": [
        ("n_sprints_90", "Nº Sprints / 90"),
        ("n_acc_2_5_90", "Acc / 90"),
        ("n_dec_3_0_90", "Dec / 90"),
    ],
    "Picos de Fase": [
        ("vmax_mps_peak", "Vmax Pico"),
        ("peak_1m_m_min_peak", "Pico 1m"),
    ],
}

GAME_DISPLAY_METRICS = [
    ("duracao_min_total", "Minutos"),
    ("dist_m_total", "Distância"),
    ("m_min", "m/min"),
    ("hsr_pct", "HSR %"),
    ("active_pct", "Ativo %"),
    ("active_time_min_total", "Tempo Ativo"),
    ("hsr_dist_m_total", "Distância HSR"),
    ("sprint_dist_m_total", "Distância Sprint"),
    ("n_sprints_total", "Nº Sprints"),
    ("n_acc_2_5_total", "Acc"),
    ("n_dec_3_0_total", "Dec"),
    ("vmax_mps_peak", "Vmax Pico"),
    ("peak_1m_m_min_peak", "Pico 1m"),
]

GAME_METRIC_GROUPS = {
    "Volume": [
        ("duracao_min_total", "Minutos"),
        ("active_time_min_total", "Tempo Ativo"),
        ("dist_m_total", "Distância"),
        ("hsr_dist_m_total", "Distância HSR"),
        ("sprint_dist_m_total", "Distância Sprint"),
    ],
    "Intensidade": [
        ("m_min", "m/min"),
        ("hsr_pct", "HSR %"),
        ("active_pct", "Ativo %"),
    ],
    "Eventos": [
        ("n_sprints_total", "Nº Sprints"),
        ("n_acc_2_5_total", "Acc"),
        ("n_dec_3_0_total", "Dec"),
    ],
    "Picos de Fase": [
        ("vmax_mps_peak", "Vmax Pico"),
        ("peak_1m_m_min_peak", "Pico 1m"),
    ],
}

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
LABEL_BLOCK_HEIGHT = 34
HEADER_GAP_HEIGHT = 20


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _load_draft_perf_sessions() -> pd.DataFrame:
    ensure_draft_session_state()
    df = st.session_state.get("df_perf")
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

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
    for col in ["atleta_id", "contexto", "jogo", "selecao", "genero", "fase"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(_clean_text)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if "session_sk" not in df.columns:
        if "session_id_hex" in df.columns:
            df["session_sk"] = df["session_id_hex"].map(_clean_text)
        elif "session_fingerprint" in df.columns:
            df["session_sk"] = df["session_fingerprint"].map(_clean_text)
        else:
            df["session_sk"] = "draft-session"
    else:
        df["session_sk"] = df["session_sk"].map(_clean_text)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _build_draft_athletes(perf_df: pd.DataFrame) -> pd.DataFrame:
    if perf_df is None or perf_df.empty or "atleta_id" not in perf_df.columns:
        return pd.DataFrame(columns=["atleta_id", "nome", "foto_url", "posicao", "genero", "ativo"])

    athletes_df = pd.DataFrame({
        "atleta_id": perf_df["atleta_id"].map(_clean_text),
    })
    athletes_df = athletes_df[athletes_df["atleta_id"].ne("")].drop_duplicates(subset=["atleta_id"], keep="last")
    athletes_df["nome"] = athletes_df["atleta_id"]
    athletes_df["foto_url"] = ""
    athletes_df["posicao"] = ""
    athletes_df["genero"] = ""
    athletes_df["ativo"] = True
    return athletes_df.sort_values(["nome", "atleta_id"], na_position="last").reset_index(drop=True)


def _build_session_event_label(row: pd.Series) -> str:
    data_txt = pd.to_datetime(row.get("data")).strftime("%d/%m/%Y") if pd.notna(row.get("data")) else "-"
    contexto = _clean_text(row.get("contexto"))
    jogo = _clean_text(row.get("jogo"))
    if contexto == "Jogo" and jogo:
        event_txt = f"Jogo | {jogo}"
    elif contexto:
        event_txt = contexto
    else:
        event_txt = "Sessao"
    return f"{data_txt} | {event_txt}"


def _build_draft_session_option_map(perf_df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    if perf_df is None or perf_df.empty:
        return [], {}

    session_rows = (
        perf_df.sort_values(["data", "session_sk"], ascending=[False, False])
        .drop_duplicates(subset=["session_sk"], keep="first")
    )
    options = []
    labels = {}
    for _, row in session_rows.iterrows():
        session_sk = _clean_text(row.get("session_sk"))
        if not session_sk:
            continue
        labels[session_sk] = _build_session_event_label(row)
        options.append(session_sk)
    return options, labels


def _build_athlete_options_for_session(perf_df: pd.DataFrame, session_sk: str) -> list[str]:
    if perf_df is None or perf_df.empty:
        return []
    session_df = perf_df[perf_df["session_sk"].eq(session_sk)].copy()
    if session_df.empty:
        return []
    return sorted(session_df["atleta_id"].dropna().astype(str).map(_clean_text).loc[lambda s: s.ne("")].unique().tolist())


def _build_phase_options(perf_df: pd.DataFrame, session_sk: str, athlete_id: str) -> list[str]:
    if perf_df is None or perf_df.empty:
        return []
    df_selected = perf_df[
        perf_df["session_sk"].eq(session_sk) &
        perf_df["atleta_id"].eq(athlete_id)
    ].copy()
    if df_selected.empty:
        return []
    preferred_order = ["Warm-Up", "1P", "2P", "Total"]
    available = df_selected["fase"].dropna().astype(str).map(_clean_text)
    return [phase for phase in preferred_order if phase in available.tolist()]


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
    agg["duracao_min_media"] = (dur / agg["n_sessoes"]) if pd.notna(dur) and agg["n_sessoes"] > 0 else np.nan
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


def _build_session_option_map(perf_df: pd.DataFrame, athlete_id: str, contexto: str = "Jogo") -> tuple[list[str], dict[str, str]]:
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
        session_sk = _clean_text(row.get("session_sk"))
        if not session_sk:
            continue
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
    session_sk: str,
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
    if metric_key in {
        "duracao_min_total",
        "duracao_min_media",
        "dist_m_total",
        "dist_m_90",
        "active_time_min_total",
        "active_time_min_90",
        "hsr_dist_m_total",
        "hsr_dist_m_90",
        "sprint_dist_m_total",
        "sprint_dist_m_90",
        "peak_1m_m_min_peak",
    }:
        return "-" if value is None or pd.isna(value) else f"{round(float(value)):.0f}"
    if metric_key in {
        "n_sprints_total",
        "n_acc_2_5_total",
        "n_dec_3_0_total",
        "n_sprints_90",
        "n_acc_2_5_90",
        "n_dec_3_0_90",
    }:
        return "-" if value is None or pd.isna(value) else f"{round(float(value)):.0f}"
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


def _get_snapshot_display_metrics(snapshot: dict | None) -> list[tuple[str, str]]:
    if snapshot and snapshot.get("kind") == "session":
        return GAME_DISPLAY_METRICS
    return DISPLAY_METRICS


def _get_snapshot_metric_groups(snapshot: dict | None) -> dict[str, list[tuple[str, str]]]:
    if snapshot and snapshot.get("kind") == "session":
        return GAME_METRIC_GROUPS
    return PROFILE_METRIC_GROUPS


def _build_comparison_table(left_snapshot: dict, right_snapshot: dict) -> pd.DataFrame:
    rows = []
    left_metrics = left_snapshot.get("metrics", {}) if left_snapshot else {}
    right_metrics = right_snapshot.get("metrics", {}) if right_snapshot else {}
    left_metric_groups = _get_snapshot_metric_groups(left_snapshot)
    right_metric_groups = _get_snapshot_metric_groups(right_snapshot)

    for category, left_group_metrics in left_metric_groups.items():
        right_group_metrics = right_metric_groups.get(category, [])
        rows.append(
            {
                "Métrica": category,
                "Valor": "",
                "Delta": "",
                "Delta %": "",
                "Valor ": "",
                "Métrica ": "",
                "_is_category": True,
            }
        )
        for (left_key, left_label), (right_key, right_label) in zip(left_group_metrics, right_group_metrics):
            left_value = left_metrics.get(left_key, np.nan)
            right_value = right_metrics.get(right_key, np.nan)
            values_are_comparable = (
                left_key == right_key
                or (left_snapshot and right_snapshot and left_snapshot.get("kind") != right_snapshot.get("kind"))
            )
            delta_abs = (
                float(left_value) - float(right_value)
                if values_are_comparable and pd.notna(left_value) and pd.notna(right_value)
                else np.nan
            )
            delta_pct = (
                (delta_abs / float(right_value) * 100.0)
                if values_are_comparable and pd.notna(delta_abs) and pd.notna(right_value) and float(right_value) != 0
                else np.nan
            )
            rows.append(
                {
                    "Métrica": left_label,
                    "Valor": _format_profile_value(left_key, left_value),
                    "Delta": _format_profile_value(left_key, delta_abs) if values_are_comparable else "-",
                    "Delta %": _format_number(delta_pct, "%"),
                    "Valor ": _format_profile_value(right_key, right_value),
                    "Métrica ": right_label,
                    "_is_category": False,
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


def _style_category_row(row: pd.Series) -> list[str]:
    if bool(row.get("_is_category", False)):
        return ["font-weight: 700; background-color: rgba(148, 163, 184, 0.12);"] * (len(row) - 1)
    return [""] * (len(row) - 1)


def _render_comparison_table(df: pd.DataFrame) -> None:
    row_height = 35
    header_height = 38
    table_height = header_height + max(len(df), 1) * row_height + 2
    style_df = df.reset_index(drop=True)
    styled = (
        style_df.drop(columns=["_is_category"], errors="ignore")
        .style
        .apply(lambda row: _style_category_row(style_df.loc[row.name]), axis=1)
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


def _render_label_block(snapshot: dict | None) -> None:
    safe_label = ""
    st.markdown(
        f"""
        <div style="
            height: {LABEL_BLOCK_HEIGHT}px;
            display: flex;
            align-items: flex-start;
            color: rgb(107, 114, 128);
            font-size: 0.875rem;
            line-height: 1.2;
            overflow: hidden;
        ">
            {safe_label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_comparison_header(left_snapshot: dict, right_snapshot: dict) -> None:
    header_left, header_right = st.columns(2, gap="large")
    with header_left:
        _render_header_block(left_snapshot)
    with header_right:
        _render_header_block(right_snapshot)


def _render_comparison_section(left_snapshot: dict, right_snapshot: dict) -> None:
    if not left_snapshot or not right_snapshot:
        st.info("Seleciona um item em ambos os lados para comparar.")
        return

    _render_comparison_header(left_snapshot, right_snapshot)
    st.markdown(f"<div style='height: {HEADER_GAP_HEIGHT}px;'></div>", unsafe_allow_html=True)
    _render_comparison_table(_build_comparison_table(left_snapshot, right_snapshot))
    if left_snapshot.get("kind") != right_snapshot.get("kind"):
        st.caption("Nota: nesta comparação, os totais do jogo são comparados com o perfil de referência normalizado a 90 minutos.")


def _resolve_modes(comparison_mode: str) -> tuple[str, str]:
    if comparison_mode == "Jogo vs Perfil":
        return "Jogo", "Perfil"
    if comparison_mode == "Jogo vs Jogo":
        return "Jogo", "Jogo"
    return "Perfil", "Perfil"


def _build_session_phase_snapshot(
    athletes_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    session_sk: str,
    athlete_id: str,
    fase: str,
) -> dict:
    athlete_row = athletes_df[athletes_df["atleta_id"].eq(athlete_id)].head(1)
    if athlete_row.empty:
        return {}

    athlete_info = athlete_row.iloc[0].to_dict()
    df_session = perf_df[
        perf_df["session_sk"].eq(session_sk) &
        perf_df["atleta_id"].eq(athlete_id) &
        perf_df["fase"].eq(fase)
    ].copy()
    if df_session.empty:
        return {}

    return {
        "kind": "session",
        "athlete": athlete_info,
        "label": _build_session_event_label(df_session.iloc[0]),
        "phase_label": fase,
        "sessions_df": df_session,
        "metrics": _aggregate_profile(df_session),
    }


st.title("Analise de Perfis")
st.caption("Comparação Jogo vs Jogo entre os atleta_id carregados na sessão em draft.")

perf_df = _load_draft_perf_sessions()
athletes_df = _build_draft_athletes(perf_df)

if perf_df.empty:
    st.warning("Sem dados em draft disponiveis. Processa uma sessao e usa 'Visualizar / Download' para ativar esta pagina.")
    st.stop()

if athletes_df.empty:
    st.warning("Sem atleta_id carregados no draft atual.")
    st.stop()

athlete_display_map = {
    row["atleta_id"]: _clean_text(row["atleta_id"])
    for _, row in athletes_df.iterrows()
}
athlete_options = list(athlete_display_map.keys())

left_mode, right_mode = "Jogo", "Jogo"
session_options, session_label_map = _build_draft_session_option_map(perf_df)
if not session_options:
    st.warning("Sem sessões em draft disponíveis para comparação.")
    st.stop()

current_session = session_options[0]
st.markdown("**Sessão**")
st.markdown(session_label_map.get(current_session, current_session))

selector_left, selector_right = st.columns(2, gap="large")

with selector_left:
    left_session = current_session
    left_athlete_options = _build_athlete_options_for_session(perf_df, left_session)
    athlete_left = st.selectbox(
        "Atleta ID",
        options=left_athlete_options,
        format_func=lambda aid: athlete_display_map.get(aid, aid),
        key="athlete_compare_left",
    ) if left_athlete_options else None
    left_phase_options = _build_phase_options(perf_df, left_session, athlete_left) if athlete_left else []
    if left_phase_options:
        left_phase = st.selectbox("Fase", options=left_phase_options, key="phase_compare_left")
        left_snapshot = _build_session_phase_snapshot(athletes_df, perf_df, left_session, athlete_left, left_phase)
    else:
        st.info("Sem fases disponiveis para esta combinação.")
        left_snapshot = {}

with selector_right:
    right_session = current_session
    right_athlete_options = _build_athlete_options_for_session(perf_df, right_session)
    default_right_index = 1 if len(right_athlete_options) > 1 else 0
    athlete_right = st.selectbox(
        "Atleta ID",
        options=right_athlete_options,
        index=default_right_index,
        format_func=lambda aid: athlete_display_map.get(aid, aid),
        key="athlete_compare_right",
    ) if right_athlete_options else None
    right_phase_options = _build_phase_options(perf_df, right_session, athlete_right) if athlete_right else []
    if right_phase_options:
        right_phase = st.selectbox("Fase", options=right_phase_options, key="phase_compare_right")
        right_snapshot = _build_session_phase_snapshot(athletes_df, perf_df, right_session, athlete_right, right_phase)
    else:
        st.info("Sem fases disponiveis para esta combinação.")
        right_snapshot = {}

_render_comparison_section(left_snapshot, right_snapshot)

st.markdown("## Sessões Utilizadas")
sessions_left, sessions_right = st.columns(2, gap="large")

with sessions_left:
    st.caption(f"Sessões usadas no lado esquerdo ({left_mode.lower()})")
    left_sessions = left_snapshot.get("sessions_df", pd.DataFrame()) if left_snapshot else pd.DataFrame()
    if left_sessions.empty:
        st.info("Sem sessões para este lado.")
    else:
        cols = [c for c in ["data", "contexto", "jogo", "duracao_min"] if c in left_sessions.columns]
        left_display = left_sessions[cols].sort_values("data", ascending=False).copy()
        if "duracao_min" in left_display.columns:
            left_display["duracao_min"] = pd.to_numeric(left_display["duracao_min"], errors="coerce").round().astype("Int64")
        st.dataframe(left_display, use_container_width=True, hide_index=True)

with sessions_right:
    st.caption(f"Sessões usadas no lado direito ({right_mode.lower()})")
    right_sessions = right_snapshot.get("sessions_df", pd.DataFrame()) if right_snapshot else pd.DataFrame()
    if right_sessions.empty:
        st.info("Sem sessões para este lado.")
    else:
        cols = [c for c in ["data", "contexto", "jogo", "duracao_min"] if c in right_sessions.columns]
        right_display = right_sessions[cols].sort_values("data", ascending=False).copy()
        if "duracao_min" in right_display.columns:
            right_display["duracao_min"] = pd.to_numeric(right_display["duracao_min"], errors="coerce").round().astype("Int64")
        st.dataframe(right_display, use_container_width=True, hide_index=True)
