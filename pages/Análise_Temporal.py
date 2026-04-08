import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from fpf_modules.supabase_manager import initialize_schema, read_table
from fpf_modules.metrics import estimate_hr_max_from_birthdate


st.set_page_config(page_title="Analise Temporal", layout="wide")


TEMPORAL_METRICS = [
    ("dist_m", "Distancia", "m"),
    ("duracao_min", "Duracao", "min"),
    ("m_min", "m/min", ""),
    ("hsr_dist_m", "Distancia HSR", "m"),
    ("hsr_pct", "HSR %", "%"),
    ("sprint_dist_m", "Distancia Sprint", "m"),
    ("n_sprints", "N Sprints", ""),
    ("n_acc_2_5", "Acc", ""),
    ("n_dec_3_0", "Dec", ""),
    ("active_time_min", "Tempo Ativo", "min"),
    ("active_pct", "Ativo %", "%"),
    ("hr_avg_bpm", "HR Media", "bpm"),
    ("hr_peak_bpm", "HR Pico", "bpm"),
    ("beats_total", "Batimentos Totais", ""),
    ("dist_per_beat_m", "Distancia / Batimento", "m/bat"),
    ("external_load_score", "Carga Externa", ""),
    ("total_load_score", "Carga Total", ""),
    ("player_load", "Player Load", ""),
    ("rhie_bouts", "RHIE", ""),
    ("trimp_banister", "TRIMP", ""),
    ("vmax_mps", "Vmax", "m/s"),
    ("peak_1m_m_min", "Pico 1m", "m/min"),
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _banister_trimp(duracao_min, hr_avg_bpm, hr_max_bpm, hr_rest_bpm=60.0, genero=""):
    if any(pd.isna(v) for v in [duracao_min, hr_avg_bpm, hr_max_bpm]):
        return np.nan
    if float(hr_max_bpm) <= float(hr_rest_bpm) or float(duracao_min) <= 0:
        return np.nan
    hrr = (float(hr_avg_bpm) - float(hr_rest_bpm)) / (float(hr_max_bpm) - float(hr_rest_bpm))
    hrr = float(np.clip(hrr, 0.0, 1.0))
    k = 1.67 if str(genero or "").strip().upper().startswith("F") else 1.92
    return float(float(duracao_min) * hrr * 0.64 * np.exp(k * hrr))


def _compute_acwr(series: pd.Series, acute_window: int = 7, chronic_window: int = 28) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    acute = values.rolling(window=acute_window, min_periods=1).mean()
    chronic = values.rolling(window=chronic_window, min_periods=max(acute_window, 3)).mean()
    return acute / chronic.replace(0, np.nan)


def _safe_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or column_name not in df.columns:
        return pd.Series(dtype=float)
    values = df[column_name]
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    elif not isinstance(values, pd.Series):
        if np.isscalar(values):
            values = pd.Series([values], dtype="float64")
        else:
            values = pd.Series(list(values))

    numeric_values = pd.to_numeric(values, errors="coerce")
    if not isinstance(numeric_values, pd.Series):
        if np.isscalar(numeric_values):
            numeric_values = pd.Series([numeric_values], dtype="float64")
        else:
            numeric_values = pd.Series(list(numeric_values))
    return numeric_values.dropna()


def _safe_scalar(value):
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return np.nan
        value = value.iloc[0, 0]
    elif isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        value = value.dropna().iloc[0] if value.dropna().any() else np.nan
    elif not np.isscalar(value) and value is not None:
        try:
            seq = list(value)
            value = seq[0] if seq else np.nan
        except Exception:
            value = np.nan
    return value


@st.cache_data(show_spinner=False, ttl=300)
def _load_athletes() -> pd.DataFrame:
    initialize_schema()
    df = read_table("athletes")
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["atleta_id", "nome", "foto_url", "posicao", "ativo", "genero", "hr_max_bpm", "hr_rest_bpm", "data_nascimento"]
        )

    for col in ["atleta_id", "nome", "foto_url", "posicao", "genero"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(_clean_text)
    if "data_nascimento" not in df.columns:
        df["data_nascimento"] = pd.NaT
    df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce")

    for col in ["hr_max_bpm", "hr_rest_bpm"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

    text_cols = ["atleta_id", "contexto", "jogo", "selecao", "genero"]
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
        "hr_avg_bpm",
        "hr_peak_bpm",
        "hr_time_min",
        "hr_time_valid_pct",
        "beats_total",
        "dist_per_beat_m",
        "hsr_per_beat_m",
        "sprint_per_beat_m",
        "sprints_per_1000_beats",
        "acc_per_1000_beats",
        "dec_per_1000_beats",
        "external_load_score",
        "total_load_score",
        "player_load",
        "rhie_bouts",
        "rhie_actions",
        "trimp_banister",
        "trimp_per_min",
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(_clean_text)

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    athletes_df = _load_athletes()
    athlete_cols = [c for c in ["atleta_id", "genero", "hr_max_bpm", "hr_rest_bpm", "data_nascimento"] if c in athletes_df.columns]
    if athlete_cols:
        df = df.merge(
            athletes_df[athlete_cols].drop_duplicates(subset=["atleta_id"], keep="last"),
            on="atleta_id",
            how="left",
            suffixes=("", "_ath"),
        )
        if "genero_ath" in df.columns:
            df["genero"] = df["genero"].where(df["genero"].map(_clean_text).ne(""), df["genero_ath"])
            df = df.drop(columns=["genero_ath"])

    if "hr_max_bpm" not in df.columns:
        df["hr_max_bpm"] = np.nan
    estimated_hrmax = df.apply(
        lambda row: estimate_hr_max_from_birthdate(row.get("data_nascimento"), row.get("data")),
        axis=1,
    ) if "data_nascimento" in df.columns else pd.Series(np.nan, index=df.index)
    df["hr_max_bpm_effective"] = pd.to_numeric(df["hr_max_bpm"], errors="coerce").where(
        pd.to_numeric(df["hr_max_bpm"], errors="coerce").notna(),
        estimated_hrmax,
    )

    if "trimp_banister" not in df.columns:
        df["trimp_banister"] = np.nan
    missing_trimp = df["trimp_banister"].isna()
    if missing_trimp.any():
        df.loc[missing_trimp, "trimp_banister"] = df.loc[missing_trimp].apply(
            lambda row: _banister_trimp(
                row.get("duracao_min"),
                row.get("hr_avg_bpm"),
                row.get("hr_max_bpm_effective"),
                hr_rest_bpm=row.get("hr_rest_bpm") if pd.notna(row.get("hr_rest_bpm")) else 60.0,
                genero=row.get("genero"),
            ),
            axis=1,
        )

    if "player_load" not in df.columns:
        df["player_load"] = np.nan
    if "rhie_bouts" not in df.columns:
        df["rhie_bouts"] = np.nan

    return df


def _build_session_label(row: pd.Series) -> str:
    data_txt = pd.to_datetime(row.get("data")).strftime("%d/%m/%Y") if pd.notna(row.get("data")) else "-"
    contexto_txt = _clean_text(row.get("contexto")) or "-"
    jogo_txt = _clean_text(row.get("jogo")) or f"Sessao {row.get('session_sk', '-')}"
    return f"{data_txt} | {contexto_txt} | {jogo_txt}"


def _build_temporal_figure(df: pd.DataFrame, metric_key: str, metric_label: str, suffix: str, rolling_window: int):
    plot_df = df.copy()
    plot_df["ordem"] = np.arange(len(plot_df))
    plot_df["label"] = plot_df.apply(_build_session_label, axis=1)
    if metric_key not in plot_df.columns:
        plot_df[metric_key] = np.nan
    plot_df[metric_key] = pd.to_numeric(plot_df[metric_key], errors="coerce")
    plot_df["rolling"] = pd.to_numeric(plot_df[metric_key], errors="coerce").rolling(window=rolling_window, min_periods=1).mean()

    if go is None:
        return plot_df

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["ordem"],
            y=plot_df[metric_key],
            mode="lines+markers",
            name=metric_label,
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            customdata=plot_df[["label", "contexto"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Contexto: %{customdata[1]}<br>Valor: %{y:.2f}"
            + suffix
            + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["ordem"],
            y=plot_df["rolling"],
            mode="lines",
            name=f"Media movel ({rolling_window})",
            line=dict(color="#ef4444", width=2, dash="dash"),
            hovertemplate="Media movel: %{y:.2f}" + suffix + "<extra></extra>",
        )
    )

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Sessoes", tickmode="array", tickvals=plot_df["ordem"], ticktext=plot_df["label"]),
        yaxis=dict(title=f"{metric_label}{f' ({suffix})' if suffix else ''}"),
        hovermode="x unified",
    )
    return fig


st.title("Analise Temporal")
st.caption("Evolucao temporal das metricas de performance por atleta.")

athletes_df = _load_athletes()
perf_df = _load_perf_sessions()

if athletes_df.empty:
    st.warning("Sem atletas ativos na base de dados.")
    st.stop()

if perf_df.empty:
    st.warning("Sem dados de performance disponiveis para analise temporal.")
    st.stop()

athlete_display_map = {
    row["atleta_id"]: f"{_clean_text(row['nome'])} ({_clean_text(row['atleta_id'])})"
    for _, row in athletes_df.iterrows()
}

selector_left, selector_right, selector_extra = st.columns([1.5, 1.0, 1.0], gap="large")

with selector_left:
    athlete_id = st.selectbox(
        "Atleta",
        options=list(athlete_display_map.keys()),
        format_func=lambda aid: athlete_display_map.get(aid, aid),
    )

with selector_right:
    contexto_filter = st.selectbox("Contexto", options=["Todos", "Jogo", "Treino"], index=0)

with selector_extra:
    rolling_window = st.selectbox("Media movel", options=[2, 3, 5], index=1)

df_athlete = perf_df[perf_df["atleta_id"].eq(athlete_id)].copy()
if contexto_filter != "Todos":
    df_athlete = df_athlete[df_athlete["contexto"].eq(contexto_filter)].copy()

if df_athlete.empty:
    st.info("Sem sessoes disponiveis para os filtros selecionados.")
    st.stop()

sort_cols = [c for c in ["data", "session_sk"] if c in df_athlete.columns]
if sort_cols:
    df_athlete = df_athlete.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)

available_metrics = [
    (key, label, suffix)
    for key, label, suffix in TEMPORAL_METRICS
    if key in df_athlete.columns
]
if not available_metrics:
    st.info("Sem metricas disponiveis para as sessoes selecionadas.")
    st.stop()

metric_labels = {label: (key, suffix) for key, label, suffix in available_metrics}
selected_label = st.selectbox("Metrica", options=list(metric_labels.keys()), index=0)
metric_key, metric_suffix = metric_labels[selected_label]

athlete_info = athletes_df[athletes_df["atleta_id"].eq(athlete_id)].head(1)
athlete_row = athlete_info.iloc[0] if not athlete_info.empty else {}

top_left, top_right = st.columns([0.95, 2.2], gap="large")
with top_left:
    foto_url = _clean_text(athlete_row.get("foto_url")) if not athlete_info.empty else ""
    if foto_url:
        st.image(foto_url, width=120)
with top_right:
    st.markdown(f"**Atleta:** {_clean_text(athlete_row.get('nome')) if not athlete_info.empty else '-'}")
    st.markdown(f"**Posicao:** {_clean_text(athlete_row.get('posicao')) if not athlete_info.empty else '-'}")
    st.markdown(f"**Sessoes consideradas:** {len(df_athlete)}")
    hrmax_measured = _safe_scalar(athlete_row.get("hr_max_bpm"))
    ref_session_date = df_athlete["data"].max() if "data" in df_athlete.columns and df_athlete["data"].notna().any() else None
    hrmax_estimated = _safe_scalar(estimate_hr_max_from_birthdate(
        _safe_scalar(athlete_row.get("data_nascimento")),
        reference_date=ref_session_date,
    ))
    if pd.notna(hrmax_measured):
        st.markdown(f"**HRmax:** {int(float(hrmax_measured))} bpm")
    elif pd.notna(hrmax_estimated):
        st.markdown(f"**HRmax estimada:** {int(round(float(hrmax_estimated)))} bpm")
    if "data" in df_athlete.columns and df_athlete["data"].notna().any():
        st.markdown(f"**Periodo:** {df_athlete['data'].min().strftime('%d/%m/%Y')} a {df_athlete['data'].max().strftime('%d/%m/%Y')}")

chart_data = _build_temporal_figure(df_athlete, metric_key, selected_label, metric_suffix, rolling_window)
if go is None:
    st.info("Plotly nao esta instalado neste ambiente; a pagina usa um grafico nativo do Streamlit.")
    st.line_chart(
        chart_data.set_index("label")[[metric_key, "rolling"]].rename(
            columns={metric_key: selected_label, "rolling": f"Media movel ({rolling_window})"}
        ),
        use_container_width=True,
        height=440,
    )
else:
    st.plotly_chart(chart_data, use_container_width=True)

table_cols = list(
    dict.fromkeys(
        c
        for c in [
            "data",
            "contexto",
            "jogo",
            "duracao_min",
        ]
        if c in df_athlete.columns
    )
)
table_df = df_athlete[table_cols].copy()
if "data" in table_df.columns:
    table_df["data"] = table_df["data"].dt.strftime("%d/%m/%Y")
if "duracao_min" in table_df.columns:
    table_df = table_df.rename(columns={"duracao_min": "Minutos"})

st.markdown("## Sessoes")
st.dataframe(table_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
