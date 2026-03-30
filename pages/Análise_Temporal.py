import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from fpf_modules.supabase_manager import initialize_schema, read_table


st.set_page_config(page_title="Análise Temporal", layout="wide")


TEMPORAL_METRICS = [
    ("dist_m", "Distância", "m"),
    ("duracao_min", "Duração", "min"),
    ("m_min", "m/min", ""),
    ("hsr_dist_m", "Distância HSR", "m"),
    ("hsr_pct", "HSR %", "%"),
    ("sprint_dist_m", "Distância Sprint", "m"),
    ("n_sprints", "Nº Sprints", ""),
    ("n_acc_2_5", "Acc", ""),
    ("n_dec_3_0", "Dec", ""),
    ("active_time_min", "Tempo Ativo", "min"),
    ("active_pct", "Ativo %", "%"),
    ("vmax_mps", "Vmax", "m/s"),
    ("peak_1m_m_min", "Pico 1m", "m/min"),
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _format_value(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}{suffix}".strip()


@st.cache_data(show_spinner=False, ttl=300)
def _load_athletes() -> pd.DataFrame:
    initialize_schema()
    df = read_table("athletes")
    if df is None or df.empty:
        return pd.DataFrame(columns=["atleta_id", "nome", "foto_url", "posicao", "ativo"])

    for col in ["atleta_id", "nome", "foto_url", "posicao"]:
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

    return df


def _build_session_label(row: pd.Series) -> str:
    data_txt = pd.to_datetime(row.get("data")).strftime("%d/%m/%Y") if pd.notna(row.get("data")) else "-"
    contexto_txt = _clean_text(row.get("contexto")) or "-"
    jogo_txt = _clean_text(row.get("jogo")) or f"Sessão {row.get('session_sk', '-')}"
    return f"{data_txt} | {contexto_txt} | {jogo_txt}"


def _build_temporal_figure(df: pd.DataFrame, metric_key: str, metric_label: str, suffix: str, rolling_window: int):
    plot_df = df.copy()
    plot_df["ordem"] = np.arange(len(plot_df))
    plot_df["label"] = plot_df.apply(_build_session_label, axis=1)
    plot_df["rolling"] = plot_df[metric_key].rolling(window=rolling_window, min_periods=1).mean()

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
            hovertemplate="<b>%{customdata[0]}</b><br>Contexto: %{customdata[1]}<br>Valor: %{y:.1f}" + suffix + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["ordem"],
            y=plot_df["rolling"],
            mode="lines",
            name=f"Média móvel ({rolling_window})",
            line=dict(color="#ef4444", width=2, dash="dash"),
            hovertemplate="Média móvel: %{y:.1f}" + suffix + "<extra></extra>",
        )
    )

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Sessões",
            tickmode="array",
            tickvals=plot_df["ordem"],
            ticktext=plot_df["label"],
        ),
        yaxis=dict(title=f"{metric_label}{f' ({suffix})' if suffix else ''}"),
        hovermode="x unified",
    )
    return fig


st.title("Análise Temporal")
st.caption("Evolução temporal das métricas de performance por atleta.")

athletes_df = _load_athletes()
perf_df = _load_perf_sessions()

if athletes_df.empty:
    st.warning("Sem atletas ativos na base de dados.")
    st.stop()

if perf_df.empty:
    st.warning("Sem dados de performance disponíveis para análise temporal.")
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
    rolling_window = st.selectbox("Média móvel", options=[2, 3, 5], index=1)

metric_labels = {label: (key, suffix) for key, label, suffix in TEMPORAL_METRICS}
selected_label = st.selectbox("Métrica", options=list(metric_labels.keys()), index=0)
metric_key, metric_suffix = metric_labels[selected_label]

df_athlete = perf_df[perf_df["atleta_id"].eq(athlete_id)].copy()
if contexto_filter != "Todos":
    df_athlete = df_athlete[df_athlete["contexto"].eq(contexto_filter)].copy()

if df_athlete.empty:
    st.info("Sem sessões disponíveis para os filtros selecionados.")
    st.stop()

sort_cols = [c for c in ["data", "session_sk"] if c in df_athlete.columns]
ascending = [True] * len(sort_cols)
if sort_cols:
    df_athlete = df_athlete.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

athlete_info = athletes_df[athletes_df["atleta_id"].eq(athlete_id)].head(1)
athlete_row = athlete_info.iloc[0] if not athlete_info.empty else {}

top_left, top_right = st.columns([0.95, 2.2], gap="large")
with top_left:
    foto_url = _clean_text(athlete_row.get("foto_url")) if not athlete_info.empty else ""
    if foto_url:
        st.image(foto_url, width=120)
with top_right:
    st.markdown(f"**Atleta:** {_clean_text(athlete_row.get('nome')) if not athlete_info.empty else '-'}")
    st.markdown(f"**Posição:** {_clean_text(athlete_row.get('posicao')) if not athlete_info.empty else '-'}")
    st.markdown(f"**Sessões consideradas:** {len(df_athlete)}")
    if "data" in df_athlete.columns and df_athlete["data"].notna().any():
        st.markdown(
            f"**Período:** {df_athlete['data'].min().strftime('%d/%m/%Y')} a {df_athlete['data'].max().strftime('%d/%m/%Y')}"
        )

chart_data = _build_temporal_figure(df_athlete, metric_key, selected_label, metric_suffix, rolling_window)
if go is None:
    st.info("Plotly não está instalado neste ambiente; a página está a usar um gráfico nativo do Streamlit.")
    st.line_chart(
        chart_data.set_index("label")[[metric_key, "rolling"]].rename(
            columns={metric_key: selected_label, "rolling": f"Média móvel ({rolling_window})"}
        ),
        use_container_width=True,
        height=440,
    )
else:
    st.plotly_chart(chart_data, use_container_width=True)

table_cols = list(
    dict.fromkeys(
        c for c in ["data", "contexto", "jogo", metric_key, "duracao_min", "dist_m", "m_min", "vmax_mps"] if c in df_athlete.columns
    )
)
table_df = df_athlete[table_cols].copy()
if "data" in table_df.columns:
    table_df["data"] = table_df["data"].dt.strftime("%d/%m/%Y")

st.markdown("## Sessões")
st.dataframe(table_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
