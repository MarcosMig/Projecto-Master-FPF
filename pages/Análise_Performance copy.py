import matplotlib.pyplot as plt
import mplsoccer as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull, QhullError

import fpf_modules.visualization as visual
from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES
from fpf_modules.metrics import calcular_area_cache, calcular_compactacao_cache
from fpf_modules.utils import converter_para_relogio_fpf, fmt

# Fixar max largura da pagina, de forma ao campo não esticar em demasia
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Estilo para as métricas de partida
st.markdown(
    """
    <style>
    /* Estilo Cartas Métricas Partida */
    .fpt-kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 20px;
    }
    .fpt-kpi-card {
        background-color: #1e1e1e;
        border-left: 5px solid #E30613; /* Linha vermelha */
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        flex: 1;
    }
    .fpt-kpi-label {
        color: #9aa0a6;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .fpt-kpi-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="FPF | Análise Performance", layout="wide")

st.title("Análise Performance")


# --- CARREGAMENTO E PROCESSAMENTO ---
@st.cache_data
def load_and_merge_data():
    perf = pd.read_parquet(f"{CLEANDATA_DIR}/performance_metrics.parquet")
    tracking = pd.read_parquet(f"{CLEANDATA_DIR}/tracking.parquet")
    sessions = pd.read_parquet(f"{CLEANDATA_DIR}/sessions.parquet")

    # Merge de metadados no tracking
    cols_meta = [
        "data",
        "selecao",
        "contexto",
        "jogo",
        "session_sk",
        "dist_x",
        "dist_y",
    ]
    df_merged = pd.merge(tracking, sessions[cols_meta], how="left", on="session_sk")

    perf["data"] = pd.to_datetime(perf["data"], format="%d/%m/%Y")
    df_merged["data"] = pd.to_datetime(df_merged["data"], format="%d/%m/%Y")
    return perf, df_merged


try:
    performance_df, tracking_df = load_and_merge_data()
except Exception as e:
    st.error(f"Erro ao processar dados: {e}")
    st.stop()


# -------------------------------------------------------------------------
# SIDEBAR: FILTROS E CONTROLES
# -------------------------------------------------------------------------

with st.sidebar:
    st.title("⚽ Filtros de Sessão")

    # --- 1. FILTROS PRINCIPAIS — Seleção, Contexto, Ano --- #

    selecao = st.selectbox("Seleção", options=SELECOES_OPCOES)
    contexto = st.selectbox("Contexto", options=["Treino", "Jogo"])

    df_contexto = performance_df[
        (performance_df["selecao"] == selecao)
        & (performance_df["contexto"] == contexto)
    ]

    if df_contexto.empty:
        st.warning("Sem dados para este contexto!")
        st.stop()

    anos_disponiveis = sorted(df_contexto["data"].dt.year.unique(), reverse=True)
    ano = st.selectbox("Ano", options=anos_disponiveis)

    df_ano = df_contexto[df_contexto["data"].dt.year == ano]

    # --- 2. SELEÇÃO DE SESSÃO — Jogo ou Treino --- #

    jogo = ""
    data = None

    if contexto == "Jogo":
        df_ano["jogo_label"] = (
            df_ano["jogo"].astype(str)
            + " ("
            + df_ano["data"].dt.strftime("%d-%m-%Y")
            + ")"
        )
        jogos_disponiveis = df_ano["jogo_label"].unique()

        if jogos_disponiveis.size == 0:
            st.warning("Nenhum jogo registado para esta seleção.")
            st.stop()

        jogo_label = st.selectbox("Adversário", options=jogos_disponiveis)
        linha = df_ano[df_ano["jogo_label"] == jogo_label].iloc[0]
        jogo = linha["jogo"]
        data = linha["data"]

    else:  # Treino
        datas_disponiveis = sorted(df_ano["data"].unique(), reverse=True)

        if len(datas_disponiveis) == 0:
            st.warning("Nenhum treino registado para esta seleção.")
            st.stop()

        data = st.selectbox(
            "Data da Sessão",
            options=datas_disponiveis,
            format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
        )

    # --- 3. SESSÕES DISPONÍVEIS para a seleção actual --- #

    perf_df = performance_df[
        (performance_df["selecao"] == selecao)
        & (performance_df["contexto"] == contexto)
        & (performance_df["jogo"] == jogo)
        & (performance_df["data"] == data)
    ]

    # ---  4. NAVEGAÇÃO TEMPORAL — Fase e Slider de Tempo --- #

    st.divider()
    st.header("⏱️ Navegação Temporal")

    fase_selected = st.radio("Fase", options=["Warm-Up", "1P", "2P"], index=1)

    track_df = tracking_df[
        (tracking_df["selecao"] == selecao)
        & (tracking_df["contexto"] == contexto)
        & (tracking_df["jogo"] == jogo)
        & (tracking_df["data"] == data)
        & (tracking_df["fase"] == fase_selected)
    ]

    if track_df.empty:
        st.warning(f"Sem dados de tracking para a fase {fase_selected}.")
        selected_time = None
        dist_x, dist_y = None, None
    else:
        dist_x = track_df["dist_x"].iloc[0]
        dist_y = track_df["dist_y"].iloc[0]

        timestamps = sorted(track_df["time_evento_s"].unique())
        selected_time = st.select_slider(
            "Momento do Jogo",
            options=timestamps,
            format_func=converter_para_relogio_fpf,
        )

# -------------------------------------------------------------------------
# PAGINA
# -------------------------------------------------------------------------

# Info sessão
st.markdown(
    f"""
<div style="margin-bottom:24px; border-left:3px solid #E30613; padding-left:12px;">
    <div style="font-size:0.75rem; color:#9aa0a6; letter-spacing:1px; text-transform:uppercase;">
        {selecao} · {contexto} · {data.strftime("%d-%m-%Y") if contexto == "Jogo" else ""}
    </div>
    <div style="font-size:1.4rem; font-weight:600; color:#ffffff; margin-top:2px;">
        {jogo if jogo else data.strftime("%d-%m-%Y")}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Incializar Paineis
tab_metrics, tab_visual = st.tabs(
    ["📊 Métricas de Performance", "📍 Análise Posicional"]
)


# -------------------------------------------------------------------------
# TAB 1: MÉTRICAS DE PERFORMANCE
# -------------------------------------------------------------------------

# ── Constantes ─────────────────────────────────────────────────────

FASES = ["Warm-Up", "1P", "2P"]

METRICS_CONFIG = {
    "dist_m": "Distância (m)",
    "m_min": "m/min",
    "vmax_mps": "Vel. Máx (m/s)",
    "n_sprints": "Sprints",
    "hsr_pct": "HSR (%)",
    "n_acc_2_5": "Acelerações",
}

# ── Pagina ─────────────────────────────────────────────────────

with tab_metrics:
    # ── Remover linhas sem atividade ─────────────────────────────────

    df_clean = perf_df[
        perf_df["duracao_min"].notna() & (perf_df["duracao_min"] > 0)
    ].copy()

    if df_clean.empty:
        st.info("Sem dados de performance para a sessão selecionada.")
        st.stop()

    # ── Filtros ─────────────────────────────────────────────────────

    st.subheader("Comparação de Métricas")

    c1, c2 = st.columns([1, 2])

    with c1:
        fase_chart = st.selectbox("Fase", FASES, key="chart_fase2")
    with c2:
        metric_label = st.selectbox(
            "Métrica", list(METRICS_CONFIG.values()), key="chart_metric2"
        )

    metric_col = next(k for k, v in METRICS_CONFIG.items() if v == metric_label)

    df_chart = df_clean[df_clean["fase"] == fase_chart].dropna(subset=[metric_col])

    # ── KPI Cards ─────────────────────────────────────────────────────

    if not df_chart.empty:
        n_players = df_chart["atleta_id"].nunique()
        avg_dist = df_chart["dist_m"].mean()
        avg_mmin = df_chart["m_min"].mean()
        total_spr = df_chart["n_sprints"].sum()
        avg_vmax = df_chart["vmax_mps"].max()

        labels = [
            "Jogadores",
            "Dist. Média (m)",
            "m/min Médio",
            "Total Sprints",
            "Vel. Máx (m/s)",
        ]
        values = [n_players, avg_dist, avg_mmin, total_spr, avg_vmax]

        for col, label, value in zip(st.columns(5), labels, values):
            with col:
                visual.kpi_card(label, f"{value:.0f}")

        st.divider()

        # ── Comparison chart ─────────────────────────────────────────────────────

        df_chart_sorted = df_chart.sort_values(metric_col, ascending=True)
        n = len(df_chart_sorted)
        colours = ["#E30613" if i >= n - 3 else "#3a3a3a" for i in range(n)]
        fig = go.Figure(
            go.Bar(
                x=df_chart_sorted[metric_col],
                y=df_chart_sorted["atleta_id"],
                orientation="h",
                marker=dict(color=colours),
                text=df_chart_sorted[metric_col].round(1),
                textposition="outside",
                textfont=dict(color="#fff", size=11),
                hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"{metric_label} · {fase_chart}", y=0.98, x=0.5, xanchor="center"
            ),
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="#fff"),
            xaxis=dict(showgrid=True, gridcolor="#2a2a2a", zeroline=False),
            yaxis=dict(automargin=True),
            margin=dict(l=0, r=50, t=40, b=10),
            height=max(300, n * 36),
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.divider()

    # ── Expander Jogadores ──────────────────────────────────────────────────

    st.markdown(
        "#### Detalhe por Jogador",
        help="A distância que aparece no cartão apenas engloba as fases de jogo!",
    )

    players = sorted(df_clean["atleta_id"].unique())

    pairs = [players[i : i + 2] for i in range(0, len(players), 2)]

    for pair in pairs:
        col_left, col_right = st.columns(2)

        for col, player in zip([col_left, col_right], pair):
            with col:
                # Obter dados jogador
                df_player = df_clean.loc[(df_clean["atleta_id"] == player)]

                df_player_m = df_player[df_player["fase"].isin(["1P", "2P"])]

                dist_label = ""

                # Obter total distancia partida em km
                if not df_player_m.empty and df_player_m["dist_m"].notna().any():
                    dist_km = round(df_player_m["dist_m"].sum() / 1000, 2)

                    dist_label = f"  |  Distância {dist_km}km"

                with st.expander(
                    f"**:red[{player}]**  {dist_label}", icon=":material/person:"
                ):
                    phase_cols = st.columns(3)

                    for phase_col, fase in zip(phase_cols, FASES):
                        with phase_col:
                            # Estitlo titulo fase
                            st.markdown(
                                f'<p style="color:#E30613; font-weight:700; text-align:center; font-size:20px; border-bottom:1px solid #E30613;">{fase}</p>',
                                unsafe_allow_html=True,
                            )

                            row = df_player[df_player["fase"] == fase]

                            if row.empty:
                                st.caption("Sem dados")
                                continue

                            r = row.iloc[0]
                            for (
                                metric_col_name,
                                metric_display,
                            ) in METRICS_CONFIG.items():
                                visual.render_metric(
                                    metric_display,
                                    fmt(r.get(metric_col_name), metric_col_name),
                                )

# -------------------------------------------------------------------------
# TAB 2: VISUALIZAÇÃO DO CAMPO (METRICAS DE TRACKING)
# -------------------------------------------------------------------------

with tab_visual:
    st.subheader(f"Métricas da Partida · {fase_selected}")

    # ── Calculo Métricas ──────────────────────────────────────────────────

    if fase_selected != "Warm-Up" and contexto != "Treino":
        df_compactacao = calcular_compactacao_cache(track_df, dist_x, dist_y)

        df_area = calcular_area_cache(track_df, dist_x, dist_y)

        if not df_compactacao.empty:
            avg_comp_vert = df_compactacao["comp_vertical"].mean().round(2)
            avg_comp_hor = df_compactacao["comp_horizontal"].mean().round(2)
            avg_area = df_area["area"].mean().round(2)

            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            with col1:
                visual.kpi_card("COMPACTAÇÃO VERTICAL (m)", f"{avg_comp_vert:.1f}")

            with col2:
                visual.kpi_card("COMPACTAÇÃO HORIZONTAL (m)", f"{avg_comp_hor:.1f}")

            with col3:
                visual.kpi_card("ÁREA OCUPADA (m²)", f"{avg_area:.1f}")

            st.divider()

    else:
        st.info("Métricas calculadas apenas para fase de jogo!")

    # ── Snapshot ──────────────────────────────────────────────────

    if selected_time is not None:
        # 1. Obter snapshot
        snapshot = track_df[
            (track_df["time_evento_s"] == selected_time)
            & (track_df["x_tr"].notna())
            & (track_df["y_tr"].notna())
        ]

        if fase_selected != "Warm-Up":
            # Obter metricas para o frame
            compactacao_frame = df_compactacao.loc[
                df_compactacao["time_evento_s"] == selected_time
            ]
            area_frame = df_area.loc[df_area["time_evento_s"] == selected_time]

        col1, col2 = st.columns(2)

        with col1:
            campo = st.selectbox(
                label="Visualização Campo",
                options=["Convex Hull"],
                index=None,
                placeholder="Seleciona outro metodo de visualizar o campo",
            )

        # ── Heatmap ──────────────────────────────────────────────────

        if "selected_player" not in st.session_state:
            st.session_state.selected_player = None

        with col2:
            st.markdown(
                '<p style="font-size:14px; margin-bottom:4px;">Visualizar Heatmap do Jogador</p>',
                unsafe_allow_html=True,
            )

            texto_popup = "Selecione um jogador no campo, para verificar o seu heatmap!"

            if st.session_state.selected_player:
                player_name = st.session_state.selected_player
                texto_popup = f"Jogador selecionado: {player_name}"

            with st.popover(texto_popup, width=500):
                if st.session_state.selected_player:
                    # Obter dados do atleta
                    player_frames = track_df[track_df["atleta_id"] == player_name]

                    st.markdown(f"### {player_name} — Heatmap")

                    # Vamos usar Plotly, pois o kdeplot do mplsoccer
                    # Demora muito tempo a carregar
                    fig_heat = go.Figure()
                    fig_heat.update_layout(
                        shapes=visual.draw_statsbomb_pitch_horizontal(),
                        plot_bgcolor="#22312b",
                        paper_bgcolor="#22312b",
                        xaxis=dict(range=[0, 120], visible=False),
                        yaxis=dict(range=[0, 80], visible=False),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=300,
                    )

                    # Calcular Histograma de Posição
                    fig_heat.add_trace(
                        go.Histogram2dContour(
                            x=player_frames["x_tr"],
                            y=player_frames["y_tr"],
                            colorscale=visual.custom_hot,
                            reversescale=False,
                            showscale=False,
                            ncontours=20,
                            opacity=1,
                            contours=dict(
                                coloring="fill",
                            ),
                            line=dict(width=0),
                        )
                    )

                    # Render Heatmap
                    st.plotly_chart(
                        fig_heat,
                        config={
                            "scrollZoom": False,
                            "displayModeBar": False,
                            "staticPlot": True,
                        },
                        width="stretch",
                    )

                else:
                    st.warning(
                        "Selecione um jogador no campo, para verificar a sua posição durante o jogo!"
                    )

        # ── Campo ──────────────────────────────────────────────────

        col_map, col_info = st.columns([2.5, 1])

        # Campo
        with col_map:
            # --- Main pitch ---
            fig_pitch = go.Figure()

            # Configurar Campo
            fig_pitch.update_layout(
                shapes=visual.draw_statsbomb_pitch_horizontal(),
                plot_bgcolor="#22312b",
                paper_bgcolor="#22312b",
                xaxis=dict(
                    range=[-2, 122],
                    visible=False,
                ),
                yaxis=dict(
                    range=[-2, 82],
                    visible=False,
                    fixedrange=False,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=550,
                autosize=True,
                dragmode=False,
            )

            # Desenhar Jogadores
            fig_pitch.add_trace(
                go.Scatter(
                    x=snapshot["x_tr"],
                    y=snapshot["y_tr"],
                    mode="markers",
                    marker=dict(
                        size=14, color="red", line=dict(color="white", width=1)
                    ),
                    customdata=snapshot[["atleta_id"]].values,
                    hovertemplate="<b>Jogador: %{customdata[0]}</b><extra></extra>",
                )
            )

            # Campo Default
            if campo is None:
                # Capture clicks
                clicked = st.plotly_chart(
                    fig_pitch,
                    on_select="rerun",
                    width="stretch",
                    config={
                        "scrollZoom": False,
                        "responsive": True,
                        "displayModeBar": False,
                    },
                )

                if clicked and clicked["selection"]["points"]:
                    st.session_state.selected_player = clicked["selection"]["points"][
                        0
                    ]["customdata"][0]

            # ── Convex Hull ──────────────────────────────────────────────────
            if campo == "Convex Hull":
                try:
                    pts = snapshot[["x_tr", "y_tr"]].dropna().values

                    hull = ConvexHull(pts)

                    # Get hull vertices in order and close the polygon
                    # by repeating the first point
                    hull_pts = pts[hull.vertices]
                    hull_pts_closed = np.vstack([hull_pts, hull_pts[0]])

                    fig_pitch.add_trace(
                        go.Scatter(
                            x=hull_pts_closed[:, 0],
                            y=hull_pts_closed[:, 1],
                            mode="lines",
                            fill="toself",
                            fillcolor="rgba(227, 6, 19, 0.3)",
                            line=dict(color="#E30613", width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    fig_pitch.update_layout(showlegend=False)

                except QhullError:
                    pass

                st.plotly_chart(
                    fig_pitch,
                    config={
                        "scrollZoom": False,
                        "responsive": True,
                        "displayModeBar": False,
                        #    'staticPlot': True
                    },
                )

        # ── Frame Info ──────────────────────────────────────────────────
        with col_info:
            st.subheader("Analise de Frame")
            st.metric(
                "Tempo Selecionado", f"{converter_para_relogio_fpf(selected_time)}s"
            )

            if fase_selected != "Warm-Up":
                st.metric(
                    "Compactação Vertical (m)", compactacao_frame["comp_vertical"]
                )
                st.metric(
                    "Compactação Horizontal (m)", compactacao_frame["comp_horizontal"]
                )
                st.metric("Área Ocupada (m²)", area_frame["area"])


# --- FOOTER ---
st.divider()
st.caption(
    f"FPF UTM Engine v16 | Data Shape: {performance_df.shape[0]} sessions loaded."
)
