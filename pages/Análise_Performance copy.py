from itertools import combinations

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

PITCH_X = 120
PITCH_Y = 80

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

    campo_visualizacao = st.session_state.get("campo_visualizacao")
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
        & (tracking_df["x_tr"].notna())
        & (tracking_df["y_tr"].notna())
    ]

    if track_df.empty:
        st.warning(f"Sem dados de tracking para a fase {fase_selected}.")
        selected_time = None
        dist_x, dist_y = None, None
    else:
        dist_x = track_df["dist_x"].iloc[0]
        dist_y = track_df["dist_y"].iloc[0]

        timestamps = sorted(track_df["time_evento_s"].unique())

        # Se campo for movimento slider tempo muda para pagina principal
        if campo_visualizacao == "Movimento Relativo de Jogadores ao Longo do Tempo":
            selected_time = st.session_state.get("selected_time_jogo", timestamps[0])

        else:
            # Slider para navegar no tempo
            selected_time = st.sidebar.select_slider(
                "Momento do Jogo (s)",
                options=timestamps,
                key="selected_time_jogo",
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

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False},
            key="metricas_barras",
        )

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
    campo_visualizacao = st.session_state.get("campo_visualizacao")

    ocultar_metricas_partida = campo_visualizacao in [
        "Distância entre Jogadores",
        "Movimento Relativo de Jogadores ao Longo do Tempo",
    ]

    if not ocultar_metricas_partida:
        st.subheader(f"Métricas da Partida · {fase_selected}")

    # ── Calculo Métricas ──────────────────────────────────────────────────

    ##### !!!!!! TESTE
    def kpi_card(label, value, tooltip=None):
        title_attr = f'title="{tooltip}"' if tooltip else ""
        st.markdown(
            f"""
            <div class="fpt-kpi-card" {title_attr}>
                <div class="fpt-kpi-label">{label}</div>
                <div class="fpt-kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if fase_selected != "Warm-Up" and contexto != "Treino":
        df_compactacao = calcular_compactacao_cache(track_df, dist_x, dist_y)

        df_area = calcular_area_cache(track_df, dist_x, dist_y)

        if not df_compactacao.empty and not ocultar_metricas_partida:
            avg_comp_vert = df_compactacao["comp_vertical"].mean().round(2)
            avg_comp_hor = df_compactacao["comp_horizontal"].mean().round(2)
            avg_area = df_area["area"].mean().round(2)

            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            with col1:
                kpi_card("COMPACTAÇÃO VERTICAL (m)", f"{avg_comp_vert:.1f}", "151")

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

        jogadores_disponiveis = sorted(
            snapshot["atleta_id"].dropna().astype(str).unique().tolist()
        )

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
                options=[
                    "Convex Hull",
                    "Distância entre Jogadores",
                    "Movimento Relativo de Jogadores ao Longo do Tempo",
                ],
                index=None,
                placeholder="Seleciona outro metodo de visualizar o campo",
                key="campo_visualizacao",
            )

        jogadores_selecionados = []
        pares_opcoes = []
        pares_selecionados = []
        distancias_pares = []
        jogador_rel_1 = None
        jogador_rel_2 = None
        tempo_relativo = None
        janela_segundos = 1
        distancia_rel = None
        delta_distancia_rel = None
        distancia_media_rel = None
        distancia_min_rel = None
        distancia_max_rel = None
        tendencia_rel = None
        serie_distancias_rel = pd.DataFrame()
        cores_pares = [
            ("Amarelo", "#FFD166"),
            ("Verde", "#06D6A0"),
            ("Azul Claro", "#4CC9F0"),
            ("Rosa", "#EF476F"),
            ("Laranja", "#F77F00"),
            ("Verde Lima", "#90BE6D"),
        ]
        if campo == "Distância entre Jogadores" and len(jogadores_disponiveis) >= 2:
            jogadores_default = st.session_state.get(
                "dist_jogadores_selecionados", jogadores_disponiveis[:2]
            )
            jogadores_selecionados = [
                str(jogador)
                for jogador in jogadores_default
                if str(jogador) in jogadores_disponiveis
            ][:4]

            if len(jogadores_selecionados) < 2:
                jogadores_selecionados = jogadores_disponiveis[:2]

            pares_opcoes = [
                f"{jogador_1} - {jogador_2}"
                for jogador_1, jogador_2 in combinations(jogadores_selecionados, 2)
            ]
            pares_default = st.session_state.get(
                "dist_pares_selecionados", pares_opcoes[:1]
            )
            pares_selecionados = [par for par in pares_default if par in pares_opcoes]

            if not pares_selecionados and pares_opcoes:
                pares_selecionados = pares_opcoes[:1]

        elif (
            campo == "Movimento Relativo de Jogadores ao Longo do Tempo"
            and len(jogadores_disponiveis) >= 2
        ):
            jogador_rel_1 = st.session_state.get(
                "mov_rel_jogador_1", jogadores_disponiveis[0]
            )
            jogador_rel_2 = st.session_state.get(
                "mov_rel_jogador_2",
                jogadores_disponiveis[1 if len(jogadores_disponiveis) > 1 else 0],
            )
            janela_segundos = int(st.session_state.get("mov_rel_janela_segundos", 1))

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
                        key="heatmap",
                    )

                else:
                    st.warning(
                        "Selecione um jogador no campo, para verificar a sua posição durante o jogo!"
                    )

        # ── Campo ──────────────────────────────────────────────────

        col_map, col_info = st.columns([2.5, 1])

        # Campo
        with col_map:
            plot_placeholder = st.empty()
            slider_placeholder = st.empty()
            graph_placeholder = st.empty()

            timestamps_relativos = []
            timestamps_amostrados = []
            tempo_visualizacao = selected_time

            if campo == "Movimento Relativo de Jogadores ao Longo do Tempo":
                timestamps_relativos = sorted(
                    track_df["time_evento_s"].dropna().unique().tolist()
                )
                timestamps_amostrados = (
                    timestamps_relativos[::10] if timestamps_relativos else []
                )
                if (
                    timestamps_relativos
                    and timestamps_relativos[-1] not in timestamps_amostrados
                ):
                    timestamps_amostrados.append(timestamps_relativos[-1])

                if timestamps_amostrados:
                    tempo_relativo = st.session_state.get(
                        "mov_rel_slider", timestamps_amostrados[0]
                    )
                    if tempo_relativo not in timestamps_amostrados:
                        tempo_relativo = timestamps_amostrados[0]
                    tempo_visualizacao = tempo_relativo

            snapshot = track_df[
                (track_df["time_evento_s"] == tempo_visualizacao)
                & (track_df["x_tr"].notna())
                & (track_df["y_tr"].notna())
            ]
            jogadores_disponiveis = sorted(
                snapshot["atleta_id"].dropna().astype(str).unique().tolist()
            )

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
                    showlegend=False,
                )
            )

            # Campo Default
            if campo is None:
                # Capture clicks
                clicked = plot_placeholder.plotly_chart(
                    fig_pitch,
                    on_select="rerun",
                    width="stretch",
                    config={
                        "scrollZoom": False,
                        "responsive": True,
                        "displayModeBar": False,
                    },
                    key="main_field",
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

            # ── Distância entre Jogadores ──────────────────────────────────────────────────

            elif (
                campo == "Distância entre Jogadores"
                and len(jogadores_selecionados) >= 2
            ):
                coords_jogadores = {}
                for jogador_id in jogadores_selecionados:
                    jogador_data = snapshot[
                        snapshot["atleta_id"].astype(str) == str(jogador_id)
                    ]
                    if not jogador_data.empty:
                        coords_jogadores[str(jogador_id)] = tuple(
                            jogador_data[["x_tr", "y_tr"]].iloc[0]
                        )

                if coords_jogadores:
                    xs_sel = [
                        coords_jogadores[jogador_id][0]
                        for jogador_id in coords_jogadores
                    ]
                    ys_sel = [
                        coords_jogadores[jogador_id][1]
                        for jogador_id in coords_jogadores
                    ]

                    # Adicionar jogador selecionado com cor diferente
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=xs_sel,
                            y=ys_sel,
                            mode="markers",
                            marker=dict(
                                size=18,
                                color="#2ECC71",
                                line=dict(color="black", width=1.5),
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                for idx, par in enumerate(pares_selecionados):
                    jogador_1_id, jogador_2_id = par.split(" - ")
                    if (
                        jogador_1_id not in coords_jogadores
                        or jogador_2_id not in coords_jogadores
                    ):
                        continue

                    x1, y1 = coords_jogadores[jogador_1_id]
                    x2, y2 = coords_jogadores[jogador_2_id]

                    # Converter em metros reais
                    distancia_par = float(
                        np.hypot(
                            (x2 - x1) * (dist_x / PITCH_X),
                            (y2 - y1) * (dist_y / PITCH_Y),
                        )
                    )

                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    nome_cor, cor_par = cores_pares[idx % len(cores_pares)]

                    distancias_pares.append(
                        {
                            "Par": par,
                            "Cor": nome_cor,
                            "Distância": round(distancia_par, 2),
                        }
                    )

                    # Line between two points with distance label
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color=cor_par, width=2.5),
                            opacity=0.95,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    fig_pitch.add_annotation(
                        x=x_mid,
                        y=y_mid,
                        text=f"{distancia_par:.2f} m",
                        showarrow=False,
                        font=dict(color="black", size=10, family="Arial Black"),
                        bgcolor=cor_par,
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.95,
                    )

                    # Player ID annotations
                    for _, row in snapshot[
                        snapshot["atleta_id"].astype(str).isin(coords_jogadores.keys())
                    ].iterrows():
                        fig_pitch.add_annotation(
                            x=row["x_tr"],
                            y=row["y_tr"],
                            text=str(row["atleta_id"]),
                            showarrow=False,
                            font=dict(color="black", size=10, family="Arial Black"),
                            xanchor="center",
                            yanchor="middle",
                        )

            # ── Movimento Jogadores ──────────────────────────────────────────────────

            elif (
                campo == "Movimento Relativo de Jogadores ao Longo do Tempo"
                and jogador_rel_1 is not None
                and jogador_rel_2 is not None
                and jogador_rel_1 != jogador_rel_2
                and tempo_relativo is not None
            ):
                idx_tempo = (
                    timestamps_relativos.index(tempo_relativo)
                    if tempo_relativo in timestamps_relativos
                    else len(timestamps_relativos) - 1
                )
                janela_frames = max(1, int(janela_segundos) * 10)
                janela_timestamps = timestamps_relativos[
                    max(0, idx_tempo - janela_frames + 1) : idx_tempo + 1
                ]

                trilho_df = track_df.loc[
                    (track_df["time_evento_s"].isin(janela_timestamps))
                    & (
                        track_df["atleta_id"]
                        .astype(str)
                        .isin([str(jogador_rel_1), str(jogador_rel_2)])
                    )
                    & (track_df["x_tr"].notna())
                    & (track_df["y_tr"].notna())
                ].copy()

                jogador_1_trilho = trilho_df[
                    trilho_df["atleta_id"].astype(str) == str(jogador_rel_1)
                ]
                jogador_2_trilho = trilho_df[
                    trilho_df["atleta_id"].astype(str) == str(jogador_rel_2)
                ]

                if not jogador_1_trilho.empty:
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=jogador_1_trilho["x_tr"],
                            y=jogador_1_trilho["y_tr"],
                            mode="lines",
                            line=dict(color="#2ECC71", width=2.2),
                            opacity=0.95,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    if len(jogador_1_trilho) >= 2:
                        x_prev_1, y_prev_1 = jogador_1_trilho[["x_tr", "y_tr"]].iloc[-2]
                        x_last_1, y_last_1 = jogador_1_trilho[["x_tr", "y_tr"]].iloc[-1]
                        fig_pitch.add_annotation(
                            x=x_last_1,
                            y=y_last_1,
                            ax=x_prev_1,
                            ay=y_prev_1,
                            axref="x",
                            ayref="y",
                            xref="x",
                            yref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.2,
                            arrowwidth=2.2,
                            arrowcolor="#2ECC71",
                            text="",
                        )

                if not jogador_2_trilho.empty:
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=jogador_2_trilho["x_tr"],
                            y=jogador_2_trilho["y_tr"],
                            mode="lines",
                            line=dict(color="#F2F2F2", width=2.2),
                            opacity=0.95,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    if len(jogador_2_trilho) >= 2:
                        x_prev_2, y_prev_2 = jogador_2_trilho[["x_tr", "y_tr"]].iloc[-2]
                        x_last_2, y_last_2 = jogador_2_trilho[["x_tr", "y_tr"]].iloc[-1]
                        fig_pitch.add_annotation(
                            x=x_last_2,
                            y=y_last_2,
                            ax=x_prev_2,
                            ay=y_prev_2,
                            axref="x",
                            ayref="y",
                            xref="x",
                            yref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.2,
                            arrowwidth=2.2,
                            arrowcolor="#F2F2F2",
                            text="",
                        )

                snapshot_rel = trilho_df.loc[
                    trilho_df["time_evento_s"] == tempo_relativo
                ].copy()
                jogador_1_frame = snapshot_rel[
                    snapshot_rel["atleta_id"].astype(str) == str(jogador_rel_1)
                ]
                jogador_2_frame = snapshot_rel[
                    snapshot_rel["atleta_id"].astype(str) == str(jogador_rel_2)
                ]

                jogador_1_serie = (
                    trilho_df.loc[
                        trilho_df["atleta_id"].astype(str) == str(jogador_rel_1),
                        ["time_evento_s", "x_tr", "y_tr"],
                    ]
                    .drop_duplicates(subset=["time_evento_s"])
                    .rename(columns={"x_tr": "x_1", "y_tr": "y_1"})
                )
                jogador_2_serie = (
                    trilho_df.loc[
                        trilho_df["atleta_id"].astype(str) == str(jogador_rel_2),
                        ["time_evento_s", "x_tr", "y_tr"],
                    ]
                    .drop_duplicates(subset=["time_evento_s"])
                    .rename(columns={"x_tr": "x_2", "y_tr": "y_2"})
                )
                serie_distancias_rel = pd.merge(
                    jogador_1_serie, jogador_2_serie, on="time_evento_s", how="inner"
                ).sort_values("time_evento_s")

                if not serie_distancias_rel.empty:
                    serie_distancias_rel["distancia_m"] = np.hypot(
                        (serie_distancias_rel["x_2"] - serie_distancias_rel["x_1"])
                        * (dist_x / PITCH_X),
                        (serie_distancias_rel["y_2"] - serie_distancias_rel["y_1"])
                        * (dist_y / PITCH_Y),
                    )
                    serie_distancias_rel["tempo_label"] = serie_distancias_rel[
                        "time_evento_s"
                    ].map(converter_para_relogio_fpf)

                    distancia_media_rel = float(
                        serie_distancias_rel["distancia_m"].mean()
                    )
                    distancia_min_rel = float(serie_distancias_rel["distancia_m"].min())
                    distancia_max_rel = float(serie_distancias_rel["distancia_m"].max())

                    distancia_inicial = float(
                        serie_distancias_rel["distancia_m"].iloc[0]
                    )
                    distancia_final = float(
                        serie_distancias_rel["distancia_m"].iloc[-1]
                    )
                    delta_distancia_rel = distancia_final - distancia_inicial

                    if delta_distancia_rel <= -1:
                        tendencia_rel = "Aproximação"
                    elif delta_distancia_rel >= 1:
                        tendencia_rel = "Afastamento"
                    else:
                        tendencia_rel = "Estável"

                if not jogador_1_frame.empty and not jogador_2_frame.empty:
                    x1, y1 = jogador_1_frame[["x_tr", "y_tr"]].iloc[0]
                    x2, y2 = jogador_2_frame[["x_tr", "y_tr"]].iloc[0]
                    distancia_rel = float(
                        np.hypot(
                            (x2 - x1) * (dist_x / PITCH_X),
                            (y2 - y1) * (dist_y / PITCH_Y),
                        )
                    )

                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2

                    # Dots
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=[x1],
                            y=[y1],
                            mode="markers",
                            marker=dict(
                                size=18,
                                color="#2ECC71",
                                line=dict(color="white", width=1.5),
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=[x2],
                            y=[y2],
                            mode="markers",
                            marker=dict(
                                size=18,
                                color="#F2F2F2",
                                line=dict(color="black", width=1.5),
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    # Line between players
                    fig_pitch.add_trace(
                        go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color="#FFC857", width=2.5),
                            opacity=0.95,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    # Player ID labels
                    for x, y, label in [
                        (x1, y1, str(jogador_rel_1)),
                        (x2, y2, str(jogador_rel_2)),
                    ]:
                        fig_pitch.add_annotation(
                            x=x,
                            y=y,
                            text=label,
                            showarrow=False,
                            font=dict(color="black", size=10, family="Arial Black"),
                            xanchor="center",
                            yanchor="middle",
                        )

                    # Distance label
                    fig_pitch.add_annotation(
                        x=x_mid,
                        y=y_mid,
                        text=f"{distancia_rel:.2f} m",
                        showarrow=False,
                        font=dict(color="black", size=10, family="Arial Black"),
                        bgcolor="#FFC857",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.95,
                    )

            # ── RENDERIZAR CAMPOS ───────────────────────────────────────────────────────────

            if campo is not None or campo is None:
                plot_placeholder.plotly_chart(
                    fig_pitch,
                    config={
                        "scrollZoom": False,
                        "responsive": True,
                        "displayModeBar": False,
                    },
                    key=f"pitch_{campo}",
                )

            # ── Grafico Distancia ───────────────────────────────────────────────────────────

            if (
                campo == "Movimento Relativo de Jogadores ao Longo do Tempo"
                and timestamps_amostrados
            ):
                tempo_relativo = slider_placeholder.select_slider(
                    "Momento do Movimento (1 em 1 segundo)",
                    options=timestamps_amostrados,
                    value=tempo_relativo
                    if tempo_relativo in timestamps_amostrados
                    else timestamps_amostrados[0],
                    format_func=converter_para_relogio_fpf,
                    key="mov_rel_slider",
                )

            if (
                campo == "Movimento Relativo de Jogadores ao Longo do Tempo"
                and not serie_distancias_rel.empty
            ):
                tick_positions = serie_distancias_rel["time_evento_s"].tolist()
                tick_labels = serie_distancias_rel["tempo_label"].tolist()
                step_ticks = max(1, len(tick_positions) // 5)

                fig_dist = go.Figure()

                # Main line
                fig_dist.add_trace(
                    go.Scatter(
                        x=serie_distancias_rel["time_evento_s"],
                        y=serie_distancias_rel["distancia_m"],
                        mode="lines",
                        line=dict(color="#E30613", width=2.2),
                        customdata=serie_distancias_rel[["tempo_label"]].values,
                        hovertemplate=(
                            "<b>Tempo %{customdata[0]}</b><br>"
                            "Distância: <b>%{y:.2f} m</b>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                # Current-moment dot
                fig_dist.add_trace(
                    go.Scatter(
                        x=[serie_distancias_rel["time_evento_s"].iloc[-1]],
                        y=[serie_distancias_rel["distancia_m"].iloc[-1]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="#FFC857",
                            line=dict(color="black", width=1.5),
                        ),
                        customdata=[[serie_distancias_rel["tempo_label"].iloc[-1]]],
                        hovertemplate=(
                            "<span style='color:#FFC857'><b>⏱ Momento Atual</b></span><br><br>"
                            "<b>Tempo %{customdata[0]}</b><br>"
                            "Distância: <b>%{y:.2f} m</b>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                # Mean dashed line
                fig_dist.add_hline(
                    y=serie_distancias_rel["distancia_m"].mean(),
                    line=dict(color="#F2F2F2", width=1.3, dash="dash"),
                )

                fig_dist.update_layout(
                    title=dict(
                        text="Distância entre jogadores ao longo da janela",
                        font=dict(color="white", size=11),
                    ),
                    paper_bgcolor="#1e1e1e",
                    plot_bgcolor="#1e1e1e",
                    height=260,
                    margin=dict(l=50, r=20, t=40, b=40),
                    xaxis=dict(
                        tickvals=tick_positions[::step_ticks],
                        ticktext=tick_labels[::step_ticks],
                        tickfont=dict(color="white", size=9),
                        title=dict(text="Tempo", font=dict(color="white")),
                        gridcolor="#444444",
                        linecolor="#666666",
                    ),
                    yaxis=dict(
                        tickfont=dict(color="white", size=9),
                        title=dict(text="Distância (m)", font=dict(color="white")),
                        gridcolor="#444444",
                        griddash="dot",
                        linecolor="#666666",
                    ),
                )

                graph_placeholder.plotly_chart(
                    fig_dist, use_container_width=True, key="distancia_linhas"
                )

        # ── Frame Info ──────────────────────────────────────────────────
        with col_info:
            if campo == "Distância entre Jogadores":
                st.subheader("Selecionar Jogadores")

                if len(jogadores_disponiveis) >= 2:
                    jogadores_escolhidos = st.multiselect(
                        "Jogadores (Máx. 4)",
                        options=jogadores_disponiveis,
                        default=jogadores_selecionados,
                        key="dist_jogadores_selecionados",
                        max_selections=4,
                    )

                    if len(jogadores_escolhidos) < 2:
                        st.warning("Seleciona pelo menos 2 jogadores.")
                    else:
                        pares_widget_opcoes = [
                            f"{jogador_1} - {jogador_2}"
                            for jogador_1, jogador_2 in combinations(
                                jogadores_escolhidos, 2
                            )
                        ]
                        pares_default_widget = [
                            par
                            for par in pares_selecionados
                            if par in pares_widget_opcoes
                        ]
                        if not pares_default_widget and pares_widget_opcoes:
                            pares_default_widget = pares_widget_opcoes[:1]

                        st.multiselect(
                            "Pares",
                            options=pares_widget_opcoes,
                            default=pares_default_widget,
                            key="dist_pares_selecionados",
                        )
                else:
                    st.info(
                        "São necessários pelo menos dois jogadores no frame para esta visualização."
                    )

                st.divider()

            elif campo == "Movimento Relativo de Jogadores ao Longo do Tempo":
                st.subheader("Selecionar Jogadores")

                if len(jogadores_disponiveis) >= 2:
                    jogador_rel_1 = st.selectbox(
                        "Jogador 1",
                        options=jogadores_disponiveis,
                        index=0,
                        key="mov_rel_jogador_1",
                    )
                    jogador_rel_2 = st.selectbox(
                        "Jogador 2",
                        options=jogadores_disponiveis,
                        index=1 if len(jogadores_disponiveis) > 1 else 0,
                        key="mov_rel_jogador_2",
                    )

                    if jogador_rel_1 == jogador_rel_2:
                        st.warning("Seleciona dois jogadores diferentes.")

                    st.selectbox(
                        "Janela Temporal",
                        options=[1, 2, 3, 5, 10],
                        index=[1, 2, 3, 5, 10].index(janela_segundos)
                        if janela_segundos in [1, 2, 3, 5, 10]
                        else 0,
                        key="mov_rel_janela_segundos",
                        format_func=lambda valor: f"{valor}s",
                    )
                else:
                    st.info(
                        "São necessários pelo menos dois jogadores no frame para esta visualização."
                    )

                st.divider()

            st.subheader("Analise de Frame")
            st.metric(
                "Tempo Selecionado", f"{converter_para_relogio_fpf(selected_time)}s"
            )
            if campo == "Distância entre Jogadores":
                if distancias_pares:
                    if len(distancias_pares) == 1:
                        st.metric(
                            "Distância entre Jogadores",
                            f"{distancias_pares[0]['Distância']:.2f} m",
                        )
                    else:
                        df_distancias = pd.DataFrame(distancias_pares)
                        df_distancias["Distância"] = df_distancias["Distância"].map(
                            lambda valor: f"{valor:.2f} m"
                        )
                        st.dataframe(df_distancias, width="stretch", hide_index=True)
                else:
                    st.metric("Distância entre Jogadores", "N/A")
            elif campo == "Movimento Relativo de Jogadores ao Longo do Tempo":
                if tempo_relativo is not None:
                    st.metric(
                        "Tempo do Movimento",
                        f"{converter_para_relogio_fpf(tempo_relativo)}s",
                    )
                else:
                    st.metric("Tempo do Movimento", "N/A")

                if distancia_rel is not None:
                    st.metric("Distância Atual", f"{distancia_rel:.2f} m")
                else:
                    st.metric("Distância Atual", "N/A")

                if distancia_media_rel is not None:
                    st.metric("Distância Média", f"{distancia_media_rel:.2f} m")
                else:
                    st.metric("Distância Média", "N/A")

                if distancia_min_rel is not None:
                    st.metric("Distância Mínima", f"{distancia_min_rel:.2f} m")
                else:
                    st.metric("Distância Mínima", "N/A")

                if distancia_max_rel is not None:
                    st.metric("Distância Máxima", f"{distancia_max_rel:.2f} m")
                else:
                    st.metric("Distância Máxima", "N/A")

                if delta_distancia_rel is not None:
                    st.metric(
                        f"Variação em {janela_segundos}s",
                        f"{delta_distancia_rel:+.2f} m",
                        delta=tendencia_rel,
                    )
                else:
                    st.metric(f"Variação em {janela_segundos}s", "N/A")

                if tendencia_rel is not None:
                    st.metric("Tendência", tendencia_rel)
                else:
                    st.metric("Tendência", "N/A")
            else:
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
