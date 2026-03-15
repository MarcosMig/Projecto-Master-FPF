import pandas as pd
import numpy as np
import streamlit as st
import mplsoccer as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
import plotly.graph_objects as go
import numpy as np

from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES
from fpf_modules.metrics import calcular_area
import fpf_modules.visualization as visual


# Estilo para as métricas de partida
st.markdown("""
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
    """, unsafe_allow_html=True)

def kpi_card(label, value):
    st.markdown(f"""
        <div class="fpt-kpi-card">
            <div class="fpt-kpi-label">{label}</div>
            <div class="fpt-kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="FPF | Positional Analysis", layout="wide")

# Incializar Paineis
tab_metrics, tab_visual = st.tabs(["📊 Métricas de Performance", "📍 Análise Posicional"])

# --- CARREGAMENTO E PROCESSAMENTO ---
@st.cache_data
def load_and_merge_data():
    perf = pd.read_parquet(f"{CLEANDATA_DIR}/performance_metrics.parquet")
    tracking = pd.read_parquet(f"{CLEANDATA_DIR}/tracking.parquet")
    sessions = pd.read_parquet(f"{CLEANDATA_DIR}/sessions.parquet")

    # Merge de metadados no tracking
    cols_meta = ['data', 'selecao', 'contexto', 'jogo', 'session_sk', 'dist_x', 'dist_y']
    df_merged = pd.merge(
        tracking,
        sessions[cols_meta],
        how='left',
        on='session_sk'
    )
    return perf, df_merged

try:
    df_perf, tracking_df = load_and_merge_data()
except Exception as e:
    st.error(f"Erro ao processar dados: {e}")
    st.stop()

@st.cache_data
def calcular_compactacao(tracking_df, dist_x, dist_y, pitch_x=120, pitch_y=80):
    """Calcula compactação Vertical e Horizontal por Frame, converte de campo (default StatsBomb) para metros.

    Args:
        tracking_df (DataFrame): DataFrame que contem tracking data.
        dist_x (Float): Comprimento original do campo
        dist_y (Float): Largura original do campo.
        pitch_x (Integer): Comprimento do campo, default 120 (Statsbomb)
        pitch_y (Integer): Largura do campo, default 80 (Statsbomb)
    Returns:
        pd.DataFrame: DataFrame com a compactação vertical e horizontal por frame em metros.
    """
    if 'time_evento_s' in tracking_df.columns:

        # Agrupar por tempo
        frame_data = tracking_df.groupby('time_evento_s').agg(
            x_min=('x_tr', 'min'),
            x_max=('x_tr', 'max'),
            y_min=('y_tr', 'min'),
            y_max=('y_tr', 'max')
        )

        frame_data['comp_vertical'] = round( (frame_data['x_max'] - frame_data['x_min']) * (pitch_x / dist_x), 2)
        frame_data['comp_horizontal'] = round( (frame_data['y_max'] - frame_data['y_min']) * (pitch_y / dist_y), 2)

        frame_data = frame_data.drop(columns={'x_min', 'x_max', 'y_min', 'y_max'}).reset_index()

        return frame_data


@st.cache_data
def calcular_area_media(tracking_df, dist_x, dist_y, pitch_x=120, pitch_y=80):
    """Calcula a area ocupada média da partida, convertendo de campo (default StatsBomb) em m², usando as distancias originais do campo.

    Args:
        tracking_df (DataFrame): DataFrame que contem tracking data.
        dist_x (Float): Comprimento original do campo
        dist_y (Float): Largura original do campo.
        pitch_x (Integer): Comprimento do campo, default 120 (Statsbomb)
        pitch_y (Integer): Largura do campo, default 80 (Statsbomb)

    Returns:
        Dictionary: Média e Mediana da Área Ocupada.
    """
    df = tracking_df.copy()

    fator_conversao = (dist_x * dist_y) / (pitch_x * pitch_y)
    areas = []

    for time, frame in df.groupby('time_evento_s'):
        pts = frame[['x_tr', 'y_tr']].dropna().values

        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                areas.append(
                    {
                        'time_evento_s':time,
                        'area': round(hull.volume * fator_conversao, 2)
                     }
                )

            except QhullError:
                continue

    areas_df = pd.DataFrame(areas)

    return areas_df if not areas_df.empty else None

# --- SIDEBAR: FILTROS E CONTROLES ---
def converter_para_relogio_fpf(segundos_totais):
    """
    Exemplo: 4150.3s -> '69:10.3'
    (Minuto 69, Segundo 10, Frame 3)
    """
    minutos = int(segundos_totais // 60)
    segundos = int(segundos_totais % 60)
    frame = int(round((segundos_totais % 1) * 10))
    if frame == 10: frame = 0; segundos += 1 # Ajuste de arredondamento

    return f"{minutos:02d}:{segundos:02d}.{frame}"

with st.sidebar:
    st.title("⚽ Filtros de Sessão")

    selecao = st.selectbox("Seleção", options=SELECOES_OPCOES)
    contexto = st.selectbox("Contexto", options=["Treino", "Jogo"])

    # Inicializamos o jogo como vazio por defeito
    jogo = ""

    if contexto == 'Jogo':

        # Filtramos a lista de jogos disponíveis para esta seleção
        jogos_disponiveis = df_perf[
            (df_perf.selecao == selecao) &
            (df_perf.contexto == "Jogo")
        ]['jogo'].unique()

        if len(jogos_disponiveis) > 0:
            jogo = st.selectbox('Adversário', options=jogos_disponiveis)
        else:
            st.warning("Nenhum jogo registado para esta seleção.")

    # Filtrar sessões disponíveis para o slider ou seleção
    sessoes_disponiveis = df_perf.loc[
        (df_perf.selecao == selecao)
        & (df_perf.contexto == contexto)
        & (df_perf.jogo == jogo)
    ]

    st.divider()

    st.sidebar.header("⏱️ Navegação Temporal")

    fase_selected = st.sidebar.radio("Fase", ["Warm-Up", "1P", "2P"], index=1)

    # Filtrar tracking pela fase para pegar os timestamps
    df_fase = tracking_df.loc[
        (tracking_df.selecao == selecao)
        & (tracking_df.contexto == contexto)
        & (tracking_df.jogo == jogo)
        & (tracking_df.fase == fase_selected)
    ]

    # Obter dimensões originais do campo
    dist_x = df_fase['dist_x'].unique()[0].squeeze()
    dist_y = df_fase['dist_y'].unique()[0].squeeze()

    if not df_fase.empty:
        timestamps = sorted(df_fase['time_evento_s'].unique())
        # Slider para navegar no tempo
        selected_time = st.sidebar.select_slider(
            "Momento do Jogo (s)",
            options=timestamps,
         format_func=converter_para_relogio_fpf
        )

    else:
        st.sidebar.warning(f"Sem dados de tracking para a fase {fase_selected}")
        selected_time = None

# --- PAINEL PRINCIPAL ---


# TAB 1: MÉTRICAS

with tab_metrics:

    st.subheader(f"Métricas: {selecao} | {contexto}")
    if sessoes_disponiveis.empty:
        st.warning("Nenhuma métrica encontrada para estes filtros.")
    else:
        st.dataframe(sessoes_disponiveis, width='stretch', hide_index=True)

# TAB 2: VISUALIZAÇÃO DO CAMPO

with tab_visual:

    st.subheader('Métricas da Partida')

    if fase_selected != 'Warm-Up':

        df_compactacao = calcular_compactacao(
            df_fase,
            dist_x,
            dist_y
            )

        df_area = calcular_area_media(
            df_fase,
            dist_x,
            dist_y
            )

        if not df_compactacao.empty:

            avg_comp_vert = df_compactacao['comp_vertical'].mean().round(2)
            avg_comp_hor = df_compactacao['comp_horizontal'].mean().round(2)
            avg_area = df_area['area'].mean().round(2)

            col1, col2, col3, col4 = st.columns([1.2,1.2,1,1])

            with col1:
                kpi_card("COMPACTAÇÃO VERTICAL (m)", f"{avg_comp_vert:.1f}")

            with col2:
                kpi_card("COMPACTAÇÃO HORIZONTAL (m)", f"{avg_comp_hor:.1f}")

            with col3:
                kpi_card("ÁREA OCUPADA (m²)", f"{avg_area:.1f}")

            st.divider()

    else:
        st.info('Métricas calculadas apenas para fase de jogo!')


    if selected_time is not None:

        # 1. Obter snapshot
        snapshot = df_fase[
            (df_fase['time_evento_s'] == selected_time)
            & (df_fase['x_tr'].notna())
            & (df_fase['y_tr'].notna())
        ]

        if fase_selected != 'Warm-Up':
            # Obter metricas para o frame
            compactacao_frame = df_compactacao.loc[ df_compactacao['time_evento_s'] == selected_time]
            area_frame = df_area.loc[ df_area['time_evento_s'] == selected_time]

        col1, col2 = st.columns(2)

        with col1:
            campo = st.selectbox(label='Visualização Campo', options=['Convex Hull'],index=None, placeholder='Seleciona outro metodo de visualizar o campo')

        if 'selected_player' not in st.session_state:
            st.session_state.selected_player = None

        with col2:
                st.markdown('<p style="font-size:14px; margin-bottom:4px;">Visualizar Heatmap do Jogador</p>', unsafe_allow_html=True)

                texto_popup = 'Selecione um jogador no campo, para verificar o seu heatmap!'

                if st.session_state.selected_player:
                        player_name = st.session_state.selected_player
                        texto_popup = f'Jogador selecionado: {player_name}'

                with st.popover(texto_popup, width=500):

                    if st.session_state.selected_player:

                        # Filter that player's tracking data across all frames
                        player_frames = tracking_df[tracking_df['atleta_id'] == player_name]

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

                        # Cores do Heatmap
                        custom_hot = [
                            [0.0,  "rgba(34, 49, 43, 0)"],      # transparente — campo limpo
                            [0.08, "rgba(34, 49, 43, 0.5)"],    # verde campo — densidade muito baixa
                            [0.2,  "rgba(80, 80, 20, 0.8)"],    # transição rápida para quente
                            [0.35, "rgba(180, 160, 0, 0.9)"],   # amarelo escuro
                            [0.5,  "rgba(240, 220, 0, 1.0)"],   # amarelo brilhante
                            [0.65, "rgba(255, 140, 0, 1.0)"],   # laranja
                            [0.8,  "rgba(220, 50, 0, 1.0)"],    # vermelho alaranjado
                            [1.0,  "rgba(120, 0, 0, 1.0)"],     # vermelho escuro
                        ]

                        # Calcular Histograma de Posição
                        fig_heat.add_trace(go.Histogram2dContour(
                            x=player_frames['x_tr'],
                            y=player_frames['y_tr'],
                            colorscale=custom_hot,
                            reversescale=False,
                            showscale=False,
                            ncontours=20,
                            opacity=1,
                            contours=dict(
                                coloring='fill',
                            ),
                            line=dict(width=0),
                        ))


                        # Render Heatmap
                        st.plotly_chart(
                            fig_heat,
                            config = {
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'staticPlot': True,
                            },
                            use_container_width=True
                        )

                    else:
                        st.warning('Selecione um jogador no campo, para verificar a sua posição durante o jogo!')

        col_map, col_info = st.columns([3, 1])

        # CAMPO
        with col_map:

            # --- Main pitch ---
            fig_pitch = go.Figure()

            # Configurar Campo
            fig_pitch.update_layout(
                shapes=visual.draw_statsbomb_pitch_horizontal(),
                plot_bgcolor="#22312b",
                paper_bgcolor="#22312b",
                xaxis=dict(range=[-2, 122], visible=False, fixedrange=True),
                yaxis=dict(range=[-2, 82], visible=False, scaleanchor="x", fixedrange=True),
                margin=dict(l=0, r=0, t=0, b=0),
                height=450,
                dragmode=False,
            )

            # Desenhar Jogadores
            fig_pitch.add_trace(go.Scatter(
                x=snapshot['x_tr'],
                y=snapshot['y_tr'],
                mode='markers',
                marker=dict(size=14, color='red', line=dict(color='white', width=1)),
                customdata=snapshot[['atleta_id']].values,
                hovertemplate="<b>Jogador: %{customdata[0]}</b><extra></extra>"
            ))

            # Campo Default
            if campo is None:
                # Capture clicks
                clicked = st.plotly_chart(
                    fig_pitch,
                    on_select='rerun',
                    config = {
                        'scrollZoom': False,
                        'responsive': True,
                        'displayModeBar': False,
                    }
                )

                if clicked and clicked['selection']['points']:
                    st.session_state.selected_player = clicked['selection']['points'][0]['customdata'][0]

            # Calculo do Convex Hull
            area = calcular_area(
                snapshot,
                df_fase['dist_x'].unique()[0].squeeze(),
                df_fase['dist_y'].unique()[0].squeeze()
            )

            # Visualizar Convex Hull
            if campo == 'Convex Hull':
                try:
                    pts = snapshot[['x_tr', 'y_tr']].dropna().values

                    hull = ConvexHull(pts)

                    # Get hull vertices in order and close the polygon by repeating the first point
                    hull_pts = pts[hull.vertices]
                    hull_pts_closed = np.vstack([hull_pts, hull_pts[0]])

                    fig_pitch.add_trace(go.Scatter(
                        x=hull_pts_closed[:, 0],
                        y=hull_pts_closed[:, 1],
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(227, 6, 19, 0.3)',
                        line=dict(color='#E30613', width=2),
                        hoverinfo='skip',
                        showlegend=False
                    ))

                    fig_pitch.update_layout(showlegend=False)

                except QhullError:
                    pass

                st.plotly_chart(
                    fig_pitch,
                    config = {
                        'scrollZoom': False,
                        'responsive': True,
                        'displayModeBar': False,
                    #    'staticPlot': True
                    }
                )

        # INFO FRAME
        with col_info:

            st.subheader("Analise de Frame")
            st.metric("Tempo Selecionado", f"{converter_para_relogio_fpf(selected_time)}s")

            if fase_selected != 'Warm-Up':
                st.metric("Compactação Vertical (m)", compactacao_frame['comp_vertical'])
                st.metric("Compactação Horizontal (m)", compactacao_frame['comp_horizontal'])
                st.metric("Área Ocupada (m²)", area_frame['area'])
    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")


# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
