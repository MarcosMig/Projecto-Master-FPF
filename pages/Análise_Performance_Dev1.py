import pandas as pd
import numpy as np
import streamlit as st
import mplsoccer as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES
from scipy.spatial import ConvexHull, QhullError

# TODO 1. Converter métricas em m2
# TODO 2. Alterar Heat map posicional de plotly para mplsoccer

# Estilo para as métricas
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
        border-left: 5px solid #E30613; /* Linha vermelha FPF */
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        flex: 1;
    }
    .fpt-kpi-label {
        color: #9aa0a6;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
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
    cols_meta = ['data', 'selecao', 'contexto', 'jogo', 'session_sk']
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
def calcular_compactacao(tracking_df):
    """Calcula compactação Vertical e Horizontal por Frame.

    Args:
        tracking_df (_type_): DataFrame que contem tracking data.

    Returns:
        pd.DataFrame: DataFrame com a compactação vertical e horizontal por frame.
    """
    if 'time_evento_s' in tracking_df.columns:

        # Agrupar por tempo
        frame_data = tracking_df.groupby('time_evento_s').agg(
            x_min=('x_tr', 'min'),
            x_max=('x_tr', 'max'),
            y_min=('y_tr', 'min'),
            y_max=('y_tr', 'max')
        )

        frame_data['comp_vertical'] = round( frame_data['x_max'] - frame_data['x_min'], 2)
        frame_data['comp_horizontal'] = round( frame_data['y_max'] - frame_data['y_min'], 2)

        frame_data = frame_data.drop(columns={'x_min', 'x_max', 'y_min', 'y_max'}).reset_index()

        return frame_data


@st.cache_data
def calcular_area_media(tracking_df, dist_x, dist_y):
    df = tracking_df.copy()

   # fator_conversao = (dist_x * dist_y) / (120 * 80)  # StatsBomb → m²
    areas = []

    for _, frame in df.groupby('time'):
        pts = frame[['x_tr', 'y_tr']].dropna().values

        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                areas.append(hull.volume) # * fator_conversao)
            except QhullError:
                continue

    return {
        'mean': round(np.mean(areas), 2),
        'median': round(np.median(areas), 2),
        'std': round(np.std(areas), 2)
    } if areas else None

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

    campo_visualizacao = st.session_state.get('campo_visualizacao')

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
        & (tracking_df['fase'] == fase_selected)
    ]

    if not df_fase.empty:
        timestamps = sorted(df_fase['time_evento_s'].unique())
        if campo_visualizacao == 'Movimento Relativo de Jogadores ao Longo do Tempo':
            selected_time = st.session_state.get('selected_time_jogo', timestamps[0])
        else:
            # Slider para navegar no tempo
            selected_time = st.sidebar.select_slider(
                "Momento do Jogo (s)",
                options=timestamps,
                key='selected_time_jogo',
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
    campo_visualizacao = st.session_state.get('campo_visualizacao')
    ocultar_metricas_partida = campo_visualizacao in [
        'Distância entre Jogadores',
        'Movimento Relativo de Jogadores ao Longo do Tempo',
    ]

    if not ocultar_metricas_partida:
        st.subheader('Métricas da Partida')

    df_compactacao = calcular_compactacao(df_fase)
    area_dict = calcular_area_media(df_fase, 0, 0)

    if not ocultar_metricas_partida and not df_compactacao.empty:

        avg_comp_vert = df_compactacao['comp_vertical'].mean().round(2)
        avg_comp_hor = df_compactacao['comp_horizontal'].mean().round(2)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            kpi_card("Compactação Vertical", f"{avg_comp_vert:.1f}")

        with col2:
            kpi_card("Compactação Horizontal", f"{avg_comp_hor:.1f}")

        with col3:
            kpi_card("Área Ocupada", f"{area_dict['median']:.1f}")

        st.divider()


    if selected_time is not None:
        snapshot = df_fase[
            (df_fase['time_evento_s'] == selected_time)
            & (df_fase['x_tr'].notna())
            & (df_fase['y_tr'].notna())
        ]
        jogadores_disponiveis = sorted(snapshot['atleta_id'].dropna().astype(str).unique().tolist())

        # Obter metricas para o frame
        compactacao_frame = df_compactacao.loc[ df_compactacao['time_evento_s'] == selected_time]

        campo = st.selectbox(label='Visualização Campo', options=['Convex Hull', 'Distância entre Jogadores', 'Movimento Relativo de Jogadores ao Longo do Tempo', 'Teste'],index=None, placeholder='Seleciona outro metodo de visualizar o campo', key='campo_visualizacao')

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
            ('Amarelo', '#FFD166'),
            ('Verde', '#06D6A0'),
            ('Azul Claro', '#4CC9F0'),
            ('Rosa', '#EF476F'),
            ('Laranja', '#F77F00'),
            ('Verde Lima', '#90BE6D'),
        ]
        if campo == 'Distância entre Jogadores' and len(jogadores_disponiveis) >= 2:
            jogadores_default = st.session_state.get('dist_jogadores_selecionados', jogadores_disponiveis[:2])
            jogadores_selecionados = [
                str(jogador) for jogador in jogadores_default
                if str(jogador) in jogadores_disponiveis
            ][:4]

            if len(jogadores_selecionados) < 2:
                jogadores_selecionados = jogadores_disponiveis[:2]

            pares_opcoes = [
                f"{jogador_1} - {jogador_2}"
                for jogador_1, jogador_2 in combinations(jogadores_selecionados, 2)
            ]
            pares_default = st.session_state.get('dist_pares_selecionados', pares_opcoes[:1])
            pares_selecionados = [par for par in pares_default if par in pares_opcoes]

            if not pares_selecionados and pares_opcoes:
                pares_selecionados = pares_opcoes[:1]

        elif campo == 'Movimento Relativo de Jogadores ao Longo do Tempo' and len(jogadores_disponiveis) >= 2:
            jogador_rel_1 = st.session_state.get('mov_rel_jogador_1', jogadores_disponiveis[0])
            jogador_rel_2 = st.session_state.get('mov_rel_jogador_2', jogadores_disponiveis[1 if len(jogadores_disponiveis) > 1 else 0])
            janela_segundos = int(st.session_state.get('mov_rel_janela_segundos', 1))

        col_map, col_info = st.columns([3, 1])

        # CAMPO
        with col_map:
            plot_placeholder = st.empty()
            slider_placeholder = st.empty()
            graph_placeholder = st.empty()

            timestamps_relativos = []
            timestamps_amostrados = []
            tempo_visualizacao = selected_time

            if campo == 'Movimento Relativo de Jogadores ao Longo do Tempo':
                timestamps_relativos = sorted(df_fase['time_evento_s'].dropna().unique().tolist())
                timestamps_amostrados = timestamps_relativos[::10] if timestamps_relativos else []
                if timestamps_relativos and timestamps_relativos[-1] not in timestamps_amostrados:
                    timestamps_amostrados.append(timestamps_relativos[-1])

                if timestamps_amostrados:
                    tempo_relativo = st.session_state.get('mov_rel_slider', timestamps_amostrados[0])
                    if tempo_relativo not in timestamps_amostrados:
                        tempo_relativo = timestamps_amostrados[0]
                    tempo_visualizacao = tempo_relativo

            snapshot = df_fase[
                (df_fase['time_evento_s'] == tempo_visualizacao)
                & (df_fase['x_tr'].notna())
                & (df_fase['y_tr'].notna())
            ]
            jogadores_disponiveis = sorted(snapshot['atleta_id'].dropna().astype(str).unique().tolist())

            # 2. Configurar Pitch
            pitch = mpl.Pitch(
                pitch_type='statsbomb', # Se escalou para 120x80
                pitch_color='#22312b',
                line_color='#c7d5cc'
            )

            fig, ax = pitch.draw(figsize=(10, 7))

            # 3. Desenhar Jogadores
            pitch.scatter(
                snapshot.x_tr,
                snapshot.y_tr,
                s=400, c='#E30613', edgecolors='white', linewidth=1, alpha=0.9, ax=ax
            )

            # 4. Adicionar IDs dos Atletas (Labels)
            for i, row in snapshot.iterrows():
                pitch.annotate(
                    row['atleta_id'],
                    (row['x_tr'], row['y_tr']),
                    ax=ax, color='white', fontsize=10, fontweight='bold',
                    va='center', ha='center'
                )

            # 5. Calculo do Convex Hull (inicializamos antes para calculo da area por frame)
            convex_hull = pitch.convexhull(
                    snapshot.x_tr,
                    snapshot.y_tr,
            )

            # Calculo da area
            # Convex Hull Shape is (1, n, 2) — index accordingly
            vertices = convex_hull[0]  # shape (n, 2)
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # Visualizar Convex Hull
            if campo == 'Convex Hull':

                polygon = pitch.polygon(
                    convex_hull, ax=ax,
                    edgecolor='E30613',
                    color='#E30613', alpha=0.3
                )

            elif campo == 'Distância entre Jogadores' and len(jogadores_selecionados) >= 2:
                coords_jogadores = {}
                for jogador_id in jogadores_selecionados:
                    jogador_data = snapshot[snapshot['atleta_id'].astype(str) == str(jogador_id)]
                    if not jogador_data.empty:
                        coords_jogadores[str(jogador_id)] = tuple(jogador_data[['x_tr', 'y_tr']].iloc[0])

                if coords_jogadores:
                    xs_sel = [coords_jogadores[jogador_id][0] for jogador_id in coords_jogadores]
                    ys_sel = [coords_jogadores[jogador_id][1] for jogador_id in coords_jogadores]
                    pitch.scatter(
                        xs_sel,
                        ys_sel,
                        s=500,
                        c='#2ECC71',
                        edgecolors='black',
                        linewidth=1.5,
                        ax=ax,
                        zorder=3,
                    )

                for idx, par in enumerate(pares_selecionados):
                    jogador_1_id, jogador_2_id = par.split(' - ')
                    if jogador_1_id not in coords_jogadores or jogador_2_id not in coords_jogadores:
                        continue

                    x1, y1 = coords_jogadores[jogador_1_id]
                    x2, y2 = coords_jogadores[jogador_2_id]
                    distancia_par = float(np.hypot(x2 - x1, y2 - y1))
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    nome_cor, cor_par = cores_pares[idx % len(cores_pares)]

                    distancias_pares.append({
                        'Par': par,
                        'Cor': nome_cor,
                        'Distância': round(distancia_par, 2),
                    })

                    ax.plot([x1, x2], [y1, y2], color=cor_par, linewidth=2.5, alpha=0.95)
                    ax.text(
                        x_mid,
                        y_mid,
                        f"{distancia_par:.2f} m",
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=cor_par, edgecolor='black', alpha=0.95),
                    )

                for _, row in snapshot[snapshot['atleta_id'].astype(str).isin(coords_jogadores.keys())].iterrows():
                    pitch.annotate(
                        row['atleta_id'],
                        (row['x_tr'], row['y_tr']),
                        ax=ax,
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        va='center',
                        ha='center',
                        zorder=5,
                    )

            elif (
                campo == 'Movimento Relativo de Jogadores ao Longo do Tempo'
                and jogador_rel_1 is not None
                and jogador_rel_2 is not None
                and jogador_rel_1 != jogador_rel_2
                and tempo_relativo is not None
            ):
                idx_tempo = timestamps_relativos.index(tempo_relativo) if tempo_relativo in timestamps_relativos else len(timestamps_relativos) - 1
                janela_frames = max(1, int(janela_segundos) * 10)
                janela_timestamps = timestamps_relativos[max(0, idx_tempo - janela_frames + 1): idx_tempo + 1]

                trilho_df = df_fase.loc[
                    (df_fase['time_evento_s'].isin(janela_timestamps))
                    & (df_fase['atleta_id'].astype(str).isin([str(jogador_rel_1), str(jogador_rel_2)]))
                    & (df_fase['x_tr'].notna())
                    & (df_fase['y_tr'].notna())
                ].copy()

                jogador_1_trilho = trilho_df[trilho_df['atleta_id'].astype(str) == str(jogador_rel_1)]
                jogador_2_trilho = trilho_df[trilho_df['atleta_id'].astype(str) == str(jogador_rel_2)]

                if not jogador_1_trilho.empty:
                    ax.plot(
                        jogador_1_trilho['x_tr'],
                        jogador_1_trilho['y_tr'],
                        color='#2ECC71',
                        linewidth=2.2,
                        alpha=0.95,
                    )

                    if len(jogador_1_trilho) >= 2:
                        x_prev_1, y_prev_1 = jogador_1_trilho[['x_tr', 'y_tr']].iloc[-2]
                        x_last_1, y_last_1 = jogador_1_trilho[['x_tr', 'y_tr']].iloc[-1]
                        ax.annotate(
                            '',
                            xy=(x_last_1, y_last_1),
                            xytext=(x_prev_1, y_prev_1),
                            arrowprops=dict(arrowstyle='-|>', color='#2ECC71', lw=2.2, shrinkA=0, shrinkB=0),
                            zorder=4,
                        )

                if not jogador_2_trilho.empty:
                    ax.plot(
                        jogador_2_trilho['x_tr'],
                        jogador_2_trilho['y_tr'],
                        color='#F2F2F2',
                        linewidth=2.2,
                        alpha=0.95,
                    )

                    if len(jogador_2_trilho) >= 2:
                        x_prev_2, y_prev_2 = jogador_2_trilho[['x_tr', 'y_tr']].iloc[-2]
                        x_last_2, y_last_2 = jogador_2_trilho[['x_tr', 'y_tr']].iloc[-1]
                        ax.annotate(
                            '',
                            xy=(x_last_2, y_last_2),
                            xytext=(x_prev_2, y_prev_2),
                            arrowprops=dict(arrowstyle='-|>', color='#F2F2F2', lw=2.2, shrinkA=0, shrinkB=0),
                            zorder=4,
                        )

                snapshot_rel = trilho_df.loc[trilho_df['time_evento_s'] == tempo_relativo].copy()
                jogador_1_frame = snapshot_rel[snapshot_rel['atleta_id'].astype(str) == str(jogador_rel_1)]
                jogador_2_frame = snapshot_rel[snapshot_rel['atleta_id'].astype(str) == str(jogador_rel_2)]

                jogador_1_serie = (
                    trilho_df.loc[trilho_df['atleta_id'].astype(str) == str(jogador_rel_1), ['time_evento_s', 'x_tr', 'y_tr']]
                    .drop_duplicates(subset=['time_evento_s'])
                    .rename(columns={'x_tr': 'x_1', 'y_tr': 'y_1'})
                )
                jogador_2_serie = (
                    trilho_df.loc[trilho_df['atleta_id'].astype(str) == str(jogador_rel_2), ['time_evento_s', 'x_tr', 'y_tr']]
                    .drop_duplicates(subset=['time_evento_s'])
                    .rename(columns={'x_tr': 'x_2', 'y_tr': 'y_2'})
                )
                serie_distancias_rel = (
                    pd.merge(jogador_1_serie, jogador_2_serie, on='time_evento_s', how='inner')
                    .sort_values('time_evento_s')
                )

                if not serie_distancias_rel.empty:
                    serie_distancias_rel['distancia_m'] = np.hypot(
                        serie_distancias_rel['x_2'] - serie_distancias_rel['x_1'],
                        serie_distancias_rel['y_2'] - serie_distancias_rel['y_1'],
                    )
                    serie_distancias_rel['tempo_label'] = serie_distancias_rel['time_evento_s'].map(converter_para_relogio_fpf)

                    distancia_media_rel = float(serie_distancias_rel['distancia_m'].mean())
                    distancia_min_rel = float(serie_distancias_rel['distancia_m'].min())
                    distancia_max_rel = float(serie_distancias_rel['distancia_m'].max())

                    distancia_inicial = float(serie_distancias_rel['distancia_m'].iloc[0])
                    distancia_final = float(serie_distancias_rel['distancia_m'].iloc[-1])
                    delta_distancia_rel = distancia_final - distancia_inicial

                    if delta_distancia_rel <= -0.25:
                        tendencia_rel = 'Aproximação'
                    elif delta_distancia_rel >= 0.25:
                        tendencia_rel = 'Afastamento'
                    else:
                        tendencia_rel = 'Estável'

                if not jogador_1_frame.empty and not jogador_2_frame.empty:
                    x1, y1 = jogador_1_frame[['x_tr', 'y_tr']].iloc[0]
                    x2, y2 = jogador_2_frame[['x_tr', 'y_tr']].iloc[0]
                    distancia_rel = float(np.hypot(x2 - x1, y2 - y1))

                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2

                    pitch.scatter([x1], [y1], s=520, c='#2ECC71', edgecolors='white', linewidth=1.5, ax=ax, zorder=4)
                    pitch.scatter([x2], [y2], s=520, c='#F2F2F2', edgecolors='black', linewidth=1.5, ax=ax, zorder=4)
                    ax.plot([x1, x2], [y1, y2], color='#FFC857', linewidth=2.5, alpha=0.95)
                    pitch.annotate(str(jogador_rel_1), (x1, y1), ax=ax, color='black', fontsize=10, fontweight='bold', va='center', ha='center', zorder=5)
                    pitch.annotate(str(jogador_rel_2), (x2, y2), ax=ax, color='black', fontsize=10, fontweight='bold', va='center', ha='center', zorder=5)
                    ax.text(
                        x_mid,
                        y_mid,
                        f'{distancia_rel:.2f} m',
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFC857', edgecolor='black', alpha=0.95),
                    )

            plot_placeholder.pyplot(fig)

            if campo == 'Movimento Relativo de Jogadores ao Longo do Tempo' and timestamps_amostrados:
                tempo_relativo = slider_placeholder.select_slider(
                    'Momento do Movimento (1 em 1 segundo)',
                    options=timestamps_amostrados,
                    value=tempo_relativo if tempo_relativo in timestamps_amostrados else timestamps_amostrados[0],
                    format_func=converter_para_relogio_fpf,
                    key='mov_rel_slider',
                )

            if campo == 'Movimento Relativo de Jogadores ao Longo do Tempo' and not serie_distancias_rel.empty:
                fig_dist, ax_dist = plt.subplots(figsize=(10, 3.2))
                ax_dist.plot(
                    serie_distancias_rel['time_evento_s'],
                    serie_distancias_rel['distancia_m'],
                    color='#E30613',
                    linewidth=2.2,
                )
                ax_dist.scatter(
                    serie_distancias_rel['time_evento_s'].iloc[-1],
                    serie_distancias_rel['distancia_m'].iloc[-1],
                    color='#FFC857',
                    edgecolors='black',
                    s=70,
                    zorder=3,
                )
                ax_dist.axhline(serie_distancias_rel['distancia_m'].mean(), color='#F2F2F2', linewidth=1.3, linestyle='--')
                ax_dist.set_facecolor('#1e1e1e')
                fig_dist.patch.set_facecolor('#1e1e1e')
                ax_dist.tick_params(colors='white', labelsize=9)
                ax_dist.set_ylabel('Distância (m)', color='white')
                ax_dist.set_xlabel('Tempo', color='white')
                tick_positions = serie_distancias_rel['time_evento_s'].tolist()
                tick_labels = serie_distancias_rel['tempo_label'].tolist()
                step_ticks = max(1, len(tick_positions) // 5)
                ax_dist.set_xticks(tick_positions[::step_ticks])
                ax_dist.set_xticklabels(tick_labels[::step_ticks], rotation=0)
                for spine in ax_dist.spines.values():
                    spine.set_color('#666666')
                ax_dist.grid(axis='y', color='#444444', linestyle=':', linewidth=0.7, alpha=0.8)
                ax_dist.set_title('Distância entre jogadores ao longo da janela', color='white', fontsize=11)
                fig_dist.tight_layout()
                graph_placeholder.pyplot(fig_dist)

        # INFO FRAME
        with col_info:
            if campo == 'Distância entre Jogadores':
                st.subheader("Selecionar Jogadores")

                if len(jogadores_disponiveis) >= 2:
                    jogadores_escolhidos = st.multiselect(
                        "Jogadores (Máx. 4)",
                        options=jogadores_disponiveis,
                        default=jogadores_selecionados,
                        key='dist_jogadores_selecionados',
                        max_selections=4,
                    )

                    if len(jogadores_escolhidos) < 2:
                        st.warning("Seleciona pelo menos 2 jogadores.")
                    else:
                        pares_widget_opcoes = [
                            f"{jogador_1} - {jogador_2}"
                            for jogador_1, jogador_2 in combinations(jogadores_escolhidos, 2)
                        ]
                        pares_default_widget = [par for par in pares_selecionados if par in pares_widget_opcoes]
                        if not pares_default_widget and pares_widget_opcoes:
                            pares_default_widget = pares_widget_opcoes[:1]

                        st.multiselect(
                            "Pares",
                            options=pares_widget_opcoes,
                            default=pares_default_widget,
                            key='dist_pares_selecionados',
                        )
                else:
                    st.info("São necessários pelo menos dois jogadores no frame para esta visualização.")

                st.divider()

            elif campo == 'Movimento Relativo de Jogadores ao Longo do Tempo':
                st.subheader('Selecionar Jogadores')

                if len(jogadores_disponiveis) >= 2:
                    jogador_rel_1 = st.selectbox(
                        'Jogador 1',
                        options=jogadores_disponiveis,
                        index=0,
                        key='mov_rel_jogador_1',
                    )
                    jogador_rel_2 = st.selectbox(
                        'Jogador 2',
                        options=jogadores_disponiveis,
                        index=1 if len(jogadores_disponiveis) > 1 else 0,
                        key='mov_rel_jogador_2',
                    )

                    if jogador_rel_1 == jogador_rel_2:
                        st.warning('Seleciona dois jogadores diferentes.')

                    st.selectbox(
                        'Janela Temporal',
                        options=[1, 2, 3, 5, 10],
                        index=[1, 2, 3, 5, 10].index(janela_segundos) if janela_segundos in [1, 2, 3, 5, 10] else 0,
                        key='mov_rel_janela_segundos',
                        format_func=lambda valor: f'{valor}s',
                    )
                else:
                    st.info('São necessários pelo menos dois jogadores no frame para esta visualização.')

                st.divider()

            st.subheader("Analise de Frame")
            st.metric("Tempo Selecionado", f"{converter_para_relogio_fpf(selected_time)}s")
            if campo == 'Distância entre Jogadores':
                if distancias_pares:
                    if len(distancias_pares) == 1:
                        st.metric("Distância entre Jogadores", f"{distancias_pares[0]['Distância']:.2f} m")
                    else:
                        df_distancias = pd.DataFrame(distancias_pares)
                        df_distancias['Distância'] = df_distancias['Distância'].map(lambda valor: f"{valor:.2f} m")
                        st.dataframe(df_distancias, width='stretch', hide_index=True)
                else:
                    st.metric("Distância entre Jogadores", "N/A")
            elif campo == 'Movimento Relativo de Jogadores ao Longo do Tempo':
                if tempo_relativo is not None:
                    st.metric('Tempo do Movimento', f"{converter_para_relogio_fpf(tempo_relativo)}s")
                else:
                    st.metric('Tempo do Movimento', 'N/A')

                if distancia_rel is not None:
                    st.metric('Distância Atual', f'{distancia_rel:.2f} m')
                else:
                    st.metric('Distância Atual', 'N/A')

                if distancia_media_rel is not None:
                    st.metric('Distância Média', f'{distancia_media_rel:.2f} m')
                else:
                    st.metric('Distância Média', 'N/A')

                if distancia_min_rel is not None:
                    st.metric('Distância Mínima', f'{distancia_min_rel:.2f} m')
                else:
                    st.metric('Distância Mínima', 'N/A')

                if distancia_max_rel is not None:
                    st.metric('Distância Máxima', f'{distancia_max_rel:.2f} m')
                else:
                    st.metric('Distância Máxima', 'N/A')

                if delta_distancia_rel is not None:
                    st.metric(f'Variação em {janela_segundos}s', f'{delta_distancia_rel:+.2f} m', delta=tendencia_rel)
                else:
                    st.metric(f'Variação em {janela_segundos}s', 'N/A')

                if tendencia_rel is not None:
                    st.metric('Tendência', tendencia_rel)
                else:
                    st.metric('Tendência', 'N/A')
            else:
                st.metric("Compactação Vertical", compactacao_frame['comp_vertical'])
                st.metric("Compactação Horizontal", compactacao_frame['comp_horizontal'])
                st.metric("Area", round(area, 2))
    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")


# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
