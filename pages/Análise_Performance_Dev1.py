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

    df_compactacao = calcular_compactacao(df_fase)
    area_dict = calcular_area_media(df_fase, 0, 0)

    if not df_compactacao.empty:

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

        # 1. Obter snapshot
        snapshot = df_fase[
            (df_fase['time_evento_s'] == selected_time)
            & (df_fase['x_tr'].notna())
            & (df_fase['y_tr'].notna())
        ]
        jogadores_disponiveis = sorted(snapshot['atleta_id'].dropna().astype(str).unique().tolist())

        # Obter metricas para o frame
        compactacao_frame = df_compactacao.loc[ df_compactacao['time_evento_s'] == selected_time]

        campo = st.selectbox(label='Visualização Campo', options=['Convex Hull', 'Distância entre Jogadores', 'Teste'],index=None, placeholder='Seleciona outro metodo de visualizar o campo')

        jogadores_selecionados = []
        pares_opcoes = []
        pares_selecionados = []
        distancias_pares = []
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

        col_map, col_info = st.columns([3, 1])

        # CAMPO
        with col_map:

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

            st.pyplot(fig)

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
            else:
                st.metric("Compactação Vertical", compactacao_frame['comp_vertical'])
                st.metric("Compactação Horizontal", compactacao_frame['comp_horizontal'])
                st.metric("Area", round(area, 2))
    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")


# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
