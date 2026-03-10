import pandas as pd
import streamlit as st
import mplsoccer as mpl
import matplotlib.pyplot as plt
from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES
from fpf_modules.metrics import calcular_compactacao

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
        pd.DataFrame: DataFrame com o vol
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
        st.dataframe(sessoes_disponiveis, use_container_width=True, hide_index=True)

# TAB 2: VISUALIZAÇÃO DO CAMPO

with tab_visual:

    df_compactacao = calcular_compactacao(df_fase)
    avg_comp_vert = df_compactacao['comp_vertical'].mean().round(2)
    # Estilo para as métricas

    st.markdown("""
    <style>
        /* Estilo exclusivo para as nossas cartas de topo */
        .fpt-kpi-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 20px;
        }
        .fpt-kpi-card {
            background-color: #1e1e1e; /* Fundo escuro premium */
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Compactação Vertical", f"{avg_comp_vert:.1f} m")

    st.divider()

    if selected_time is not None:

        # 1. Obter snapshot
        snapshot = df_fase[df_fase['time_evento_s'] == selected_time]

        # Obter metricas para o frame
        compactacao_frame = df_compactacao.loc[ df_compactacao['time_evento_s'] == selected_time]

        col_map, col_info = st.columns([3, 1])

        with col_map:
            # 2. Configurar Pitch
            # DICA: Se o seu x_tr vai até 105, o pitch_type deve ser 'custom' ou 'uefa'
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

            st.pyplot(fig)



        with col_info:

            st.metric("Tempo Selecionado", f"{converter_para_relogio_fpf(selected_time)}s")
            st.metric("Compactação Vertical", compactacao_frame['comp_vertical'])
            st.metric("Compactação Horizontal", compactacao_frame['comp_horizontal'])
    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")

# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
