import pandas as pd
import streamlit as st
import mplsoccer as mpl
import matplotlib.pyplot as plt
from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES

# TODO 2. Aplicar filtro dados da sessão aos dados de tracking


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
        timestamps = sorted(df_fase['time'].unique())
        # Slider para navegar no tempo
        selected_time = st.sidebar.select_slider(
            "Momento do Jogo (s)",
            options=timestamps,
       #  format_func=converter_para_relogio_fpf
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

    if selected_time is not None:
        # 1. Obter snapshot
        snapshot = df_fase[df_fase['time'] == selected_time]

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
            st.metric("Tempo Selecionado", f"{selected_time}s")
    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")

# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
