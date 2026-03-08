import pandas as pd
import streamlit as st
import mplsoccer as mpl
import matplotlib.pyplot as plt
from fpf_modules.constants import CLEANDATA_DIR, SELECOES_OPCOES

# TODO 1. Alinhar dados de tracking de forma a atacar da esquerda para direita
# TODO 2. Aplicar filtro dados da sessão aos dados de tracking


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="FPF | Positional Analysis", layout="wide")

# --- FUNÇÕES DE CARREGAMENTO (Com Cache) ---
@st.cache_data
def load_data():
    perf = pd.read_parquet(f"{CLEANDATA_DIR}/performance_metrics.parquet")
    # atletas = pd.read_parquet(f"{CLEANDATA_DIR}/athletes.parquet")
    tracking = pd.read_parquet(f"{CLEANDATA_DIR}/tracking.parquet")
    return perf, tracking

try:
    df_perf, df_tracking = load_data()
except Exception as e:
    st.error(f"Erro ao carregar arquivos Parquet: {e}")
    st.stop()

# --- SIDEBAR: FILTROS E CONTROLES ---
with st.sidebar:
    st.title("⚽ Filtros de Sessão")

    selecao = st.selectbox("Seleção", options=SELECOES_OPCOES)
    contexto = st.selectbox("Contexto", options=["Treino", "Jogo"])

    # Filtrar sessões disponíveis para o slider ou seleção
    sessoes_disponiveis = df_perf[(df_perf.selecao == selecao) & (df_perf.contexto == contexto)]

    st.divider()
    st.header("⏱️ Navegação Temporal")

    fase_selected = st.radio("Fase", ["Warm-Up", "1P", "2P"], index=1)

    # Filtrar tracking pela fase para pegar os timestamps
    df_fase = df_tracking[df_tracking['fase'] == fase_selected]

    if not df_fase.empty:
        timestamps = sorted(df_fase['time'].unique())
        # Slider para navegar no tempo
        selected_time = st.select_slider(
            "Momento do Jogo (s)",
            options=timestamps,
           # format_func=lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}s"
        )
    else:
        st.warning(f"Sem dados de tracking para a fase {fase_selected}")
        selected_time = None

# --- PAINEL PRINCIPAL ---
tab_metrics, tab_visual = st.tabs(["📊 Métricas de Performance", "📍 Análise Posicional"])

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
            st.metric("Jogadores em Campo", len(snapshot))
            st.write("Coordenadas Atuais:")
            st.dataframe(snapshot[['atleta_id', 'x_tr', 'y_tr']], hide_index=True)

    else:
        st.info("Selecione uma fase com dados para visualizar o campo.")

st.dataframe(df_tracking.sample(20))
# --- FOOTER ---
st.divider()
st.caption(f"FPF UTM Engine v16 | Data Shape: {df_perf.shape[0]} sessions loaded.")
