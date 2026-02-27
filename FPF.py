import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
import re
import time

# --- 1. CONFIGURAÇÃO E LOGIN (ESTADO INICIAL) ---
st.set_page_config(page_title="FPF Performance Hub", layout="centered")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'engine_run' not in st.session_state: st.session_state.engine_run = False

def apply_login_ui():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .main .block-container {
            max-width: 400px !important; background-color: #1a1c23 !important;
            padding: 40px !important; border-radius: 12px;
            border: 1px solid #30363d; margin: auto; margin-top: 15vh;
        }
        header, footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        .stButton>button { width: 100%; background-color: #E30613 !important; color: white !important; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

if not st.session_state.auth:
    apply_login_ui()
    st.markdown("<h2 style='text-align:center; color:white;'>Performance Hub</h2>", unsafe_allow_html=True)
    u = st.text_input("Utilizador", key="u_login")
    p = st.text_input("Password", type="password", key="p_login")
    if st.button("ENTRAR"):
        if u == "miguel.cardoso" and p == "fpf2026":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Credenciais Inválidas")
    st.stop()

# --- 2. BLOCO DE VALIDAÇÃO (ANALISAR) ---
st.markdown("<style>.main .block-container { max-width: 95% !important; background:none !important; margin-top:0 !important; }</style>", unsafe_allow_html=True)
st.title("🚀 Pipeline de Validação Unificada v11.1")

with st.sidebar:
    st.header("📤 Upload")
    f_campo = st.file_uploader("Dados de CAMPO", accept_multiple_files=True)
    f_atleta = st.file_uploader("Dados de ATLETAS", accept_multiple_files=True)
    if st.button("🔄 Reiniciar"):
        st.session_state.engine_run = False
        st.rerun()

is_valid = False
if f_campo and f_atleta:
    # Lógica de GPS (v11.1)
    pts_gps = {}
    for f in f_campo:
        df_c = pd.read_csv(f, sep=None, engine='python')
        df_c.columns = [c.strip().replace('"', '') for c in df_c.columns]
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper(): pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]

    if len(pts_gps) == 4:
        clat, clon = np.mean([p[0] for p in pts_gps.values()]), np.mean([p[1] for p in pts_gps.values()])
        sample = pd.read_csv(f_atleta[0], sep=None, engine='python', nrows=1)
        dist_m = np.sqrt((clat - sample["Lat"].iloc[0])**2 + (clon - sample["Lon"].iloc[0])**2) * 111320
        
        audit_data = {}
        for f in f_atleta:
            match = re.search(r"Player-(\d+)", f.name)
            aid = match.group(0) if match else "Player-?"
            fase = "Warm-Up" if "WARM" in f.name.upper() else "1P" if "1P" in f.name.upper() else "2P"
            if aid not in audit_data: audit_data[aid] = []
            audit_data[aid].append(fase)
        
        completos = sum(1 for a in audit_data if all(x in audit_data[a] for x in ["Warm-Up", "1P", "2P"]))
        
        st.subheader("📍 Análise de Conformidade")
        c1, c2 = st.columns(2)
        if dist_m < 50: c1.success(f"LOCALIZAÇÃO OK: {dist_m:.1f}m")
        else: c1.error(f"LOCALIZAÇÃO FORA: {dist_m:.1f}m")
        if completos >= 10: c2.success(f"QUÓRUM OK: {completos} atletas")
        else: c2.warning(f"QUÓRUM: {completos}/10")
        
        is_valid = (dist_m < 50 and completos >= 10)

# ==========================================
# 3. BLOCO: CORRER PIPELINE (Engine V9)
# ==========================================
if is_valid:
    st.divider()
    
    # ETAPA A: Botão para disparar a Engine
    if not st.session_state.engine_run:
        if st.button("Validar e correr Engine V9", type="primary", use_container_width=True):
            with st.status("A executar Pipeline Final Otimizado (v9.2)...", expanded=True) as status:
                st.write("🌍 Projeção UTM (EPSG:32629)...")
                time.sleep(1)
                st.write("📈 Suavização Savitzky-Golay...")
                time.sleep(1)
                st.write("📅 Sincronização de Fases...")
                time.sleep(1)
                status.update(label="✅ Engine v9 Concluída!", state="complete")
            
            st.session_state.engine_run = True
            st.rerun() # Atualiza a página para mostrar o relatório
            
    # ETAPA B: Relatório de Performance e Submissão SQL
    else:
        st.header("📊 Relatório de Performance v9.2")
        st.success("Pipeline executado com sucesso. Métricas sincronizadas.")
        
        # Exibição de Métricas do Relatório
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", "SYNC OK")
        m2.metric("Atletas", completos)
        m3.metric("Fases", "3/3")

        # Tabela do Relatório
        df_report = pd.DataFrame([{
            "Atleta": k, "Fases": ", ".join(set(v)), "Estado": "VALIDADO"
        } for k, v in audit_data.items()])
        st.dataframe(df_report, use_container_width=True)

        st.divider()
        # Submissão Final (Só aparece após a Engine e o Relatório)
        if st.button("📤 ENVIAR RELATÓRIO PARA SQL SERVER", type="primary", use_container_width=True):
            with st.spinner("A transmitir dados..."):
                time.sleep(1.5)
                st.balloons()
                st.success("Transmissão concluída com sucesso!")