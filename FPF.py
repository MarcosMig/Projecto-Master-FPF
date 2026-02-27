import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from pyproj import Transformer
from scipy.signal import savgol_filter

# --- 1. CONFIGURAÇÃO E LOGIN (ESTÁVEL 2) ---
st.set_page_config(page_title="FPF Performance Hub", layout="centered")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'engine_run' not in st.session_state: st.session_state.engine_run = False

def apply_login_style():
    st.markdown("""<style>
        .stApp { background-color: #0e1117; }
        .main .block-container { max-width: 400px !important; background-color: #1a1c23 !important; padding: 40px !important; border-radius: 12px; border: 1px solid #30363d; margin: auto; margin-top: 15vh; }
        .stButton>button { width: 100%; background-color: #E30613 !important; color: white !important; font-weight: bold; height: 3.2em; }
        header, footer {visibility: hidden;}
    </style>""", unsafe_allow_html=True)

if not st.session_state.auth:
    apply_login_style()
    st.markdown("<h2 style='text-align:center; color:white;'>FPF Performance Hub</h2>", unsafe_allow_html=True)
    u = st.text_input("Utilizador", key="u_val")
    p = st.text_input("Password", type="password", key="p_val")
    if st.button("ENTRAR"):
        if u == "miguel.cardoso" and p == "fpf2026":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Acesso Negado")
    st.stop()

# --- 2. VALIDAÇÃO (ANALISAR) ---
st.markdown("<style>header {visibility: visible !important;} [data-testid='stSidebar'] {display: block !important;} .main .block-container {max-width: 95% !important; margin-top: 0 !important;}</style>", unsafe_allow_html=True)
st.title("Validação e Engine V9.2")

with st.sidebar:
    st.header("📤 Upload")
    f_campo = st.file_uploader("Dados de CAMPO", accept_multiple_files=True)
    f_atleta = st.file_uploader("Dados de ATLETAS", accept_multiple_files=True)
    if st.button("🔄 Reiniciar App"):
        st.session_state.engine_run = False
        st.rerun()

# Inicialização de variáveis de segurança
is_geo_valid = False
quorum_ok = False
pts_gps = {}

if f_campo and f_atleta:
    # Validação do Campo (Normalização de colunas para evitar KeyError 'Lat')
    for f in f_campo:
        df_c = pd.read_csv(f, sep=None, engine='python')
        df_c.columns = [c.strip().replace('"', '').replace("'", "") for c in df_c.columns]
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper() and 'Lat' in df_c.columns:
                pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]

    if len(pts_gps) == 4:
        # Cálculo de distância ao centro (v11.1)
        clat = np.mean([p[0] for p in pts_gps.values()])
        clon = np.mean([p[1] for p in pts_gps.values()])
        
        # Amostra do primeiro atleta para geofencing
        df_sample = pd.read_csv(f_atleta[0], sep=None, engine='python', nrows=1)
        df_sample.columns = [c.strip().replace('"', '').replace("'", "") for c in df_sample.columns]
        
        dist_m = np.sqrt((clat - df_sample["Lat"].iloc[0])**2 + (clon - df_sample["Lon"].iloc[0])**2) * 111320
        is_geo_valid = dist_m < 50
        
        # Auditoria de Atletas (Sincronização de Fases)
        audit = {}
        for f in f_atleta:
            match = re.search(r"Player-(\d+)", f.name)
            aid = match.group(0) if match else "ID_Desconhecido"
            fase = "Warm-Up" if "WARM" in f.name.upper() else "1P" if "1P" in f.name.upper() else "2P"
            if aid not in audit: audit[aid] = []
            audit[aid].append(fase)
        
        num_completos = sum(1 for a in audit if all(x in audit[a] for x in ["Warm-Up", "1P", "2P"]))
        quorum_ok = num_completos >= 10

        # Feedback de Validação
        c1, c2 = st.columns(2)
        if is_geo_valid: c1.success(f"📍 GPS OK: {dist_m:.1f}m")
        else: c1.error(f"📍 Fora do Campo: {dist_m:.1f}m")
        if quorum_ok: c2.success(f"👥 Quórum OK: {num_completos} atletas")
        else: c2.warning(f"👥 Quórum: {num_completos}/10")

# --- 3. ENGINE V9.2 (O TEU MOTOR REAL) ---
if is_geo_valid and quorum_ok:
    st.divider()
    if not st.session_state.engine_run:
        st.subheader("⚙️ Executamos UTM Engine V9... ?")
        if st.button("VALIDAR E CORRER ENGINE V9", type="primary", use_container_width=True):
            with st.status("A executar FPF UTM Engine v9.2...", expanded=True) as status:
                
                # FASE 1: CALIBRAÇÃO REAL (v9.2)
                st.write("1.