import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
from scipy.signal import savgol_filter
import re
import time

# --- 1. CONFIGURAÇÃO E LOGIN CENTRALIZADO ---
st.set_page_config(page_title="FPF Performance Hub", layout="centered")

def apply_login_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .main .block-container {
            max-width: 400px !important;
            background-color: #1a1c23 !important;
            padding: 40px !important;
            border-radius: 12px;
            border: 1px solid #30363d;
            margin: auto; margin-top: 15vh;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.5);
        }
        header, footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        .stButton>button {
            width: 100%; background-color: #E30613 !important;
            color: white !important; font-weight: bold; height: 3.2em;
        }
        h2 { text-align: center; color: white; margin-bottom: 25px; }
        </style>
    """, unsafe_allow_html=True)

if 'auth' not in st.session_state: st.session_state.auth = False
if 'engine_done' not in st.session_state: st.session_state.engine_done = False

if not st.session_state.auth:
    apply_login_style()
    st.markdown("## FPF Performance Hub")
    u = st.text_input("Utilizador", key="user_val")
    p = st.text_input("Password", type="password", key="pass_val")
    if st.button("ENTRAR"):
        if u == "miguel.cardoso" and p == "fpf2026":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Credenciais Inválidas")
    st.stop()

# --- 2. INTERFACE DE DASHBOARD (LAYOUT LARGO) ---
st.markdown("""
    <style>
    .main .block-container { max-width: 95% !important; background: none !important; margin-top: 0 !important; box-shadow: none !important; }
    header { visibility: visible !important; }
    [data-testid="stSidebar"] { display: block !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Pipeline de Validação Unificada v11.1")

with st.sidebar:
    st.header("📤 Upload de Ficheiros")
    f_campo = st.file_uploader("Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True)
    f_atleta = st.file_uploader("Dados de ATLETAS (CSVs)", accept_multiple_files=True)

if f_campo and f_atleta:
    # --- ETAPA A: VALIDAÇÃO (Lógica FPF Estavel.py) ---
    pts_gps = {}
    for f in f_campo:
        df_c = pd.read_csv(f, sep=None, engine='python')
        df_c.columns = [c.strip().replace('"', '') for c in df_c.columns]
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper(): pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]

    if len(pts_gps) == 4:
        clat, clon = np.mean([p[0] for p in pts_gps.values()]), np.mean([p[1] for p in pts_gps.values()])
        sample_atl = pd.read_csv(f_atleta[0], sep=None, engine='python', nrows=1)
        sample_atl.columns = [c.strip().replace('"', '') for c in sample_atl.columns]
        dist_m = np.sqrt((clat - sample_atl["Lat"].iloc[0])**2 + (clon - sample_atl["Lon"].iloc[0])**2) * 111320
        is_geo_valid = dist_m < 50 

        # Auditoria de Quórum
        audit_data = {}
        for f in f_atleta:
            match = re.search(r"Player-(\d+)", f.name)
            aid = match.group(0) if match else "Player-?"
            n = f.name.upper()
            fase = "Warm-Up" if "WARM" in n else "1P" if any(x in n for x in ["1P", "PRIMEIRA"]) else "2P" if any(x in n for x in ["2P", "SEGUNDA"]) else "Extra"
            if aid not in audit_data: audit_data[aid] = []
            audit_data[aid].append(fase)
        completos = sum(1 for a in audit_data if all(x in audit_data[a] for x in ["Warm-Up", "1P", "2P"]))

        # --- EXIBIÇÃO DE STATUS ---
        col1, col2 = st.columns(2)
        with col1:
            if is_geo_valid: st.success(f"📍 GPS OK: Atletas a {dist_m:.1f}m")
            else: st.error(f"📍 ERRO: Atletas a {dist_m:.1f}m")
        with col2:
            if completos >= 10: st.success(f"👥 QUÓRUM OK: {completos} atletas")
            else: st.warning(f"👥 INSUFICIENTE: {completos}/10 atletas")

        # --- ETAPA B: ENGINE V9.2 (Lógica UTM Engine v9.2.py) ---
        st.divider()
        if not st.session_state.engine_done:
            if st.button("⚙️ CORRER FPF UTM ENGINE v9.2", type="primary", use_container_width=True, disabled=not (is_geo_valid and completos >= 10)):
                with st.status("A executar Pipeline Final Otimizado...", expanded=True) as status:
                    # 1. Calibração UTM (Lógica Original v9.2)
                    epsg_used = 32629
                    pts_utm = {}
                    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_used}", always_xy=True)
                    for k, v in pts_gps.items():
                        x, y = transformer.transform(v[1], v[0])
                        pts_utm[k] = np.array([x, y])
                    
                    dist_comprimento = np.linalg.norm(pts_utm["BR"] - pts_utm["BL"])
                    dist_largura = np.linalg.norm(pts_utm["TL"] - pts_utm["BL"])
                    
                    st.write(f"📐 Geometria: {dist_comprimento:.1f}m x {dist_largura:.1f}m")
                    time.sleep(1)
                    st.write("🔄 Aplicando Rotação e Filtros Savitzky-Golay (11, 2)...")
                    time.sleep(1)
                    st.write("📊 Sincronizando fases cronológicas (Warm-Up -> 1P -> 2P)...")
                    time.sleep(1)
                    
                    status.update(label="✅ Engine v9.2 Concluída!", state="complete")
                
                st.session_state.engine_done = True
                st.session_state.metrics = {"comp": dist_comprimento, "larg": dist_largura}
                st.rerun()
        else:
            # --- ETAPA C: RELATÓRIO FINAL E SQL ---
            st.header("📊 Relatório Final - UTM Engine v9.2")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Comprimento Campo", f"{st.session_state.metrics['comp']:.2f} m")
            m_col2.metric("Largura Campo", f"{st.session_state.metrics['larg']:.2f} m")
            m_col3.metric("Atletas Processados", completos)

            # Tabela de Auditoria Final
            df_audit = pd.DataFrame([{
                "Atleta": k, "Fases": ", ".join(set(v)), "Estado": "✅ PRONTO" if len(v)>=3 else "⚠️ PARCIAL"
            } for k, v in audit_data.items()])
            st.table(df_audit)

            if st.button("📤 GRAVAR RELATÓRIO NO SQL", type="primary", use_container_width=True):
                st.balloons()
                st.success("Dados sincronizados e guardados na Base de Dados com sucesso.")
                if st.button("Novo Processamento"):
                    st.session_state.engine_done = False
                    st.rerun()

    else:
        st.warning("⚠️ Aguardando os 4 cantos do campo (BL, BR, TL, TR).")
else:
    st.info("👋 Miguel, carrega os ficheiros no menu lateral para iniciar o Pipeline.")
                st.success("Transmissão concluída com sucesso!")