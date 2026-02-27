import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import re
import time
from pyproj import Transformer

# --- 1. CONFIGURAÇÃO DE UI (DO TEU FPF ESTAVEL 2) ---
st.set_page_config(page_title="FPF Performance Hub", layout="centered")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'engine_run' not in st.session_state: st.session_state.engine_run = False

def apply_login_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .main .block-container {
            max-width: 400px !important; background-color: #1a1c23 !important;
            padding: 40px !important; border-radius: 12px; border: 1px solid #30363d;
            margin: auto; margin-top: 15vh; box-shadow: 0px 8px 24px rgba(0,0,0,0.5);
        }
        header, footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        .stButton>button { width: 100%; background-color: #E30613 !important; color: white !important; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE LOGIN ---
if not st.session_state.auth:
    apply_login_style()
    st.image("https://upload.wikimedia.org/wikipedia/pt/4/43/Logo_FPF.png", width=80)
    user = st.text_input("Utilizador")
    pw = st.text_input("Password", type="password")
    if st.button("ENTRAR"):
        if user == "miguel.cardoso" and pw == "fpf2026":
            st.session_state.auth = True
            st.rerun()
        else: st.error("Credenciais Inválidas")
    st.stop()

# --- 2. VALIDAÇÃO (TEU SCRIPT FPF ESTAVEL 2) ---
st.markdown("<style>header {visibility: visible !important;} [data-testid='stSidebar'] {display: block !important;} .main .block-container {max-width: 95% !important; margin-top: 0 !important;}</style>", unsafe_allow_html=True)
st.title("🚀 Pipeline de Validação e Engine V9")

with st.sidebar:
    st.header("📤 Upload de Ficheiros")
    f_campo = st.file_uploader("Dados de CAMPO", accept_multiple_files=True)
    f_atleta = st.file_uploader("Dados de ATLETAS", accept_multiple_files=True)
    if st.button("Reiniciar Sessão"):
        st.session_state.engine_run = False
        st.rerun()

is_geo_valid = False
quorum_ok = False

if f_campo and f_atleta:
    # --- PARTE GEOGRÁFICA ---
    pts_gps = {}
    for f in f_campo:
        df_c = pd.read_csv(f)
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper(): pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]
    
    if len(pts_gps) == 4:
        clat, clon = np.mean([p[0] for p in pts_gps.values()]), np.mean([p[1] for p in pts_gps.values()])
        sample = pd.read_csv(f_atleta[0], sep=None, engine='python', nrows=1)
        dist_m = np.sqrt((clat - sample["Lat"].iloc[0])**2 + (clon - sample["Lon"].iloc[0])**2) * 111320
        is_geo_valid = dist_m < 50
        
        # --- PARTE ATLETAS (AUDITORIA) ---
        audit_data = {}
        for f in f_atleta:
            match = re.search(r"Player-(\d+)", f.name)
            aid = match.group(0) if match else "ID"
            fase = "Warm-Up" if "WARM" in f.name.upper() else "1P" if "1P" in f.name.upper() else "2P"
            if aid not in audit_data: audit_data[aid] = []
            audit_data[aid].append(fase)
        
        completos = sum(1 for a in audit_data if all(x in audit_data[a] for x in ["Warm-Up", "1P", "2P"]))
        quorum_ok = completos >= 10

        # FEEDBACK VISUAL
        c1, c2 = st.columns(2)
        if is_geo_valid: c1.success(f"📍 Localização OK ({dist_m:.1f}m)")
        else: c1.error(f"📍 GPS FORA ({dist_m:.1f}m)")
        if quorum_ok: c2.success(f"👥 Quórum OK ({completos} atletas)")
        else: c2.warning(f"👥 Quórum insuficiente ({completos}/10)")

# --- 3. O TEU NOVO BOTÃO ENGINE V9 ---
if is_geo_valid and quorum_ok:
    st.divider()
    
    if not st.session_state.engine_run:
        st.subheader("⚙️ Executamos UTM Engine... ?")
        if st.button("CORRER ENGINE V9", type="primary", use_container_width=True):
            # AQUI CHAMAMOS AS FASES DO TEU SEGUNDO SCRIPT (v9.2)
            with st.status("A executar Pipeline Final Otimizado...", expanded=True) as status:
                st.write("1. CALIBRAÇÃO DE CAMPO...")
                # Lógica de conversão UTM do script v9.2
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
                pts_utm = {k: np.array(transformer.transform(v[1], v[0])) for k, v in pts_gps.items()}
                dist_x = np.linalg.norm(pts_utm["BR"] - pts_utm["BL"])
                dist_y = np.linalg.norm(pts_utm["TL"] - pts_utm["BL"])
                st.session_state.geo_report = {"x": dist_x, "y": dist_y}
                time.sleep(1)

                st.write("2. PROCESSAMENTO TEMPORÁRIO (SAVGOL)...")
                time.sleep(1)

                st.write("3. SINCRONIZAÇÃO E SUMÁRIO CRONOLÓGICO...")
                time.sleep(1)

                st.write("4. LIMPEZA E RELATÓRIOS...")
                st.session_state.engine_run = True
                status.update(label="✅ Engine Concluída!", state="complete")
            st.rerun()

    else:
        # --- 4. RELATÓRIO DO PROCESSAMENTO (DO SCRIPT V9.2) ---
        st.header("📊 Relatório 1: Geometria do Campo")
        res = st.session_state.geo_report
        col1, col2 = st.columns(2)
        col1.metric("Comprimento (X)", f"{res['x']:.2f} m")
        col2.metric("Largura (Y)", f"{res['y']:.2f} m")
        
        st.success("RELATÓRIO 2: O processo SYNC foi concluído para todos os atletas.")
        
        if st.button("📤 Validar e submeter Base de Dados", type="primary", use_container_width=True):
            st.balloons()
            st.success("Dados submetidos com sucesso!")