import streamlit as st
import pandas as pd
import numpy as np
import folium
from pyproj import Geod
GEOD = Geod(ellps='WGS84')  # WGS84 geodesic distance

from streamlit_folium import st_folium
import re

# --- 1. CONFIGURAÇÃO DE UI (LOG IN CENTRALIZADO) ---
st.set_page_config(page_title="FPF Performance Hub", layout="centered")

def apply_login_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        /* Caixa de Login Compacta e Centralizada */
        .main .block-container {
            max-width: 400px !important;
            background-color: #1a1c23 !important;
            padding: 40px !important;
            border-radius: 12px;
            border: 1px solid #30363d;
            margin: auto;
            margin-top: 15vh;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.5);
        }
        header, footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        /* Botão ENTRAR Vermelho FPF */
        .stButton>button {
            width: 100%;
            background-color: #E30613 !important;
            color: white !important;
            font-weight: bold;
            border: none;
            height: 3.2em;
        }
        h2 { text-align: center; color: white; margin-bottom: 25px; }
        </style>
    """, unsafe_allow_html=True)

# --- 2. FASE 1: LOG IN (Utilizando a tua lógica funcional) ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    apply_login_style()
    st.markdown("## FPF Performance Hub")
    
    # Adicionadas chaves (key) para os inputs não perderem o valor
    user_input = st.text_input("Utilizador", key="user_val")
    pass_input = st.text_input("Password", type="password", key="pass_val")
    
    if st.button("ENTRAR"):
        if user_input == "miguel.cardoso" and pass_input == "fpf2026":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Credenciais Inválidas") #
    st.stop()

# --- INTERFACE SINGLE PAGE ---
st.title("Validação de Dados")

with st.sidebar:
    st.header("📤 Upload de Ficheiros")
    f_campo = st.file_uploader("Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True)
    f_atleta = st.file_uploader("Dados de ATLETAS (CSVs)", accept_multiple_files=True)

if f_campo and f_atleta:
    # 1. PROCESSAMENTO DE CAMPO
    pts_gps = {}
    for f in f_campo:
        df_c = pd.read_csv(f, sep=None, engine='python')
        df_c.columns = [c.strip().replace('"', '') for c in df_c.columns]
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper():
                pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]

    if len(pts_gps) == 4:
        # Coordenadas Reais do Campo
        clat, clon = np.mean([p[0] for p in pts_gps.values()]), np.mean([p[1] for p in pts_gps.values()])
        
        
# 2. VALIDAÇÃO CRUZADA (GEO-FENCING APERTADO - 50m)
atleta_file = f_atleta[0]

# Garantir que o ponteiro do ficheiro está no início (upload stream)
try:
    atleta_file.seek(0)
except Exception:
    pass

# Amostra robusta: usa até 500 linhas e mediana (evita outliers do primeiro registo)
sample_atl = pd.read_csv(atleta_file, sep=None, engine='python', nrows=500)
sample_atl.columns = [c.strip().replace('"', '') for c in sample_atl.columns]

if "Lat" not in sample_atl.columns or "Lon" not in sample_atl.columns:
    st.error("❌ CSV do atleta não contém colunas 'Lat' e 'Lon'.")
    st.stop()

sub = sample_atl[["Lat", "Lon"]].dropna()
if sub.empty:
    st.error("❌ Sem amostras Lat/Lon válidas no CSV do atleta (NaNs).")
    st.stop()

alat = float(sub["Lat"].median())
alon = float(sub["Lon"].median())

# Distância geodésica (m) entre atleta e centro do campo (WGS84)
_, _, dist_metros = GEOD.inv(alon, alat, clon, clat)

# Validação: atleta deve estar dentro de 50 m do centro
is_geo_valid = dist_metros < 50

# Reset para não afetar leituras posteriores do mesmo ficheiro
try:
    atleta_file.seek(0)
except Exception:
    pass

        # --- EXIBIÇÃO ---
        st.header("📍 Identificação do Campo")
        if is_geo_valid:
            st.success(f"✅ LOCALIZAÇÃO VALIDADA: Atletas e Campo na mesma localizaçã. Atletas a {dist_metros:.1f}m do centro do campo.")
        else:
            st.error(f"❌ ERRO CRÍTICO DE LOCALIZAÇÃO: Os atletas estão a {dist_metros:.1f}m do campo. Limite máximo: 50m.")

        m = folium.Map(location=[clat, clon], zoom_start=18)
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery', name='Esri (Satélite)').add_to(m)
        for k, v in pts_gps.items():
            folium.Marker(v, popup=f"Canto {k}", icon=folium.Icon(color='red' if is_geo_valid else 'black')).add_to(m)
        st_folium(m, width=1100, height=450, key="mapa_v11_1")

        st.divider()

        # 3. AUDITORIA DE ATLETAS
        st.header("Auditoria de Atletas")
        audit_data = {}
        for f in f_atleta:
            match = re.search(r"Player-(\d+)", f.name)
            aid = match.group(0) if match else "Player-?"
            n = f.name.upper()
            fase = "Warm-Up" if "WARM" in n else "1P" if any(x in n for x in ["1P", "PRIMEIRA"]) else "2P" if any(x in n for x in ["2P", "SEGUNDA"]) else "Extra"
            if aid not in audit_data: audit_data[aid] = []
            audit_data[aid].append(fase)

        rows = []
        completos = 0
        for aid in sorted(audit_data.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
            f_list = audit_data[aid]
            is_ok = all(x in f_list for x in ["Warm-Up", "1P", "2P"])
            if is_ok: completos += 1
            rows.append({
                "ID Atleta": aid,
                "Ficheiros": len(f_list),
                "Estado": "✅ OK" if is_ok else "❌ INCOMPLETO",
                "Fases": ", ".join(set(f_list))
            })
        
        st.table(pd.DataFrame(rows))
        
        # Verificação de Quórum
        quorum_ok = completos >= 10
        if quorum_ok:
            st.success(f"✅ Quórum Atingido: {completos} atletas validados.")
        else:
            st.warning(f"⚠️ Quórum Insuficiente: Precisamos de 10 atletas completos. Tens apenas {completos}.")

        st.divider()

        # 4. SUBMISSÃO FINAL
        pode_submeter = is_geo_valid and quorum_ok
        if st.button("Validar e submeter Base de Dados", type="primary", use_container_width=True, disabled=not pode_submeter):
            st.success("Tudo em conformidade. Dados prontos para integração.")

    else:
        st.warning("⚠️ Aguardando os 4 cantos do campo (BL, BR, TL, TR).")
else:
    st.info("👋 Por Favor, carregar os dados no menu lateral para iniciar.")

