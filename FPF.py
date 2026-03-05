# -*- coding: utf-8 -*-
"""
FPF UTM Engine v11.1 (fix indent + report + metrics)
Autor: Marcos (base) + ajustes de estabilidade/indentação
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from pyproj import Geod, Transformer
from pathlib import Path
import tempfile
import hashlib
import uuid
import requests
import re
import io

from streamlit_folium import st_folium
from scipy.signal import savgol_filter

# -- Local Modules -- #
from fpf_modules.constants import (
    HSR_MPS, SPRINT_MPS, ACC_THR, DEC_THR, SPRINT_BOUT_MIN_S, ENGINE_VERSION,
    COL_LAT, COL_LON, COL_TIME, COL_FASE,
    SELECOES_OPCOES
)

from fpf_modules.metrics import (
    time_to_seconds,
    count_bouts,
    audit_timebase,
    compute_metrics_for_df
)

from fpf_modules.io_utils import (
    hash_session,
    clean_cols,
    get_atleta_id,
    infer_fase,
    read_csv_upload
)

from fpf_modules.geo import (
    calibrar_campo,
    order_corners_latlon,
    retangularizar_cantos_latlon,
    geo_validacao_por_atleta,
    calibrar_campo_from_pts_gps,
    sample_athlete_track_latlon,
    reverse_geocode_place_city_country,
    get_atletas_centroid_latlon
)

from fpf_modules.pipeline import (
    processar_atletas_para_temp,
    sincronizar
)

GEOD = Geod(ellps="WGS84")  # WGS84 geodesic distance (metros reais)

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v11.1", layout="wide")
if "auth" not in st.session_state:
    st.session_state.auth = False
if "login_user" not in st.session_state:
    st.session_state.login_user = ""

# --- Persistência de outputs (evita desaparecer após zoom/scroll no mapa) ---
if "df_metrics" not in st.session_state:
    st.session_state.df_metrics = None
if "report_txt" not in st.session_state:
    st.session_state.report_txt = None
if "process_done" not in st.session_state:
    st.session_state.process_done = False

# --- Persistência para 'Pick no mapa' (cantos do campo) ---
if "pick_corners" not in st.session_state:
    st.session_state.pick_corners = []  # lista [(lat, lon), ...]
if "pts_gps_picked" not in st.session_state:
    st.session_state.pts_gps_picked = None  # dict com BL/BR/TL/TR após ordenação
if "pick_last_click_sig" not in st.session_state:
    st.session_state.pick_last_click_sig = None  # evita duplicar o mesmo clique após rerun

# --- LOGIN (CENTRADO + st.secrets) ---
def _apply_login_style():
    css = """
<style>
  .stApp { background-color: #0e1117; }
  header, footer {visibility: hidden;}
  [data-testid="stSidebar"] {display: none;}

  /* Container geral transparente */
  .main .block-container {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding-top: 2rem !important;
  }

  .login-card{
    background: #1a1c23;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 34px 34px 26px 34px;
    box-shadow: 0px 10px 28px rgba(0,0,0,0.55);
  }

  .login-title{
    text-align: center;
    color: #ffffff;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 1.25rem 0;
  }

  .login-card .stTextInput input{
    background: #0e1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
  }

  .login-card .stButton > button{
    width: 100%;
    background: #E30613 !important;
    color: #fff !important;
    font-weight: 800;
    border: 0;
    height: 3.1em;
    border-radius: 10px;
  }

  .login-card .stButton > button:hover{
    filter: brightness(0.95);
  }
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

def _get_auth_from_secrets():
    """
    Espera em st.secrets:
    [auth]
    username = "..."
    password = "..."
    """
    try:
        auth = st.secrets["auth"]
        return auth.get("username"), auth.get("password")
    except Exception:
        return None, None


if not st.session_state.auth:
    _apply_login_style()

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="login-title">FPF Performance Hub</div>', unsafe_allow_html=True)

        u = st.text_input("Utilizador", key="user_val")
        p = st.text_input("Password", type="password", key="pass_val")

        secrets_user, secrets_pass = _get_auth_from_secrets()
        if not secrets_user:
            st.warning("⚠️ Credenciais não configuradas em st.secrets. Defina [auth] no secrets.toml / Streamlit Cloud.")

        if st.button("Entrar"):
            if secrets_user and u == secrets_user and p == secrets_pass:
                st.session_state.login_user = u
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Credenciais inválidas")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# --- INTERFACE SINGLE PAGE ---
st.title("Validação de Dados")

with st.sidebar:
    st.markdown(
    f"<div style='text-align:left; font-size:18px; color:#9aa0a6;'>User: {st.session_state.login_user} | FPF</div>",
    unsafe_allow_html=True
)
    st.header("Dados da Sessão")
    # Estádio agora é inferido automaticamente pela localização do campo (sem input manual)
    estadio = None
    data_sessao = st.date_input("Data do Evento")
    selecao = st.selectbox("Seleção", options=SELECOES_OPCOES, index=0)
    # Género é inferido da seleção (M/F), não é input manual
    genero = selecao.split()[-1] if selecao.split() and selecao.split()[-1] in ["M", "F"] else ""
    contexto = st.selectbox("Contexto", options=["Treino", "Jogo"], index=0)

    adversario = ""
    if contexto == "Jogo":
        adversario = st.text_input("Adversário")
    st.divider()

    st.header("🗺️ Calibração do Campo")
    st.caption("Define os 4 cantos via upload (BL/BR/TL/TR) ou usando o modo 'Pick no mapa'.")

    metodo_campo = st.radio(

        "Como queres definir os 4 cantos?",

        options=["Upload (BL/BR/TL/TR)", "Pick no mapa (clicar 4 cantos)"],

        index=0,

        help="Alternativa ao upload: usa um mapa satélite e clica nos 4 cantos do campo.",

    )

    st.caption("Se escolheres 'Pick no mapa', não precisas de carregar os 4 CSVs do campo.")



    f_campo = st.file_uploader(
        "Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True, type=["csv"]
    )
    st.caption(
        "Atletas: CSVs com Player-<id> e indicação de fase (Warm/Primeira/Segunda/1P/2P) no nome do ficheiro."
    )
    f_atleta = st.file_uploader(
        "Dados de ATLETAS (CSVs)", accept_multiple_files=True, type=["csv"]
    )

st.divider()

# Defaults (menu de opções removido)
epsg_used = 32629
raio_validacao_m = 50
amostra_geo_n = 500
min_pct_atletas_ok = 0.80
aplicar_suavizacao = True
janela_savgol = 11
poly_savgol = 2


@st.cache_data(show_spinner=False, ttl=86400)
def _reverse_geocode_city_country(lat: float, lon: float):
    """Reverse geocode via OpenStreetMap Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1}
        headers = {"User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, None
        data = r.json()
        addr = data.get("address", {}) if isinstance(data, dict) else {}
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or addr.get("county")
        )
        country = addr.get("country")
        return city, country
    except Exception:
        return None, None




@st.cache_data(show_spinner=False, ttl=86400)



def _processar_atletas_para_temp(
    f_atleta_files, epsg, origin, R, aplicar_suav, janela, poly, temp_dir: Path
):
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    groups = {}
    for f in f_atleta_files:
        aid = get_atleta_id(f.name)
        groups.setdefault(aid, []).append(f)

    temp_files = []
    audit = {}  # aid -> list of fases
    issues = []

    for aid, files in groups.items():
        atleta_data = []
        audit[aid] = []
        for uf in files:
            try:
                df = read_csv_upload(uf)
                fase_n = infer_fase(uf.name)
                audit[aid].append(fase_n)

                if df.empty:
                    continue
                if COL_TIME not in df.columns:
                    issues.append((aid, uf.name, "Sem coluna Time"))
                    continue
                if COL_LAT not in df.columns or COL_LON not in df.columns:
                    issues.append((aid, uf.name, "Sem colunas Lat/Lon"))
                    continue

                lon = df[COL_LON].astype(float)
                lat = df[COL_LAT].astype(float)
                ux, uy = trans.transform(lon.values, lat.values)
                p = np.vstack([ux, uy]).T
                p_loc = (R @ (p - origin).T).T  # rotate around BL

                df["X_UTM"] = p_loc[:, 0]
                df["Y_UTM"] = p_loc[:, 1]


                # Micro-gaps (≤1 amostra consecutiva): contagem + preenchimento
                nan_before = int(df["X_UTM"].isna().sum() + df["Y_UTM"].isna().sum())
                df["X_UTM"] = df["X_UTM"].interpolate(limit=1, limit_direction="both")
                df["Y_UTM"] = df["Y_UTM"].interpolate(limit=1, limit_direction="both")
                nan_after = int(df["X_UTM"].isna().sum() + df["Y_UTM"].isna().sum())
                df["_micro_gaps_corrigidos"] = max(0, nan_before - nan_after)
                # Suavização opcional Savitzky–Golay

                if aplicar_suavizacao and len(df) >= int(janela) and int(janela) % 2 == 1:
                    x = pd.Series(df["X_UTM"]).interpolate()
                    y = pd.Series(df["Y_UTM"]).interpolate()
                    try:
                        df["X_UTM"] = savgol_filter(x, int(janela), int(poly))
                        df["Y_UTM"] = savgol_filter(y, int(janela), int(poly))
                    except Exception:
                        pass

                df[COL_FASE] = fase_n
                df["Atleta_ID"] = aid
                atleta_data.append(
                    df[
                        [
                            COL_TIME,
                            "Atleta_ID",
                            COL_FASE,
                            COL_LAT,
                            COL_LON,
                            "X_UTM",
                            "Y_UTM",
                        ]
                    ]
                )
            except Exception as e:
                issues.append((aid, uf.name, f"Erro a processar: {e}"))

        if atleta_data:
            out = pd.concat(atleta_data, ignore_index=True)
            out = out.sort_values(by=COL_TIME)
            out_path = temp_dir / f"T_{aid}.csv"
            out.to_csv(out_path, sep=";", index=False)
            temp_files.append(out_path)

    return temp_files, audit, issues


def _sincronizar(temp_files, out_dir: Path):
    fases_dict = {}
    all_unique_times = set()

    # Pass 1: collect times and phase windows
    for f in temp_files:
        df = pd.read_csv(f, sep=";", usecols=[COL_TIME, COL_FASE])
        df = df.dropna(subset=[COL_TIME])
        all_unique_times.update(df[COL_TIME].tolist())
        for fs in df[COL_FASE].dropna().unique():
            t_fase = df.loc[df[COL_FASE] == fs, COL_TIME]
            t_s, t_e = t_fase.min(), t_fase.max()
            if fs not in fases_dict:
                fases_dict[fs] = [t_s, t_e]
            else:
                fases_dict[fs][0] = min(fases_dict[fs][0], t_s)
                fases_dict[fs][1] = max(fases_dict[fs][1], t_e)

    master_df = pd.DataFrame({COL_TIME: sorted(list(all_unique_times))})

    out_files = []
    for f in temp_files:
        df_atl = pd.read_csv(f, sep=";")
        df_sync = pd.merge(master_df, df_atl, on=COL_TIME, how="left")
        aid = (
            str(df_atl["Atleta_ID"].iloc[0])
            if "Atleta_ID" in df_atl.columns
            else f.stem.replace("T_", "")
        )
        df_sync["Atleta_ID"] = aid

        # fill phase windows
        for fase, (t_s, t_e) in fases_dict.items():
            mask = (df_sync[COL_TIME] >= t_s) & (df_sync[COL_TIME] <= t_e)
            df_sync.loc[mask, COL_FASE] = fase

        out_path = out_dir / f"Player_{aid}_SYNC.csv"
        df_sync.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
        out_files.append(out_path)

    ordem_fases = {"Warm-Up": 0, "1P": 1, "2P": 2}
    fases_ordenadas = sorted(fases_dict.items(), key=lambda x: ordem_fases.get(x[0], 99))

    return out_files, fases_ordenadas, fases_dict, len(master_df)


# -------------------------------
# Main flow
# -------------------------------
if not f_atleta:
    st.info("👋 Carrega os ficheiros de ATLETAS na barra lateral para iniciar.")
    st.stop()

have_upload_corners = bool(f_campo)
have_picked_corners = st.session_state.get("pts_gps_picked") is not None

if metodo_campo == "Upload (BL/BR/TL/TR)" and not have_upload_corners:
    st.info("👋 Selecionaste 'Upload', mas ainda não carregaste os 4 CSVs do campo (BL/BR/TL/TR).")
    st.stop()

if metodo_campo == "Pick no mapa (clicar 4 cantos)" and not have_picked_corners:
    st.warning("ℹ️ Selecionaste 'Pick no mapa'. Define os 4 cantos no mapa abaixo e depois continua.")



if metodo_campo == "Pick no mapa (clicar 4 cantos)" and st.session_state.get("pts_gps_picked") is None:
    # --- UI: Pick dos 4 cantos no mapa (alternativa ao upload) ---
    # Cursor crosshair para maior precisão no click
    st.markdown(
        '''
        <style>
          .leaflet-container { cursor: crosshair !important; }
          div[data-testid="stFOLIUM"] * { cursor: crosshair !important; }
        </style>
        ''',
        unsafe_allow_html=True,
    )

    alat0, alon0 = get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)
    if alat0 is None or alon0 is None:
        pts_fallback = sample_athlete_track_latlon(f_atleta, max_points=10)
        if pts_fallback:
            alat0, alon0 = pts_fallback[0]
        else:
            alat0, alon0 = 0.0, 0.0

    st.header("Pick dos 4 cantos do campo")
    st.caption(
        "Clica no mapa 4 vezes (um por canto). Depois de 4 picks, aplico uma retangularização automática "
        "(corrige desvios) e mostro os pontos ajustados + o retângulo final."
    )

    m_pick = folium.Map(location=[alat0, alon0], zoom_start=18)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri (Satélite)",
    ).add_to(m_pick)

    pts_track = sample_athlete_track_latlon(f_atleta, max_points=600)
    if pts_track:
        folium.PolyLine(pts_track, weight=2, opacity=0.8).add_to(m_pick)

    # 1) Markers: pontos clicados
    for i, (lat, lon) in enumerate(st.session_state.pick_corners, start=1):
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="yellow",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Clicado {i}",
        ).add_to(m_pick)

    # 1.1) Feedback visual imediato: polígono dos pontos clicados (fecha quando tiver 4)
    if len(st.session_state.pick_corners) >= 2:
        poly_clicked = list(st.session_state.pick_corners)
        if len(st.session_state.pick_corners) == 4:
            poly_clicked = poly_clicked + [poly_clicked[0]]
        folium.PolyLine(
            locations=poly_clicked,
            color="yellow",
            weight=2,
            opacity=0.9,
            dash_array="6,6",
            tooltip="Perímetro (pontos clicados)",
        ).add_to(m_pick)

    # 2) Se já temos 4 pontos, calcular retangularização e desenhar versão ajustada
    pts_clicked_dict = None
    pts_rect_dict = None
    if len(st.session_state.pick_corners) == 4:
        try:
            pts_clicked_dict, pts_rect_dict = retangularizar_cantos_latlon(
                st.session_state.pick_corners, epsg=int(epsg_used)
            )

            # markers ajustados (cores diferentes)
            for k, (lat, lon) in pts_rect_dict.items():
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="cyan",
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=f"Ajustado {k}",
                ).add_to(m_pick)

            # polígono final (retângulo ajustado)
            poly = [
                pts_rect_dict["TL"],
                pts_rect_dict["TR"],
                pts_rect_dict["BR"],
                pts_rect_dict["BL"],
            ]
            folium.Polygon(
                locations=poly,
                color="cyan",
                weight=3,
                fill=False,
                tooltip="Retângulo final (ajustado)",
            ).add_to(m_pick)

        except Exception as e:
            st.error(f"Falha ao retangularizar cantos: {e}")
            pts_clicked_dict, pts_rect_dict = None, None

    out_pick = st_folium(m_pick, width=1100, height=520, key="mapa_pick_cantos")

    # Capturar clique (com deduplicação para evitar reprocessar o mesmo ponto após rerun)
    if out_pick and out_pick.get("last_clicked"):
        lat = float(out_pick["last_clicked"]["lat"])
        lon = float(out_pick["last_clicked"]["lng"])
        click_sig = f"{lat:.7f},{lon:.7f}"
        if len(st.session_state.pick_corners) < 4 and click_sig != st.session_state.pick_last_click_sig:
            st.session_state.pick_last_click_sig = click_sig
            st.session_state.pick_corners.append((lat, lon))
            st.rerun()

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("↩️ Desfazer", disabled=(len(st.session_state.pick_corners) == 0)):
            st.session_state.pick_corners.pop()
            st.session_state.pick_last_click_sig = None
            st.rerun()
    with c2:
        if st.button("🧹 Reset"):
            st.session_state.pick_corners = []
            st.session_state.pick_last_click_sig = None
            st.session_state.pts_gps_picked = None
            st.rerun()

    if len(st.session_state.pick_corners) < 4:
        st.info(f"Pontos escolhidos: {len(st.session_state.pick_corners)}/4")
    else:
        if pts_clicked_dict and pts_rect_dict:
            st.subheader("Cantos (clicados)")
            st.json(pts_clicked_dict)
            st.subheader("Cantos (ajustados - usados no pipeline)")
            st.json(pts_rect_dict)

            # Guardar já ajustado para o pipeline
            st.session_state.pts_gps_picked = pts_rect_dict
            st.success("✅ Cantos ajustados guardados. Agora o pipeline continua normalmente.")
        else:
            st.warning("Tens 4 pontos, mas não consegui ajustar. Faz Reset e tenta com mais zoom.")

    st.stop()

try:
    if metodo_campo == "Pick no mapa (clicar 4 cantos)":
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo_from_pts_gps(
            st.session_state.pts_gps_picked, int(epsg_used)
        )
    else:
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo(
            f_campo, int(epsg_used)
        )

    estadio, cidade, pais = reverse_geocode_place_city_country(clat, clon)
except Exception as e:

    st.error(f"❌ Erro na calibração do campo: {e}")
    st.stop()

passed_geo, pct_ok, ok_list, fora_list, geo_errors = geo_validacao_por_atleta(
    f_atleta, clat, clon, float(raio_validacao_m), int(amostra_geo_n), float(min_pct_atletas_ok)
)

st.header("Validação de Localização (Campo ↔ Atletas)")

# Campo
_, cidade_campo, pais_campo = reverse_geocode_place_city_country(clat, clon)

# Atletas (centro estimado)
alat, alon = get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)

cidade_atl, pais_atl = None, None
if alat is not None and alon is not None:
    _, cidade_atl, pais_atl = reverse_geocode_place_city_country(alat, alon)
# ----- Campo -----
campo_local = ", ".join([p for p in [cidade_campo, pais_campo] if p]) or "—"

# ----- Atletas (centro médio → Cidade/País) -----
atletas_parts = [p for p in [cidade_atl, pais_atl] if p]
atletas_local = ", ".join(atletas_parts) if atletas_parts else "—"

st.markdown(f"**Campo, Local:** {campo_local}")
st.markdown(f"**Atletas, Local:** {atletas_local}")
st.markdown(f"**% Atletas OK:** {pct_ok*100:.0f}%")

st.markdown("---")


if passed_geo:
    st.success("✅ Validação geográfica aprovada.")
else:
    st.error("❌ Validação geográfica falhou (percentagem insuficiente dentro do raio).")


# Map
m = folium.Map(location=[clat, clon], zoom_start=18)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri (Satélite)",
).add_to(m)
for k, v in pts_gps.items():
    folium.Marker(v, popup=f"Canto {k}").add_to(m)
st_folium(m, width=1100, height=450, key="mapa_pipeline")

st.divider()

# Audit by athlete phases
st.header("Auditoria de Atletas")
audit_data = {}
for f in f_atleta:
    aid = get_atleta_id(f.name)
    audit_data.setdefault(aid, [])
    audit_data[aid].append(infer_fase(f.name))

rows = []
completos = 0
for aid in sorted(
    audit_data.keys(),
    key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
):
    fases = audit_data[aid]
    is_ok = all(x in fases for x in ["Warm-Up", "1P", "2P"])
    if is_ok:
        completos += 1
    rows.append(
        {
            "ID Atleta": aid,
            "Ficheiros": len(fases),
            "Estado": "✅ OK" if is_ok else "❌ INCOMPLETO",
            "Fases": ", ".join(sorted(set(fases))),
        }
    )
st.table(pd.DataFrame(rows))
st.write(f"**Atletas completos (Warm-Up + 1P + 2P):** {completos} / {len(audit_data)}")

st.divider()

# Normalization + export
st.header("Normalização | Calculo Métricas")

if not passed_geo:
    st.warning(
        "A exportação está desativada porque a validação geográfica falhou. Ajusta o raio/% mínimo ou verifica os ficheiros."
    )
    st.stop()

btn = st.button("⚙️ Processar e Gerar Relatório", type="primary", use_container_width=True)

# outputs (para UI) — manter em session_state para sobreviver a reruns
df_metrics = st.session_state.df_metrics
report_txt = st.session_state.report_txt

if btn:
    with st.status("A iniciar processamento...", expanded=True) as status:
        if not passed_geo:
            status.update(label="Validação geográfica falhou. Processamento interrompido.", state="error")
            st.error("Validação geográfica falhou. O processamento foi interrompido.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_dir = td_path / "Temp_Processing"
            out_dir = td_path / "Output_UTM_Sincronizado"
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            status.update(label="Processamento e limpeza de dados GPS...", state="running")
            temp_files, audit_proc, issues = _processar_atletas_para_temp(
                f_atleta,
                int(epsg_used),
                origin,
                R,
                aplicar_suavizacao,
                int(janela_savgol),
                int(poly_savgol),
                temp_dir,
            )

            if not temp_files:
                st.error("❌ Não foi possível gerar ficheiros temporários (verifica colunas Time/Lat/Lon e nomes).")
                st.stop()

            status.update(label="Sincronização temporal...", state="running")
            out_files, fases_ordenadas, fases_dict, n_master = _sincronizar(temp_files, out_dir)

            # Session identifiers (auditoria/dedup)
            session_uuid = uuid.uuid4()
            session_id_hex = session_uuid.hex
            session_fingerprint = hash_session(
                data_sessao, selecao, genero, contexto, estadio, f_campo, f_atleta
            )

            # Métricas individuais a partir dos SYNC (por fase + Total)
            status.update(label="Cálculo de métricas individuais...", state="running")
            metrics_rows = []
            audit_time_rows = []
            total_micro_gaps = 0

            fases_target = ["Warm-Up", "1P", "2P"]

            for pth in out_files:
                df_sync = pd.read_csv(pth, sep=";")
                aid = (
                    str(df_sync["Atleta_ID"].dropna().iloc[0])
                    if "Atleta_ID" in df_sync.columns and df_sync["Atleta_ID"].dropna().any()
                    else Path(pth).stem
                )

                # micro-gaps acumulados (gravados na etapa de processamento)
                if "_micro_gaps_corrigidos" in df_sync.columns and df_sync["_micro_gaps_corrigidos"].notna().any():
                    try:
                        total_micro_gaps += int(df_sync["_micro_gaps_corrigidos"].dropna().iloc[0])
                    except Exception:
                        pass

                fase_mets = {}

                for fase in fases_target:
                    df_f = df_sync[df_sync[COL_FASE] == fase].copy()
                    met = compute_metrics_for_df(df_f)
                    fase_mets[fase] = met

                    aud = audit_timebase(df_f, COL_TIME, expected_hz=10.0)
                    audit_time_rows.append({"atleta_id": aid, "fase": fase, **aud})

                    metrics_rows.append(
                        {
                            "session_id_hex": session_id_hex,
                            "session_fingerprint": session_fingerprint,
                            "data": data_sessao,
                            "selecao": selecao,
                            "genero": genero,
                            "contexto": contexto,
                            "jogo": (
                                adversario.strip()
                                if contexto == "Jogo"
                                else ""
                            ),
                            "estadio": estadio,  # apenas para BD
                            "cidade": cidade,
                            "pais": pais,
                            "atleta_id": aid,
                            "fase": fase,
                            **met,
                            "engine_version": ENGINE_VERSION,
                        }
                    )

                # -------- TOTAL POR SOMA DAS FASES --------
                met_total = {}
                met_total["dist_m"] = sum(fase_mets.get(f, {}).get("dist_m", 0.0) for f in fases_target)
                met_total["duracao_min"] = sum(fase_mets.get(f, {}).get("duracao_min", 0.0) for f in fases_target)
                met_total["m_min"] = (
                    met_total["dist_m"] / met_total["duracao_min"]
                    if met_total["duracao_min"] > 0
                    else np.nan
                )

                met_total["hsr_dist_m"] = sum(fase_mets.get(f, {}).get("hsr_dist_m", 0.0) for f in fases_target)
                met_total["sprint_dist_m"] = sum(fase_mets.get(f, {}).get("sprint_dist_m", 0.0) for f in fases_target)
                met_total["n_sprints"] = sum(fase_mets.get(f, {}).get("n_sprints", 0) for f in fases_target)
                met_total["n_acc_2_5"] = sum(fase_mets.get(f, {}).get("n_acc_2_5", 0) for f in fases_target)
                met_total["n_dec_3_0"] = sum(fase_mets.get(f, {}).get("n_dec_3_0", 0) for f in fases_target)
                met_total["n_points"] = sum(fase_mets.get(f, {}).get("n_points", 0) for f in fases_target)

                met_total["vmax_mps"] = max((fase_mets.get(f, {}).get("vmax_mps", np.nan) for f in fases_target), default=np.nan)
                met_total["peak_1m_m_min"] = max((fase_mets.get(f, {}).get("peak_1m_m_min", np.nan) for f in fases_target), default=np.nan)

                met_total["hsr_pct"] = (
                    met_total["hsr_dist_m"] / met_total["dist_m"] * 100.0
                    if met_total["dist_m"] > 0
                    else np.nan
                )

                # Active time total
                met_total["active_time_min"] = sum(fase_mets.get(f, {}).get("active_time_min", 0.0) for f in fases_target)
                dur_total_s = met_total["duracao_min"] * 60.0
                met_total["active_pct"] = (
                    (met_total["active_time_min"] * 60.0) / dur_total_s * 100.0
                    if dur_total_s > 0
                    else np.nan
                )

                # Qualidade: pct_time_valid e gaps>2s — média ponderada simples por pontos válidos
                try:
                    w = np.array([max(1, fase_mets[f]["n_points"]) for f in fases_target], dtype=float)
                    met_total["pct_time_valid"] = float(
                        np.average([fase_mets[f]["pct_time_valid"] for f in fases_target], weights=w)
                    )
                    met_total["n_gaps_gt2s"] = int(sum(fase_mets[f]["n_gaps_gt2s"] for f in fases_target))
                except Exception:
                    met_total["pct_time_valid"] = np.nan
                    met_total["n_gaps_gt2s"] = int(sum(fase_mets[f]["n_gaps_gt2s"] for f in fases_target))

                metrics_rows.append(
                    {
                        "session_id_hex": session_id_hex,
                        "session_fingerprint": session_fingerprint,
                        "data": data_sessao,
                        "selecao": selecao,
                        "genero": genero,
                        "contexto": contexto,
                        "jogo": (
                            adversario.strip()
                            if contexto == "Jogo"
                            else ""
                        ),
                        "estadio": estadio,  # apenas para BD
                        "cidade": cidade,
                        "pais": pais,
                        "atleta_id": aid,
                        "fase": "Total",
                        **met_total,
                        "engine_version": ENGINE_VERSION,
                    }
                )

            df_metrics = pd.DataFrame(metrics_rows)
            df_time_audit = pd.DataFrame(audit_time_rows)
            st.session_state.df_time_audit = df_time_audit


            status.update(label="Construção do relatório...", state="running")
            # Build report (rotação mantida)
            rot_deg = float(np.degrees(angulo_rad))
            report_lines = []
            report_lines.append("FPF Performance Hub — Relatório de Validação e Normalização")
            report_lines.append("=" * 70)

            report_lines.append("Dados da Sessão")
            report_lines.append(
                f"  Data: {data_sessao.strftime('%d/%m/%Y') if hasattr(data_sessao, 'strftime') else data_sessao}"
            )
            report_lines.append(f"  Seleção: {selecao} | Género: {genero} | Contexto: {contexto}")
            if contexto == "Jogo":
                vs_txt = adversario.strip()
                if vs_txt:
                    report_lines.append(f"  Jogo: {vs_txt}")

            loc_part = ", ".join([p for p in [cidade, pais] if p]) or "—"
            report_lines.append(f"  Localização: {loc_part}")

            report_lines.append(f"EPSG (UTM): {epsg_used}")
            report_lines.append(f"Comprimento (BL→BR): {dist_x:.2f} m")
            report_lines.append(f"Largura (BL→TL):     {dist_y:.2f} m")
            report_lines.append(f"Rotação aplicada:    {rot_deg:.2f}° (alinhamento BL→BR com eixo X)")

            report_lines.append("-" * 70)
            report_lines.append("Validação geográfica")
            report_lines.append(
                f"  Raio: {raio_validacao_m:.0f} m | Amostra: {amostra_geo_n} linhas/atleta | % OK: {pct_ok*100:.0f}% "
                f"(mínimo {min_pct_atletas_ok*100:.0f}%)"
            )
            report_lines.append(
                f"  Dentro do raio: {len({a for a,_ in ok_list})} atletas | "
                f"Fora: {len({a for a,_ in fora_list})} atletas | Erros: {len(geo_errors)}"
            )

            report_lines.append("-" * 70)
            report_lines.append("Auditoria de atletas (submissão)")
            report_lines.append(
                f"  Atletas totais: {len(audit_data)} | Atletas completos (Warm-Up+1P+2P): {completos}"
            )

            report_lines.append("-" * 70)
            report_lines.append("Sincronização")
            report_lines.append(f"  Timestamps mestre: {n_master}")
            report_lines.append(f"  Ficheiros gerados: {len(out_files)}")
            report_lines.append("  Fases (ordem cronológica):")
            for fase, (t_s, t_e) in fases_ordenadas:
                report_lines.append(f"    - {fase:8} | início: {t_s} | fim: {t_e}")

            if issues:
                report_lines.append("-" * 70)
                report_lines.append("Avisos/Problemas (exemplos):")
                for aid, fn, msg in issues[:25]:
                    report_lines.append(f"  - {aid} | {fn} | {msg}")

            report_lines.append("-" * 70)
            report_lines.append("Métricas Individuais (GPS-only) — thresholds fixos")
            report_lines.append(
                f"  HSR ≥ {HSR_MPS:.1f} m/s | Sprint ≥ {SPRINT_MPS:.1f} m/s | "
                f"Acc ≥ {ACC_THR:.1f} m/s² | Dec ≤ {DEC_THR:.1f} m/s²"
            )
            report_lines.append(f"  Session ID (hex): {session_id_hex}")
            report_lines.append(f"  Session fingerprint (sha1): {session_fingerprint}")

            if df_metrics is not None and not df_metrics.empty:
                report_lines.append("  (Métricas calculadas com sucesso)")
            else:
                report_lines.append("  (Sem métricas calculadas)")


            report_lines.append("-" * 70)
            report_lines.append("Qualidade do Sinal GPS")
            report_lines.append(f"  Micro-gaps corrigidos (≤1 amostra consecutiva): {total_micro_gaps}")

            # Auditoria de timestamp (resumo)
            try:
                dfta = st.session_state.get("df_time_audit", None)
                if dfta is not None and isinstance(dfta, pd.DataFrame) and not dfta.empty:
                    hz_med = float(dfta["hz_est"].dropna().median()) if dfta["hz_est"].dropna().any() else np.nan
                    n_dup = int((dfta["n_dt_zero"] > 0).sum()) if "n_dt_zero" in dfta.columns else 0
                    n_g2 = int((dfta["n_gaps_gt_2s"] > 0).sum()) if "n_gaps_gt_2s" in dfta.columns else 0

                    report_lines.append("-" * 70)
                    report_lines.append("Auditoria de Timestamp")
                    if np.isfinite(hz_med):
                        report_lines.append(f"  Hz mediano estimado (por atleta/fase): {hz_med:.1f} Hz")
                    else:
                        report_lines.append("  Hz mediano estimado (por atleta/fase): —")
                    report_lines.append(f"  Atleta×fase com timestamps duplicados: {n_dup}")
                    report_lines.append(f"  Atleta×fase com gaps >2s: {n_g2}")
            except Exception:
                pass

            report_txt = "\n".join(report_lines)

            # Persistir outputs (map zoom/scroll dispara rerun do Streamlit)
            st.session_state.df_metrics = df_metrics
            st.session_state.report_txt = report_txt
            st.session_state.process_done = True
            status.update(label="Finalizado.", state="complete")

    st.success("✅ Processamento concluído. Relatório e métricas disponíveis abaixo.")


# ---------- UI (fora do if btn) ----------
df_metrics = st.session_state.df_metrics
report_txt = st.session_state.report_txt

if st.session_state.process_done and df_metrics is not None and isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:

    # 1️⃣ Identificar coluna atleta
    col_inicio = None
    for possible in ["atleta_id", "ID_atleta", "Atleta_ID", "atleta"]:
        if possible in df_metrics.columns:
            col_inicio = possible
            break

    if col_inicio is None:
        st.error("Não encontrei a coluna do atleta.")
        st.write("Colunas disponíveis:", list(df_metrics.columns))
        st.stop()

    # 2️⃣ Cortar a partir da coluna do atleta
    df_display = df_metrics.loc[:, col_inicio:].copy()

    # 3️⃣ Remover engine_version (se existir)
    if "engine_version" in df_display.columns:
        df_display = df_display.drop(columns=["engine_version"])

    # 4️⃣ Ordenação por atleta + fase (ordem personalizada)
    if "fase" in df_display.columns:
        ordem_fases = {
            "Warm-Up": 0,
            "1P": 1,
            "2P": 2,
            "Total": 3,
        }

        df_display["__fase_ord"] = df_display["fase"].map(ordem_fases).fillna(99)
        df_display = df_display.sort_values(
            by=[col_inicio, "__fase_ord"]
        ).drop(columns="__fase_ord")

    else:
        df_display = df_display.sort_values(by=[col_inicio])

    # 5️⃣ Mostrar
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 6️⃣ Download coerente com o display
    st.download_button(
        "⬇️ Download Métricas (.csv)",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="metricas_individuais_FPF.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Auditoria de timestamp (diagnóstico)
    if "df_time_audit" in st.session_state and st.session_state.df_time_audit is not None:
        dfta = st.session_state.df_time_audit
        if isinstance(dfta, pd.DataFrame) and not dfta.empty:
            st.subheader("Auditoria de Timestamp (por atleta e fase)")
            st.dataframe(dfta, use_container_width=True, hide_index=True)

    st.subheader("Relatório")
    if report_txt:
        st.code(report_txt, language="text")
        st.download_button(
            "⬇️ Download Relatório (.txt)",
            data=report_txt.encode("utf-8"),
            file_name="relatorio_FPF.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.warning("Sem relatório para mostrar (processa novamente).")

    # (Opcional) botão para limpar resultados
    if st.button("🧹 Limpar resultados", use_container_width=True):
        st.session_state.df_metrics = None
        st.session_state.report_txt = None
        st.session_state.process_done = False
        st.rerun()
