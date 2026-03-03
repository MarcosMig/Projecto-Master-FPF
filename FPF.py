# -*- coding: utf-8 -*-
"""
FPF UTM Engine v11.1 — CÓDIGO COMPLETO E LIMPO (Streamlit)

Notas:
- Este ficheiro assume que tens estes módulos disponíveis no teu projeto:
    - fpf_modules.constants: ENGINE_VERSION
    - fpf_modules.io_utils: hash_session
    - fpf_modules.metrics: audit_timebase, compute_metrics_for_df
    - fpf_modules.pipeline: processar_atletas_para_temp, sincronizar

- O “Pick no mapa” está incluído e gera BL/BR/TL/TR (ordenado) e faz retangularização.
- A validação geográfica é simples e robusta (amostragem por atleta vs. centro do campo).
- A rotação é calculada para alinhar BL→BR com o eixo X em UTM (aplicada na pipeline via R/origin).

Autor: base Marcos + limpeza/robustez
"""

import io
import re
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from pyproj import Geod, Transformer
from streamlit_folium import st_folium

# ===== Imports do teu projeto =====
from fpf_modules.constants import ENGINE_VERSION
from fpf_modules.io_utils import hash_session
from fpf_modules.metrics import audit_timebase, compute_metrics_for_df
from fpf_modules.pipeline import processar_atletas_para_temp, sincronizar

GEOD = Geod(ellps="WGS84")

COL_LAT, COL_LON, COL_TIME, COL_FASE = "Lat", "Lon", "Time", "Fase"

# ==============================
# Streamlit base + session_state
# ==============================
st.set_page_config(page_title="FPF UTM Engine v11.1", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

# outputs persistentes
if "df_metrics" not in st.session_state:
    st.session_state.df_metrics = None
if "df_time_audit" not in st.session_state:
    st.session_state.df_time_audit = None
if "report_txt" not in st.session_state:
    st.session_state.report_txt = None
if "process_done" not in st.session_state:
    st.session_state.process_done = False
if "metrics_parquet_bytes" not in st.session_state:
    st.session_state.metrics_parquet_bytes = None
if "metrics_parquet_error" not in st.session_state:
    st.session_state.metrics_parquet_error = None

# pick corners
if "pick_corners" not in st.session_state:
    st.session_state.pick_corners = []  # [(lat, lon), ...]
if "pts_gps_picked" not in st.session_state:
    st.session_state.pts_gps_picked = None  # {"BL":(lat,lon),...}


# ==============================
# UI helpers
# ==============================
def _apply_login_style():
    css = """
    <style>
      .stApp { background-color: #0e1117; }
      header, footer {visibility: hidden;}
      [data-testid="stSidebar"] {display: block;}
      .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
      }
      /* crosshair feel: aplica ao folium container */
      iframe { cursor: crosshair !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ==============================
# Geometria / Campo
# ==============================
def _to_utm_transformer(epsg: int):
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def _latlon_to_xy(lat, lon, tr: Transformer):
    # always_xy=True => (lon, lat)
    x, y = tr.transform(lon, lat)
    return float(x), float(y)


def _xy_to_latlon(x, y, tr: Transformer):
    inv = tr.inverse
    lon, lat = inv.transform(x, y)
    return float(lat), float(lon)


def _order_corners_approx_latlon(pts_latlon):
    """
    Ordena 4 pontos (lat,lon) em BL/BR/TL/TR por heurística:
    - Converte para plano local (UTM) e usa soma/diferença.
    """
    if pts_latlon is None or len(pts_latlon) != 4:
        raise ValueError("Precisas de exatamente 4 pontos para ordenar cantos.")

    # EPSG default 32629 só para ordenar; depois o engine usa epsg_used real do user
    tr = _to_utm_transformer(32629)
    xy = np.array([_latlon_to_xy(lat, lon, tr) for lat, lon in pts_latlon], dtype=float)

    xs = xy[:, 0]
    ys = xy[:, 1]

    # BL ~ menor y e menor x (aprox), TR ~ maior y e maior x
    # usar soma/dif é mais estável:
    s = xs + ys
    d = xs - ys

    bl_i = int(np.argmin(s))
    tr_i = int(np.argmax(s))
    br_i = int(np.argmax(d))
    tl_i = int(np.argmin(d))

    # garantir índices únicos; se houver colisão (cliques muito tortos), cair para um fallback robusto
    idxs = {bl_i, br_i, tl_i, tr_i}
    if len(idxs) != 4:
        # fallback: ordena por y, separa 2 de baixo e 2 de cima, depois por x
        order_y = np.argsort(ys)
        bottom = order_y[:2]
        top = order_y[2:]
        bottom = bottom[np.argsort(xs[bottom])]
        top = top[np.argsort(xs[top])]
        bl_i, br_i = int(bottom[0]), int(bottom[1])
        tl_i, tr_i = int(top[0]), int(top[1])

    return {
        "BL": pts_latlon[bl_i],
        "BR": pts_latlon[br_i],
        "TL": pts_latlon[tl_i],
        "TR": pts_latlon[tr_i],
    }


def _rectangularize_utm(corners_latlon, epsg_used: int):
    """
    Retangulariza cantos em UTM:
    - usa BL como origem;
    - direção X: BL->BR
    - direção Y: BL->TL
    - projeta BR e TL para eixos ortogonais e reconstrói TR.
    Retorna:
      - corners_clean_latlon (BL/BR/TL/TR)
      - origin (x0,y0)
      - R (2x2) para alinhar campo com X
      - dist_x, dist_y, angulo_rad
    """
    tr = _to_utm_transformer(epsg_used)

    BL = corners_latlon["BL"]
    BR = corners_latlon["BR"]
    TL = corners_latlon["TL"]
    TR = corners_latlon["TR"]

    bl = np.array(_latlon_to_xy(BL[0], BL[1], tr))
    br = np.array(_latlon_to_xy(BR[0], BR[1], tr))
    tl = np.array(_latlon_to_xy(TL[0], TL[1], tr))
    trp = np.array(_latlon_to_xy(TR[0], TR[1], tr))

    vx = br - bl
    vy = tl - bl

    norm_vx = np.linalg.norm(vx)
    norm_vy = np.linalg.norm(vy)
    if norm_vx < 1e-6 or norm_vy < 1e-6:
        raise ValueError("Cantos inválidos (distâncias muito pequenas).")

    ex = vx / norm_vx
    ey_raw = vy / norm_vy

    # ortogonaliza ey (Gram-Schmidt)
    ey = ey_raw - np.dot(ey_raw, ex) * ex
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-6:
        # se demasiado colinear, cria ey por rotação de 90°
        ey = np.array([-ex[1], ex[0]])
    else:
        ey = ey / ey_norm

    # projetar BR e TL
    br_clean = bl + np.dot((br - bl), ex) * ex
    tl_clean = bl + np.dot((tl - bl), ey) * ey
    tr_clean = br_clean + (tl_clean - bl)

    dist_x = float(np.linalg.norm(br_clean - bl))
    dist_y = float(np.linalg.norm(tl_clean - bl))

    # rotação para alinhar ex ao eixo X: R * (p - origin)
    angulo_rad = float(np.arctan2(ex[1], ex[0]))
    c = float(np.cos(-angulo_rad))
    s = float(np.sin(-angulo_rad))
    R = np.array([[c, -s], [s, c]], dtype=float)

    corners_clean_xy = {
        "BL": bl,
        "BR": br_clean,
        "TL": tl_clean,
        "TR": tr_clean,
    }
    corners_clean_latlon = {k: _xy_to_latlon(v[0], v[1], tr) for k, v in corners_clean_xy.items()}
    origin = (float(bl[0]), float(bl[1]))

    return corners_clean_latlon, origin, R, dist_x, dist_y, angulo_rad


def _field_center_latlon(corners_latlon):
    pts = np.array([(lat, lon) for (lat, lon) in corners_latlon.values()], dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _read_minimal_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # tolerância de nomes
    rename = {}
    for c in df.columns:
        c_strip = c.strip()
        if c_strip.lower() in ["lat", "latitude"]:
            rename[c] = "Lat"
        elif c_strip.lower() in ["lon", "lng", "long", "longitude"]:
            rename[c] = "Lon"
        elif c_strip.lower() in ["time", "timestamp", "ts"]:
            rename[c] = "Time"
        elif c_strip.lower() in ["fase", "phase"]:
            rename[c] = "Fase"
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in ["Lat", "Lon", "Time"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")
    return df


def _extract_atleta_id(filename: str) -> str:
    # tenta "Player-123" ou "player_123"
    m = re.search(r"(?:player|Player)[-_ ]?(\w+)", filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def _geo_validate_atletas(
    f_atleta_files,
    center_latlon,
    raio_validacao_m=50,
    amostra_geo_n=500,
    min_pct_atletas_ok=0.80,
):
    """
    Para cada atleta (ficheiro), amostra N linhas e verifica % de pontos dentro do raio do centro.
    Critério:
      - atleta OK se >= 50% das amostras dentro do raio
      - sessão OK se pct_atletas_OK >= min_pct_atletas_ok
    """
    if not f_atleta_files:
        return False, 0.0, [], [], []

    clat, clon = center_latlon
    ok_list = []
    fora_list = []
    errors = []

    for uf in f_atleta_files:
        try:
            df = _read_minimal_csv(uf)
            n = min(int(amostra_geo_n), len(df))
            if n <= 0:
                raise ValueError("CSV vazio.")

            idx = np.linspace(0, len(df) - 1, n).astype(int)
            samp = df.iloc[idx][["Lat", "Lon"]].dropna()
            if samp.empty:
                raise ValueError("Sem Lat/Lon válidos na amostra.")

            lats = samp["Lat"].astype(float).to_numpy()
            lons = samp["Lon"].astype(float).to_numpy()

            _, _, dist = GEOD.inv(np.full_like(lons, clon), np.full_like(lats, clat), lons, lats)
            dist = np.asarray(dist, dtype=float)

            inside = (dist <= float(raio_validacao_m)).mean() if len(dist) else 0.0
            atleta_ok = inside >= 0.50

            aid = _extract_atleta_id(getattr(uf, "name", str(uf)))
            if atleta_ok:
                ok_list.append((aid, inside))
            else:
                fora_list.append((aid, inside))
        except Exception as e:
            aid = _extract_atleta_id(getattr(uf, "name", str(uf)))
            errors.append((aid, getattr(uf, "name", str(uf)), str(e)))

    n_total = len(ok_list) + len(fora_list) + len(errors)
    pct_ok = (len(ok_list) / n_total) if n_total else 0.0
    passed = pct_ok >= float(min_pct_atletas_ok)
    return passed, pct_ok, ok_list, fora_list, errors


# ==============================
# Thresholds métricas (GPS-only)
# ==============================
HSR_MPS = 5.5
SPRINT_MPS = 7.0
ACC_THR = 2.5
DEC_THR = -3.0
SPRINT_BOUT_MIN_S = 1.0


# ==============================
# Layout
# ==============================
_apply_login_style()

st.title("FPF UTM Engine v11.1")

with st.sidebar:
    st.subheader("Sessão")

    nome_user = st.text_input("Nome (login)", value="")
    instituicao_user = st.text_input("Instituição", value="")

    data_sessao = st.date_input("Data")
    selecao = st.text_input("Seleção", value="")
    genero = st.selectbox("Género", ["Masculino", "Feminino", "Misto"], index=0)
    contexto = st.selectbox("Contexto", ["Treino", "Jogo"], index=0)

    c1, c2 = st.columns(2)
    adversario_a = c1.text_input("Adversário A", value="") if contexto == "Jogo" else ""
    adversario_b = c2.text_input("Adversário B", value="") if contexto == "Jogo" else ""

    st.subheader("Local")
    estadio = st.text_input("Estádio", value="")
    cidade = st.text_input("Cidade", value="")
    pais = st.text_input("País", value="")

    st.subheader("Inputs")
    st.caption("Campo: CSV com cantos ou usa o Pick no mapa. Atletas: CSVs com Player-<id> e fase no nome.")
    f_campo = st.file_uploader("Dados de CAMPO (CSV opcional)", type=["csv"], accept_multiple_files=False)
    f_atleta = st.file_uploader("Dados de ATLETAS (CSVs)", accept_multiple_files=True, type=["csv"])

    st.divider()

    st.subheader("Parâmetros")
    epsg_used = st.number_input("EPSG (UTM)", min_value=20000, max_value=39999, value=32629, step=1)
    raio_validacao_m = st.slider("Raio validação (m)", 5, 200, 50, 5)
    amostra_geo_n = st.slider("Amostra linhas/atleta", 50, 2000, 500, 50)
    min_pct_atletas_ok = st.slider("% mínimo atletas OK", 0.50, 1.00, 0.80, 0.05)

    aplicar_suavizacao = st.checkbox("Suavização (Savgol)", value=True)
    janela_savgol = st.slider("Janela Savgol", 5, 51, 11, 2)
    poly_savgol = st.slider("Polinómio Savgol", 1, 5, 2, 1)

    st.divider()
    st.info(f"ENGINE_VERSION: {ENGINE_VERSION}")


st.subheader("Calibração do Campo (cantos)")

tab_pick, tab_csv = st.tabs(["🧭 Pick no mapa", "📄 Ler CSV do Campo"])

with tab_csv:
    st.caption("Se usares CSV, precisa de 4 linhas com colunas Lat/Lon (ou variantes). A ordem não importa.")
    if f_campo is not None:
        try:
            dfc = pd.read_csv(f_campo)
            rename = {}
            for c in dfc.columns:
                cl = c.strip().lower()
                if cl in ["lat", "latitude"]:
                    rename[c] = "Lat"
                if cl in ["lon", "lng", "long", "longitude"]:
                    rename[c] = "Lon"
            if rename:
                dfc = dfc.rename(columns=rename)

            if "Lat" not in dfc.columns or "Lon" not in dfc.columns:
                st.error("CSV do campo precisa de colunas Lat e Lon.")
            else:
                pts = list(zip(dfc["Lat"].astype(float).tolist(), dfc["Lon"].astype(float).tolist()))
                pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
                if len(pts) < 4:
                    st.error("CSV do campo precisa de pelo menos 4 pontos válidos.")
                else:
                    pts = pts[:4]
                    corners = _order_corners_approx_latlon(pts)
                    st.session_state.pts_gps_picked = corners
                    st.success("Cantos carregados e ordenados a partir do CSV.")
                    st.write(corners)
        except Exception as e:
            st.error(f"Erro a ler CSV do campo: {e}")

with tab_pick:
    st.caption("Clica 4 cantos (qualquer ordem). Depois o motor ordena e retangulariza.")
    c_lat = st.number_input("Centro (lat) para iniciar mapa", value=38.7223, format="%.6f")
    c_lon = st.number_input("Centro (lon) para iniciar mapa", value=-9.1393, format="%.6f")
    zoom = st.slider("Zoom", 12, 20, 17, 1)

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("➕ Novo Pick (limpar)", use_container_width=True):
            st.session_state.pick_corners = []
            st.session_state.pts_gps_picked = None
            st.rerun()
    with colB:
        pick_enabled = st.toggle("Modo Pick", value=True)
    with colC:
        st.write("Pontos:", len(st.session_state.pick_corners), st.session_state.pick_corners)

    m = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, tiles="OpenStreetMap")

    for i, (lat, lon) in enumerate(st.session_state.pick_corners, start=1):
        folium.CircleMarker(
            location=[lat, lon], radius=6, tooltip=f"Pick {i}", fill=True
        ).add_to(m)

    if st.session_state.pts_gps_picked is not None:
        corners = st.session_state.pts_gps_picked
        poly = [corners["BL"], corners["BR"], corners["TR"], corners["TL"], corners["BL"]]
        folium.PolyLine([(p[0], p[1]) for p in poly], weight=4).add_to(m)
        for k in ["BL", "BR", "TL", "TR"]:
            lat, lon = corners[k]
            folium.Marker([lat, lon], tooltip=k).add_to(m)

    out = st_folium(m, height=520, width=None)

    if pick_enabled and out and out.get("last_clicked"):
        lat = out["last_clicked"]["lat"]
        lon = out["last_clicked"]["lng"]
        if len(st.session_state.pick_corners) < 4:
            st.session_state.pick_corners.append((float(lat), float(lon)))
            st.rerun()

    if len(st.session_state.pick_corners) == 4 and st.session_state.pts_gps_picked is None:
        try:
            corners = _order_corners_approx_latlon(st.session_state.pick_corners)
            st.session_state.pts_gps_picked = corners
            st.success("4 pontos recolhidos → cantos ordenados (BL/BR/TL/TR).")
            st.write(corners)
        except Exception as e:
            st.error(f"Erro a ordenar cantos: {e}")


if st.session_state.pts_gps_picked is None:
    st.warning("Define os 4 cantos do campo (Pick ou CSV) para ativar o processamento.")
    st.stop()

try:
    corners_clean, origin, R, dist_x, dist_y, angulo_rad = _rectangularize_utm(
        st.session_state.pts_gps_picked, int(epsg_used)
    )
except Exception as e:
    st.error(f"Erro na retangularização/rotação do campo: {e}")
    st.stop()

center_latlon = _field_center_latlon(corners_clean)
rot_deg = float(np.degrees(angulo_rad))

st.markdown("#### Campo (após ajuste)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Comprimento (BL→BR)", f"{dist_x:.2f} m")
c2.metric("Largura (BL→TL)", f"{dist_y:.2f} m")
c3.metric("Rotação aplicada", f"{rot_deg:.2f}°")
c4.metric("EPSG (UTM)", str(epsg_used))

with st.expander("Ver cantos (limpos)"):
    st.write(corners_clean)


st.subheader("Validação Geográfica (submissão)")

passed_geo, pct_ok, ok_list, fora_list, geo_errors = _geo_validate_atletas(
    f_atleta_files=f_atleta,
    center_latlon=center_latlon,
    raio_validacao_m=raio_validacao_m,
    amostra_geo_n=amostra_geo_n,
    min_pct_atletas_ok=min_pct_atletas_ok,
)

colv1, colv2, colv3 = st.columns(3)
colv1.metric("Atletas OK", str(len({a for a, _ in ok_list})))
colv2.metric("Atletas fora", str(len({a for a, _ in fora_list})))
colv3.metric("% OK", f"{pct_ok*100:.0f}%")

if geo_errors:
    with st.expander("Erros de leitura (exemplos)"):
        st.write(geo_errors[:25])

if not passed_geo:
    st.error(
        "A validação geográfica falhou. A exportação está desativada. "
        "Ajusta raio/% mínimo ou verifica os ficheiros de atletas."
    )
    st.stop()


st.divider()
btn = st.button("⚙️ Processar e Gerar Relatório", type="primary", use_container_width=True)

if btn:
    with st.status("A iniciar processamento...", expanded=True) as status:
        if not f_atleta:
            status.update(label="Sem ficheiros de atletas.", state="error")
            st.error("Faz upload dos CSVs de atletas.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_dir = td_path / "Temp_Processing"
            out_dir = td_path / "Output_UTM_Sincronizado"
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            status.update(label="Processamento e limpeza de dados GPS...", state="running")
            temp_files, audit_proc, issues = processar_atletas_para_temp(
                f_atleta,
                int(epsg_used),
                origin,
                R,
                bool(aplicar_suavizacao),
                int(janela_savgol),
                int(poly_savgol),
                temp_dir,
            )

            if not temp_files:
                status.update(label="Falha: não gerou ficheiros temporários.", state="error")
                st.error("❌ Não foi possível gerar ficheiros temporários (verifica colunas Time/Lat/Lon e nomes).")
                st.stop()

            status.update(label="Sincronização temporal...", state="running")
            sync_result = sincronizar(temp_files, out_dir, include_event_clock=True)
            if isinstance(sync_result, tuple) and len(sync_result) == 5:
                out_files, fases_ordenadas, fases_dict, n_master, event_clock = sync_result
            elif isinstance(sync_result, tuple) and len(sync_result) == 4:
                out_files, fases_ordenadas, fases_dict, n_master = sync_result
                event_clock = {}
            else:
                raise ValueError("Formato inesperado do retorno de sincronizar().")

            session_uuid = uuid.uuid4()
            session_id_hex = session_uuid.hex
            session_fingerprint = hash_session(
                data_sessao, selecao, genero, contexto, estadio, f_campo, f_atleta
            )

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

                if "_micro_gaps_corrigidos" in df_sync.columns and df_sync["_micro_gaps_corrigidos"].notna().any():
                    try:
                        total_micro_gaps += int(df_sync["_micro_gaps_corrigidos"].dropna().iloc[0])
                    except Exception:
                        pass

                fase_mets = {}
                for fase in fases_target:
                    if COL_FASE not in df_sync.columns:
                        df_sync[COL_FASE] = "Total"

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
                                " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
                                if contexto == "Jogo"
                                else ""
                            ),
                            "estadio": estadio,
                            "cidade": cidade,
                            "pais": pais,
                            "atleta_id": aid,
                            "fase": fase,
                            **met,
                            "engine_version": ENGINE_VERSION,
                        }
                    )

                met_total = {}
                keys = set().union(*[set(fase_mets[f].keys()) for f in fases_target if f in fase_mets])
                for k in keys:
                    vals = []
                    for f in fases_target:
                        if f in fase_mets and k in fase_mets[f]:
                            v = fase_mets[f][k]
                            if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                                vals.append(float(v))
                    if vals:
                        met_total[k] = float(np.sum(vals))

                metrics_rows.append(
                    {
                        "session_id_hex": session_id_hex,
                        "session_fingerprint": session_fingerprint,
                        "data": data_sessao,
                        "selecao": selecao,
                        "genero": genero,
                        "contexto": contexto,
                        "jogo": (
                            " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
                            if contexto == "Jogo"
                            else ""
                        ),
                        "estadio": estadio,
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
            st.session_state.df_metrics = df_metrics
            st.session_state.df_time_audit = df_time_audit

            try:
                parquet_buffer = io.BytesIO()
                df_metrics.to_parquet(parquet_buffer, index=False)
                st.session_state.metrics_parquet_bytes = parquet_buffer.getvalue()
                st.session_state.metrics_parquet_error = None
            except Exception as e:
                st.session_state.metrics_parquet_bytes = None
                st.session_state.metrics_parquet_error = str(e)

            status.update(label="Construção do relatório...", state="running")
            report_lines = []
            report_lines.append("FPF Performance Hub — Relatório de Validação e Normalização")
            report_lines.append("=" * 70)
            report_lines.append("Dados da Sessão")
            report_lines.append(f"  Data: {data_sessao.strftime('%d/%m/%Y')}")
            report_lines.append(f"  Seleção: {selecao} | Género: {genero} | Contexto: {contexto}")
            if contexto == "Jogo":
                vs_txt = " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
                if vs_txt:
                    report_lines.append(f"  Jogo: {vs_txt}")

            loc_part = ", ".join([p for p in [cidade, pais] if p]) or "—"
            report_lines.append(f"  Localização: {loc_part}")
            report_lines.append(f"  Estádio (meta): {estadio or '—'}")
            report_lines.append("")
            report_lines.append("Campo (UTM)")
            report_lines.append(f"  EPSG: {epsg_used}")
            report_lines.append(f"  Comprimento (BL→BR): {dist_x:.2f} m")
            report_lines.append(f"  Largura (BL→TL):     {dist_y:.2f} m")
            report_lines.append(f"  Rotação aplicada:    {rot_deg:.2f}° (alinhamento BL→BR com eixo X)")
            report_lines.append("")
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
            report_lines.append("Sincronização")
            report_lines.append(f"  Timestamps mestre: {n_master}")
            report_lines.append(f"  Ficheiros gerados: {len(out_files)}")
            report_lines.append("  Fases (ordem cronológica):")
            for fase, (t_s, t_e) in fases_ordenadas:
                report_lines.append(f"    - {fase:8} | início: {t_s} | fim: {t_e}")

            if event_clock:
                report_lines.append("  Timeline de Jogo (uniformizada):")
                for fase in ["Warm-Up", "1P", "2P"]:
                    if fase in event_clock:
                        ec = event_clock[fase]
                        ini = int(round(ec.get("start_s", 0.0)))
                        fim = int(round(ec.get("end_s", 0.0)))
                        ext = int(round(ec.get("extra_s", 0.0)))

                        def _fmt(sec):
                            sign = "-" if sec < 0 else ""
                            sec = abs(sec)
                            h = sec // 3600
                            m = (sec % 3600) // 60
                            s = sec % 60
                            return f"{sign}{h:02d}:{m:02d}:{s:02d}"

                        report_lines.append(
                            f"    - {fase:8} | evento: {_fmt(ini)} → {_fmt(fim)} | extra: {_fmt(ext)}"
                        )

            if issues:
                report_lines.append("-" * 70)
                report_lines.append("Avisos/Problemas (exemplos):")
                for aid_, fn_, msg_ in issues[:25]:
                    report_lines.append(f"  - {aid_} | {fn_} | {msg_}")

            report_lines.append("-" * 70)
            report_lines.append("Métricas Individuais (GPS-only) — thresholds fixos")
            report_lines.append(
                f"  HSR ≥ {HSR_MPS:.1f} m/s | Sprint ≥ {SPRINT_MPS:.1f} m/s | "
                f"Acc ≥ {ACC_THR:.1f} m/s² | Dec ≤ {DEC_THR:.1f} m/s²"
            )
            report_lines.append(f"  Session ID (hex): {session_id_hex}")
            report_lines.append(f"  Session fingerprint (sha1): {session_fingerprint}")
            report_lines.append("-" * 70)
            report_lines.append("Qualidade do Sinal GPS")
            report_lines.append(f"  Micro-gaps corrigidos (≤1 amostra consecutiva): {total_micro_gaps}")

            st.session_state.report_txt = "\n".join(report_lines)
            st.session_state.process_done = True

            status.update(label="Concluído ✅", state="complete")


if st.session_state.process_done and st.session_state.df_metrics is not None:
    st.subheader("Métricas Individuais")

    df_metrics = st.session_state.df_metrics.copy()

    if "fase" in df_metrics.columns:
        ordem = {"Warm-Up": 0, "1P": 1, "2P": 2, "Total": 3}
        df_metrics["__fase_ord"] = df_metrics["fase"].map(ordem).fillna(99)
        df_metrics = df_metrics.sort_values(by=["atleta_id", "__fase_ord"]).drop(columns="__fase_ord")

    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Métricas (.csv)",
        data=df_metrics.to_csv(index=False).encode("utf-8"),
        file_name="metricas_individuais_FPF.csv",
        mime="text/csv",
        use_container_width=True,
    )

    parquet_bytes = st.session_state.get("metrics_parquet_bytes")
    parquet_error = st.session_state.get("metrics_parquet_error")
    if parquet_bytes:
        st.download_button(
            "⬇️ Download Métricas Completas (.parquet)",
            data=parquet_bytes,
            file_name="metricas_individuais_FPF_full.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )
    elif parquet_error:
        st.warning(f"Não foi possível gerar Parquet (pandas/engine): {parquet_error}")

    dfta = st.session_state.df_time_audit
    if isinstance(dfta, pd.DataFrame) and not dfta.empty:
        st.subheader("Auditoria de Timestamp (por atleta e fase)")
        st.dataframe(dfta, use_container_width=True, hide_index=True)

    st.subheader("Relatório")
    report_txt = st.session_state.report_txt
    if report_txt:
        st.code(report_txt, language="text")
        st.download_button(
            "⬇️ Download Relatório (.txt)",
            data=report_txt.encode("utf-8"),
            file_name="relatorio_FPF.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("🧹 Limpar resultados", use_container_width=True):
        st.session_state.df_metrics = None
        st.session_state.df_time_audit = None
        st.session_state.report_txt = None
        st.session_state.metrics_parquet_bytes = None
        st.session_state.metrics_parquet_error = None
        st.session_state.process_done = False
        st.rerun()
