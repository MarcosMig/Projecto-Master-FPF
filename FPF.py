# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 19:38:12 2026

@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 19:38:12 2026

@author: marco
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from pyproj import Geod
from pathlib import Path
import json
import tempfile
import hashlib
import uuid
import requests
GEOD = Geod(ellps='WGS84')  # WGS84 geodesic distance (metros reais)
from streamlit_folium import st_folium
import re

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v11.1", layout="wide")
if 'auth' not in st.session_state: st.session_state.auth = False



# --- LOGIN (CENTRADO + st.secrets) ---
def _apply_login_style():
    st.markdown("""
    <style>
      .stApp { background-color: #0e1117; }
      header, footer {visibility: hidden;}
      [data-testid="stSidebar"] {display: none;}

      /* Card visual */
      .login-card{
        background: #1a1c23;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 34px 34px 26px 34px;
        box-shadow: 0px 10px 28px rgba(0,0,0,0.55);
      .login-title{
        text-align: center;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 1.25rem 0;
      /* Inputs */
      .stTextInput > div > div > input {
        background: #0e1117;
        border: 1px solid #30363d;
      /* Botão */
      .stButton > button{
        width: 100%;
        background: #E30613 !important;
        color: #fff !important;
        font-weight: 800;
        border: 0;
        height: 3.1em;
        border-radius: 10px;
      .stButton > button:hover{ filter: brightness(0.95); }
    </style>
    """, unsafe_allow_html=True)

def _get_auth_from_secrets():
    """Espera em st.secrets:
    [auth]
    username = "..."
    password = "..."
    """
    try:
        auth = st.secrets["auth"]
        return auth.get("username"), auth.get("password")
    except Exception:
        return None, None

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    _apply_login_style()

    # Centrar com colunas (robusto no Streamlit)
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">⚽ FPF Performance Hub</div>', unsafe_allow_html=True)

        u = st.text_input("Utilizador", key="user_val")
        p = st.text_input("Password", type="password", key="pass_val")

        secrets_user, secrets_pass = _get_auth_from_secrets()
        if not secrets_user:
            st.warning("⚠️ Credenciais não configuradas em st.secrets. Defina [auth] no secrets.toml / Streamlit Cloud.")

        if st.button("Entrar"):
            if secrets_user and u == secrets_user and p == secrets_pass:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Credenciais inválidas")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- INTERFACE SINGLE PAGE ---
# --- INTERFACE SINGLE PAGE ---
st.title("🚀 Pipeline de Validação + Normalização (UTM/Rotação)")

with st.sidebar:
    st.header("🧾 Dados da Sessão")
    estadio = st.text_input("Estádio")
    data_sessao = st.date_input("Data")
    selecao = st.text_input("Seleção (ex.: U19)")
    genero = st.selectbox("Género", options=["M", "F"], index=0)
    contexto = st.selectbox("Contexto", options=["Treino", "Jogo"], index=0)
    adversario_a = ""
    adversario_b = ""
    if contexto == "Jogo":
        col_a, col_b = st.columns(2)
        with col_a:
            adversario_a = st.text_input("Equipa A (ex.: Portugal)")
        with col_b:
            adversario_b = st.text_input("Equipa B (ex.: Espanha)")

    st.divider()
    st.header("📤 Upload de Ficheiros")
    st.caption("Campo: 4 CSVs com BL, BR, TL, TR no nome do ficheiro.")
    f_campo = st.file_uploader("Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True, type=["csv"])
    st.caption("Atletas: CSVs com Player-<id> e indicação de fase (Warm/Primeira/Segunda/1P/2P) no nome do ficheiro.")
    f_atleta = st.file_uploader("Dados de ATLETAS (CSVs)", accept_multiple_files=True, type=["csv"])

st.divider()


# Defaults (menu de opções removido)
epsg_used = 32629
raio_validacao_m = 50
amostra_geo_n = 500
min_pct_atletas_ok = 0.80
aplicar_suavizacao = True
janela_savgol = 11
poly_savgol = 2

# -------------------------------
# Helpers (Streamlit-friendly)
# -------------------------------
from pyproj import Transformer
from scipy.signal import savgol_filter
import math
import io

# -------------------------------
# Métricas Individuais (GPS-only)
# -------------------------------
HSR_MPS = 5.5      # High-Speed Running (m/s) ~19.8 km/h
SPRINT_MPS = 7.0   # Sprint (m/s) ~25.2 km/h
ACC_THR = 2.5      # m/s^2
DEC_THR = -3.0     # m/s^2
SPRINT_BOUT_MIN_S = 1.0  # duração mínima do bout de sprint (s)
ENGINE_VERSION = "v12-metrics"

def _time_to_seconds(series: pd.Series) -> pd.Series:
    """Converte Time para segundos (float). Suporta numérico (s/ms) ou datetime-like string."""
    s = series.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        med = float(s_num.dropna().median()) if s_num.notna().any() else 0.0
        if med > 1e12:  # epoch ms
            return s_num / 1000.0
        if med > 1e9:   # epoch s
            return s_num.astype(float)
        return s_num.astype(float)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        t0 = dt.dropna().iloc[0]
        return (dt - t0).dt.total_seconds()
    return pd.Series([np.nan]*len(s), index=s.index, dtype="float64")

def _count_bouts(t: np.ndarray, mask: np.ndarray, min_dur_s: float) -> int:
    """Conta episódios consecutivos onde mask==True com duração >= min_dur_s."""
    if len(t) == 0:
        return 0
    bouts = 0
    in_bout = False
    t_start = None
    for i in range(len(t)):
        if mask[i] and not in_bout:
            in_bout = True
            t_start = t[i]
        if (not mask[i]) and in_bout:
            dur = t[i-1] - t_start if t_start is not None else 0.0
            if dur >= min_dur_s:
                bouts += 1
            in_bout = False
            t_start = None
    if in_bout and t_start is not None:
        dur = t[-1] - t_start
        if dur >= min_dur_s:
            bouts += 1
    return bouts

def _compute_metrics_for_df(df: pd.DataFrame) -> dict:
    """Calcula métricas para um atleta numa fase (df filtrado)."""
    if df.empty:
        return {
            "duracao_min": 0.0, "dist_m": 0.0, "m_min": np.nan,
            "peak_1m_m_min": np.nan, "vmax_mps": np.nan,
            "hsr_dist_m": 0.0, "hsr_pct": np.nan,
            "sprint_dist_m": 0.0, "n_sprints": 0,
            "n_acc_2_5": 0, "n_dec_3_0": 0,
            "n_points": 0, "pct_time_valid": 0.0, "n_gaps_gt2s": 0
        }

    t_sec = _time_to_seconds(df[COL_TIME])
    x = pd.to_numeric(df["X_UTM"], errors="coerce")
    y = pd.to_numeric(df["Y_UTM"], errors="coerce")

    valid = t_sec.notna() & x.notna() & y.notna()
    dfv = pd.DataFrame({"t": t_sec[valid], "x": x[valid], "y": y[valid]}).sort_values("t")

    if len(dfv) < 2:
        return {
            "duracao_min": 0.0, "dist_m": 0.0, "m_min": np.nan,
            "peak_1m_m_min": np.nan, "vmax_mps": np.nan,
            "hsr_dist_m": 0.0, "hsr_pct": np.nan,
            "sprint_dist_m": 0.0, "n_sprints": 0,
            "n_acc_2_5": 0, "n_dec_3_0": 0,
            "n_points": int(len(dfv)), "pct_time_valid": float(valid.mean()*100.0),
            "n_gaps_gt2s": 0
        }

    t = dfv["t"].to_numpy(dtype=float)
    dx = np.diff(dfv["x"].to_numpy(dtype=float))
    dy = np.diff(dfv["y"].to_numpy(dtype=float))
    dt = np.diff(t)

    good = dt > 0
    n_gaps = int(np.sum(dt[good] > 2.0)) if np.any(good) else 0

    dx, dy, dt = dx[good], dy[good], dt[good]
    if len(dt) == 0:
        return {
            "duracao_min": 0.0, "dist_m": 0.0, "m_min": np.nan,
            "peak_1m_m_min": np.nan, "vmax_mps": np.nan,
            "hsr_dist_m": 0.0, "hsr_pct": np.nan,
            "sprint_dist_m": 0.0, "n_sprints": 0,
            "n_acc_2_5": 0, "n_dec_3_0": 0,
            "n_points": int(len(dfv)), "pct_time_valid": float(valid.mean()*100.0),
            "n_gaps_gt2s": n_gaps
        }

    dist_step = np.hypot(dx, dy)
    dist_total = float(np.nansum(dist_step))
    dur_s = float(np.nansum(dt))
    dur_min = dur_s / 60.0 if dur_s > 0 else 0.0
    m_min = (dist_total / dur_min) if dur_min > 0 else np.nan

    v = dist_step / dt
    vmax = float(np.nanmax(v)) if len(v) else np.nan

    dv = np.diff(v)
    dt2 = dt[1:]
    acc = np.where(dt2 > 0, dv / dt2, np.nan)

    hsr_dist = float(np.nansum(dist_step[v >= HSR_MPS]))
    sprint_dist = float(np.nansum(dist_step[v >= SPRINT_MPS]))
    hsr_pct = (hsr_dist / dist_total * 100.0) if dist_total > 0 else np.nan

    n_acc = int(np.nansum(acc >= ACC_THR))
    n_dec = int(np.nansum(acc <= DEC_THR))

    dist_cum = np.concatenate([[0.0], np.cumsum(dist_step)])
    # t_cum alinhado com dist_cum (1+len(dist_step)); usamos o t original pós-filter
    t_cum = np.concatenate([[t[0]], t[1:][good]])
    peak_1m = np.nan
    if len(t_cum) == len(dist_cum) and len(t_cum) > 1:
        best = 0.0
        j = 0
        for i in range(len(t_cum)):
            while j < len(t_cum) and t_cum[j] - t_cum[i] <= 60.0:
                j += 1
            if j-1 >= i:
                dj = dist_cum[j-1] - dist_cum[i]
                if dj > best:
                    best = dj
        peak_1m = float(best)

    t_mid = t[1:][good]
    sprint_mask = v >= SPRINT_MPS
    n_sprints = _count_bouts(t_mid, sprint_mask, SPRINT_BOUT_MIN_S)

    return {
        "duracao_min": dur_min,
        "dist_m": dist_total,
        "m_min": m_min,
        "peak_1m_m_min": peak_1m,
        "vmax_mps": vmax,
        "hsr_dist_m": hsr_dist,
        "hsr_pct": hsr_pct,
        "sprint_dist_m": sprint_dist,
        "n_sprints": n_sprints,
        "n_acc_2_5": n_acc,
        "n_dec_3_0": n_dec,
        "n_points": int(len(dfv)),
        "pct_time_valid": float(valid.mean()*100.0),
        "n_gaps_gt2s": n_gaps
    }

def _hash_session(data_sessao, selecao, genero, contexto, estadio, f_campo_files, f_atleta_files) -> str:
    """Fingerprint determinístico (para deduplicação futura)."""
    h = hashlib.sha1()
    h.update(str(data_sessao).encode("utf-8"))
    h.update(str(selecao).encode("utf-8"))
    h.update(str(genero).encode("utf-8"))
    h.update(str(contexto).encode("utf-8"))
    h.update(str(estadio).encode("utf-8"))

    def _feed_files(files):
        for uf in sorted(files, key=lambda x: x.name):
            h.update(uf.name.encode("utf-8"))
            try:
                h.update(str(getattr(uf, "size", "")).encode("utf-8"))
            except Exception:
                pass
            try:
                uf.seek(0)
                chunk = uf.read(8192)
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="ignore")
                h.update(chunk or b"")
                uf.seek(0)
            except Exception:
                pass

    _feed_files(f_campo_files)
    _feed_files(f_atleta_files)
    return h.hexdigest()


COL_LAT, COL_LON, COL_TIME, COL_FASE = "Lat", "Lon", "Time", "Fase"

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    return df

def _get_atleta_id(fname: str) -> str:
    m = re.search(r"Player-(\d+)", fname, flags=re.I)
    return m.group(1) if m else (fname.split('-')[0] if '-' in fname else fname)

def _infer_fase(fname: str) -> str:
    n = fname.upper()
    if "WARM" in n or "WUP" in n:
        return "Warm-Up"
    if "1P" in n or "PRIMEIRA" in n or "FIRST" in n:
        return "1P"
    if "2P" in n or "SEGUNDA" in n or "SECOND" in n:
        return "2P"
    return "Extra"

def _read_csv_upload(upload, nrows=None) -> pd.DataFrame:
    # Streamlit uploaded file behaves like a file-like object
    try:
        upload.seek(0)
    except Exception:
        pass
    df = pd.read_csv(upload, sep=None, engine="python", nrows=nrows)
    try:
        upload.seek(0)
    except Exception:
        pass
    return _clean_cols(df)

def _calibrar_campo(f_campo_files, epsg: int):
    pts_gps = {}
    for f in f_campo_files:
        df_c = _read_csv_upload(f)
        if COL_LAT not in df_c.columns or COL_LON not in df_c.columns:
            continue
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper():
                pts_gps[key] = [df_c[COL_LAT].mean(), df_c[COL_LON].mean()]

    if len(pts_gps) != 4:
        missing = [k for k in ["BL","BR","TL","TR"] if k not in pts_gps]
        raise ValueError(f"Campo incompleto. Em falta: {', '.join(missing)}")

    clat = float(np.mean([p[0] for p in pts_gps.values()]))
    clon = float(np.mean([p[1] for p in pts_gps.values()]))

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts_utm = {}
    for key, (lat, lon) in pts_gps.items():
        x, y = transformer.transform(lon, lat)
        pts_utm[key] = np.array([x, y], dtype=float)

    dist_comprimento = float(np.linalg.norm(pts_utm["BR"] - pts_utm["BL"]))
    dist_largura = float(np.linalg.norm(pts_utm["TL"] - pts_utm["BL"]))

    origin = pts_utm["BL"]
    v_base = pts_utm["BR"] - origin
    angulo_rad = float(np.arctan2(v_base[1], v_base[0]))

    R = np.array([
        [np.cos(-angulo_rad), -np.sin(-angulo_rad)],
        [np.sin(-angulo_rad),  np.cos(-angulo_rad)]
    ], dtype=float)

    return pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_comprimento, dist_largura

@st.cache_data(show_spinner=False, ttl=86400)
def _reverse_geocode_city_country(lat: float, lon: float):
    """Reverse geocode via OpenStreetMap Nominatim.
    Nota: depende de acesso à internet no ambiente Streamlit Cloud.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1}
        headers = {"User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, None
        data = r.json()
        addr = data.get("address", {}) if isinstance(data, dict) else {}
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality") or addr.get("county")
        country = addr.get("country")
        return city, country
    except Exception:
        return None, None


def _geo_validacao_por_atleta(f_atleta_files, centroid_lat, centroid_lon, raio_m, amostra_n, min_pct_ok):
    ok, fora, erros = [], [], []
    for f in f_atleta_files:
        aid = _get_atleta_id(f.name)
        try:
            df = _read_csv_upload(f, nrows=int(amostra_n))
            if COL_LAT not in df.columns or COL_LON not in df.columns:
                erros.append((aid, "Sem colunas Lat/Lon"))
                continue
            sub = df[[COL_LAT, COL_LON]].dropna()
            if sub.empty:
                erros.append((aid, "Sem amostras Lat/Lon válidas"))
                continue
            lat_med = float(sub[COL_LAT].median())
            lon_med = float(sub[COL_LON].median())
            _, _, dist_m = GEOD.inv(lon_med, lat_med, centroid_lon, centroid_lat)
            if dist_m <= raio_m:
                ok.append((aid, dist_m))
            else:
                fora.append((aid, dist_m))
        except Exception as e:
            erros.append((aid, str(e)))

    total = len(set([_get_atleta_id(f.name) for f in f_atleta_files]))
    pct_ok = (len({a for a,_ in ok}) / max(1,total))
    passed = pct_ok >= min_pct_ok
    return passed, pct_ok, ok, fora, erros

def _processar_atletas_para_temp(f_atleta_files, epsg, origin, R, aplicar_suav, janela, poly, temp_dir: Path):
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # group uploads by athlete id
    groups = {}
    for f in f_atleta_files:
        aid = _get_atleta_id(f.name)
        groups.setdefault(aid, []).append(f)

    temp_files = []
    audit = {}  # aid -> list of fases
    issues = []

    for aid, files in groups.items():
        atleta_data = []
        audit[aid] = []
        for uf in files:
            try:
                df = _read_csv_upload(uf)
                fase_n = _infer_fase(uf.name)
                audit[aid].append(fase_n)

                if df.empty:
                    continue
                if COL_TIME not in df.columns:
                    issues.append((aid, uf.name, "Sem coluna Time"))
                    continue
                if COL_LAT not in df.columns or COL_LON not in df.columns:
                    issues.append((aid, uf.name, "Sem colunas Lat/Lon"))
                    continue

                # Transform
                lon = df[COL_LON].astype(float)
                lat = df[COL_LAT].astype(float)
                ux, uy = trans.transform(lon.values, lat.values)
                p = np.vstack([ux, uy]).T
                p_loc = (R @ (p - origin).T).T  # rotate around BL

                df["X_UTM"] = p_loc[:,0]
                df["Y_UTM"] = p_loc[:,1]

                # optional smoothing (requires enough points)
                if aplicar_suav and len(df) >= int(janela) and int(janela) % 2 == 1:
                    x = pd.Series(df["X_UTM"]).interpolate()
                    y = pd.Series(df["Y_UTM"]).interpolate()
                    try:
                        df["X_UTM"] = savgol_filter(x, int(janela), int(poly))
                        df["Y_UTM"] = savgol_filter(y, int(janela), int(poly))
                    except Exception:
                        # fallback: keep unsmoothed
                        pass

                df[COL_FASE] = fase_n
                df["Atleta_ID"] = aid
                atleta_data.append(df[[COL_TIME, "Atleta_ID", COL_FASE, COL_LAT, COL_LON, "X_UTM", "Y_UTM"]])
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
        aid = str(df_atl["Atleta_ID"].iloc[0]) if "Atleta_ID" in df_atl.columns else f.stem.replace("T_","")
        df_sync["Atleta_ID"] = aid

        # fill phase windows
        for fase, (t_s, t_e) in fases_dict.items():
            mask = (df_sync[COL_TIME] >= t_s) & (df_sync[COL_TIME] <= t_e)
            df_sync.loc[mask, COL_FASE] = fase

        out_path = out_dir / f"Player_{aid}_SYNC.csv"
        df_sync.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
        out_files.append(out_path)

    # Order phases for reporting
    ordem_fases = {"Warm-Up": 0, "1P": 1, "2P": 2}
    fases_ordenadas = sorted(fases_dict.items(), key=lambda x: ordem_fases.get(x[0], 99))

    return out_files, fases_ordenadas, fases_dict, len(master_df)

# -------------------------------
# Main flow
# -------------------------------
if not f_campo or not f_atleta:
    st.info("👋 Carrega os ficheiros na barra lateral para iniciar.")
    st.stop()

try:
    pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = _calibrar_campo(f_campo, int(epsg_used))

    # Local (Cidade/País) derivado das coordenadas do campo (centro)
    cidade, pais = _reverse_geocode_city_country(clat, clon)
except Exception as e:
    st.error(f"❌ Erro na calibração do campo: {e}")
    st.stop()

# Geo-validation (multi-atleta)
passed_geo, pct_ok, ok_list, fora_list, geo_errors = _geo_validacao_por_atleta(
    f_atleta, clat, clon, float(raio_validacao_m), int(amostra_geo_n), float(min_pct_atletas_ok)
)

st.header("📍 Validação de Localização (Campo ↔ Atletas)")

loc_txt = "—"
parts = [p for p in [estadio.strip() if estadio else "", cidade, pais] if p]
if parts:
    loc_txt = ", ".join(parts)

c1, c2 = st.columns([1, 2])
c1.metric("% atletas OK", f"{pct_ok*100:.0f}%")
c2.markdown(f"**Local:** {loc_txt}")

if passed_geo:
    st.success("✅ Validação geográfica aprovada.")
else:
    st.error("❌ Validação geográfica falhou (percentagem insuficiente dentro do raio).")

with st.expander("Detalhes geo-check"):
    if ok_list:
        st.write("**Dentro do raio (exemplos):**", ", ".join([f"{a} ({d:.0f} m)" for a, d in ok_list[:10]]))
    if fora_list:
        st.write("**Fora do raio (exemplos):**", ", ".join([f"{a} ({d:.0f} m)" for a, d in fora_list[:10]]))
    if geo_errors:
        st.write("**Erros (exemplos):**", ", ".join([f"{a}: {m}" for a, m in geo_errors[:10]]))

# Map
m = folium.Map(location=[clat, clon], zoom_start=18)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery', name='Esri (Satélite)').add_to(m)
for k, v in pts_gps.items():
    folium.Marker(v, popup=f"Canto {k}").add_to(m)
st_folium(m, width=1100, height=450, key="mapa_pipeline")

st.divider()

# Audit by athlete phases
st.header("👥 Auditoria de Atletas (ficheiros submetidos)")
audit_data = {}
for f in f_atleta:
    aid = _get_atleta_id(f.name)
    audit_data.setdefault(aid, [])
    audit_data[aid].append(_infer_fase(f.name))

rows = []
completos = 0
for aid in sorted(audit_data.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
    fases = audit_data[aid]
    is_ok = all(x in fases for x in ["Warm-Up", "1P", "2P"])
    if is_ok:
        completos += 1
    rows.append({
        "ID Atleta": aid,
        "Ficheiros": len(fases),
        "Estado": "✅ OK" if is_ok else "❌ INCOMPLETO",
        "Fases": ", ".join(sorted(set(fases)))
    })
st.table(pd.DataFrame(rows))
st.write(f"**Atletas completos (Warm-Up + 1P + 2P):** {completos} / {len(audit_data)}")

st.divider()

# Normalization + export
st.header("🧭 Normalização (UTM + rotação) e Exportação")

if not passed_geo:
    st.warning("A exportação está desativada porque a validação geográfica falhou. Ajusta o raio/% mínimo ou verifica os ficheiros.")
    st.stop()

btn = st.button("⚙️ Processar e Gerar Relatório", type="primary", use_container_width=True)

if btn:
    with st.spinner("A processar..."):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_dir = td_path / "Temp_Processing"
            out_dir = td_path / "Output_UTM_Sincronizado"
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            temp_files, audit_proc, issues = _processar_atletas_para_temp(
                f_atleta, int(epsg_used), origin, R,
                aplicar_suavizacao, int(janela_savgol), int(poly_savgol),
                temp_dir
            )

            if not temp_files:
                st.error("❌ Não foi possível gerar ficheiros temporários (verifica colunas Time/Lat/Lon e nomes).")
                st.stop()

            out_files, fases_ordenadas, fases_dict, n_master = _sincronizar(temp_files, out_dir)

# Session identifiers (auditoria/dedup)
session_uuid = uuid.uuid4()
session_id_hex = session_uuid.hex
session_fingerprint = _hash_session(data_sessao, selecao, genero, contexto, estadio, f_campo, f_atleta)

# Métricas individuais a partir dos SYNC (por fase + Total)
metrics_rows = []
for pth in out_files:
    df_sync = pd.read_csv(pth, sep=";")
    aid = str(df_sync["Atleta_ID"].dropna().iloc[0]) if "Atleta_ID" in df_sync.columns and df_sync["Atleta_ID"].dropna().any() else Path(pth).stem
    fases = [f for f in df_sync.get(COL_FASE, pd.Series(dtype=str)).dropna().unique().tolist() if f]
    for fase in sorted(fases):
        met = _compute_metrics_for_df(df_sync[df_sync[COL_FASE] == fase])
        metrics_rows.append({
            "session_id_hex": session_id_hex,
            "session_fingerprint": session_fingerprint,
            "data": data_sessao,
            "selecao": selecao,
            "genero": genero,
            "contexto": contexto,
            "jogo": (" vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t]) if contexto == "Jogo" else ""),
            "estadio": estadio,
            "cidade": cidade,
            "pais": pais,
            "atleta_id": aid,
            "fase": fase,
            **met,
            "engine_version": ENGINE_VERSION
        })
    met_t = _compute_metrics_for_df(df_sync)
    metrics_rows.append({
        "session_id_hex": session_id_hex,
        "session_fingerprint": session_fingerprint,
        "data": data_sessao,
        "selecao": selecao,
        "genero": genero,
        "contexto": contexto,
        "jogo": (" vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t]) if contexto == "Jogo" else ""),
        "estadio": estadio,
        "cidade": cidade,
        "pais": pais,
        "atleta_id": aid,
        "fase": "Total",
        **met_t,
        "engine_version": ENGINE_VERSION
    })

df_metrics = pd.DataFrame(metrics_rows)

            # Build report
            rot_deg = float(np.degrees(angulo_rad))
            report_lines = []
            report_lines.append("FPF Performance Hub — Relatório de Validação e Normalização")
            report_lines.append("="*70)
            report_lines.append("Dados da Sessão")
            report_lines.append(f"  Data: {data_sessao.strftime('%d/%m/%Y') if hasattr(data_sessao, 'strftime') else data_sessao}")
            report_lines.append(f"  Seleção: {selecao} | Género: {genero} | Contexto: {contexto}")
            if contexto == "Jogo":
                vs_txt = " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
                if vs_txt:
                    report_lines.append(f"  Jogo: {vs_txt}")
            loc_txt = "—"
            if estadio or cidade or pais:
                parts = [p for p in [estadio.strip(), cidade.strip(), pais.strip()] if p]
                loc_txt = ", ".join(parts)
            report_lines.append(f"  Estádio: {estadio or '—'}")
            loc_part = ", ".join([p for p in [cidade, pais] if p]) or "—"
            report_lines.append(f"  Localização: {loc_part}")
            report_lines.append(f"EPSG (UTM): {epsg_used}")
            report_lines.append(f"Comprimento (BL→BR): {dist_x:.2f} m")
            report_lines.append(f"Largura (BL→TL):     {dist_y:.2f} m")
            report_lines.append(f"Rotação aplicada:    {rot_deg:.2f}° (alinhamento BL→BR com eixo X)")
            report_lines.append("-"*70)
            report_lines.append("Validação geográfica")
            report_lines.append(f"  Raio: {raio_validacao_m:.0f} m | Amostra: {amostra_geo_n} linhas/atleta | % OK: {pct_ok*100:.0f}% (mínimo {min_pct_atletas_ok*100:.0f}%)")
            report_lines.append(f"  Dentro do raio: {len({a for a,_ in ok_list})} atletas | Fora: {len({a for a,_ in fora_list})} atletas | Erros: {len(geo_errors)}")
            report_lines.append("-"*70)
            report_lines.append("Auditoria de atletas (submissão)")
            report_lines.append(f"  Atletas totais: {len(audit_data)} | Atletas completos (Warm-Up+1P+2P): {completos}")
            report_lines.append("-"*70)
            report_lines.append("Sincronização")
            report_lines.append(f"  Timestamps mestre: {n_master}")
            report_lines.append(f"  Ficheiros gerados: {len(out_files)}")
            report_lines.append("  Fases (ordem cronológica):")
            for fase, (t_s, t_e) in fases_ordenadas:
                report_lines.append(f"    - {fase:8} | início: {t_s} | fim: {t_e}")
            if issues:
                report_lines.append("-"*70)
                report_lines.append("Avisos/Problemas (exemplos):")
                for aid, fn, msg in issues[:25]:
                    report_lines.append(f"  - {aid} | {fn} | {msg}")
report_lines.append("-"*70)
report_lines.append("Métricas Individuais (GPS-only) — thresholds fixos")
report_lines.append(f"  HSR ≥ {HSR_MPS:.1f} m/s | Sprint ≥ {SPRINT_MPS:.1f} m/s | Acc ≥ {ACC_THR:.1f} m/s² | Dec ≤ {DEC_THR:.1f} m/s²")
report_lines.append(f"  Session ID (hex): {session_id_hex}")
report_lines.append(f"  Session fingerprint (sha1): {session_fingerprint}")

if 'df_metrics' in locals() and not df_metrics.empty:
    try:
        df_total = df_metrics[df_metrics["fase"] == "Total"].copy()
        df_total = df_total.dropna(subset=["m_min"])
        report_lines.append("  Top 3 (Total) — m/min:")
        top = df_total.sort_values("m_min", ascending=False).head(3)
        for _, r in top.iterrows():
            report_lines.append(f"    - Atleta {r['atleta_id']}: {r['m_min']:.1f} m/min | Dist {r['dist_m']:.0f} m | Dur {r['duracao_min']:.1f} min")
    except Exception:
        pass
    try:
        df_total2 = df_metrics[df_metrics["fase"] == "Total"].copy()
        df_total2 = df_total2.dropna(subset=["peak_1m_m_min"])
        report_lines.append("  Top 3 (Total) — Peak 1' (m/min):")
        top2 = df_total2.sort_values("peak_1m_m_min", ascending=False).head(3)
        for _, r in top2.iterrows():
            report_lines.append(f"    - Atleta {r['atleta_id']}: {r['peak_1m_m_min']:.0f} m/min | Vmax {r['vmax_mps']:.2f} m/s")
    except Exception:
        pass
else:
    report_lines.append("  (Sem métricas calculadas)")

            report_txt = "\n".join(report_lines)

    st.success("✅ Processamento concluído. Relatório disponível abaixo.")

st.subheader("📊 Métricas Individuais (pré-visualização)")
st.caption(f"Thresholds fixos: HSR ≥ {HSR_MPS:.1f} m/s | Sprint ≥ {SPRINT_MPS:.1f} m/s | Acc ≥ {ACC_THR:.1f} m/s² | Dec ≤ {DEC_THR:.1f} m/s²")
if 'df_metrics' in locals() and isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Métricas (.csv)",
        data=df_metrics.to_csv(index=False).encode("utf-8"),
        file_name="metricas_individuais_FPF.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.warning("Sem métricas para mostrar (verifica se os SYNC foram gerados corretamente).")


    st.subheader("📄 Relatório (pré-visualização)")
    st.code(report_txt, language="text")
    st.download_button(
        "⬇️ Download Relatório (.txt)",
        data=report_txt.encode("utf-8"),
        file_name="relatorio_FPF.txt",
        mime="text/plain",
        use_container_width=True
    )
