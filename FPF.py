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

GEOD = Geod(ellps="WGS84")  # WGS84 geodesic distance (metros reais)

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v11.1", layout="wide")
if "auth" not in st.session_state:
    st.session_state.auth = False



if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "user_org" not in st.session_state:
    st.session_state.user_org = None

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


def _get_auth_profile_from_secrets():
    """
    Opcional: Nome/Entidade para mostrar no topo após login.
    No secrets.toml podes definir:
      [auth]
      username = "..."
      password = "..."
      display_name = "Marcos Cardoso"
      org = "FPF"
    """
    try:
        a = st.secrets.get("auth", {})
        display_name = a.get("display_name") or a.get("name") or None
        org = a.get("org") or a.get("entidade") or a.get("institution") or None
        return display_name, org
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

                st.session_state.auth = True


                # Nome/Entidade para header (fallback: usa o utilizador e "FPF")

                dn, org = _get_auth_profile_from_secrets()

                st.session_state.user_name = dn or u

                st.session_state.user_org = org or "FPF"


                st.rerun()

            else:

                st.error("Credenciais inválidas")


        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()




# --- HEADER UTILIZADOR (topo esquerdo) ---
def _render_user_badge():
    nome = st.session_state.get("user_name") or "Utilizador"
    org = st.session_state.get("user_org") or ""

    st.markdown(
        f"""
        <style>
          /* Reserva espaço no topo para a barra (evita tapar conteúdo) */
          .main .block-container {{
            padding-top: 4.25rem !important;
          }}

          .topbar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px 0 14px;
            background: rgba(15, 17, 23, 0.92);
            border-bottom: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(10px);
          }}

          .topbar-left {{
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 280px;
          }}

          .pill {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
          }}

          .avatar {{
            width: 30px;
            height: 30px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(99,102,241,0.22);
            border: 1px solid rgba(99,102,241,0.35);
            font-size: 16px;
          }}

          .who {{
            display: flex;
            flex-direction: column;
            line-height: 1.05;
          }}

          .who .name {{
            font-weight: 800;
            font-size: 13.5px;
            color: rgba(255,255,255,0.95);
          }}

          .who .org {{
            font-weight: 600;
            font-size: 11.5px;
            color: rgba(255,255,255,0.70);
            margin-top: 2px;
          }}

          .topbar-right {{
            display: flex;
            align-items: center;
            gap: 10px;
            opacity: 0.95;
          }}

          .app-tag {{
            font-weight: 800;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.75);
          }}
        </style>

        <div class="topbar">
          <div class="topbar-left">
            <div class="pill">
              <div class="avatar">👤</div>
              <div class="who">
                <div class="name">{nome}</div>
                <div class="org">{org}</div>
              </div>
            </div>
          </div>
          <div class="topbar-right">
            <div class="app-tag">FPF Performance Hub</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

_render_user_badge()

# --- INTERFACE SINGLE PAGE ---
st.title("Validação de Dados")

with st.sidebar:
    st.header("🧾 Dados da Sessão")
    # Estádio agora é inferido automaticamente pela localização do campo (sem input manual)
    estadio = None
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
    st.caption("Campo: podes fazer upload de 4 CSVs (BL, BR, TL, TR) **ou** usar o modo 'Pick no mapa'.")
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
    st.header("🗺️ Calibração do Campo")
    metodo_campo = st.radio(
        "Como queres definir os 4 cantos?",
        options=["Upload (BL/BR/TL/TR)", "Pick no mapa (clicar 4 cantos)"],
        index=0,
        help="Alternativa ao upload: usa um mapa satélite e clica nos 4 cantos do campo.",
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

# -------------------------------
# Métricas Individuais (GPS-only)
# -------------------------------
HSR_MPS = 5.5  # High-Speed Running (m/s) ~19.8 km/h
SPRINT_MPS = 7.0  # Sprint (m/s) ~25.2 km/h
ACC_THR = 2.5  # m/s^2
DEC_THR = -3.0  # m/s^2
SPRINT_BOUT_MIN_S = 1.0  # duração mínima do bout de sprint (s)
ENGINE_VERSION = "v12-metrics"

COL_LAT, COL_LON, COL_TIME, COL_FASE = "Lat", "Lon", "Time", "Fase"


def _time_to_seconds(series: pd.Series) -> pd.Series:
    """Converte Time para segundos (float). Suporta numérico (s/ms) ou datetime-like string."""
    s = series.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        med = float(s_num.dropna().median()) if s_num.notna().any() else 0.0
        if med > 1e12:  # epoch ms
            return s_num / 1000.0
        if med > 1e9:  # epoch s
            return s_num.astype(float)
        return s_num.astype(float)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        t0 = dt.dropna().iloc[0]
        return (dt - t0).dt.total_seconds()
    return pd.Series([np.nan] * len(s), index=s.index, dtype="float64")


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
            dur = t[i - 1] - t_start if t_start is not None else 0.0
            if dur >= min_dur_s:
                bouts += 1
            in_bout = False
            t_start = None
    if in_bout and t_start is not None:
        dur = t[-1] - t_start
        if dur >= min_dur_s:
            bouts += 1
    return bouts



def _audit_timebase(df: pd.DataFrame, col_time: str, expected_hz: float = 10.0) -> dict:
    """Audita base temporal (por atleta/fase) para diagnosticar duplicados, gaps e Hz."""
    if df is None or df.empty or col_time not in df.columns:
        return {
            "n_rows": 0,
            "n_valid_t": 0,
            "wallclock_s": np.nan,
            "dt_median_s": np.nan,
            "hz_est": np.nan,
            "n_dt_neg": 0,
            "n_dt_zero": 0,
            "n_gaps_gt_0_2s": 0,
            "n_gaps_gt_2s": 0,
        }

    t = _time_to_seconds(df[col_time])
    t = pd.to_numeric(t, errors="coerce").dropna()
    if t.shape[0] < 2:
        return {
            "n_rows": int(len(df)),
            "n_valid_t": int(t.shape[0]),
            "wallclock_s": 0.0,
            "dt_median_s": np.nan,
            "hz_est": np.nan,
            "n_dt_neg": 0,
            "n_dt_zero": 0,
            "n_gaps_gt_0_2s": 0,
            "n_gaps_gt_2s": 0,
        }

    t = t.sort_values().to_numpy(dtype=float)
    dt = np.diff(t)

    n_dt_neg = int((dt < 0).sum())
    n_dt_zero = int((dt == 0).sum())
    dt_pos = dt[dt > 0]

    dt_median = float(np.median(dt_pos)) if dt_pos.size else np.nan
    hz_est = float(1.0 / dt_median) if (dt_median and dt_median > 0) else np.nan

    n_gaps_0_2 = int((dt_pos > 0.2).sum())  # para 10Hz: >0.2s é gap relevante
    n_gaps_2 = int((dt_pos > 2.0).sum())

    wallclock_s = float(t[-1] - t[0])
    return {
        "n_rows": int(len(df)),
        "n_valid_t": int(len(t)),
        "wallclock_s": wallclock_s,
        "dt_median_s": dt_median,
        "hz_est": hz_est,
        "n_dt_neg": n_dt_neg,
        "n_dt_zero": n_dt_zero,
        "n_gaps_gt_0_2s": n_gaps_0_2,
        "n_gaps_gt_2s": n_gaps_2,
    }


def _compute_metrics_for_df(df: pd.DataFrame) -> dict:
    # Defaults (evita KeyError em fases vazias/curtas)
    DEFAULT_METRICS = {
        "duracao_min": 0.0,
        "dist_m": 0.0,
        "m_min": float("nan"),
        "hsr_dist_m": 0.0,
        "sprint_dist_m": 0.0,
        "n_sprints": 0,
        "n_acc_2_5": 0,
        "n_dec_3_0": 0,
        "vmax_mps": float("nan"),
        "peak_1m_m_min": float("nan"),
        "hsr_pct": float("nan"),
        "pct_time_valid": float("nan"),
        "n_gaps_gt2s": 0,
        "n_points": 0,
        "active_time_min": 0.0,
        "active_pct": float("nan"),
    }

    """Calcula métricas para um atleta numa fase (df filtrado)."""
    if df.empty:
        return {
            "duracao_min": 0.0,
            "dist_m": 0.0,
            "m_min": np.nan,
            "peak_1m_m_min": np.nan,
            "vmax_mps": np.nan,
            "hsr_dist_m": 0.0,
            "hsr_pct": np.nan,
            "sprint_dist_m": 0.0,
            "n_sprints": 0,
            "n_acc_2_5": 0,
            "n_dec_3_0": 0,
            "n_points": 0,
            "pct_time_valid": 0.0,
            "n_gaps_gt2s": 0,
        }

    t_sec = _time_to_seconds(df[COL_TIME])
    x = pd.to_numeric(df["X_UTM"], errors="coerce")
    y = pd.to_numeric(df["Y_UTM"], errors="coerce")

    valid = t_sec.notna() & x.notna() & y.notna()
    dfv = pd.DataFrame({"t": t_sec[valid], "x": x[valid], "y": y[valid]}).sort_values("t")

    if len(dfv) < 2:
        return {
            "duracao_min": 0.0,
            "dist_m": 0.0,
            "m_min": np.nan,
            "peak_1m_m_min": np.nan,
            "vmax_mps": np.nan,
            "hsr_dist_m": 0.0,
            "hsr_pct": np.nan,
            "sprint_dist_m": 0.0,
            "n_sprints": 0,
            "n_acc_2_5": 0,
            "n_dec_3_0": 0,
            "n_points": int(len(dfv)),
            "pct_time_valid": float(valid.mean() * 100.0),
            "n_gaps_gt2s": 0,
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
            "duracao_min": 0.0,
            "dist_m": 0.0,
            "m_min": np.nan,
            "peak_1m_m_min": np.nan,
            "vmax_mps": np.nan,
            "hsr_dist_m": 0.0,
            "hsr_pct": np.nan,
            "sprint_dist_m": 0.0,
            "n_sprints": 0,
            "n_acc_2_5": 0,
            "n_dec_3_0": 0,
            "n_points": int(len(dfv)),
            "pct_time_valid": float(valid.mean() * 100.0),
            "n_gaps_gt2s": n_gaps,
        "active_time_min": active_time_min,
        "active_pct": active_pct,
        }

    dist_step = np.hypot(dx, dy)
    dist_total = float(np.nansum(dist_step))
    dur_s = float(np.nansum(dt))
    dur_min = dur_s / 60.0 if dur_s > 0 else 0.0
    m_min = (dist_total / dur_min) if dur_min > 0 else np.nan

    v = dist_step / dt
    vmax = float(np.nanmax(v)) if len(v) else np.nan

    # Active time (tempo em movimento)
    ACTIVE_V_THR = 0.5  # m/s
    active_time_s = float(np.nansum(dt[v >= ACTIVE_V_THR])) if len(v) else 0.0
    active_time_min = active_time_s / 60.0 if active_time_s > 0 else 0.0
    active_pct = (active_time_s / dur_s * 100.0) if dur_s > 0 else np.nan

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
    # Nota: t[1:] tem comprimento igual a np.diff(t) (antes de filtrar); usamos a mesma máscara good
    t_cum = np.concatenate([[t[0]], t[1:][good]])

    peak_1m = np.nan
    if len(t_cum) == len(dist_cum) and len(t_cum) > 1:
        best = 0.0
        j = 0
        for i in range(len(t_cum)):
            while j < len(t_cum) and t_cum[j] - t_cum[i] <= 60.0:
                j += 1
            if j - 1 >= i:
                dj = dist_cum[j - 1] - dist_cum[i]
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
        "pct_time_valid": float(valid.mean() * 100.0),
        "n_gaps_gt2s": n_gaps,
        "active_time_min": active_time_min,
        "active_pct": active_pct,
    }


def _hash_session(
    data_sessao, selecao, genero, contexto, estadio, f_campo_files, f_atleta_files
) -> str:
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


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    return df


def _get_atleta_id(fname: str) -> str:
    m = re.search(r"Player-(\d+)", fname, flags=re.I)
    return m.group(1) if m else (fname.split("-")[0] if "-" in fname else fname)


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
        missing = [k for k in ["BL", "BR", "TL", "TR"] if k not in pts_gps]
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

    # Rotação para alinhar BL->BR no eixo X
    R = np.array(
        [
            [np.cos(-angulo_rad), -np.sin(-angulo_rad)],
            [np.sin(-angulo_rad), np.cos(-angulo_rad)],
        ],
        dtype=float,
    )

    return (
        pts_gps,
        (clat, clon),
        pts_utm,
        origin,
        R,
        angulo_rad,
        dist_comprimento,
        dist_largura,
    )


def _order_corners_latlon(points):
    """
    Recebe lista de 4 pontos [(lat, lon), ...] e devolve dict com chaves BL, BR, TL, TR.
    Regra:
      - divide por lat (top 2 e bottom 2)
      - dentro de cada par, ordena por lon (esq/dir)
    """
    if points is None or len(points) != 4:
        raise ValueError("São necessários exatamente 4 pontos para ordenar cantos.")

    pts = [(float(lat), float(lon)) for lat, lon in points]
    pts_sorted_lat = sorted(pts, key=lambda p: p[0], reverse=True)  # maior lat = norte (topo)
    top = pts_sorted_lat[:2]
    bottom = pts_sorted_lat[2:]

    top_sorted = sorted(top, key=lambda p: p[1])      # menor lon = esquerda
    bottom_sorted = sorted(bottom, key=lambda p: p[1])

    TL = top_sorted[0]
    TR = top_sorted[1]
    BL = bottom_sorted[0]
    BR = bottom_sorted[1]

    return {"BL": [BL[0], BL[1]], "BR": [BR[0], BR[1]], "TL": [TL[0], TL[1]], "TR": [TR[0], TR[1]]}


def _calibrar_campo_from_pts_gps(pts_gps: dict, epsg: int):
    """
    Igual ao _calibrar_campo, mas recebe os 4 cantos já como lat/lon.
    pts_gps: {"BL":[lat,lon], "BR":[lat,lon], "TL":[lat,lon], "TR":[lat,lon]}
    """
    for k in ["BL", "BR", "TL", "TR"]:
        if k not in pts_gps:
            raise ValueError(f"Campo incompleto. Em falta: {k}")

    clat = float(np.mean([pts_gps[k][0] for k in pts_gps]))
    clon = float(np.mean([pts_gps[k][1] for k in pts_gps]))

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts_utm = {}
    for key, (lat, lon) in pts_gps.items():
        x, y = transformer.transform(float(lon), float(lat))
        pts_utm[key] = np.array([x, y], dtype=float)

    dist_comprimento = float(np.linalg.norm(pts_utm["BR"] - pts_utm["BL"]))
    dist_largura = float(np.linalg.norm(pts_utm["TL"] - pts_utm["BL"]))

    origin = pts_utm["BL"]
    v_base = pts_utm["BR"] - origin
    angulo_rad = float(np.arctan2(v_base[1], v_base[0]))

    # Rotação para alinhar BL->BR no eixo X
    R = np.array(
        [
            [np.cos(-angulo_rad), -np.sin(-angulo_rad)],
            [np.sin(-angulo_rad), np.cos(-angulo_rad)],
        ],
        dtype=float,
    )

    return (
        pts_gps,
        (clat, clon),
        pts_utm,
        origin,
        R,
        angulo_rad,
        dist_comprimento,
        dist_largura,
    )


def _sample_athlete_track_latlon(f_atleta_files, max_points=600):
    """
    Lê um atleta (primeiro ficheiro) e devolve uma amostra de pontos lat/lon para desenhar no mapa.
    Serve apenas para orientar o utilizador no 'pick' dos cantos.
    """
    if not f_atleta_files:
        return []
    try:
        df = _read_csv_upload(f_atleta_files[0], nrows=5000)
        if COL_LAT not in df.columns or COL_LON not in df.columns:
            return []
        sub = df[[COL_LAT, COL_LON]].dropna()
        if sub.empty:
            return []
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=7)
        pts = sub.values.tolist()
        return [(float(lat), float(lon)) for lat, lon in pts if np.isfinite(lat) and np.isfinite(lon)]
    except Exception:
        return []





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
def _reverse_geocode_place_city_country(lat: float, lon: float):
    """Reverse geocode via OpenStreetMap Nominatim.
    Devolve (place_name, city, country). 'place_name' tenta capturar estádio/recinto quando disponível.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1}
        headers = {"User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, None, None
        data = r.json()
        if not isinstance(data, dict):
            return None, None, None

        addr = data.get("address", {}) or {}
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or addr.get("county")
        )
        country = addr.get("country")

        # Melhor esforço para capturar um nome de recinto/estádio
        place = (
            data.get("name")
            or addr.get("stadium")
            or addr.get("sports_centre")
            or addr.get("amenity")
            or data.get("display_name")
        )
        return place, city, country
    except Exception:
        return None, None, None

def _geo_validacao_por_atleta(
    f_atleta_files, centroid_lat, centroid_lon, raio_m, amostra_n, min_pct_ok
):
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
    pct_ok = (len({a for a, _ in ok}) / max(1, total))
    passed = pct_ok >= min_pct_ok
    return passed, pct_ok, ok, fora, erros

def _get_atletas_centroid_latlon(f_atleta_files, amostra_n=500):
    per_atleta = {}

    for f in f_atleta_files:
        aid = _get_atleta_id(f.name)
        try:
            df = _read_csv_upload(f, nrows=int(amostra_n))
            if COL_LAT not in df.columns or COL_LON not in df.columns:
                continue

            sub = df[[COL_LAT, COL_LON]].dropna()
            if sub.empty:
                continue

            lat_med = float(pd.to_numeric(sub[COL_LAT], errors="coerce").dropna().median())
            lon_med = float(pd.to_numeric(sub[COL_LON], errors="coerce").dropna().median())

            if np.isfinite(lat_med) and np.isfinite(lon_med):
                per_atleta.setdefault(aid, []).append((lat_med, lon_med))

        except Exception:
            continue

    if not per_atleta:
        return None, None

    atleta_meds = []
    for vals in per_atleta.values():
        lats = [v[0] for v in vals]
        lons = [v[1] for v in vals]
        atleta_meds.append((float(np.median(lats)), float(np.median(lons))))

    lat_c = float(np.median([x[0] for x in atleta_meds]))
    lon_c = float(np.median([x[1] for x in atleta_meds]))

    return lat_c, lon_c


def _processar_atletas_para_temp(
    f_atleta_files, epsg, origin, R, aplicar_suav, janela, poly, temp_dir: Path
):
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

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
    alat0, alon0 = _get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)
    if alat0 is None or alon0 is None:
        pts_fallback = _sample_athlete_track_latlon(f_atleta, max_points=10)
        if pts_fallback:
            alat0, alon0 = pts_fallback[0]
        else:
            alat0, alon0 = 0.0, 0.0

    st.header("Pick dos 4 cantos do campo")
    st.caption(
        "Clica no mapa 4 vezes (um por canto). O sistema tenta ordenar automaticamente em TL/TR/BR/BL. "
        "Se falhar, usa 'Reset' e volta a clicar com mais zoom."
    )

    m_pick = folium.Map(location=[alat0, alon0], zoom_start=18)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri (Satélite)",
    ).add_to(m_pick)

    pts_track = _sample_athlete_track_latlon(f_atleta, max_points=600)
    if pts_track:
        folium.PolyLine(pts_track, weight=2, opacity=0.8).add_to(m_pick)

    for i, (lat, lon) in enumerate(st.session_state.pick_corners, start=1):
        folium.Marker([lat, lon], tooltip=f"Pick {i}").add_to(m_pick)

    if len(st.session_state.pick_corners) == 4:
        folium.Polygon(st.session_state.pick_corners, tooltip="Cantos (pick)").add_to(m_pick)

    out_pick = st_folium(m_pick, width=1100, height=520, key="mapa_pick_cantos")

    if out_pick and out_pick.get("last_clicked"):
        lat = float(out_pick["last_clicked"]["lat"])
        lon = float(out_pick["last_clicked"]["lng"])
        if len(st.session_state.pick_corners) < 4:
            st.session_state.pick_corners.append((lat, lon))
            st.rerun()

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("↩️ Desfazer", disabled=(len(st.session_state.pick_corners) == 0)):
            st.session_state.pick_corners.pop()
            st.rerun()
    with c2:
        if st.button("🧹 Reset"):
            st.session_state.pick_corners = []
            st.rerun()

    if len(st.session_state.pick_corners) == 4:
        try:
            st.session_state.pts_gps_picked = _order_corners_latlon(st.session_state.pick_corners)
            st.subheader("Cantos (ordenados)")
            st.json(st.session_state.pts_gps_picked)
            st.success("✅ Cantos definidos. Agora o pipeline continua normalmente.")
        except Exception as e:
            st.error(f"Não consegui ordenar os cantos automaticamente: {e}")
            st.info("Sugestão: faz mais zoom e clica exatamente nos 4 cantos; se necessário, usa Reset.")
    else:
        st.info(f"Pontos escolhidos: {len(st.session_state.pick_corners)}/4")

    st.stop()

try:
    if metodo_campo == "Pick no mapa (clicar 4 cantos)":
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = _calibrar_campo_from_pts_gps(
            st.session_state.pts_gps_picked, int(epsg_used)
        )
    else:
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = _calibrar_campo(
            f_campo, int(epsg_used)
        )

    estadio, cidade, pais = _reverse_geocode_place_city_country(clat, clon)
except Exception as e:

    st.error(f"❌ Erro na calibração do campo: {e}")
    st.stop()

passed_geo, pct_ok, ok_list, fora_list, geo_errors = _geo_validacao_por_atleta(
    f_atleta, clat, clon, float(raio_validacao_m), int(amostra_geo_n), float(min_pct_atletas_ok)
)

st.header("Validação de Localização (Campo ↔ Atletas)")

# Campo
_, cidade_campo, pais_campo = _reverse_geocode_place_city_country(clat, clon)

# Atletas (centro estimado)
alat, alon = _get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)

cidade_atl, pais_atl = None, None
if alat is not None and alon is not None:
    _, cidade_atl, pais_atl = _reverse_geocode_place_city_country(alat, alon)
# ----- Campo -----
campo_local = ", ".join([p for p in [cidade_campo, pais_campo] if p]) or "—"

# ----- Atletas (centro médio → Cidade/País) -----
atletas_local = ", ".join([p for p in [cidade_atl, pais_atl] if p]) or "—"

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
    aid = _get_atleta_id(f.name)
    audit_data.setdefault(aid, [])
    audit_data[aid].append(_infer_fase(f.name))

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
            session_fingerprint = _hash_session(
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
                    met = _compute_metrics_for_df(df_f)
                    fase_mets[fase] = met

                    aud = _audit_timebase(df_f, COL_TIME, expected_hz=10.0)
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
                            " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
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
                vs_txt = " vs ".join([t for t in [adversario_a.strip(), adversario_b.strip()] if t])
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
