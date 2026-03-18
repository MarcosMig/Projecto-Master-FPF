# -*- coding: utf-8 -*-
"""
FPF UTM Engine v16 (parquet downloads)
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

# testing
from streamlit_folium import st_folium
from scipy.signal import savgol_filter

# -- Local Modules -- #
from fpf_modules.constants import (
    HSR_MPS, SPRINT_MPS, ACC_THR, DEC_THR, SPRINT_BOUT_MIN_S, ENGINE_VERSION,
    COL_LAT, COL_LON, COL_TIME, COL_FASE,
    SELECOES_OPCOES, CLEANDATA_DIR
)

from fpf_modules.metrics import (
    time_to_seconds,
    count_bouts,
    audit_timebase,
    compute_metrics_for_df
)

from fpf_modules.qc import qc_gps_df

from fpf_modules.io_utils import (
    hash_session,
    clean_cols,
    get_atleta_id,
    infer_fase,
    read_csv_upload
)

from fpf_modules.utils import round_metrics_dataframe, file_to_bytes

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

from fpf_modules.supabase_manager import (
    save_field_to_parquet,
    read_field_from_parquet,
    append_dedup_parquet,
    resolve_athlete_sk,
    resolve_session_sk,
    write_session_data,
)

from fpf_modules.normalize import (
    normalize_tracking_data
)
GEOD = Geod(ellps="WGS84")  # WGS84 geodesic distance (metros reais)


PHASE_MAP = {
    "Warm-Up": 0,
    "1P": 1,
    "2P": 2,
    "Total": 3,
}

HR_CANDIDATE_COLS = [
    "HR_bpm", "HR", "HeartRate", "Heart Rate", "Heart_Rate", "BPM", "Pulse"
]


def _detect_hr_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for col in HR_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


def _build_samples_export(out_files, session_sk: int, athlete_map: dict):
    sample_frames = []
    athlete_session_rows = []
    processed_at = pd.Timestamp.utcnow()

    for pth in out_files:
        df_sync = pd.read_csv(pth, sep=";")
        if df_sync.empty:
            continue

        athlete_id = str(df_sync["Atleta_ID"].dropna().iloc[0]) if "Atleta_ID" in df_sync.columns and df_sync["Atleta_ID"].dropna().any() else Path(pth).stem
        athlete_sk = athlete_map.get(str(athlete_id))
        hr_col = _detect_hr_col(df_sync)

        sample_df = pd.DataFrame({
            "session_sk": session_sk,
            "athlete_sk": athlete_sk,
            "atleta_id": str(athlete_id),
            "fase": df_sync[COL_FASE] if COL_FASE in df_sync.columns else pd.NA,
            "time": df_sync[COL_TIME] if COL_TIME in df_sync.columns else pd.NA,
            "time_evento_s": df_sync["Time_Evento_s"] if "Time_Evento_s" in df_sync.columns else np.nan,
            "time_evento": df_sync["Time_Evento"] if "Time_Evento" in df_sync.columns else pd.NA,
            "periodo_jogo": df_sync["Periodo_Jogo"] if "Periodo_Jogo" in df_sync.columns else pd.NA,
            "minuto_jogo": df_sync["Minuto_Jogo"] if "Minuto_Jogo" in df_sync.columns else pd.NA,
            "lat": df_sync[COL_LAT] if COL_LAT in df_sync.columns else np.nan,
            "lon": df_sync[COL_LON] if COL_LON in df_sync.columns else np.nan,
            "x_utm": df_sync["X_UTM"] if "X_UTM" in df_sync.columns else np.nan,
            "y_utm": df_sync["Y_UTM"] if "Y_UTM" in df_sync.columns else np.nan,
            "hr_bpm": df_sync[hr_col] if hr_col else np.nan,
        })

        # Convert time column to proper timestamp format for DuckDB
        if "time" in sample_df.columns and not sample_df["time"].isna().all():
            # Handle relative time format (HH:MM:SS.s) by combining with a base date
            try:
                # Check if time values contain date separators
                time_strs = sample_df["time"].astype(str)
                has_dates = time_strs.str.contains('-', na=False)
                
                if has_dates.any():
                    # Some values have dates, try full timestamp format first
                    sample_df["time"] = pd.to_datetime(sample_df["time"], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    # Fill any NaT values with time-only parsing
                    still_nat = sample_df["time"].isna()
                    if still_nat.any():
                        time_only = pd.to_datetime(sample_df.loc[still_nat, "time"], format='%H:%M:%S.%f', errors='coerce')
                        base_date = pd.Timestamp('2023-01-01')
                        sample_df.loc[still_nat, "time"] = base_date + (time_only - time_only.dt.normalize())
                else:
                    # Handle relative time format by adding a base date (e.g., 2023-01-01)
                    base_date = pd.Timestamp('2023-01-01')
                    # Parse time strings and add to base date
                    time_parsed = pd.to_datetime(sample_df["time"], format='%H:%M:%S.%f', errors='coerce')
                    sample_df["time"] = base_date + (time_parsed - time_parsed.dt.normalize())
            except:
                # Fallback: try direct conversion with specific formats
                try:
                    sample_df["time"] = pd.to_datetime(sample_df["time"], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                except:
                    try:
                        sample_df["time"] = pd.to_datetime(sample_df["time"], format='%H:%M:%S.%f', errors='coerce')
                        base_date = pd.Timestamp('2023-01-01')
                        sample_df["time"] = base_date + (sample_df["time"] - sample_df["time"].dt.normalize())
                    except:
                        # Final fallback
                        sample_df["time"] = pd.to_datetime(sample_df["time"], errors="coerce")
            
            # Keep as datetime for DuckDB TIMESTAMP compatibility (don't convert to string)

        sample_df["phase_id"] = sample_df["fase"].map(PHASE_MAP)
        sample_frames.append(sample_df)

        fases_presentes = sorted([f for f in sample_df["fase"].dropna().astype(str).unique().tolist() if f in ["Warm-Up", "1P", "2P"]], key=lambda x: PHASE_MAP.get(x, 99))
        athlete_session_rows.append({
            "session_sk": session_sk,
            "athlete_sk": athlete_sk,
            "atleta_id": str(athlete_id),
            "participou_warmup": "Warm-Up" in fases_presentes,
            "participou_1p": "1P" in fases_presentes,
            "participou_2p": "2P" in fases_presentes,
            "fases_disponiveis": ",".join(fases_presentes),
            "n_samples": int(len(sample_df)),
            "tem_hr": bool(hr_col is not None),
            "processado_em": processed_at,
        })

    df_samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    df_athlete_session = pd.DataFrame(athlete_session_rows)
    
    # Ensure timestamp columns are properly formatted for DuckDB and parquet
    if not df_athlete_session.empty and "processado_em" in df_athlete_session.columns:
        # Ensure processado_em is datetime type - recreate if necessary
        current_time = pd.Timestamp.utcnow()
        df_athlete_session["processado_em"] = current_time
        # Keep as datetime for parquet compatibility (don't convert to string)
    
    return df_samples, df_athlete_session

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v16", layout="wide", initial_sidebar_state="collapsed")
if "auth" not in st.session_state:
    st.session_state.auth = False
if "login_user" not in st.session_state:
    st.session_state.login_user = ""

if not st.session_state.auth:
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Persistência de outputs (evita desaparecer após zoom/scroll no mapa) ---
if "df_metrics" not in st.session_state:
    st.session_state.df_metrics = None
if "report_txt" not in st.session_state:
    st.session_state.report_txt = None
if "manual_metricas_txt" not in st.session_state:
    st.session_state.manual_metricas_txt = None
if "process_done" not in st.session_state:
    st.session_state.process_done = False
if "df_perf" not in st.session_state:
    st.session_state.df_perf = None
if "df_qc" not in st.session_state:
    st.session_state.df_qc = None
if "df_samples" not in st.session_state:
    st.session_state.df_samples = None
if "df_athlete_session" not in st.session_state:
    st.session_state.df_athlete_session = None

# --- Persistência para 'Pick no mapa' (cantos do campo) ---
if "pick_corners" not in st.session_state:
    st.session_state.pick_corners = []  # lista [(lat, lon), ...]
if "pts_gps_picked" not in st.session_state:
    st.session_state.pts_gps_picked = None  # dict com BL/BR/TL/TR após ordenação
if "pick_last_click_sig" not in st.session_state:
    # evita duplicar o mesmo clique após rerun
    st.session_state.pick_last_click_sig = None

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
        u = auth.get("username")
        p = auth.get("password")
        # DEBUG: remove isto depois
        print(f"[DEBUG] Auth from secrets: username={repr(u)}, password={'*' * len(p) if p else None}")
        return u, p
    except Exception as e:
        print(f"[DEBUG] Secrets error: {e}")
        return None, None


if not st.session_state.auth:
    _apply_login_style()

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(
            '<div class="login-title">FPF Performance Hub</div>', unsafe_allow_html=True)

        u = st.text_input("Utilizador", key="user_val")
        p = st.text_input("Password", type="password", key="pass_val")

        secrets_user, secrets_pass = _get_auth_from_secrets()
        if not secrets_user:
            st.warning(
                "⚠️ Credenciais não configuradas em st.secrets. Defina [auth] no secrets.toml / Streamlit Cloud.")

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

    st.header("Dados de ATLETAS")
    st.caption(
        "CSVs com Player-<id> e indicação de fase (Warm/Primeira/Segunda/1P/2P) no nome do ficheiro."
    )
    f_atleta = st.file_uploader(
        "Dados de ATLETAS (CSVs)", accept_multiple_files=True, type=["csv"]
    )
    st.divider()

    st.header("🗺️ Calibração do Campo")
    st.caption(
        "Define os 4 cantos via upload (BL/BR/TL/TR) ou usando o modo 'Pick no mapa'.")

    metodo_campo = st.radio(

        "Como queres definir os 4 cantos?",

        options=["Upload (BL/BR/TL/TR)", "Pick no mapa (clicar 4 cantos)",
                 "Escolher um campo guardado anteriormente"],

        index=0,

        help="Alternativa ao upload: usa um mapa satélite e clica nos 4 cantos do campo.",

    )

    st.caption(
        "Se escolheres 'Pick no mapa', não precisas de carregar os 4 CSVs do campo.")

    f_campo = st.file_uploader(
        "Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True, type=["csv"]
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
        params = {"format": "jsonv2", "lat": lat,
                  "lon": lon, "zoom": 10, "addressdetails": 1}
        headers = {
            "User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)"}
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
def _fmt_match_clock(seconds):
    if seconds is None or pd.isna(seconds):
        return "—"
    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    minutes = total // 60
    secs = total % 60
    return f"{sign}00:{minutes:02d}:{secs:02d}"


def _build_metric_groups():
    return {
        "Performance": {
            "Volume": [
                "duracao_min",
                "dist_m",
                "hsr_dist_m",
                "sprint_dist_m",
                "active_time_min",
            ],
            "Intensidade": [
                "m_min",
                "hsr_pct",
                "active_pct",
            ],
            "Eventos": [
                "n_sprints",
                "n_acc_2_5",
                "n_dec_3_0",
            ],
            "Picos de Fase": [
                "vmax_mps",
                "peak_1m_m_min",
            ],
        },
        "Disponibilidade / Integridade": {
            "Completude do sinal": ["n_points", "pct_time_valid", "n_gaps_gt2s"],
        },
        "QC / Confiabilidade": {
            "QC": ["qc_grade", "qc_flags", "vmax_mps_qc", "n_jumps_gt15m", "n_gaps_gt2s_qc"],
        },
    }


METRIC_INFO = {
    "duracao_min": {"unidade": "min", "definicao": "Duração útil da fase em minutos, calculada a partir dos intervalos temporais válidos.", "calculo": "Soma dos dt válidos convertida para minutos.", "interpretacao": "Representa o tempo efetivo de exposição analisado na fase."},
    "dist_m": {"unidade": "m", "definicao": "Distância total percorrida pelo atleta na fase.", "calculo": "Soma dos deslocamentos ponto a ponto em X_UTM/Y_UTM.", "interpretacao": "Mede o volume locomotor total da fase."},
    "m_min": {"unidade": "m/min", "definicao": "Distância relativa por minuto.", "calculo": "dist_m dividido por duracao_min.", "interpretacao": "Representa a intensidade média locomotora da fase."},
    "vmax_mps": {"unidade": "m/s", "definicao": "Velocidade máxima instantânea estimada na fase.", "calculo": "Máximo de distância por intervalo de tempo entre amostras válidas.", "interpretacao": "Representa o pico de velocidade do atleta na fase."},
    "peak_1m_m_min": {"unidade": "m", "definicao": "Maior distância percorrida em qualquer janela contínua de 60 segundos dentro da fase.", "calculo": "Maior distância acumulada em qualquer janela móvel de 60 s.", "interpretacao": "Representa o pico locomotor da fase; não é volume acumulado nem média."},
    "hsr_dist_m": {"unidade": "m", "definicao": "Distância percorrida acima do limiar de high-speed running.", "calculo": f"Soma da distância quando v >= {HSR_MPS:.1f} m/s.", "interpretacao": "Quantifica a exposição a corrida de alta velocidade."},
    "hsr_pct": {"unidade": "%", "definicao": "Percentagem da distância total realizada em HSR.", "calculo": "hsr_dist_m dividido por dist_m, multiplicado por 100.", "interpretacao": "Representa o peso relativo da alta velocidade no volume total."},
    "sprint_dist_m": {"unidade": "m", "definicao": "Distância percorrida acima do limiar de sprint.", "calculo": f"Soma da distância quando v >= {SPRINT_MPS:.1f} m/s.", "interpretacao": "Quantifica a exposição a corrida de sprint."},
    "n_sprints": {"unidade": "contagem", "definicao": "Número de episódios de sprint.", "calculo": f"Conta bouts consecutivos com v >= {SPRINT_MPS:.1f} m/s e duração mínima de {SPRINT_BOUT_MIN_S:.1f} s.", "interpretacao": "Conta sprints válidos e evita picos isolados como sprint real."},
    "n_acc_2_5": {"unidade": "contagem", "definicao": "Número de instantes com aceleração acima do threshold operacional.", "calculo": f"Conta amostras com aceleração >= {ACC_THR:.1f} m/s².", "interpretacao": "Reflete a exigência de ações de aceleração."},
    "n_dec_3_0": {"unidade": "contagem", "definicao": "Número de instantes com desaceleração abaixo do threshold operacional.", "calculo": f"Conta amostras com desaceleração <= {DEC_THR:.1f} m/s².", "interpretacao": "Reflete a exigência de ações de desaceleração e controlo neuromuscular."},
    "active_time_min": {"unidade": "min", "definicao": "Tempo ativo em movimento durante a fase.", "calculo": "Soma do tempo em que a velocidade estimada é >= 0.5 m/s.", "interpretacao": "Distingue exposição total de tempo efetivamente ativo."},
    "active_pct": {"unidade": "%", "definicao": "Percentagem do tempo da fase em atividade motora.", "calculo": "active_time_min dividido pela duração da fase, multiplicado por 100.", "interpretacao": "Permite comparar fases com diferente tempo de inatividade."},
    "n_points": {"unidade": "contagem", "definicao": "Número de pontos válidos usados no cálculo das métricas.", "calculo": "Conta linhas com Time, X_UTM e Y_UTM válidos.", "interpretacao": "Quanto maior, mais robusta tende a ser a estimativa."},
    "pct_time_valid": {"unidade": "%", "definicao": "Percentagem de amostras válidas na fase.", "calculo": "Proporção de linhas com tempo e coordenadas válidos, multiplicada por 100.", "interpretacao": "Resume a completude do sinal disponível para cálculo."},
    "n_gaps_gt2s": {"unidade": "contagem", "definicao": "Número de gaps temporais superiores a 2 segundos.", "calculo": "Conta intervalos dt > 2.0 s entre amostras válidas.", "interpretacao": "Sinaliza perdas relevantes de continuidade temporal."},
    "qc_grade": {"unidade": "categórica", "definicao": "Classificação global da qualidade do sinal da fase.", "calculo": "Resultado das regras de QC: PASS, WARN, FAIL ou NA.", "interpretacao": "Apoia a decisão de aceitar, rever ou excluir a fase."},
    "qc_flags": {"unidade": "texto", "definicao": "Lista de flags de qualidade atribuídas à fase.", "calculo": "Concatenação dos alertas ativados pelo motor de QC.", "interpretacao": "Explica por que razão a fase recebeu o qc_grade observado."},
    "vmax_mps_qc": {"unidade": "m/s", "definicao": "Velocidade máxima observada para verificação de plausibilidade.", "calculo": "Máximo de velocidade no módulo de QC.", "interpretacao": "Ajuda a identificar picos implausíveis de velocidade."},
    "n_jumps_gt15m": {"unidade": "contagem", "definicao": "Número de saltos espaciais abruptos detetados entre amostras.", "calculo": "Conta deslocamentos excessivos consecutivos segundo o threshold interno de QC.", "interpretacao": "Ajuda a detetar teleports, ruído ou erro de posicionamento."},
    "n_gaps_gt2s_qc": {"unidade": "contagem", "definicao": "Número de gaps >2 s usado pelo módulo de QC.", "calculo": "Conta intervalos temporais superiores a 2.0 s para classificação QC.", "interpretacao": "Complementa a leitura da continuidade temporal no contexto de confiabilidade."},
}


def _build_manual_metricas_txt():
    lines = []
    lines.append("FPF Performance Hub — Manual de Métricas GPS")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Enquadramento")
    lines.append("- Dados GPS processados por atleta e por fase: Warm-Up, 1P, 2P e Total.")
    lines.append("- Organização das métricas em três blocos: Performance, Disponibilidade / Integridade e QC / Confiabilidade.")
    lines.append(
        f"- Thresholds operacionais atuais: HSR >= {HSR_MPS:.1f} m/s | Sprint >= {SPRINT_MPS:.1f} m/s | "
        f"Acc >= {ACC_THR:.1f} m/s² | Dec <= {DEC_THR:.1f} m/s² | Sprint bout mínimo >= {SPRINT_BOUT_MIN_S:.1f} s."
    )
    lines.append("")

    groups = _build_metric_groups()
    for categoria, familias in groups.items():
        lines.append(categoria)
        lines.append("-" * len(categoria))
        for familia, cols in familias.items():
            lines.append(f"{familia}")
            for col in cols:
                info = METRIC_INFO[col]
                lines.append(f"  • {col}")
                lines.append(f"    Unidade: {info['unidade']}")
                lines.append(f"    Definição: {info['definicao']}")
                lines.append(f"    Cálculo: {info['calculo']}")
                lines.append(f"    Interpretação: {info['interpretacao']}")
            lines.append("")

    lines.append("Notas metodológicas")
    lines.append("- Total resulta da agregação das fases Warm-Up, 1P e 2P no motor atual.")
    lines.append("- Métricas de Performance devem ser interpretadas em conjunto com Disponibilidade / Integridade e QC / Confiabilidade.")
    lines.append("- Flags ou grades QC desfavoráveis podem justificar revisão manual ou exclusão analítica da fase.")
    return "\n".join(lines)


def _normalize_xy_canonical(df: pd.DataFrame, dist_x: float, dist_y: float):
    """Rebase X/Y para iniciar em 0 e clip para [0,dist_x]/[0,dist_y]."""
    if df is None or df.empty:
        return df
    if "X_UTM" not in df.columns or "Y_UTM" not in df.columns:
        return df

    out = df.copy()

    x = pd.to_numeric(out["X_UTM"], errors="coerce")
    y = pd.to_numeric(out["Y_UTM"], errors="coerce")

    if x.notna().any():
        x_min = float(x.min())
        out["X_UTM"] = x - x_min

    if y.notna().any():
        y_min = float(y.min())
        out["Y_UTM"] = y - y_min

    try:
        out["X_UTM"] = pd.to_numeric(out["X_UTM"], errors="coerce").clip(
            lower=0.0, upper=float(dist_x)
        )
        out["Y_UTM"] = pd.to_numeric(out["Y_UTM"], errors="coerce").clip(
            lower=0.0, upper=float(dist_y)
        )
    except Exception:
        pass

    return out


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def _file_to_bytes(pathlike) -> bytes:
    with open(pathlike, "rb") as f:
        return f.read()

# -------------------------------
# Main flow
# -------------------------------
if not f_atleta:
    st.warning("⚠️ Ainda não carregaste ficheiros de atletas. Algumas funcionalidades podem não estar disponíveis.")

have_upload_corners = bool(f_campo)
have_picked_corners = st.session_state.get("pts_gps_picked") is not None

if metodo_campo == "Upload (BL/BR/TL/TR)" and not have_upload_corners:
    st.info("👋 Selecionaste 'Upload', mas ainda não carregaste os 4 CSVs do campo (BL/BR/TL/TR).")
    st.stop()

if metodo_campo == "Pick no mapa (clicar 4 cantos)" and not have_picked_corners:
    st.warning(
        "ℹ️ Selecionaste 'Pick no mapa'. Define os 4 cantos no mapa abaixo e depois continua.")

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

    if f_atleta:
        alat0, alon0 = get_atletas_centroid_latlon(
            f_atleta, amostra_n=amostra_geo_n)
    else:
        alat0, alon0 = None, None

    if alat0 is None or alon0 is None:
        pts_fallback = sample_athlete_track_latlon(f_atleta, max_points=10) if f_atleta else []
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

    out_pick = st_folium(m_pick, width=1100, height=520,
                         key="mapa_pick_cantos")

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
            st.success(
                "✅ Cantos ajustados guardados. Agora o pipeline continua normalmente.")
        else:
            st.warning(
                "Tens 4 pontos, mas não consegui ajustar. Faz Reset e tenta com mais zoom.")

    st.stop()

try:
    if metodo_campo == "Pick no mapa (clicar 4 cantos)":
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo_from_pts_gps(
            st.session_state.pts_gps_picked, int(epsg_used)
        )

    elif metodo_campo == "Escolher um campo guardado anteriormente":
        df_campos = read_field_from_parquet()

        if df_campos is None or df_campos.empty:
            st.warning("Ainda não existem campos guardados.")
            st.stop()

        df_campos = df_campos.copy()
        df_campos["display_name"] = (
            df_campos["estadio"].astype(str) + " (" + df_campos["campo_local"].astype(str) + ")"
        )

        campo_selecionado = st.selectbox(
            "Seleciona o Estádio/Campo",
            options=df_campos["display_name"].tolist()
        )

        row = df_campos[df_campos["display_name"] == campo_selecionado].iloc[0]

        pts_gps_recuperado = {
            "BL": [float(row["BL_lat"]), float(row["BL_lon"])],
            "BR": [float(row["BR_lat"]), float(row["BR_lon"])],
            "TL": [float(row["TL_lat"]), float(row["TL_lon"])],
            "TR": [float(row["TR_lat"]), float(row["TR_lon"])],
        }

        # guardar cantos limpos em sessão
        st.session_state.pts_gps_picked = pts_gps_recuperado

        # recalibrar -> define origin, R, pts_utm, dist_x, dist_y, angulo_rad, clat, clon
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo_from_pts_gps(
            pts_gps_recuperado, int(epsg_used)
        )

        st.success(f"✅ Campo '{row['estadio']}' carregado e calibrado com sucesso!")

    else:
        pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo(
            f_campo, int(epsg_used)
        )

    cidade, pais = _reverse_geocode_city_country(clat, clon)
    estadio = None

except Exception as e:
    st.error(f"❌ Erro na calibração do campo: {e}")
    st.stop()
except Exception as e:

    st.error(f"❌ Erro na calibração do campo: {e}")
    st.stop()

passed_geo, pct_ok, ok_list, fora_list, geo_errors = geo_validacao_por_atleta(
    f_atleta, clat, clon, float(raio_validacao_m), int(
        amostra_geo_n), float(min_pct_atletas_ok)
)

st.header("Validação de Localização (Campo ↔ Atletas)")

# Campo (usa helper local cacheado, que era o comportamento funcional anterior)
cidade_campo, pais_campo = _reverse_geocode_city_country(clat, clon)

# Atletas (centro estimado)
alat, alon = get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)

cidade_atl, pais_atl = None, None
if alat is not None and alon is not None:
    cidade_atl, pais_atl = _reverse_geocode_city_country(alat, alon)
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
    st.error(
        "❌ Validação geográfica falhou (percentagem insuficiente dentro do raio).")


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
    key=lambda x: int(re.search(r"\d+", x).group()
                      ) if re.search(r"\d+", x) else 0,
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
st.write(
    f"**Atletas completos (Warm-Up + 1P + 2P):** {completos} / {len(audit_data)}")

st.divider()

# Normalization + export
st.header("Normalização | Calculo Métricas")

if not passed_geo:
    st.warning(
        "A exportação está desativada porque a validação geográfica falhou. Ajusta o raio/% mínimo ou verifica os ficheiros."
    )
    st.stop()

# -- Obter e guardar campo -- #

# Dados do campo
field_data = {
    "estadio": '',
    "campo_local": [campo_local],
    "cidade": [cidade_campo],
    "pais": [pais_campo],
    "clat": [float(clat)],
    "clon": [float(clon)],
    "dist_x": [float(dist_x)],
    "dist_y": [float(dist_y)],
    "rotation": [float(angulo_rad)],
    "epsg": [int(epsg_used)],
    # Coordenadas dos cantos para o preview do mapa
    'pts_gps': [pts_gps],
    "BL_lat": [float(pts_gps['BL'][0])], "BL_lon": [float(pts_gps['BL'][1])],
    "BR_lat": [float(pts_gps['BR'][0])], "BR_lon": [float(pts_gps['BR'][1])],
    "TL_lat": [float(pts_gps['TL'][0])], "TL_lon": [float(pts_gps['TL'][1])],
    "TR_lat": [float(pts_gps['TR'][0])], "TR_lon": [float(pts_gps['TR'][1])],
    'obs': ''
}


# Pop Up para Guardar campo na base de dados
with st.sidebar.popover('💾 Guardar Campo'):

    estadio = st.text_input('Adiciona o nome do estadio!')
    obs = st.text_input('Adiciona uma observação!')

    field_data['estadio'] = estadio
    field_data['obs'] = obs
    if st.button("Guardar"):
        campo_df = pd.DataFrame(field_data)
        save_field_to_parquet(campo_df)
        st.success("Campo Guardado!")

btn = st.button("⚙️ Processar e Gerar Relatório",
                type="primary", use_container_width=True)

# outputs (para UI) — manter em session_state para sobreviver a reruns
df_metrics = st.session_state.df_metrics
report_txt = st.session_state.report_txt

if btn:
    with st.status("A iniciar processamento...", expanded=True) as status:
        if not f_atleta:
            status.update(label="Faltam ficheiros de atletas. Processamento interrompido.", state="error")
            st.error("Carrega os ficheiros de atletas antes de processar.")
            st.stop()
        if not passed_geo:
            status.update(
                label="Validação geográfica falhou. Processamento interrompido.", state="error")
            st.error(
                "Validação geográfica falhou. O processamento foi interrompido.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_dir = td_path / "Temp_Processing"
            out_dir = td_path / "Output_UTM_Sincronizado"
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            status.update(
                label="Processamento e limpeza de dados GPS...", state="running")
            temp_files, audit_proc, issues = processar_atletas_para_temp(
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
                st.error(
                    "❌ Não foi possível gerar ficheiros temporários (verifica colunas Time/Lat/Lon e nomes).")
                st.stop()

            status.update(label="Sincronização temporal...", state="running")
            out_files, fases_ordenadas, fases_dict, n_master, event_clock = sincronizar(
                temp_files, out_dir)

            # Session identifiers (auditoria/dedup)
            session_uuid = uuid.uuid4()
            session_id_hex = session_uuid.hex
            session_fingerprint = hash_session(
                data_sessao, selecao, genero, contexto, estadio, f_campo, f_atleta
            )

            # Métricas individuais a partir dos SYNC (por fase + Total)
            status.update(
                label="Cálculo de métricas individuais...", state="running")
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
                        total_micro_gaps += int(
                            df_sync["_micro_gaps_corrigidos"].dropna().iloc[0])
                    except Exception:
                        pass

                fase_mets = {}

                for fase in fases_target:
                    df_f = df_sync[df_sync[COL_FASE] == fase].copy()

                    # Em ficheiros sincronizados, a fase pode existir na grelha temporal
                    # mesmo sem participação real do atleta. Avaliar presença por XY válidos.
                    has_xy = (
                        ("X_UTM" in df_f.columns) and ("Y_UTM" in df_f.columns) and
                        (pd.to_numeric(df_f["X_UTM"], errors="coerce").notna() &
                         pd.to_numeric(df_f["Y_UTM"], errors="coerce").notna()).any()
                    )
                    phase_present = bool(has_xy)

                    if not phase_present:
                        # Fase não jogada / não submetida → NA (não é falha de qualidade)
                        met = compute_metrics_for_df(df_f)
                        qc = qc_gps_df(df_f, phase_present=False)
                    else:
                        # Normalização canónica (rebase + clip) para comparabilidade entre campos
                        df_f = _normalize_xy_canonical(df_f, dist_x=float(dist_x), dist_y=float(dist_y))
                        met = compute_metrics_for_df(df_f)
                        qc = qc_gps_df(df_f, phase_present=True)

                    fase_mets[fase] = met

                    aud = audit_timebase(df_f, COL_TIME, expected_hz=10.0)
                    audit_time_rows.append(
                        {"atleta_id": aid, "fase": fase, **aud})

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
                            "qc_grade": qc.get("qc_grade"),
                            "qc_flags": qc.get("qc_flags"),
                            "vmax_mps_qc": qc.get("vmax_mps_qc"),
                            "n_jumps_gt15m": qc.get("n_jumps_gt15m"),
                            "n_gaps_gt2s_qc": qc.get("n_gaps_gt2s"),
                            "engine_version": ENGINE_VERSION,
                        }
                    )

                # -------- TOTAL POR AGREGAÇÃO DAS FASES --------
                # Regras:
                # - Soma: volume e eventos
                # - Recalcular: métricas relativas / percentuais
                # - Máximo: picos de fase
                met_total = {}
                met_total["dist_m"] = sum(fase_mets.get(f, {}).get(
                    "dist_m", 0.0) for f in fases_target)
                met_total["duracao_min"] = sum(fase_mets.get(f, {}).get(
                    "duracao_min", 0.0) for f in fases_target)
                met_total["m_min"] = (
                    met_total["dist_m"] / met_total["duracao_min"]
                    if met_total["duracao_min"] > 0
                    else np.nan
                )

                met_total["hsr_dist_m"] = sum(fase_mets.get(f, {}).get(
                    "hsr_dist_m", 0.0) for f in fases_target)
                met_total["sprint_dist_m"] = sum(fase_mets.get(f, {}).get(
                    "sprint_dist_m", 0.0) for f in fases_target)
                met_total["n_sprints"] = sum(fase_mets.get(
                    f, {}).get("n_sprints", 0) for f in fases_target)
                met_total["n_acc_2_5"] = sum(fase_mets.get(
                    f, {}).get("n_acc_2_5", 0) for f in fases_target)
                met_total["n_dec_3_0"] = sum(fase_mets.get(
                    f, {}).get("n_dec_3_0", 0) for f in fases_target)
                met_total["n_points"] = sum(fase_mets.get(
                    f, {}).get("n_points", 0) for f in fases_target)

                met_total["vmax_mps"] = max((fase_mets.get(f, {}).get(
                    "vmax_mps", np.nan) for f in fases_target), default=np.nan)
                met_total["peak_1m_m_min"] = max((fase_mets.get(f, {}).get(
                    "peak_1m_m_min", np.nan) for f in fases_target), default=np.nan)

                met_total["hsr_pct"] = (
                    met_total["hsr_dist_m"] / met_total["dist_m"] * 100.0
                    if met_total["dist_m"] > 0
                    else np.nan
                )

                # Active time total
                met_total["active_time_min"] = sum(fase_mets.get(f, {}).get(
                    "active_time_min", 0.0) for f in fases_target)
                dur_total_s = met_total["duracao_min"] * 60.0
                met_total["active_pct"] = (
                    (met_total["active_time_min"] * 60.0) / dur_total_s * 100.0
                    if dur_total_s > 0
                    else np.nan
                )

                # Qualidade: pct_time_valid e gaps>2s — média ponderada simples por pontos válidos
                try:
                    w = np.array([max(1, fase_mets[f]["n_points"])
                                 for f in fases_target], dtype=float)
                    met_total["pct_time_valid"] = float(
                        np.average([fase_mets[f]["pct_time_valid"]
                                   for f in fases_target], weights=w)
                    )
                    met_total["n_gaps_gt2s"] = int(
                        sum(fase_mets[f]["n_gaps_gt2s"] for f in fases_target))
                except Exception:
                    met_total["pct_time_valid"] = np.nan
                    met_total["n_gaps_gt2s"] = int(
                        sum(fase_mets[f]["n_gaps_gt2s"] for f in fases_target))

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
                        "qc_grade": None,
                        "qc_flags": None,
                        "vmax_mps_qc": None,
                        "n_jumps_gt15m": None,
                        "n_gaps_gt2s_qc": None,
                        "engine_version": ENGINE_VERSION,
                    }
                )

            df_metrics = pd.DataFrame(metrics_rows)
            df_metrics = round_metrics_dataframe(df_metrics)

            # -------------------------------
            # Persistência parquet analítica
            # -------------------------------
            session_payload = {
                "session_id_hex": session_id_hex,
                "session_fingerprint": session_fingerprint,
                "data": data_sessao,
                "selecao": selecao,
                "genero": genero,
                "contexto": contexto,
                "jogo": adversario.strip() if contexto == "Jogo" else "",
                "estadio": estadio,
                "cidade": cidade,
                "pais": pais,
                "epsg": int(epsg_used),
                "dist_x": float(dist_x),
                "dist_y": float(dist_y),
                "rotation_rad": float(angulo_rad),
                "engine_version": ENGINE_VERSION,
            }
            session_sk = resolve_session_sk(session_fingerprint, session_payload, CLEANDATA_DIR)
            athlete_map = resolve_athlete_sk(df_metrics, CLEANDATA_DIR, genero=genero)

            df_metrics["session_sk"] = session_sk
            df_metrics["athlete_sk"] = df_metrics["atleta_id"].astype(str).map(athlete_map)
            df_metrics["phase_id"] = df_metrics["fase"].map(PHASE_MAP)

            perf_cols = [
                "duracao_min", "dist_m", "m_min",
                "vmax_mps", "peak_1m_m_min",
                "hsr_dist_m", "hsr_pct", "sprint_dist_m", "n_sprints",
                "n_acc_2_5", "n_dec_3_0",
                "active_time_min", "active_pct",
            ]
            qc_cols = [
                "n_points", "pct_time_valid", "n_gaps_gt2s",
                "qc_grade", "qc_flags", "vmax_mps_qc", "n_jumps_gt15m", "n_gaps_gt2s_qc",
            ]
            base_cols = ["session_sk", "athlete_sk", "atleta_id", "phase_id", "fase", "data", "selecao", "genero", "contexto", "jogo"]

            df_perf = df_metrics[base_cols + perf_cols].copy()
            df_qc = df_metrics[["session_sk", "athlete_sk", "atleta_id", "phase_id", "fase"] + qc_cols].copy()
            df_samples, df_athlete_session = _build_samples_export(out_files, session_sk, athlete_map)

            # Normalizar coordenadas StatsBomb
            df_tracking = normalize_tracking_data(
                df_samples,
                dist_x=field_data['dist_x'][0],
                dist_y=field_data['dist_y'][0]
            )
            append_dedup_parquet(df_perf, str(Path(CLEANDATA_DIR) / "performance_metrics.parquet"), ["session_sk", "athlete_sk", "phase_id"])
            append_dedup_parquet(df_qc, str(Path(CLEANDATA_DIR) / "quality_metrics.parquet"), ["session_sk", "athlete_sk", "phase_id"])
            append_dedup_parquet(df_samples, str(Path(CLEANDATA_DIR) / "samples.parquet"), ["session_sk", "athlete_sk", "phase_id", "time"])
            append_dedup_parquet(df_athlete_session, str(Path(CLEANDATA_DIR) / "athlete_session.parquet"), ["session_sk", "athlete_sk"])
            append_dedup_parquet(df_tracking, str(Path(CLEANDATA_DIR) / "tracking.parquet"), ["session_sk", "athlete_sk", "phase_id", "time"])

            st.session_state.df_perf = df_perf
            st.session_state.df_qc = df_qc
            st.session_state.df_samples = df_samples
            st.session_state.df_athlete_session = df_athlete_session

            df_time_audit = pd.DataFrame(audit_time_rows)
            st.session_state.df_time_audit = df_time_audit
            st.session_state.manual_metricas_txt = _build_manual_metricas_txt()

            status.update(label="Construção do relatório...", state="running")
            # Build report (rotação mantida)
            rot_deg = float(np.degrees(angulo_rad))
            report_lines = []
            report_lines.append(
                "FPF Performance Hub — Relatório de Validação e Normalização")
            report_lines.append("=" * 70)

            report_lines.append("Dados da Sessão")
            report_lines.append(
                f"Data: {data_sessao.strftime('%d/%m/%Y') if hasattr(data_sessao, 'strftime') else data_sessao}"
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
                f"  Raio: {raio_validacao_m:.0f} m | Amostra: { amostra_geo_n} linhas/atleta | % OK: {pct_ok*100:.0f}% "
                f"(mínimo {min_pct_atletas_ok*100:.0f}%)"
            )
            report_lines.append(
                f"  Dentro do raio: {len({a for a, _ in ok_list})} atletas | "
                f"Fora: {len({a for a, _ in fora_list})} atletas | Erros: {len(geo_errors)}"
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
                report_lines.append(
                    f"    - {fase:8} | início: {t_s} | fim: {t_e}")

            report_lines.append("-" * 70)
            report_lines.append("Timeline do Jogo")
            if event_clock:
                if "1P" in event_clock:
                    ec = event_clock["1P"]
                    report_lines.append(
                        f"  1P:    {_fmt_match_clock(ec['start_s'])} → {_fmt_match_clock(ec['reg_end_s'])}"
                    )
                    if ec.get("extra_s", 0.0) > 0:
                        report_lines.append(
                            f"  ET_1P: {_fmt_match_clock(ec['reg_end_s'])} → {_fmt_match_clock(ec['end_s'])}"
                        )

                if "2P" in event_clock:
                    ec = event_clock["2P"]
                    report_lines.append(
                        f"  2P:    {_fmt_match_clock(ec['start_s'])} → {_fmt_match_clock(ec['reg_end_s'])}"
                    )
                    if ec.get("extra_s", 0.0) > 0:
                        report_lines.append(
                            f"  ET_2P: {_fmt_match_clock(ec['reg_end_s'])} → {_fmt_match_clock(ec['end_s'])}"
                        )
            else:
                report_lines.append("  Sem event_clock disponível.")

            if issues:
                report_lines.append("-" * 70)
                report_lines.append("Avisos/Problemas (exemplos):")
                for aid, fn, msg in issues[:25]:
                    report_lines.append(f"  - {aid} | {fn} | {msg}")

            report_lines.append("-" * 70)
            report_lines.append(
                "Métricas Individuais (GPS-only) — thresholds fixos")
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
                    hz_med = float(dfta["hz_est"].dropna().median(
                    )) if dfta["hz_est"].dropna().any() else np.nan
                    n_dup = int((dfta["n_dt_zero"] > 0).sum()
                                ) if "n_dt_zero" in dfta.columns else 0
                    n_g2 = int((dfta["n_gaps_gt_2s"] > 0).sum()
                               ) if "n_gaps_gt_2s" in dfta.columns else 0

                    report_lines.append("-" * 70)
                    report_lines.append("Auditoria de Timestamp")
                    if np.isfinite(hz_med):
                        report_lines.append(
                            f"  Hz mediano estimado (por atleta/fase): {hz_med:.1f} Hz")
                    else:
                        report_lines.append(
                            "  Hz mediano estimado (por atleta/fase): —")
                    report_lines.append(
                        f"  Atleta×fase com timestamps duplicados: {n_dup}")
                    report_lines.append(f"  Atleta×fase com gaps >2s: {n_g2}")
            except Exception:
                pass

            report_txt = "\n".join(report_lines)

            # Persistir outputs (map zoom/scroll dispara rerun do Streamlit)
            st.session_state.df_metrics = df_metrics
            st.session_state.report_txt = report_txt
            st.session_state.process_done = True
            status.update(label="Finalizado.", state="complete")

    st.success(
        "✅ Processamento concluído. Relatório e métricas disponíveis abaixo.")

    try:
        if out_files:
            sample_sync = pd.read_csv(out_files[0], sep=";")
            cols_preview = [
                c for c in ["Time", "Fase", "Periodo_Jogo", "Time_Evento", "Minuto_Jogo"]
                if c in sample_sync.columns
            ]
            if cols_preview:
                st.subheader("Preview timeline sincronizada")
                st.dataframe(
                    sample_sync[cols_preview].head(30),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception:
        pass


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

        df_display["__fase_ord"] = df_display["fase"].map(
            ordem_fases).fillna(99)
        df_display = df_display.sort_values(
            by=[col_inicio, "__fase_ord"]
        ).drop(columns="__fase_ord")

    else:
        df_display = df_display.sort_values(by=[col_inicio])

    # 5️⃣ Organização vertical por blocos e famílias
    id_cols = [c for c in [col_inicio, "fase"] if c in df_display.columns]
    metric_groups = _build_metric_groups()

    ordered_metric_cols = []
    for familias in metric_groups.values():
        for cols in familias.values():
            ordered_metric_cols.extend([c for c in cols if c in df_display.columns])

    df_export = df_display[id_cols + ordered_metric_cols].copy()

    st.subheader("Métricas organizadas por contexto")
    for categoria, familias in metric_groups.items():
        st.markdown(f"### {categoria}")
        for familia, cols in familias.items():
            cols_presentes = [c for c in cols if c in df_display.columns]
            if not cols_presentes:
                continue
            st.markdown(f"**{familia}**")
            st.dataframe(
                df_display[id_cols + cols_presentes],
                use_container_width=True,
                hide_index=True,
            )

    # 6️⃣ Downloads
    st.download_button(
        "⬇️ Download Métricas (.csv)",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name="metricas_individuais_FPF.csv",
        mime="text/csv",
        use_container_width=True,
    )


    st.subheader("Integração na Base de Dados")

    if st.session_state.get("df_perf") is not None and not st.session_state.df_perf.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✅ Dados processados e prontos para serem integrados no Supabase.")
        
        with col2:
            if st.button("💾 Gravar na Base", key="btn_save_duckdb", use_container_width=True):
                try:
                    stats = write_session_data(
                        st.session_state.df_perf,
                        st.session_state.df_qc,
                        st.session_state.df_samples,
                        st.session_state.df_athlete_session,
                    )
                    
                    # Build stats message
                    stats_msg = "📊 **Resumo da Integração:**\n\n"
                    for table, table_stats in stats.items():
                        if table_stats is not None:
                            inserted = table_stats.get('inserted', 0)
                            updated = table_stats.get('updated', 0)
                            stats_msg += f"• **{table}**: {inserted} inseridos, {updated} atualizados\n"
                    
                    st.success("✅ Dados gravados com sucesso na base DuckDB!")
                    st.markdown(stats_msg)
                except Exception as e:
                    st.error(f"❌ Erro ao gravar: {str(e)}")
    else:
        st.warning("📊 Processa a sessão primeiro para gravar os dados.")

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
        st.session_state.manual_metricas_txt = None
        st.session_state.df_perf = None
        st.session_state.df_qc = None
        st.session_state.df_samples = None
        st.session_state.df_athlete_session = None
        st.session_state.process_done = False
        st.rerun()
