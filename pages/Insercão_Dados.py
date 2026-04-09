# -*- coding: utf-8 -*-
"""
FPF UTM Engine v16 (parquet downloads)
Autor: Marcos (base) + ajustes de estabilidade/indentação
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyproj import Geod, Transformer
from pathlib import Path
import tempfile
import hashlib
import uuid
import requests
import re
import io
import textwrap

# testing
from streamlit_folium import st_folium
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages

# -- Local Modules -- #
from fpf_modules.constants import (
    HSR_MPS, SPRINT_MPS, ACC_THR, DEC_THR, SPRINT_BOUT_MIN_S, ENGINE_VERSION,
    COL_LAT, COL_LON, COL_TIME, COL_FASE,
    CLEANDATA_DIR
)
from fpf_modules.draft_manager import (
    clear_draft_session_state,
    ensure_draft_session_state,
    save_draft_outputs,
    store_draft_results,
)
from fpf_modules.reference_data import (
    load_active_athletes_by_selection,
    load_field_reference,
    load_selection_reference,
)

from fpf_modules.metrics import (
    time_to_seconds,
    count_bouts,
    audit_timebase,
    compute_metrics_for_df,
    derive_load_metrics,
)

from fpf_modules.qc import qc_gps_df

from fpf_modules.io_utils import (
    hash_session,
    clean_cols,
    get_atleta_id,
    infer_fase,
    read_csv_upload
)

from fpf_modules.utils import round_metrics_dataframe, file_to_bytes, format_metrics_display_dataframe

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
    initialize_schema,
    insert_or_update_table,
    read_table,
    save_field_reference,
    save_session_report,
    resolve_athlete_sk,
    resolve_session_sk,
    write_session_data,
)
from fpf_modules.auth_manager import authenticate_user, get_secrets_auth

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

ATHLETE_PROFILE_COLUMN_MAP = {
    "atleta_id": ["atleta_id", "id_atleta", "player_id", "player", "athlete_id", "id"],
    "nome": ["nome", "name", "jogador", "player_name"],
    "data_nascimento": ["data_nascimento", "dt_nascimento", "birth_date", "data_nasc"],
    "posicao": ["posicao", "posição", "position", "pos"],
    "pe_preferencial": ["pe_preferencial", "pé_preferencial", "preferred_foot", "foot", "pe"],
    "altura_cm": ["altura_cm", "height_cm", "altura"],
    "peso_kg": ["peso_kg", "weight_kg", "peso"],
    "escalao": ["escalao", "escalao_etario", "age_group"],
    "selecao": ["selecao", "seleção", "team", "equipa"],
}
ATHLETE_POSITIONS = ["", "GR", "DD", "DE", "DC", "MD", "ME", "MC", "MDC", "MAC", "ED", "EE", "AV", "PL"]
ATHLETE_FEET = ["", "Direito", "Esquerdo", "Ambidestro"]
ATHLETE_ESCALOES = ["", "A", "Sub-23", "Sub-21", "Sub-20", "Sub-19", "Sub-18", "Sub-17", "Sub-16", "Sub-15"]
SELECTION_OPTIONS = load_selection_reference()
GENDER_OPTIONS = ["Masculino", "Feminino"]
SESSION_REPORTS_DIR = Path(CLEANDATA_DIR) / "session_reports_pdf"


def _clean_text_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _normalize_athlete_identifier(value) -> str:
    text = _clean_text_value(value)
    if not text:
        return ""
    return re.sub(r"^0+(?=\d+$)", "", text)


def _genero_label_to_code(value: str) -> str:
    mapping = {
        "Masculino": "M",
        "Feminino": "F",
        "M": "M",
        "F": "F",
    }
    return mapping.get(_clean_text_value(value), "")


def _generate_internal_atleta_id(df: pd.DataFrame) -> str:
    if df is None or df.empty or "atleta_id" not in df.columns:
        return "ATH-00001"

    existing_ids = df["atleta_id"].astype(str).str.strip()
    numeric_suffixes = (
        existing_ids.str.extract(r"ATH-(\d+)", expand=False)
        .dropna()
        .astype(int)
    )
    next_number = 1 if numeric_suffixes.empty else int(numeric_suffixes.max()) + 1
    return f"ATH-{next_number:05d}"


def _create_inline_athlete(
    nome: str,
    sobrenome: str,
    data_nascimento,
    genero_label: str,
    posicao: str,
    selecao_value: str,
):
    initialize_schema()
    athletes_df = read_table("athletes")
    if athletes_df is None or athletes_df.empty:
        athletes_df = pd.DataFrame(columns=["atleta_id"])

    nome = _clean_text_value(nome)
    sobrenome = _clean_text_value(sobrenome)
    genero = _genero_label_to_code(genero_label)
    atleta_id = _generate_internal_atleta_id(athletes_df)

    if not nome:
        raise RuntimeError("O campo Nome e obrigatorio.")
    if not sobrenome:
        raise RuntimeError("O campo Sobrenome e obrigatorio.")
    if not genero:
        raise RuntimeError("O campo Genero e obrigatorio.")

    available_columns = set(athletes_df.columns.astype(str).tolist()) if not athletes_df.empty else set()
    payload = {
        "atleta_id": atleta_id,
        "nome": f"{nome} {sobrenome}".strip(),
        "data_nascimento": data_nascimento,
        "posicao": _clean_text_value(posicao),
        "genero": genero,
        "ativo": True,
    }
    optional_payload = {
        "foto_url": "",
        "selecao": _clean_text_value(selecao_value),
        "hr_max_bpm": None,
        "hr_rest_bpm": None,
    }
    for column, value in optional_payload.items():
        if not available_columns or column in available_columns:
            payload[column] = value

    new_row = pd.DataFrame([payload])
    stats = insert_or_update_table("athletes", new_row, pk_columns=["atleta_id"])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao criar atleta na base de dados.")
    return atleta_id


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    values = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in values):
        return np.nan
    r = 6371000.0
    phi1 = np.radians(float(lat1))
    phi2 = np.radians(float(lat2))
    dphi = np.radians(float(lat2) - float(lat1))
    dlambda = np.radians(float(lon2) - float(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _find_existing_field_match(current_field_df: pd.DataFrame, fields_df: pd.DataFrame, athletes_center=None):
    if current_field_df is None or current_field_df.empty or fields_df is None or fields_df.empty:
        return None

    current = current_field_df.iloc[0]
    athlete_lat = None
    athlete_lon = None
    if athletes_center is not None:
        try:
            athlete_lat, athlete_lon = athletes_center
        except Exception:
            athlete_lat, athlete_lon = None, None
    candidates = []
    for _, row in fields_df.iterrows():
        center_m = _haversine_m(current.get("clat"), current.get("clon"), row.get("clat"), row.get("clon"))
        athletes_m = _haversine_m(athlete_lat, athlete_lon, row.get("clat"), row.get("clon"))
        corner_distances = []
        for prefix in ["BL", "BR", "TL", "TR"]:
            dist_corner = _haversine_m(
                current.get(f"{prefix}_lat"),
                current.get(f"{prefix}_lon"),
                row.get(f"{prefix}_lat"),
                row.get(f"{prefix}_lon"),
            )
            if pd.notna(dist_corner):
                corner_distances.append(dist_corner)

        mean_corner_m = float(np.mean(corner_distances)) if corner_distances else np.nan
        max_corner_m = float(np.max(corner_distances)) if corner_distances else np.nan

        same_fp = False
        if "field_fingerprint" in current.index and "field_fingerprint" in row.index:
            same_fp = str(current.get("field_fingerprint") or "") == str(row.get("field_fingerprint") or "")

        match_strength = None
        if same_fp:
            match_strength = "forte"
        elif pd.notna(athletes_m) and athletes_m <= 120.0:
            if pd.notna(center_m) and center_m <= 100.0:
                match_strength = "forte"
            else:
                match_strength = "provavel"
        elif pd.notna(center_m) and center_m <= 80.0:
            match_strength = "provavel"
        elif pd.notna(center_m) and center_m <= 140.0 and pd.notna(mean_corner_m) and mean_corner_m <= 45.0:
            match_strength = "provavel"

        if match_strength:
            score = (
                (0.0 if same_fp else 1000.0)
                + (0.0 if pd.isna(athletes_m) else float(athletes_m) * 3.0)
                + (0.0 if pd.isna(center_m) else float(center_m) * 2.0)
                + (0.0 if pd.isna(mean_corner_m) else float(mean_corner_m) * 0.5)
            )
            candidates.append(
                {
                    "row": row,
                    "same_fingerprint": same_fp,
                    "match_strength": match_strength,
                    "athletes_m": athletes_m,
                    "center_m": center_m,
                    "mean_corner_m": mean_corner_m,
                    "max_corner_m": max_corner_m,
                    "score": score,
                }
            )

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: item["score"])
    return candidates[0]


def _find_field_match_from_athletes(fields_df: pd.DataFrame, athlete_lat, athlete_lon):
    if fields_df is None or fields_df.empty or athlete_lat is None or athlete_lon is None:
        return None

    candidates = []
    for _, row in fields_df.iterrows():
        athletes_m = _haversine_m(athlete_lat, athlete_lon, row.get("clat"), row.get("clon"))
        if pd.isna(athletes_m):
            continue

        if athletes_m <= 80.0:
            strength = "forte"
        elif athletes_m <= 150.0:
            strength = "provavel"
        else:
            continue

        candidates.append(
            {
                "row": row,
                "athletes_m": float(athletes_m),
                "match_strength": strength,
                "score": float(athletes_m),
            }
        )

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: item["score"])
    return candidates[0]


def _find_potential_duplicate_sessions(
    *,
    data_sessao,
    selecao: str,
    genero: str,
    contexto: str,
    jogo: str,
) -> pd.DataFrame:
    filters = {
        "data": str(pd.to_datetime(data_sessao).date()) if pd.notna(data_sessao) else "",
        "selecao": str(selecao or "").strip(),
        "genero": str(genero or "").strip(),
        "contexto": str(contexto or "").strip(),
        "fase": "Total",
    }

    if str(contexto or "").strip() == "Jogo":
        filters["jogo"] = str(jogo or "").strip()

    df_existing = read_table(
        "performance_metrics",
        filters=filters,
        columns="session_sk,data,selecao,genero,contexto,jogo,fase,atleta_id",
    )
    if df_existing is None or df_existing.empty:
        return pd.DataFrame()

    keep_cols = [
        col for col in [
            "session_sk", "data", "selecao", "genero", "contexto", "jogo", "fase", "atleta_id"
        ] if col in df_existing.columns
    ]
    return df_existing[keep_cols].drop_duplicates().sort_values(
        [c for c in ["data", "session_sk", "atleta_id"] if c in keep_cols]
    )


def _detect_hr_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for col in HR_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


def _evaluate_athlete_submission(audit_data: dict, contexto: str):
    """Valida a submissão por atleta segundo o contexto da sessão.

    Regras:
    - Jogo: cada atleta tem obrigatoriamente 1P e 2P; Warm-Up é opcional
    - Treino: cada atleta tem exatamente 1 ficheiro
    """
    fases_norm = {
        str(aid): sorted({str(f) for f in (fases or [])}, key=lambda x: PHASE_MAP.get(x, 99))
        for aid, fases in (audit_data or {}).items()
    }

    athlete_status = {}
    valid_count = 0
    for aid, fases in fases_norm.items():
        fases_set = set(fases)
        n_files = len(fases)

        if contexto == "Jogo":
            has_required = {"1P", "2P"}.issubset(fases_set)
            has_only_allowed = fases_set.issubset({"Warm-Up", "1P", "2P"})
            is_valid = has_required and has_only_allowed
            criterio = "1P + 2P obrigatórios; Warm-Up opcional"
        else:
            is_valid = n_files == 1
            criterio = "1 ficheiro obrigatório"

        if is_valid:
            valid_count += 1
        athlete_status[aid] = {
            "fases": fases,
            "is_valid": is_valid,
            "criterio": criterio,
            "n_files": n_files,
        }

    total_atletas = len(fases_norm)
    session_valid = bool(total_atletas) and valid_count == total_atletas
    return athlete_status, valid_count, session_valid


def _normalize_athlete_registry_df(df: pd.DataFrame, genero_default: str, selecao_default: str):
    if df is None or df.empty:
        return None, None

    if "atleta_id_ficheiro" not in df.columns:
        df["atleta_id_ficheiro"] = df.get("atleta_id")
    if "atleta_id" not in df.columns:
        return None, "A ficha de atletas tem de incluir atleta_id."

    keep_cols = [
        "atleta_id_ficheiro", "atleta_id", "nome", "numero_camisola", "data_nascimento", "posicao", "pe_preferencial",
        "altura_cm", "peso_kg", "escalao", "selecao"
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[keep_cols].copy()
    df["atleta_id_ficheiro"] = (
        df["atleta_id_ficheiro"]
        .astype(str)
        .str.strip()
        .str.replace(r"^0+(?=\d+$)", "", regex=True)
    )
    df["atleta_id"] = df["atleta_id"].astype(str).str.strip()
    df = df[df["atleta_id_ficheiro"].ne("")].drop_duplicates(subset=["atleta_id_ficheiro"], keep="last")

    if df.empty:
        return None, "O cadastro de atletas não contém atleta_id válidos."

    df["genero"] = genero_default or pd.NA
    df["selecao"] = selecao_default or pd.NA
    df["numero_camisola"] = pd.to_numeric(df["numero_camisola"], errors="coerce")
    df["altura_cm"] = pd.to_numeric(df["altura_cm"], errors="coerce")
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce")
    df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce").dt.date
    return df, None


def _build_athlete_registry_editor(f_atleta_files, genero_default: str, selecao_default: str):
    athlete_ids = sorted({
        str(get_atleta_id(f.name)).strip()
        for f in (f_atleta_files or [])
        if str(get_atleta_id(f.name)).strip()
    })
    if not athlete_ids:
        return None, None

    state_key = "athlete_registry_editor_df"
    state_selection_key = "athlete_registry_editor_selection"
    selection_filter_key = "athlete_registry_selection_filter"
    existing = st.session_state.get(state_key)

    selection_options = ["Masculino", "Feminino"]
    genero_code = str(genero_default or "").strip().upper()
    default_selection = "Masculino" if genero_code == "M" else "Feminino" if genero_code == "F" else selection_options[0]
    current_selection = st.session_state.get(selection_filter_key, default_selection)
    if current_selection not in selection_options:
        current_selection = default_selection

    selected_selection = st.selectbox(
        "Genero para procurar atletas",
        options=selection_options,
        index=selection_options.index(current_selection),
        key=selection_filter_key,
    )

    db_athletes = load_active_athletes_by_selection(selected_selection)
    athlete_options = db_athletes["atleta_id"].astype(str).tolist() if not db_athletes.empty else []
    db_profiles = (
        db_athletes.set_index("atleta_id").to_dict(orient="index")
        if not db_athletes.empty
        else {}
    )
    athlete_display_map = {"": ""}
    for athlete_id in athlete_options:
        profile = db_profiles.get(athlete_id, {})
        nome = str(profile.get("nome") or "").strip()
        athlete_display_map[athlete_id] = f"{nome} ({athlete_id})" if nome else athlete_id
    display_to_athlete_id = {
        display_label: athlete_id
        for athlete_id, display_label in athlete_display_map.items()
    }
    base_rows = pd.DataFrame({
        "atleta_id_ficheiro": athlete_ids,
        "atleta_id": ["" for _ in athlete_ids],
    })
    if (
        st.session_state.get(state_selection_key) != selected_selection
        or existing is None
        or not isinstance(existing, pd.DataFrame)
        or "atleta_id_ficheiro" not in existing.columns
    ):
        editor_df = base_rows.copy()
    else:
        editor_df = base_rows.merge(
            existing,
            on="atleta_id_ficheiro",
            how="left",
            suffixes=("", "_existing"),
        )
        if "atleta_id_existing" in editor_df.columns:
            editor_df["atleta_id"] = editor_df["atleta_id_existing"].where(
                editor_df["atleta_id_existing"].astype(str).isin(athlete_options),
                editor_df["atleta_id"],
            )
            editor_df = editor_df.drop(columns=["atleta_id_existing"])

    defaults = {
        "nome": pd.NA,
        "numero_camisola": np.nan,
        "data_nascimento": pd.NaT,
        "posicao": "",
        "pe_preferencial": "",
        "altura_cm": np.nan,
        "peso_kg": np.nan,
        "selecao": selected_selection or "",
    }
    for col, default in defaults.items():
        if col not in editor_df.columns:
            editor_df[col] = default
        else:
            editor_df[col] = editor_df[col].fillna(default)

    for idx in editor_df.index:
        selected_athlete_id = str(editor_df.at[idx, "atleta_id"]).strip()
        profile = db_profiles.get(selected_athlete_id)
        if not profile:
            continue
        for col in defaults:
            if col == "selecao":
                editor_df.at[idx, col] = selected_selection or ""
                continue
            current_value = editor_df.at[idx, col]
            if pd.isna(current_value) or current_value == "":
                editor_df.at[idx, col] = profile.get(col, current_value)

    editor_df["selecao"] = selected_selection or ""

    if selected_selection:
        st.caption(f"Atletas ativos na base de dados para {selected_selection}: {len(db_athletes)}")
    if selected_selection and not athlete_options:
        st.warning(f"Sem atletas ativos registados para {selected_selection}.")

    header_left, header_right = st.columns([1, 1.25], gap="medium")
    header_left.markdown("**ID no ficheiro**")
    header_right.markdown("**Atleta da base de dados**")

    selected_ids = set()
    resolved_rows = []
    display_labels = list(display_to_athlete_id.keys())

    for idx, row in editor_df.reset_index(drop=True).iterrows():
        row_left, row_right = st.columns([1, 1.25], gap="medium")
        atleta_ficheiro = str(row.get("atleta_id_ficheiro", "")).strip()
        current_athlete_id = str(row.get("atleta_id", "")).strip()
        if current_athlete_id and current_athlete_id in selected_ids:
            current_athlete_id = ""
        current_label = athlete_display_map.get(current_athlete_id, "")

        row_left.text(atleta_ficheiro)

        available_labels = [""]
        for label in display_labels:
            athlete_id = display_to_athlete_id.get(label, "")
            if not athlete_id:
                continue
            if athlete_id == current_athlete_id or athlete_id not in selected_ids:
                available_labels.append(label)

        available_labels = list(dict.fromkeys(available_labels))
        if current_label and current_label not in available_labels:
            available_labels.append(current_label)

        selected_label = row_right.selectbox(
            "Atleta da base de dados",
            options=available_labels,
            index=available_labels.index(current_label) if current_label in available_labels else 0,
            key=f"athlete_registry_row_{selected_selection}_{idx}",
            label_visibility="collapsed",
        )

        selected_athlete_id = display_to_athlete_id.get(selected_label, "")
        if selected_athlete_id:
            selected_ids.add(selected_athlete_id)

        row_data = row.to_dict()
        row_data["atleta_id"] = selected_athlete_id
        resolved_rows.append(row_data)

    resolved_df = pd.DataFrame(resolved_rows)
    for col in defaults:
        if col not in resolved_df.columns:
            resolved_df[col] = pd.NA

    for idx in resolved_df.index:
        selected_athlete_id = str(resolved_df.at[idx, "atleta_id"]).strip()
        profile = db_profiles.get(selected_athlete_id, {})
        for col in defaults:
            if col == "selecao":
                resolved_df.at[idx, col] = selected_selection or ""
            elif (pd.isna(resolved_df.at[idx, col]) or resolved_df.at[idx, col] == "") and profile:
                resolved_df.at[idx, col] = profile.get(col, pd.NA)

    st.session_state[state_key] = resolved_df.copy()
    st.session_state[state_selection_key] = selected_selection

    add_mode_key = "athlete_registry_add_mode"
    add_col1, add_col2 = st.columns([0.9, 1.6], gap="medium")
    if add_col1.button("Adicionar atleta", key="btn_add_inline_athlete"):
        st.session_state[add_mode_key] = not st.session_state.get(add_mode_key, False)
        st.rerun()

    if st.session_state.get(add_mode_key, False):
        with add_col2:
            st.markdown("**Novo atleta na base de dados**")
            with st.form("inline_new_athlete_form", clear_on_submit=True):
                form_col1, form_col2 = st.columns(2)
                nome = form_col1.text_input("Nome")
                sobrenome = form_col2.text_input("Sobrenome")
                form_col3, form_col4, form_col5 = st.columns(3)
                data_nascimento = form_col3.date_input("Data de nascimento", value=None, format="DD/MM/YYYY")
                genero_label = form_col4.selectbox(
                    "Genero",
                    GENDER_OPTIONS,
                    index=GENDER_OPTIONS.index(selected_selection) if selected_selection in GENDER_OPTIONS else 0,
                )
                posicao = form_col5.selectbox("Posicao", ATHLETE_POSITIONS)
                save_new = st.form_submit_button("Guardar atleta", type="primary")

            if save_new:
                try:
                    _create_inline_athlete(
                        nome=nome,
                        sobrenome=sobrenome,
                        data_nascimento=data_nascimento,
                        genero_label=genero_label,
                        posicao=posicao,
                        selecao_value=selected_selection,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[add_mode_key] = False
                    st.success("Novo atleta criado com sucesso.")
                    st.rerun()

    return _normalize_athlete_registry_df(resolved_df, genero_default, selected_selection)


def _parse_report_sections(report_txt: str):
    if not report_txt:
        return "", []

    lines = report_txt.splitlines()
    title = ""
    sections = []
    current_title = None
    current_lines = []
    separators = {"-" * 70, "=" * 70}

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not title and stripped and stripped not in separators:
            title = stripped
            continue

        if stripped in separators or not stripped:
            continue

        if not line.startswith(" ") and ":" not in stripped:
            if current_title:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped
            current_lines = []
            continue

        if current_title:
            current_lines.append(line)

    if current_title:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return title, sections


def _sanitize_filename_part(value: str) -> str:
    text = _clean_text_value(value)
    if not text:
        return "sessao"
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sessao"


def _session_pdf_paths(session_fingerprint: str, selecao: str, contexto: str, jogo: str) -> tuple[Path, Path]:
    SESSION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selecao_part = _sanitize_filename_part(selecao)
    contexto_part = _sanitize_filename_part(contexto)
    jogo_part = _sanitize_filename_part(jogo)
    base_name = f"{selecao_part}_{contexto_part}_{jogo_part}_{_sanitize_filename_part(session_fingerprint)[:16]}"
    return (
        SESSION_REPORTS_DIR / f"{base_name}_coletivo.pdf",
        SESSION_REPORTS_DIR / f"{base_name}_individual.pdf",
    )


def _pdf_add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")
    wrapped_lines = []
    for raw_line in str(body or "").splitlines():
        chunks = textwrap.wrap(raw_line, width=100) or [""]
        wrapped_lines.extend(chunks)

    lines_per_page = 48
    pages = [wrapped_lines[i:i + lines_per_page] for i in range(0, max(len(wrapped_lines), 1), lines_per_page)] or [[]]
    plt.close(fig)
    for idx, page_lines in enumerate(pages, start=1):
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
        ax.axis("off")
        ax.text(0, 1.0, title if idx == 1 else f"{title} ({idx})", fontsize=14, fontweight="bold", va="top")
        ax.text(0, 0.96, "\n".join(page_lines) if page_lines else "-", fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _pdf_add_table_pages(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 24) -> None:
    if df is None or df.empty:
        _pdf_add_text_page(pdf, title, "Sem dados disponíveis.")
        return

    display_df = df.copy().fillna("-")
    display_df.columns = [str(col) for col in display_df.columns]
    pages = [display_df.iloc[i:i + rows_per_page].copy() for i in range(0, len(display_df), rows_per_page)]
    for idx, page_df in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title if len(pages) == 1 else f"{title} ({idx})", fontsize=13, fontweight="bold", pad=12)
        table = ax.table(
            cellText=page_df.astype(str).values,
            colLabels=page_df.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _plot_grouped_bars_on_axis(
    ax,
    *,
    title: str,
    categories: list[str],
    current_values: list[float],
    reference_values: list[float],
    current_label: str,
    reference_label: str,
    metric_col: str,
    rotate_xticks: bool = False,
) -> None:
    x = np.arange(len(categories))
    width = 0.38

    current_series = pd.to_numeric(pd.Series(current_values), errors="coerce").fillna(0.0)
    reference_series = pd.to_numeric(pd.Series(reference_values), errors="coerce")
    reference_available = reference_series.notna().any()
    if not reference_available:
        reference_series = pd.Series([0.0] * len(categories))
    else:
        reference_series = reference_series.fillna(0.0)

    bars_current = ax.bar(
        x - (width / 2 if reference_available else 0),
        current_series,
        width=width if reference_available else 0.6,
        color="#7fb24d",
        edgecolor="#2f3b1f",
        linewidth=1.0,
        label=current_label,
    )
    bars_reference = None
    if reference_available:
        bars_reference = ax.bar(
            x + width / 2,
            reference_series,
            width=width,
            color="#d9dde5",
            edgecolor="#6b7280",
            linewidth=1.0,
            label=reference_label,
        )

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90 if rotate_xticks else 0)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ymax = max(float(current_series.max()) if not current_series.empty else 0.0, float(reference_series.max()) if not reference_series.empty else 0.0)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

    for bars, values in [(bars_current, current_series), (bars_reference, reference_series if reference_available else None)]:
        if bars is None or values is None:
            continue
        labels = _format_metric_chart_text(metric_col, values)
        for rect, label in zip(bars, labels):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + (ymax * 0.02 if ymax > 0 else 0.02),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#64748b",
            )

    if reference_available:
        ax.legend(loc="upper right", frameon=False)


def _pdf_add_collective_group_charts(
    pdf: PdfPages,
    *,
    family_name: str,
    metric_cols: list[str],
    df_metrics: pd.DataFrame,
    selecao: str,
    contexto: str,
) -> None:
    charts = []
    collective_family_df = _build_collective_phase_totals(df_metrics, metric_cols)
    if collective_family_df is None or collective_family_df.empty:
        return

    for metric_col in metric_cols:
        metric_spec = COLLECTIVE_PROFILE_METRIC_MAP.get(metric_col, (metric_col, metric_col, _metric_user_label(metric_col)))
        current_metric_col, _, display_label = metric_spec
        if current_metric_col not in collective_family_df.columns:
            continue
        reference_phase_df = _load_historical_collective_phase_reference(
            selecao=selecao,
            contexto=contexto,
            metric_col=metric_col,
        )
        phase_order = ["Warm-Up", "1P", "2P"]
        current_df = collective_family_df[["fase", current_metric_col]].copy()
        current_df["fase"] = current_df["fase"].astype(str).str.strip()
        current_df = current_df[current_df["fase"].isin(phase_order)].copy()
        if current_df.empty:
            continue
        current_map = dict(zip(current_df["fase"], pd.to_numeric(current_df[current_metric_col], errors="coerce")))
        reference_map = {}
        if reference_phase_df is not None and not reference_phase_df.empty:
            reference_map = dict(zip(reference_phase_df["fase"], pd.to_numeric(reference_phase_df["reference_value"], errors="coerce")))
        categories = [phase for phase in phase_order if phase in current_map]
        charts.append(
            {
                "title": display_label,
                "categories": categories,
                "current_values": [current_map.get(cat, np.nan) for cat in categories],
                "reference_values": [reference_map.get(cat, np.nan) for cat in categories],
                "metric_col": metric_col,
                "rotate_xticks": False,
            }
        )

    if not charts:
        return

    for chart in charts:
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        fig.suptitle(family_name, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.98)
        _plot_grouped_bars_on_axis(
            ax,
            title=chart["title"],
            categories=chart["categories"],
            current_values=chart["current_values"],
            reference_values=chart["reference_values"],
            current_label="Sessão Atual",
            reference_label="Média Equipa",
            metric_col=chart["metric_col"],
            rotate_xticks=chart["rotate_xticks"],
        )
        fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _pdf_add_individual_group_charts(
    pdf: PdfPages,
    *,
    family_name: str,
    metric_cols: list[str],
    totals_by_athlete: pd.DataFrame,
    selecao: str,
    contexto: str,
    athlete_name_map: dict[str, str],
    athlete_target_map: dict[str, str],
    athlete_col: str = "atleta_id",
) -> None:
    if totals_by_athlete is None or totals_by_athlete.empty or athlete_col not in totals_by_athlete.columns:
        return

    charts = []
    for metric_col in metric_cols:
        metric_spec = INDIVIDUAL_PROFILE_METRIC_MAP.get(metric_col, (metric_col, metric_col, _metric_user_label(metric_col)))
        current_metric_col, _, display_label = metric_spec
        if current_metric_col not in totals_by_athlete.columns:
            continue

        reference_df = _load_historical_individual_metric_reference(
            selecao=selecao,
            contexto=contexto,
            metric_col=metric_col,
        )
        reference_map = {}
        if reference_df is not None and not reference_df.empty:
            reference_map = (
                reference_df[["atleta_id_norm", "reference_value"]]
                .drop_duplicates(subset=["atleta_id_norm"], keep="last")
                .set_index("atleta_id_norm")["reference_value"]
                .to_dict()
            )

        chart_df = totals_by_athlete[[athlete_col, current_metric_col]].copy()
        chart_df[athlete_col] = chart_df[athlete_col].map(_normalize_athlete_identifier)
        chart_df[current_metric_col] = pd.to_numeric(chart_df[current_metric_col], errors="coerce")
        chart_df = chart_df[chart_df[athlete_col].ne("") & chart_df[current_metric_col].notna()].copy()
        if chart_df.empty:
            continue
        chart_df["athlete_ref_id"] = chart_df[athlete_col].map(lambda athlete_id: athlete_target_map.get(athlete_id, athlete_id))
        chart_df["athlete_label"] = chart_df[athlete_col].map(
            lambda athlete_id: athlete_name_map.get(_normalize_athlete_identifier(athlete_id), _normalize_athlete_identifier(athlete_id))
        )
        chart_df = chart_df.sort_values(by=current_metric_col, ascending=False).reset_index(drop=True)

        charts.append(
            {
                "title": display_label,
                "categories": chart_df["athlete_label"].tolist(),
                "current_values": chart_df[current_metric_col].tolist(),
                "reference_values": [reference_map.get(ref_id, np.nan) for ref_id in chart_df["athlete_ref_id"].tolist()],
                "metric_col": metric_col,
                "rotate_xticks": True,
            }
        )

    if not charts:
        return

    for chart in charts:
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        fig.suptitle(family_name, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.98)
        _plot_grouped_bars_on_axis(
            ax,
            title=chart["title"],
            categories=chart["categories"],
            current_values=chart["current_values"],
            reference_values=chart["reference_values"],
            current_label="Sessão Atual",
            reference_label="Média Individual",
            metric_col=chart["metric_col"],
            rotate_xticks=chart["rotate_xticks"],
        )
        fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _generate_session_report_pdfs(
    *,
    session_fingerprint: str,
    selecao: str,
    contexto: str,
    jogo: str,
    report_txt: str,
    df_metrics: pd.DataFrame,
    athlete_name_map: dict[str, str] | None = None,
    athlete_target_map: dict[str, str] | None = None,
) -> tuple[str, str]:
    collective_pdf_path, individual_pdf_path = _session_pdf_paths(session_fingerprint, selecao, contexto, jogo)
    athlete_name_map = athlete_name_map or {}
    athlete_target_map = athlete_target_map or {}

    totals_by_athlete = _build_totals_by_athlete(df_metrics)
    if totals_by_athlete is None:
        totals_by_athlete = pd.DataFrame()

    collective_groups = _build_performance_metric_groups()
    with PdfPages(collective_pdf_path) as pdf:
        for familia, cols in collective_groups.items():
            _pdf_add_collective_group_charts(
                pdf,
                family_name=familia,
                metric_cols=cols,
                df_metrics=df_metrics,
                selecao=selecao,
                contexto=contexto,
            )

    with PdfPages(individual_pdf_path) as pdf:
        for familia, cols in collective_groups.items():
            _pdf_add_individual_group_charts(
                pdf,
                family_name=familia,
                metric_cols=cols,
                totals_by_athlete=totals_by_athlete,
                selecao=selecao,
                contexto=contexto,
                athlete_name_map=athlete_name_map,
                athlete_target_map=athlete_target_map,
            )

    return str(collective_pdf_path), str(individual_pdf_path)


def _build_samples_export(out_files, session_fingerprint: str, athlete_map: dict | None = None):
    sample_frames = []
    athlete_session_rows = []
    processed_at = pd.Timestamp.utcnow()
    athlete_map = athlete_map or {}

    for pth in out_files:
        df_sync = pd.read_csv(pth, sep=";")
        if df_sync.empty:
            continue

        athlete_id = str(df_sync["Atleta_ID"].dropna().iloc[0]) if "Atleta_ID" in df_sync.columns and df_sync["Atleta_ID"].dropna().any() else Path(pth).stem
        athlete_sk = athlete_map.get(str(athlete_id))
        hr_col = _detect_hr_col(df_sync)

        sample_df = pd.DataFrame({
            "session_fingerprint": session_fingerprint,
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

        # Convert time column to a publishable timestamp. If some raw values are invalid,
        # recover them from the relative event clock when available instead of inventing samples.
        if "time" in sample_df.columns and not sample_df["time"].isna().all():
            raw_time = sample_df["time"].copy()
            # Handle relative time format (HH:MM:SS.s) by combining with a base date
            try:
                # Check if time values contain date separators
                time_strs = raw_time.astype(str)
                has_dates = time_strs.str.contains('-', na=False)
                
                if has_dates.any():
                    # Some values have dates, try full timestamp format first
                    sample_df["time"] = pd.to_datetime(raw_time, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    # Fill any NaT values with time-only parsing
                    still_nat = sample_df["time"].isna()
                    if still_nat.any():
                        time_only = pd.to_datetime(raw_time.loc[still_nat], format='%H:%M:%S.%f', errors='coerce')
                        base_date = pd.Timestamp('2023-01-01')
                        sample_df.loc[still_nat, "time"] = base_date + (time_only - time_only.dt.normalize())
                else:
                    # Handle relative time format by adding a base date (e.g., 2023-01-01)
                    base_date = pd.Timestamp('2023-01-01')
                    # Parse time strings and add to base date
                    time_parsed = pd.to_datetime(raw_time, format='%H:%M:%S.%f', errors='coerce')
                    sample_df["time"] = base_date + (time_parsed - time_parsed.dt.normalize())
            except:
                # Fallback: try direct conversion with specific formats
                try:
                    sample_df["time"] = pd.to_datetime(raw_time, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                except:
                    try:
                        sample_df["time"] = pd.to_datetime(raw_time, format='%H:%M:%S.%f', errors='coerce')
                        base_date = pd.Timestamp('2023-01-01')
                        sample_df["time"] = base_date + (sample_df["time"] - sample_df["time"].dt.normalize())
                    except:
                        # Final fallback
                        sample_df["time"] = pd.to_datetime(raw_time, errors="coerce")

            invalid_time_mask = sample_df["time"].isna()
            if invalid_time_mask.any():
                fallback_seconds = pd.Series(np.nan, index=sample_df.index, dtype="float64")
                if "time_evento_s" in sample_df.columns:
                    fallback_seconds = pd.to_numeric(sample_df["time_evento_s"], errors="coerce")
                if fallback_seconds.notna().sum() == 0 and "time_evento" in sample_df.columns:
                    fallback_seconds = time_to_seconds(sample_df["time_evento"])
                if fallback_seconds.notna().sum() == 0:
                    fallback_seconds = time_to_seconds(raw_time)

                base_timestamp = None
                anchor_mask = sample_df["time"].notna() & fallback_seconds.notna()
                if anchor_mask.any():
                    anchor_idx = anchor_mask[anchor_mask].index[0]
                    base_timestamp = sample_df.loc[anchor_idx, "time"] - pd.to_timedelta(float(fallback_seconds.loc[anchor_idx]), unit="s")
                elif sample_df["time"].notna().any():
                    first_valid_time = sample_df["time"].dropna().iloc[0]
                    base_timestamp = first_valid_time.normalize() if isinstance(first_valid_time, pd.Timestamp) else pd.Timestamp("2023-01-01")
                elif fallback_seconds.notna().any():
                    base_timestamp = pd.Timestamp("2023-01-01")

                if base_timestamp is not None:
                    recoverable_mask = invalid_time_mask & fallback_seconds.notna()
                    if recoverable_mask.any():
                        sample_df.loc[recoverable_mask, "time"] = (
                            base_timestamp + pd.to_timedelta(fallback_seconds.loc[recoverable_mask], unit="s")
                        )

            # Keep as datetime for DuckDB TIMESTAMP compatibility (don't convert to string)

        sample_df["phase_id"] = sample_df["fase"].map(PHASE_MAP)
        sample_df = sample_df[sample_df["time"].notna() & sample_df["phase_id"].notna()].copy()
        sample_frames.append(sample_df)

        fases_presentes = sorted([f for f in sample_df["fase"].dropna().astype(str).unique().tolist() if f in ["Warm-Up", "1P", "2P"]], key=lambda x: PHASE_MAP.get(x, 99))
        athlete_session_rows.append({
            "session_fingerprint": session_fingerprint,
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


def _build_draft_exports(df_metrics: pd.DataFrame, out_files, session_payload: dict, field_data: pd.DataFrame):
    df_metrics_draft = df_metrics.copy()
    df_metrics_draft["session_fingerprint"] = session_payload["session_fingerprint"]
    df_metrics_draft["session_id_hex"] = session_payload["session_id_hex"]
    df_metrics_draft["phase_id"] = df_metrics_draft["fase"].map(PHASE_MAP)
    df_collective_perf = _build_collective_performance_metrics_draft(df_metrics_draft, session_payload)

    perf_cols = [
        "duracao_min", "dist_m", "m_min",
        "vmax_mps", "peak_1m_m_min",
        "hsr_dist_m", "hsr_pct", "sprint_dist_m", "n_sprints",
        "n_acc_2_5", "n_dec_3_0",
        "active_time_min", "active_pct",
        "hr_avg_bpm", "hr_peak_bpm", "hr_time_min", "hr_time_valid_pct", "beats_total",
        "dist_per_beat_m", "hsr_per_beat_m", "sprint_per_beat_m",
        "sprints_per_1000_beats", "acc_per_1000_beats", "dec_per_1000_beats",
        "external_load_score", "total_load_score",
        "player_load", "rhie_bouts", "rhie_actions",
        "zone1_walk_time_min", "zone1_walk_dist_m",
        "zone2_jog_time_min", "zone2_jog_dist_m",
        "zone3_run_time_min", "zone3_run_dist_m",
        "zone4_hsr_time_min", "zone4_hsr_dist_m",
        "zone5_sprint_time_min", "zone5_sprint_dist_m",
        "peak_dist_1m_m", "peak_dist_3m_m", "peak_dist_5m_m",
        "peak_hsr_1m_m", "peak_hsr_3m_m", "peak_hsr_5m_m",
        "peak_sprint_1m_m", "peak_sprint_3m_m", "peak_sprint_5m_m",
        "peak_acc_actions_1m", "peak_acc_actions_3m", "peak_acc_actions_5m",
        "peak_hi_actions_1m", "peak_hi_actions_3m", "peak_hi_actions_5m",
        "trimp_banister", "trimp_per_min",
    ]
    qc_cols = [
        "n_points", "pct_time_valid", "n_gaps_gt2s",
        "qc_grade", "qc_flags", "vmax_mps_qc", "n_jumps_gt15m", "n_gaps_gt2s_qc",
    ]
    base_cols = [
        "session_fingerprint", "session_id_hex", "atleta_id", "phase_id", "fase",
        "data", "selecao", "genero", "contexto", "jogo",
    ]

    df_perf = df_metrics_draft[base_cols + perf_cols].copy()
    df_qc = df_metrics_draft[["session_fingerprint", "session_id_hex", "atleta_id", "phase_id", "fase"] + qc_cols].copy()
    df_samples, df_athlete_session = _build_samples_export(
        out_files,
        session_payload["session_fingerprint"],
    )
    if not df_samples.empty:
        df_samples = df_samples.drop_duplicates(
            subset=["session_fingerprint", "atleta_id", "phase_id", "time"],
            keep="last",
        )

    df_tracking = normalize_tracking_data(
        df_samples,
        dist_x=field_data['dist_x'][0],
        dist_y=field_data['dist_y'][0]
    )

    return df_metrics_draft, df_perf, df_collective_perf, df_qc, df_samples, df_athlete_session, df_tracking


def _validate_publish_registry(athlete_registry_df: pd.DataFrame, athlete_ids_expected: list[str]):
    if athlete_registry_df is None or athlete_registry_df.empty:
        return None, "Preenche a ficha de atletas antes de publicar na base de dados."

    registry_df = athlete_registry_df.copy()
    registry_df["atleta_id_ficheiro"] = (
        registry_df["atleta_id_ficheiro"]
        .astype(str)
        .str.strip()
        .str.replace(r"^0+(?=\d+$)", "", regex=True)
    )
    registry_df["atleta_id"] = registry_df["atleta_id"].astype(str).str.strip()

    missing_rows = registry_df[registry_df["atleta_id"].eq("")]
    if not missing_rows.empty:
        missing_ids = ", ".join(sorted(missing_rows["atleta_id_ficheiro"].tolist()))
        return None, f"Falta associar os atletas do ficheiro a uma ficha da base: {missing_ids}."

    registry_df = registry_df.drop_duplicates(subset=["atleta_id_ficheiro"], keep="last")
    athlete_ids_expected = [
        str(v).strip()
        for v in athlete_ids_expected
    ]
    athlete_ids_expected = [
        pd.Series([v]).astype(str).str.replace(r"^0+(?=\d+$)", "", regex=True).iloc[0]
        for v in athlete_ids_expected
    ]
    missing_expected = sorted(set(athlete_ids_expected) - set(registry_df["atleta_id_ficheiro"].tolist()))
    if missing_expected:
        return None, f"Faltam associações para os atletas: {', '.join(missing_expected)}."

    return registry_df, None


def _prepare_publish_payloads(
    *,
    df_perf_draft: pd.DataFrame,
    df_collective_perf_draft: pd.DataFrame,
    df_qc_draft: pd.DataFrame,
    df_samples_draft: pd.DataFrame,
    df_athlete_session_draft: pd.DataFrame,
    session_payload: dict,
    genero: str,
    selecao: str,
    athlete_registry_df: pd.DataFrame,
    base_dir: str,
):
    athlete_ids_expected = sorted(df_perf_draft["atleta_id"].dropna().astype(str).unique().tolist())
    registry_df, registry_error = _validate_publish_registry(athlete_registry_df, athlete_ids_expected)
    if registry_error:
        raise ValueError(registry_error)

    athlete_target_map = dict(
        zip(
            registry_df["atleta_id_ficheiro"].astype(str),
            registry_df["atleta_id"].astype(str),
        )
    )
    athlete_profiles = registry_df.drop(columns=["atleta_id_ficheiro"]).copy()
    athlete_profiles = athlete_profiles.drop_duplicates(subset=["atleta_id"], keep="last")

    session_sk = resolve_session_sk(
        session_payload["session_fingerprint"],
        session_payload,
        base_dir,
    )

    athlete_seed = pd.DataFrame({"atleta_id": sorted(set(athlete_target_map.values()))})
    athlete_map = resolve_athlete_sk(
        athlete_seed,
        base_dir,
        genero=genero,
        athlete_profiles=athlete_profiles,
        selecao=selecao,
    )

    def _map_for_publish(df: pd.DataFrame):
        if df is None or df.empty:
            return df
        mapped = df.copy()
        source_ids = mapped["atleta_id"].astype(str)
        mapped["atleta_id"] = source_ids.map(athlete_target_map)
        if mapped["atleta_id"].isna().any():
            missing_ids = sorted(source_ids[mapped["atleta_id"].isna()].unique().tolist())
            raise ValueError(
                f"Não foi possível mapear todos os atletas para publicação: {', '.join(missing_ids)}."
            )
        mapped["session_sk"] = session_sk
        mapped["athlete_sk"] = mapped["atleta_id"].astype(str).map(athlete_map)
        return mapped

    df_perf_publish = _map_for_publish(df_perf_draft)
    df_qc_publish = _map_for_publish(df_qc_draft)
    df_samples_publish = _map_for_publish(df_samples_draft)
    df_athlete_session_publish = _map_for_publish(df_athlete_session_draft)
    df_collective_perf_publish = (
        df_collective_perf_draft.copy()
        if df_collective_perf_draft is not None else pd.DataFrame()
    )
    if not df_collective_perf_publish.empty:
        df_collective_perf_publish["session_sk"] = session_sk

    df_perf_publish = df_perf_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "phase_id", "fase", "data", "selecao", "genero",
            "contexto", "jogo", "duracao_min", "dist_m", "m_min", "vmax_mps", "peak_1m_m_min",
            "hsr_dist_m", "hsr_pct", "sprint_dist_m", "n_sprints", "n_acc_2_5", "n_dec_3_0",
            "active_time_min", "active_pct", "hr_avg_bpm", "hr_peak_bpm", "hr_time_min",
            "hr_time_valid_pct", "beats_total", "dist_per_beat_m", "hsr_per_beat_m",
            "sprint_per_beat_m", "sprints_per_1000_beats", "acc_per_1000_beats",
            "dec_per_1000_beats", "external_load_score", "total_load_score",
            "player_load", "rhie_bouts", "rhie_actions",
            "zone1_walk_time_min", "zone1_walk_dist_m",
            "zone2_jog_time_min", "zone2_jog_dist_m",
            "zone3_run_time_min", "zone3_run_dist_m",
            "zone4_hsr_time_min", "zone4_hsr_dist_m",
            "zone5_sprint_time_min", "zone5_sprint_dist_m",
            "peak_dist_1m_m", "peak_dist_3m_m", "peak_dist_5m_m",
            "peak_hsr_1m_m", "peak_hsr_3m_m", "peak_hsr_5m_m",
            "peak_sprint_1m_m", "peak_sprint_3m_m", "peak_sprint_5m_m",
            "peak_acc_actions_1m", "peak_acc_actions_3m", "peak_acc_actions_5m",
            "peak_hi_actions_1m", "peak_hi_actions_3m", "peak_hi_actions_5m",
            "trimp_banister", "trimp_per_min",
        ]
    ].copy()
    df_qc_publish = df_qc_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "phase_id", "fase", "n_points",
            "pct_time_valid", "n_gaps_gt2s", "qc_grade", "qc_flags", "vmax_mps_qc",
            "n_jumps_gt15m", "n_gaps_gt2s_qc",
        ]
    ].copy()
    df_samples_publish = df_samples_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "fase", "time", "time_evento_s", "time_evento",
            "periodo_jogo", "minuto_jogo", "lat", "lon", "x_utm", "y_utm", "hr_bpm", "phase_id",
        ]
    ].copy()
    if not df_samples_publish.empty:
        df_samples_publish["time"] = pd.to_datetime(df_samples_publish["time"], errors="coerce")
        df_samples_publish["phase_id"] = pd.to_numeric(df_samples_publish["phase_id"], errors="coerce").astype("Int64")
        df_samples_publish = df_samples_publish[
            df_samples_publish["time"].notna() & df_samples_publish["phase_id"].notna()
        ].copy()
    df_athlete_session_publish = df_athlete_session_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "participou_warmup", "participou_1p",
            "participou_2p", "fases_disponiveis", "n_samples", "tem_hr", "processado_em",
        ]
    ].copy()
    if not df_collective_perf_publish.empty:
        df_collective_perf_publish = df_collective_perf_publish[
            [
                "session_sk", "phase_id", "fase", "data", "selecao", "genero", "contexto", "jogo",
                "duracao_min_total", "dist_m_total", "hsr_dist_m_total", "sprint_dist_m_total",
                "active_time_min_total", "m_min_avg", "hsr_pct_avg", "active_pct_avg",
                "n_sprints_total", "n_acc_2_5_total", "n_dec_3_0_total",
                "vmax_mps_max", "peak_1m_m_min_max", "hr_avg_bpm_avg",
                "external_load_score_total", "total_load_score_total",
                "player_load_total", "rhie_bouts_total", "rhie_actions_total", "trimp_banister_total",
            ]
        ].copy()
        collective_int_cols = [
            "session_sk",
            "phase_id",
            "n_sprints_total",
            "n_acc_2_5_total",
            "n_dec_3_0_total",
            "rhie_bouts_total",
            "rhie_actions_total",
        ]
        for col in collective_int_cols:
            if col in df_collective_perf_publish.columns:
                df_collective_perf_publish[col] = (
                    pd.to_numeric(df_collective_perf_publish[col], errors="coerce")
                    .round()
                    .astype("Int64")
                )

    return df_perf_publish, df_collective_perf_publish, df_qc_publish, df_samples_publish, df_athlete_session_publish

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v16", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      div.stButton > button,
      div.stDownloadButton > button {
        white-space: nowrap;
      }

      div.element-container:has(.phase-marker.phase-complete) + div.element-container details {
        background: rgba(34, 197, 94, 0.10) !important;
        border: 1px solid rgba(34, 197, 94, 0.28) !important;
        border-radius: 0.75rem !important;
      }

      div.element-container:has(.phase-marker.phase-complete) + div.element-container summary {
        color: #166534 !important;
      }

      div.element-container:has(.phase-marker.phase-complete) + div.element-container summary:hover {
        background: rgba(34, 197, 94, 0.06) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
if "auth" not in st.session_state:
    st.session_state.auth = False
if "login_user" not in st.session_state:
    st.session_state.login_user = ""
if "login_role" not in st.session_state:
    st.session_state.login_role = ""

if not st.session_state.auth:
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Persistência de outputs (evita desaparecer após zoom/scroll no mapa) ---
ensure_draft_session_state()

# --- Persistência para 'Pick no mapa' (cantos do campo) ---
if "pick_corners" not in st.session_state:
    st.session_state.pick_corners = []  # lista [(lat, lon), ...]
if "pts_gps_picked" not in st.session_state:
    st.session_state.pts_gps_picked = None  # dict com BL/BR/TL/TR após ordenação
if "pick_last_click_sig" not in st.session_state:
    # evita duplicar o mesmo clique após rerun
    st.session_state.pick_last_click_sig = None
if "last_metodo_campo" not in st.session_state:
    st.session_state.last_metodo_campo = None
if "field_save_completed_sig" not in st.session_state:
    st.session_state.field_save_completed_sig = None
if "field_save_flash_msg" not in st.session_state:
    st.session_state.field_save_flash_msg = None
if "athlete_registry_expanded" not in st.session_state:
    st.session_state.athlete_registry_expanded = True
if "show_final_report_section" not in st.session_state:
    st.session_state.show_final_report_section = False
if "phase3_complete" not in st.session_state:
    st.session_state.phase3_complete = False
if "publish_success" not in st.session_state:
    st.session_state.publish_success = False

# --- LOGIN (CENTRADO + st.secrets) ---


def _render_phase_marker(is_complete: bool):
    marker_class = "phase-marker phase-complete" if is_complete else "phase-marker"
    st.markdown(f"<div class='{marker_class}' style='display:none;'></div>", unsafe_allow_html=True)


def _phase_title(label: str, is_complete: bool) -> str:
    return f"{label}\u200b" if is_complete else label


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


if not st.session_state.auth:
    _apply_login_style()

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(
            '<div class="login-title">FPF Performance Hub</div>', unsafe_allow_html=True)

        u = st.text_input("Utilizador", key="user_val")
        p = st.text_input("Password", type="password", key="pass_val")

        secrets_user, secrets_pass = get_secrets_auth()
        if not secrets_user:
            st.caption("Sem credenciais base em st.secrets. Podes entrar com um utilizador criado em Administracao.")

        if st.button("Entrar"):
            auth_result = authenticate_user(u, p)
            user = auth_result.get("user", {})
            if auth_result.get("success"):
                st.session_state.login_user = user.get("username", u)
                st.session_state.login_role = user.get("role", "")
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Credenciais invalidas")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# --- INTERFACE SINGLE PAGE ---
st.markdown(
    f"<div style='text-align:left; font-size:18px; color:#9aa0a6;'>User: {st.session_state.login_user} | FPF</div>",
    unsafe_allow_html=True,
)

top_left_col, top_right_col = st.columns(2, gap="large")

current_data_sessao = st.session_state.get("data_sessao_input")
current_selecao = st.session_state.get("selecao_input", "")
current_contexto = st.session_state.get("contexto_input", "Treino")
current_adversario = st.session_state.get("adversario_input", "")
session_ready = bool(current_data_sessao) and bool(current_selecao) and (
    (current_contexto != "Jogo") or bool(str(current_adversario).strip())
)
_render_phase_marker(session_ready)
with st.expander(_phase_title("1. Dados da Sessão", session_ready), expanded=not session_ready):
    # Estádio agora é inferido automaticamente pela localização do campo (sem input manual)
    estadio = None
    session_col_1, session_col_2, session_col_3, session_col_4 = st.columns([1.1, 1.4, 1.0, 1.2], gap="medium")

    with session_col_1:
        data_sessao = st.date_input("Data do Evento", key="data_sessao_input")

    with session_col_2:
        selecao = st.selectbox("Seleção", options=SELECTION_OPTIONS, index=0 if SELECTION_OPTIONS else None, key="selecao_input")

    # Género é inferido da seleção (M/F), não é input manual
    genero = selecao.split()[-1] if selecao.split() and selecao.split()[-1] in ["M", "F"] else ""

    with session_col_3:
        contexto = st.selectbox("Contexto", options=["Treino", "Jogo"], index=0, key="contexto_input")

    adversario = ""
    with session_col_4:
        if contexto == "Jogo":
            adversario = st.text_input("Adversário", key="adversario_input")
        else:
            adversario = st.text_input("Adversário", value="", disabled=True, key="adversario_input")

session_ready = bool(data_sessao) and bool(selecao) and ((contexto != "Jogo") or bool(adversario.strip()))
st.session_state.session_step_complete = session_ready

athletes_loaded = bool(st.session_state.get("f_atleta_upload"))
_render_phase_marker(athletes_loaded)
with st.expander(_phase_title("2. Dados dos Atletas", athletes_loaded), expanded=not athletes_loaded):
    st.caption(
        "CSVs com Player-<id> e indicação de fase (Warm/Primeira/Segunda/1P/2P) no nome do ficheiro."
    )
    f_atleta = st.file_uploader(
        "Dados de ATLETAS (CSVs)", accept_multiple_files=True, type=["csv"], key="f_atleta_upload"
    )
    athlete_registry_df = None
    athlete_registry_error = None
    if f_atleta:
        st.caption("A associação com a base de dados é feita no bloco Auditoria de Atletas.")
    else:
        st.caption("Carrega os ficheiros de atletas para poderes fazer a associação com a base de dados.")

st.session_state.athletes_step_complete = bool(f_atleta)

metodo_campo = "Escolher um campo guardado anteriormente"

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
def _suggest_field_name(lat: float, lon: float, city: str | None, country: str | None):
    """Sugere um nome base para o campo privilegiando referencias desportivas."""
    generic_tokens = {
        "road", "residential", "footway", "path", "cycleway", "pedestrian",
        "service", "track", "street", "avenue", "highway", "route",
    }

    def _clean_candidate(value):
        text = str(value or "").strip()
        return text if len(text) >= 4 else ""

    try:
        lat = round(float(lat), 6)
        lon = round(float(lon), 6)
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "jsonv2",
            "lat": lat,
            "lon": lon,
            "zoom": 18,
            "addressdetails": 1,
            "namedetails": 1,
        }
        headers = {
            "User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)",
            "Accept-Language": "pt-PT,pt,en",
        }
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json() if r.status_code == 200 and r.content else {}
    except Exception:
        data = {}

    addr = data.get("address", {}) if isinstance(data, dict) else {}
    if not isinstance(addr, dict):
        addr = {}
    namedetails = data.get("namedetails", {}) if isinstance(data, dict) else {}
    if not isinstance(namedetails, dict):
        namedetails = {}
    osm_type = str(data.get("type") or "").strip().lower()

    priority_candidates = [
        addr.get("stadium"),
        addr.get("sports_centre"),
        addr.get("sports_center"),
        addr.get("leisure"),
        addr.get("club"),
        addr.get("amenity"),
        namedetails.get("name"),
        data.get("name"),
    ]

    for candidate in priority_candidates:
        candidate = _clean_candidate(candidate)
        if candidate and candidate.lower() not in generic_tokens:
            return candidate

    if osm_type in {"stadium", "sports_centre", "pitch", "sports_hall"}:
        candidate = _clean_candidate(data.get("display_name", "").split(",")[0])
        if candidate:
            return candidate

    city = str(city or "").strip()
    country = str(country or "").strip()
    if city and country:
        return f"Campo {city} ({country})"
    if city:
        return f"Campo {city}"
    if country:
        return f"Campo ({country})"
    return ""


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
            "Carga": [
                "hr_avg_bpm",
                "hr_peak_bpm",
                "beats_total",
                "dist_per_beat_m",
                "external_load_score",
                "total_load_score",
            ],
        },
        "Disponibilidade / Integridade": {
            "Completude do sinal": ["n_points", "pct_time_valid", "n_gaps_gt2s"],
        },
        "QC / Confiabilidade": {
            "QC": ["qc_grade", "qc_flags", "vmax_mps_qc", "n_jumps_gt15m", "n_gaps_gt2s_qc"],
        },
    }


def _build_performance_metric_groups():
    return {
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
        "Carga": [
            "hr_avg_bpm",
            "external_load_score",
            "total_load_score",
            "player_load",
            "rhie_bouts",
            "trimp_banister",
        ],
    }

def _build_technical_metric_groups():
    return {
        "Disponibilidade / Integridade | Completude do sinal": [
            "n_points",
            "pct_time_valid",
            "n_gaps_gt2s",
        ],
        "QC / Confiabilidade | QC": [
            "qc_grade",
            "qc_flags",
            "vmax_mps_qc",
            "n_jumps_gt15m",
            "n_gaps_gt2s_qc",
        ],
    }


def _build_totals_by_athlete(df_metrics: pd.DataFrame) -> pd.DataFrame:
    if df_metrics is None or df_metrics.empty or "atleta_id" not in df_metrics.columns:
        return pd.DataFrame()

    totals_df = df_metrics.copy()
    if "fase" in totals_df.columns:
        totals_df = totals_df[totals_df["fase"].astype(str).str.strip().eq("Total")].copy()
    if totals_df.empty:
        return pd.DataFrame()

    duration_col = "duracao_min"
    per90_source_cols = [
        "dist_m",
        "hsr_dist_m",
        "sprint_dist_m",
        "active_time_min",
        "n_sprints",
        "n_acc_2_5",
        "n_dec_3_0",
        "beats_total",
        "external_load_score",
        "total_load_score",
        "player_load",
        "rhie_bouts",
        "trimp_banister",
    ]

    for col in [duration_col] + per90_source_cols:
        if col in totals_df.columns:
            totals_df[col] = pd.to_numeric(totals_df[col], errors="coerce")

    if duration_col in totals_df.columns:
        duration = pd.to_numeric(totals_df[duration_col], errors="coerce")
    else:
        duration = pd.Series(np.nan, index=totals_df.index, dtype="float64")
    valid_duration = duration.notna() & (duration > 0)

    rename_map = {
        "dist_m": "dist_m_90",
        "hsr_dist_m": "hsr_dist_m_90",
        "sprint_dist_m": "sprint_dist_m_90",
        "active_time_min": "active_time_min_90",
        "n_sprints": "n_sprints_90",
        "n_acc_2_5": "n_acc_2_5_90",
        "n_dec_3_0": "n_dec_3_0_90",
        "beats_total": "beats_total_90",
        "external_load_score": "external_load_score_90",
        "total_load_score": "total_load_score_90",
        "player_load": "player_load_90",
        "rhie_bouts": "rhie_bouts_90",
        "trimp_banister": "trimp_banister_90",
    }

    for source_col, target_col in rename_map.items():
        if source_col in totals_df.columns:
            totals_df[target_col] = np.where(
                valid_duration,
                pd.to_numeric(totals_df[source_col], errors="coerce") / duration * 90.0,
                np.nan,
            )

    preferred_cols = [
        "atleta_id",
        "fase",
        "duracao_min",
        "dist_m",
        "dist_m_90",
        "m_min",
        "hsr_dist_m",
        "hsr_dist_m_90",
        "hsr_pct",
        "sprint_dist_m",
        "sprint_dist_m_90",
        "n_sprints",
        "n_sprints_90",
        "n_acc_2_5",
        "n_acc_2_5_90",
        "n_dec_3_0",
        "n_dec_3_0_90",
        "active_time_min",
        "active_time_min_90",
        "active_pct",
        "hr_avg_bpm",
        "hr_peak_bpm",
        "hr_time_min",
        "hr_time_valid_pct",
        "beats_total",
        "beats_total_90",
        "dist_per_beat_m",
        "external_load_score",
        "external_load_score_90",
        "total_load_score",
        "total_load_score_90",
        "player_load",
        "player_load_90",
        "rhie_bouts",
        "rhie_bouts_90",
        "trimp_banister",
        "trimp_banister_90",
        "vmax_mps",
        "peak_1m_m_min",
        "n_points",
        "pct_time_valid",
        "n_gaps_gt2s",
        "qc_grade",
        "qc_flags",
        "vmax_mps_qc",
        "n_jumps_gt15m",
        "n_gaps_gt2s_qc",
    ]
    available_cols = [col for col in preferred_cols if col in totals_df.columns]
    totals_df = totals_df[available_cols].copy()
    return round_metrics_dataframe(totals_df)


def _build_collective_phase_totals(df_metrics: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if df_metrics is None or df_metrics.empty or "fase" not in df_metrics.columns:
        return pd.DataFrame()

    phase_order = ["Warm-Up", "1P", "2P"]
    work_df = df_metrics.copy()
    work_df["fase"] = work_df["fase"].astype(str).str.strip()
    work_df = work_df[work_df["fase"].isin(phase_order)].copy()
    if work_df.empty:
        return pd.DataFrame()

    cols_present = [col for col in metric_cols if col in work_df.columns]
    duration_col = "duracao_min"
    if duration_col in work_df.columns and duration_col not in cols_present:
        cols_present = [duration_col] + cols_present
    if not cols_present:
        return pd.DataFrame()

    aggregation_rules = {
        "duracao_min": "sum",
        "dist_m": "sum",
        "hsr_dist_m": "sum",
        "sprint_dist_m": "sum",
        "active_time_min": "sum",
        "n_sprints": "sum",
        "n_acc_2_5": "sum",
        "n_dec_3_0": "sum",
        "player_load": "sum",
        "rhie_bouts": "sum",
        "rhie_actions": "sum",
        "trimp_banister": "sum",
        "external_load_score": "sum",
        "total_load_score": "sum",
        "m_min": "mean",
        "hsr_pct": "mean",
        "active_pct": "mean",
        "hr_avg_bpm": "mean",
        "trimp_per_min": "mean",
        "vmax_mps": "max",
        "peak_1m_m_min": "max",
        "peak_dist_1m_m": "max",
        "peak_dist_3m_m": "max",
        "peak_dist_5m_m": "max",
        "peak_hsr_1m_m": "max",
        "peak_hsr_3m_m": "max",
        "peak_hsr_5m_m": "max",
        "peak_sprint_1m_m": "max",
        "peak_sprint_3m_m": "max",
        "peak_sprint_5m_m": "max",
        "peak_acc_actions_1m": "max",
        "peak_acc_actions_3m": "max",
        "peak_acc_actions_5m": "max",
        "peak_hi_actions_1m": "max",
        "peak_hi_actions_3m": "max",
        "peak_hi_actions_5m": "max",
    }

    for metric_col in cols_present:
        work_df[metric_col] = pd.to_numeric(work_df[metric_col], errors="coerce")

    collective_rows = []
    for phase in phase_order:
        phase_df = work_df[work_df["fase"].eq(phase)].copy()
        if phase_df.empty:
            continue

        row = {"fase": phase}
        for metric_col in cols_present:
            series = pd.to_numeric(phase_df[metric_col], errors="coerce")
            rule = aggregation_rules.get(metric_col, "sum")
            if rule == "mean":
                value = float(series.mean()) if series.notna().any() else np.nan
            elif rule == "max":
                value = float(series.max()) if series.notna().any() else np.nan
            else:
                value = float(series.fillna(0.0).sum()) if not series.empty else np.nan
            row[metric_col] = value
        collective_rows.append(row)

    collective_df = pd.DataFrame(collective_rows)
    if collective_df.empty:
        return pd.DataFrame()
    if "duracao_min" in collective_df.columns:
        duration = pd.to_numeric(collective_df["duracao_min"], errors="coerce")
    else:
        duration = pd.Series(np.nan, index=collective_df.index, dtype="float64")
    valid_duration = duration.notna() & (duration > 0)
    per90_map = {
        "dist_m": "dist_m_90",
        "hsr_dist_m": "hsr_dist_m_90",
        "sprint_dist_m": "sprint_dist_m_90",
        "active_time_min": "active_time_min_90",
        "n_sprints": "n_sprints_90",
        "n_acc_2_5": "n_acc_2_5_90",
        "n_dec_3_0": "n_dec_3_0_90",
        "external_load_score": "external_load_score_90",
        "total_load_score": "total_load_score_90",
        "player_load": "player_load_90",
        "rhie_bouts": "rhie_bouts_90",
        "trimp_banister": "trimp_banister_90",
    }
    for source_col, target_col in per90_map.items():
        if source_col in collective_df.columns:
            collective_df[target_col] = np.where(
                valid_duration,
                pd.to_numeric(collective_df[source_col], errors="coerce") / duration * 90.0,
                np.nan,
            )
    collective_df["__fase_ord"] = collective_df["fase"].map({phase: idx for idx, phase in enumerate(phase_order)})
    collective_df = collective_df.sort_values("__fase_ord").drop(columns="__fase_ord").reset_index(drop=True)
    return round_metrics_dataframe(collective_df)


def _build_collective_performance_metrics_draft(df_metrics: pd.DataFrame, session_payload: dict) -> pd.DataFrame:
    if df_metrics is None or df_metrics.empty:
        return pd.DataFrame()

    collective_specs = [
        ("duracao_min", "duracao_min_total"),
        ("dist_m", "dist_m_total"),
        ("hsr_dist_m", "hsr_dist_m_total"),
        ("sprint_dist_m", "sprint_dist_m_total"),
        ("active_time_min", "active_time_min_total"),
        ("m_min", "m_min_avg"),
        ("hsr_pct", "hsr_pct_avg"),
        ("active_pct", "active_pct_avg"),
        ("n_sprints", "n_sprints_total"),
        ("n_acc_2_5", "n_acc_2_5_total"),
        ("n_dec_3_0", "n_dec_3_0_total"),
        ("vmax_mps", "vmax_mps_max"),
        ("peak_1m_m_min", "peak_1m_m_min_max"),
        ("hr_avg_bpm", "hr_avg_bpm_avg"),
        ("external_load_score", "external_load_score_total"),
        ("total_load_score", "total_load_score_total"),
        ("player_load", "player_load_total"),
        ("rhie_bouts", "rhie_bouts_total"),
        ("rhie_actions", "rhie_actions_total"),
        ("trimp_banister", "trimp_banister_total"),
    ]

    source_cols = [source for source, _ in collective_specs]
    collective_df = _build_collective_phase_totals(df_metrics, source_cols)
    if collective_df is None or collective_df.empty:
        return pd.DataFrame()

    collective_df = collective_df.copy()
    collective_df["session_fingerprint"] = session_payload["session_fingerprint"]
    collective_df["session_id_hex"] = session_payload["session_id_hex"]
    collective_df["phase_id"] = collective_df["fase"].map(PHASE_MAP)
    collective_df["data"] = session_payload.get("data")
    collective_df["selecao"] = session_payload.get("selecao")
    collective_df["genero"] = session_payload.get("genero")
    collective_df["contexto"] = session_payload.get("contexto")
    collective_df["jogo"] = session_payload.get("jogo")

    for source_col, target_col in collective_specs:
        if source_col in collective_df.columns:
            collective_df[target_col] = collective_df[source_col]

    base_cols = [
        "session_fingerprint",
        "session_id_hex",
        "phase_id",
        "fase",
        "data",
        "selecao",
        "genero",
        "contexto",
        "jogo",
    ]
    available_metric_cols = [target for _, target in collective_specs if target in collective_df.columns]
    return collective_df[base_cols + available_metric_cols].copy()


METRIC_LABELS = {
    "duracao_min": "Duração (min)",
    "dist_m": "Distância Total (m)",
    "hsr_dist_m": "Distância HSR (m)",
    "sprint_dist_m": "Distância Sprint (m)",
    "active_time_min": "Tempo Ativo (min)",
    "m_min": "Intensidade (m/min)",
    "hsr_pct": "HSR (%)",
    "active_pct": "Tempo Ativo (%)",
    "n_sprints": "N.º Sprints",
    "n_acc_2_5": "N.º Acelerações",
    "n_dec_3_0": "N.º Desacelerações",
    "vmax_mps": "Velocidade Máxima (m/s)",
    "peak_1m_m_min": "Pico 1 min (m/min)",
    "hr_avg_bpm": "FC Média (bpm)",
    "hr_peak_bpm": "FC Máxima (bpm)",
    "beats_total": "Batimentos Totais",
    "dist_per_beat_m": "Distância por Batimento",
    "external_load_score": "Carga Externa",
    "total_load_score": "Carga Total",
    "player_load": "Player Load",
    "rhie_bouts": "RHIE",
    "trimp_banister": "TRIMP",
    "n_points": "N.º Pontos",
    "pct_time_valid": "Tempo Válido (%)",
    "n_gaps_gt2s": "Falhas > 2s",
    "qc_grade": "QC Grade",
    "qc_flags": "QC Flags",
    "vmax_mps_qc": "Vmax QC (m/s)",
    "n_jumps_gt15m": "Saltos > 15m",
    "n_gaps_gt2s_qc": "Falhas QC > 2s",
}

TEAM_REFERENCE_METRICS = [
    "duracao_min",
    "dist_m",
    "m_min",
    "hsr_pct",
    "n_sprints",
    "n_acc_2_5",
    "n_dec_3_0",
    "vmax_mps",
    "peak_1m_m_min",
]

INDIVIDUAL_REFERENCE_METRICS = [
    "dist_m",
    "m_min",
    "hsr_pct",
    "n_sprints",
]

COLLECTIVE_REFERENCE_COLUMN_MAP = {
    "duracao_min": "duracao_min_total",
    "dist_m": "dist_m_total",
    "hsr_dist_m": "hsr_dist_m_total",
    "sprint_dist_m": "sprint_dist_m_total",
    "active_time_min": "active_time_min_total",
    "m_min": "m_min_avg",
    "hsr_pct": "hsr_pct_avg",
    "active_pct": "active_pct_avg",
    "n_sprints": "n_sprints_total",
    "n_acc_2_5": "n_acc_2_5_total",
    "n_dec_3_0": "n_dec_3_0_total",
    "vmax_mps": "vmax_mps_max",
    "peak_1m_m_min": "peak_1m_m_min_max",
    "hr_avg_bpm": "hr_avg_bpm_avg",
    "external_load_score": "external_load_score_total",
    "total_load_score": "total_load_score_total",
    "player_load": "player_load_total",
    "rhie_bouts": "rhie_bouts_total",
    "rhie_actions": "rhie_actions_total",
    "trimp_banister": "trimp_banister_total",
}

INDIVIDUAL_PROFILE_METRIC_MAP = {
    "duracao_min": ("duracao_min", "duracao_min_media", "Duração Média (min)"),
    "dist_m": ("dist_m_90", "dist_m_90", "Distância / 90 (m)"),
    "hsr_dist_m": ("hsr_dist_m_90", "hsr_dist_m_90", "Distância HSR / 90 (m)"),
    "sprint_dist_m": ("sprint_dist_m_90", "sprint_dist_m_90", "Distância Sprint / 90 (m)"),
    "active_time_min": ("active_time_min_90", "active_time_min_90", "Tempo Ativo / 90 (min)"),
    "m_min": ("m_min", "m_min", "Intensidade (m/min)"),
    "hsr_pct": ("hsr_pct", "hsr_pct", "HSR (%)"),
    "active_pct": ("active_pct", "active_pct", "Tempo Ativo (%)"),
    "n_sprints": ("n_sprints_90", "n_sprints_90", "N.º Sprints / 90"),
    "n_acc_2_5": ("n_acc_2_5_90", "n_acc_2_5_90", "N.º Acelerações / 90"),
    "n_dec_3_0": ("n_dec_3_0_90", "n_dec_3_0_90", "N.º Desacelerações / 90"),
    "hr_avg_bpm": ("hr_avg_bpm", "hr_avg_bpm", "FC Média (bpm)"),
    "external_load_score": ("external_load_score_90", "external_load_score_90", "Carga Externa / 90"),
    "total_load_score": ("total_load_score_90", "total_load_score_90", "Carga Total / 90"),
    "player_load": ("player_load_90", "player_load_90", "Player Load / 90"),
    "rhie_bouts": ("rhie_bouts_90", "rhie_bouts_90", "RHIE / 90"),
    "trimp_banister": ("trimp_banister_90", "trimp_banister_90", "TRIMP / 90"),
    "vmax_mps": ("vmax_mps", "vmax_mps_peak", "Velocidade Máxima (m/s)"),
    "peak_1m_m_min": ("peak_1m_m_min", "peak_1m_m_min_peak", "Pico 1 min (m/min)"),
}

COLLECTIVE_PROFILE_METRIC_MAP = {
    "duracao_min": ("duracao_min", "duracao_min", "Duração (min)"),
    "dist_m": ("dist_m_90", "dist_m_90", "Distância / 90 (m)"),
    "hsr_dist_m": ("hsr_dist_m_90", "hsr_dist_m_90", "Distância HSR / 90 (m)"),
    "sprint_dist_m": ("sprint_dist_m_90", "sprint_dist_m_90", "Distância Sprint / 90 (m)"),
    "active_time_min": ("active_time_min_90", "active_time_min_90", "Tempo Ativo / 90 (min)"),
    "m_min": ("m_min", "m_min", "Intensidade (m/min)"),
    "hsr_pct": ("hsr_pct", "hsr_pct", "HSR (%)"),
    "active_pct": ("active_pct", "active_pct", "Tempo Ativo (%)"),
    "n_sprints": ("n_sprints_90", "n_sprints_90", "N.º Sprints / 90"),
    "n_acc_2_5": ("n_acc_2_5_90", "n_acc_2_5_90", "N.º Acelerações / 90"),
    "n_dec_3_0": ("n_dec_3_0_90", "n_dec_3_0_90", "N.º Desacelerações / 90"),
    "hr_avg_bpm": ("hr_avg_bpm", "hr_avg_bpm", "FC Média (bpm)"),
    "external_load_score": ("external_load_score_90", "external_load_score_90", "Carga Externa / 90"),
    "total_load_score": ("total_load_score_90", "total_load_score_90", "Carga Total / 90"),
    "player_load": ("player_load_90", "player_load_90", "Player Load / 90"),
    "rhie_bouts": ("rhie_bouts_90", "rhie_bouts_90", "RHIE / 90"),
    "trimp_banister": ("trimp_banister_90", "trimp_banister_90", "TRIMP / 90"),
    "vmax_mps": ("vmax_mps", "vmax_mps", "Velocidade Máxima (m/s)"),
    "peak_1m_m_min": ("peak_1m_m_min", "peak_1m_m_min", "Pico 1 min (m/min)"),
}


def _metric_user_label(metric_col: str) -> str:
    return METRIC_LABELS.get(metric_col, metric_col)


def _format_metric_value_for_ui(metric_col: str, value) -> str:
    if value is None or pd.isna(value):
        return "-"
    percent_metrics = {"hsr_pct", "active_pct", "pct_time_valid", "hr_time_valid_pct"}
    decimal_metrics = {"m_min", "vmax_mps", "vmax_mps_qc", "dist_per_beat_m", "peak_1m_m_min", "player_load", "trimp_banister"}
    integer_metrics = {
        "duracao_min", "dist_m", "hsr_dist_m", "sprint_dist_m", "active_time_min",
        "n_sprints", "n_acc_2_5", "n_dec_3_0", "hr_avg_bpm", "hr_peak_bpm",
        "beats_total", "external_load_score", "total_load_score", "rhie_bouts", "n_points",
        "n_gaps_gt2s", "n_jumps_gt15m", "n_gaps_gt2s_qc",
    }
    if metric_col in percent_metrics:
        return f"{float(value):.1f}%"
    if metric_col in decimal_metrics:
        return f"{float(value):.1f}"
    if metric_col in integer_metrics:
        return f"{int(round(float(value)))}"
    return f"{float(value):.1f}" if isinstance(value, (int, float, np.number)) else str(value)


def _safe_mean(series) -> float:
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _comparison_marker(current_value, reference_value, tolerance_pct: float = 5.0) -> str:
    if pd.isna(current_value) or pd.isna(reference_value):
        return "SEM REF"
    if float(reference_value) == 0:
        if float(current_value) == 0:
            return "EM LINHA"
        return "ACIMA"
    delta_pct = ((float(current_value) - float(reference_value)) / float(reference_value)) * 100.0
    if abs(delta_pct) <= tolerance_pct:
        return "EM LINHA"
    return "ACIMA" if delta_pct > 0 else "ABAIXO"


def _format_pct_delta(current_value, reference_value) -> str:
    if pd.isna(current_value) or pd.isna(reference_value) or float(reference_value) == 0:
        return "-"
    delta_pct = ((float(current_value) - float(reference_value)) / float(reference_value)) * 100.0
    return f"{delta_pct:+.0f}%"


def _format_text_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers:
        return []
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _fmt(row_values: list[str]) -> str:
        return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row_values))

    return [
        _fmt(headers),
        "-+-".join("-" * width for width in widths),
        *[_fmt(row) for row in rows],
    ]


def _load_historical_total_rows(selecao: str, contexto: str) -> pd.DataFrame:
    try:
        hist_df = read_table(
            "performance_metrics",
            filters={"selecao": selecao, "contexto": contexto, "fase": "Total"},
            columns=(
                "session_sk,atleta_id,data,selecao,contexto,fase,duracao_min,dist_m,m_min,"
                "hsr_dist_m,hsr_pct,sprint_dist_m,n_sprints,n_acc_2_5,n_dec_3_0,"
                "active_time_min,active_pct,hr_avg_bpm,external_load_score,total_load_score,"
                "player_load,rhie_bouts,trimp_banister,vmax_mps,peak_1m_m_min"
            ),
        )
    except Exception:
        return pd.DataFrame()

    if hist_df is None or hist_df.empty:
        return pd.DataFrame()

    for col in ["session_sk", "atleta_id", "selecao", "contexto", "fase"]:
        if col not in hist_df.columns:
            hist_df[col] = ""
        hist_df[col] = hist_df[col].map(_clean_text_value)

    if "data" in hist_df.columns:
        hist_df["data"] = pd.to_datetime(hist_df["data"], errors="coerce")

    numeric_cols = [
        "duracao_min",
        "dist_m",
        "m_min",
        "hsr_dist_m",
        "hsr_pct",
        "sprint_dist_m",
        "n_sprints",
        "n_acc_2_5",
        "n_dec_3_0",
        "active_time_min",
        "active_pct",
        "hr_avg_bpm",
        "external_load_score",
        "total_load_score",
        "player_load",
        "rhie_bouts",
        "trimp_banister",
        "vmax_mps",
        "peak_1m_m_min",
    ]
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")

    hist_df["atleta_id_norm"] = hist_df["atleta_id"].map(_normalize_athlete_identifier)
    return hist_df


def _load_historical_collective_phase_reference(selecao: str, contexto: str, metric_col: str) -> pd.DataFrame:
    metric_spec = COLLECTIVE_PROFILE_METRIC_MAP.get(metric_col)
    source_col = COLLECTIVE_REFERENCE_COLUMN_MAP.get(metric_col)
    if not source_col or metric_spec is None:
        return pd.DataFrame()

    try:
        hist_df = read_table(
            "collective_performance_metrics",
            filters={"selecao": selecao, "contexto": contexto},
            columns=f"session_sk,fase,duracao_min_total,{source_col}",
        )
    except Exception:
        return pd.DataFrame()

    if hist_df is None or hist_df.empty or source_col not in hist_df.columns:
        return pd.DataFrame()

    hist_df["fase"] = hist_df["fase"].map(_clean_text_value)
    hist_df = hist_df[hist_df["fase"].isin(["Warm-Up", "1P", "2P"])].copy()
    hist_df[source_col] = pd.to_numeric(hist_df[source_col], errors="coerce")
    hist_df = hist_df[hist_df[source_col].notna()].copy()
    if hist_df.empty:
        return pd.DataFrame()

    _, profile_metric_key, _ = metric_spec
    rows = []
    for fase, fase_df in hist_df.groupby("fase", dropna=False):
        duration = pd.to_numeric(fase_df.get("duracao_min_total"), errors="coerce")
        current_series = pd.to_numeric(fase_df[source_col], errors="coerce")
        valid_duration = duration.notna() & (duration > 0)

        if profile_metric_key == "duracao_min":
            reference_value = _safe_mean(current_series)
        elif profile_metric_key.endswith("_90"):
            reference_value = _safe_mean(np.where(valid_duration, current_series / duration * 90.0, np.nan))
        else:
            reference_value = _safe_mean(current_series)

        rows.append({"fase": fase, "reference_value": reference_value})

    reference_df = pd.DataFrame(rows)
    return reference_df[reference_df["reference_value"].notna()].reset_index(drop=True)


def _load_historical_individual_metric_reference(selecao: str, contexto: str, metric_col: str) -> pd.DataFrame:
    metric_spec = INDIVIDUAL_PROFILE_METRIC_MAP.get(metric_col)
    if not metric_col or metric_spec is None:
        return pd.DataFrame()

    historical_df = _load_historical_total_rows(selecao, contexto)
    if historical_df.empty:
        return pd.DataFrame()

    _, profile_metric_key, _ = metric_spec
    work_df = historical_df.copy()
    numeric_cols = [
        "duracao_min",
        "dist_m",
        "hsr_dist_m",
        "sprint_dist_m",
        "n_sprints",
        "n_acc_2_5",
        "n_dec_3_0",
        "active_time_min",
        "m_min",
        "hsr_pct",
        "active_pct",
        "hr_avg_bpm",
        "external_load_score",
        "total_load_score",
        "player_load",
        "rhie_bouts",
        "trimp_banister",
        "vmax_mps",
        "peak_1m_m_min",
    ]
    for col in numeric_cols:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    rows = []
    for athlete_id_norm, athlete_df in work_df.groupby("atleta_id_norm", dropna=False):
        if not athlete_id_norm:
            continue
        dur = float(athlete_df["duracao_min"].sum()) if "duracao_min" in athlete_df.columns else np.nan
        dist = float(athlete_df["dist_m"].sum()) if "dist_m" in athlete_df.columns else np.nan
        hsr = float(athlete_df["hsr_dist_m"].sum()) if "hsr_dist_m" in athlete_df.columns else np.nan
        sprint = float(athlete_df["sprint_dist_m"].sum()) if "sprint_dist_m" in athlete_df.columns else np.nan
        active_time = float(athlete_df["active_time_min"].sum()) if "active_time_min" in athlete_df.columns else np.nan
        n_sprints = float(athlete_df["n_sprints"].sum()) if "n_sprints" in athlete_df.columns else np.nan
        n_acc = float(athlete_df["n_acc_2_5"].sum()) if "n_acc_2_5" in athlete_df.columns else np.nan
        n_dec = float(athlete_df["n_dec_3_0"].sum()) if "n_dec_3_0" in athlete_df.columns else np.nan
        hr_avg = _safe_mean(athlete_df["hr_avg_bpm"]) if "hr_avg_bpm" in athlete_df.columns else np.nan
        external_load = float(athlete_df["external_load_score"].sum()) if "external_load_score" in athlete_df.columns else np.nan
        total_load = float(athlete_df["total_load_score"].sum()) if "total_load_score" in athlete_df.columns else np.nan
        player_load = float(athlete_df["player_load"].sum()) if "player_load" in athlete_df.columns else np.nan
        rhie_bouts = float(athlete_df["rhie_bouts"].sum()) if "rhie_bouts" in athlete_df.columns else np.nan
        trimp_banister = float(athlete_df["trimp_banister"].sum()) if "trimp_banister" in athlete_df.columns else np.nan
        n_sessoes = int(athlete_df["session_sk"].nunique()) if "session_sk" in athlete_df.columns else int(len(athlete_df))

        profile_metrics = {
            "duracao_min_media": (dur / n_sessoes) if pd.notna(dur) and n_sessoes > 0 else np.nan,
            "m_min": (dist / dur) if pd.notna(dist) and pd.notna(dur) and dur > 0 else np.nan,
            "hsr_pct": (hsr / dist * 100.0) if pd.notna(hsr) and pd.notna(dist) and dist > 0 else np.nan,
            "active_pct": (active_time / dur * 100.0) if pd.notna(active_time) and pd.notna(dur) and dur > 0 else np.nan,
            "dist_m_90": (dist / dur * 90.0) if pd.notna(dist) and pd.notna(dur) and dur > 0 else np.nan,
            "hsr_dist_m_90": (hsr / dur * 90.0) if pd.notna(hsr) and pd.notna(dur) and dur > 0 else np.nan,
            "sprint_dist_m_90": (sprint / dur * 90.0) if pd.notna(sprint) and pd.notna(dur) and dur > 0 else np.nan,
            "n_sprints_90": (n_sprints / dur * 90.0) if pd.notna(n_sprints) and pd.notna(dur) and dur > 0 else np.nan,
            "n_acc_2_5_90": (n_acc / dur * 90.0) if pd.notna(n_acc) and pd.notna(dur) and dur > 0 else np.nan,
            "n_dec_3_0_90": (n_dec / dur * 90.0) if pd.notna(n_dec) and pd.notna(dur) and dur > 0 else np.nan,
            "active_time_min_90": (active_time / dur * 90.0) if pd.notna(active_time) and pd.notna(dur) and dur > 0 else np.nan,
            "hr_avg_bpm": hr_avg,
            "external_load_score_90": (external_load / dur * 90.0) if pd.notna(external_load) and pd.notna(dur) and dur > 0 else np.nan,
            "total_load_score_90": (total_load / dur * 90.0) if pd.notna(total_load) and pd.notna(dur) and dur > 0 else np.nan,
            "player_load_90": (player_load / dur * 90.0) if pd.notna(player_load) and pd.notna(dur) and dur > 0 else np.nan,
            "rhie_bouts_90": (rhie_bouts / dur * 90.0) if pd.notna(rhie_bouts) and pd.notna(dur) and dur > 0 else np.nan,
            "trimp_banister_90": (trimp_banister / dur * 90.0) if pd.notna(trimp_banister) and pd.notna(dur) and dur > 0 else np.nan,
            "vmax_mps_peak": float(athlete_df["vmax_mps"].max()) if "vmax_mps" in athlete_df.columns and athlete_df["vmax_mps"].notna().any() else np.nan,
            "peak_1m_m_min_peak": float(athlete_df["peak_1m_m_min"].max()) if "peak_1m_m_min" in athlete_df.columns and athlete_df["peak_1m_m_min"].notna().any() else np.nan,
        }
        rows.append(
            {
                "atleta_id_norm": athlete_id_norm,
                "reference_value": profile_metrics.get(profile_metric_key, np.nan),
            }
        )

    reference_df = pd.DataFrame(rows)
    if reference_df.empty:
        return pd.DataFrame()

    reference_df["reference_value"] = pd.to_numeric(reference_df["reference_value"], errors="coerce")
    return reference_df[reference_df["reference_value"].notna()].reset_index(drop=True)


def _build_team_reference_rows(current_totals_df: pd.DataFrame, historical_df: pd.DataFrame) -> tuple[list[list[str]], int]:
    if current_totals_df is None or current_totals_df.empty:
        return [], 0

    current_work = current_totals_df.copy()
    current_metrics = {}
    for metric_col in TEAM_REFERENCE_METRICS:
        if metric_col not in current_work.columns:
            continue
        series = pd.to_numeric(current_work[metric_col], errors="coerce")
        if metric_col in {"m_min", "hsr_pct"}:
            current_metrics[metric_col] = float(series.mean()) if series.notna().any() else np.nan
        elif metric_col in {"vmax_mps", "peak_1m_m_min"}:
            current_metrics[metric_col] = float(series.max()) if series.notna().any() else np.nan
        else:
            current_metrics[metric_col] = float(series.sum()) if not series.empty else np.nan

    session_count = 0
    reference_metrics = {}
    if historical_df is not None and not historical_df.empty and "session_sk" in historical_df.columns:
        grouped_rows = []
        for _, session_df in historical_df.groupby("session_sk", dropna=False):
            session_row = {}
            for metric_col in TEAM_REFERENCE_METRICS:
                if metric_col not in session_df.columns:
                    continue
                series = pd.to_numeric(session_df[metric_col], errors="coerce")
                if metric_col in {"m_min", "hsr_pct"}:
                    session_row[metric_col] = float(series.mean()) if series.notna().any() else np.nan
                elif metric_col in {"vmax_mps", "peak_1m_m_min"}:
                    session_row[metric_col] = float(series.max()) if series.notna().any() else np.nan
                else:
                    session_row[metric_col] = float(series.sum()) if not series.empty else np.nan
            grouped_rows.append(session_row)
        grouped_df = pd.DataFrame(grouped_rows)
        session_count = len(grouped_df)
        for metric_col in TEAM_REFERENCE_METRICS:
            if metric_col in grouped_df.columns:
                reference_metrics[metric_col] = _safe_mean(grouped_df[metric_col])

    rows = []
    for metric_col in TEAM_REFERENCE_METRICS:
        current_value = current_metrics.get(metric_col, np.nan)
        reference_value = reference_metrics.get(metric_col, np.nan)
        rows.append(
            [
                _metric_user_label(metric_col),
                _format_metric_value_for_ui(metric_col, current_value),
                _format_metric_value_for_ui(metric_col, reference_value),
                _format_pct_delta(current_value, reference_value),
                _comparison_marker(current_value, reference_value),
            ]
        )
    return rows, session_count


def _build_individual_reference_rows(current_totals_df: pd.DataFrame, historical_df: pd.DataFrame) -> list[list[str]]:
    if current_totals_df is None or current_totals_df.empty:
        return []

    current_work = current_totals_df.copy()
    current_work["atleta_id_norm"] = current_work["atleta_id"].map(_normalize_athlete_identifier)

    reference_df = pd.DataFrame()
    if historical_df is not None and not historical_df.empty:
        agg_map = {metric_col: "mean" for metric_col in INDIVIDUAL_REFERENCE_METRICS if metric_col in historical_df.columns}
        if agg_map:
            reference_df = (
                historical_df.groupby("atleta_id_norm", dropna=False)
                .agg(agg_map)
                .reset_index()
            )
            counts_df = (
                historical_df.groupby("atleta_id_norm", dropna=False)["session_sk"]
                .nunique()
                .reset_index(name="n_refs")
            )
            reference_df = reference_df.merge(counts_df, on="atleta_id_norm", how="left")

    rows = []
    for _, athlete_row in current_work.sort_values("atleta_id").iterrows():
        athlete_id = _clean_text_value(athlete_row.get("atleta_id")) or "-"
        ref_row = (
            reference_df[reference_df["atleta_id_norm"].eq(athlete_row.get("atleta_id_norm"))].head(1)
            if not reference_df.empty else pd.DataFrame()
        )
        row = [athlete_id]
        for metric_col in INDIVIDUAL_REFERENCE_METRICS:
            current_value = pd.to_numeric(pd.Series([athlete_row.get(metric_col)]), errors="coerce").iloc[0]
            reference_value = (
                pd.to_numeric(pd.Series([ref_row.iloc[0].get(metric_col)]), errors="coerce").iloc[0]
                if not ref_row.empty and metric_col in ref_row.columns
                else np.nan
            )
            row.append(_format_pct_delta(current_value, reference_value))
        n_refs = int(ref_row.iloc[0]["n_refs"]) if not ref_row.empty and pd.notna(ref_row.iloc[0].get("n_refs")) else 0
        row.append(str(n_refs))
        rows.append(row)
    return rows


def _build_historical_reference_report_lines(
    df_metrics: pd.DataFrame,
    selecao: str,
    contexto: str,
) -> list[str]:
    totals_by_athlete = _build_totals_by_athlete(df_metrics)
    if totals_by_athlete is None or totals_by_athlete.empty:
        return ["Referências históricas indisponíveis: sem métricas totais por atleta."]

    historical_df = _load_historical_total_rows(selecao, contexto)
    lines = ["Referências Históricas"]
    lines.append("  Base de comparação: médias das sessões anteriores da mesma seleção e contexto.")

    team_rows, team_sessions = _build_team_reference_rows(totals_by_athlete, historical_df)
    lines.append(f"  Equipa | sessões de referência: {team_sessions}")
    if team_rows and team_sessions > 0:
        lines.extend([f"  {line}" for line in _format_text_table(
            ["Métrica", "Atual", "Média Ant.", "Delta", "Marca"],
            team_rows,
        )])
    else:
        lines.append("  Sem histórico suficiente para referência de equipa.")

    individual_rows = _build_individual_reference_rows(totals_by_athlete, historical_df)
    lines.append("")
    lines.append("  Individual | delta % vs média histórica do próprio atleta")
    if individual_rows and any(row[-1] != "0" for row in individual_rows):
        lines.extend([f"  {line}" for line in _format_text_table(
            ["Atleta", "Dist", "Int", "HSR", "Sprint", "N Ref"],
            individual_rows,
        )])
        lines.append("  Legenda: Dist=Distância Total | Int=Intensidade | HSR=HSR (%) | Sprint=N.º Sprints")
    else:
        lines.append("  Sem histórico individual suficiente para comparação.")

    return lines


def _build_group_matrix_by_athlete(
    totals_df: pd.DataFrame,
    metric_cols: list[str],
    athlete_col: str = "atleta_id",
) -> pd.DataFrame:
    if totals_df is None or totals_df.empty or athlete_col not in totals_df.columns:
        return pd.DataFrame()

    cols_present = [col for col in metric_cols if col in totals_df.columns]
    if not cols_present:
        return pd.DataFrame()

    matrix_df = totals_df[[athlete_col] + cols_present].copy()
    matrix_df = matrix_df.sort_values(by=[athlete_col]).reset_index(drop=True)
    return matrix_df


def _transpose_group_metrics(
    source_df: pd.DataFrame,
    metric_cols: list[str],
    athlete_col: str = "atleta_id",
) -> pd.DataFrame:
    if source_df is None or source_df.empty or athlete_col not in source_df.columns:
        return pd.DataFrame()

    cols_present = [col for col in metric_cols if col in source_df.columns]
    if not cols_present:
        return pd.DataFrame()

    work_df = source_df[[athlete_col] + cols_present].copy()
    work_df[athlete_col] = work_df[athlete_col].astype(str).str.strip()
    work_df = work_df[work_df[athlete_col].ne("")]
    if work_df.empty:
        return pd.DataFrame()

    work_df = work_df.drop_duplicates(subset=[athlete_col], keep="last")
    athlete_ids = work_df[athlete_col].tolist()

    rows = []
    for metric_col in cols_present:
        row = {"Metrica": metric_col}
        for athlete_id in athlete_ids:
            metric_series = work_df.loc[work_df[athlete_col].eq(athlete_id), metric_col]
            row[athlete_id] = metric_series.iloc[0] if not metric_series.empty else "-"
        rows.append(row)

    return pd.DataFrame(rows)


def _format_metric_chart_text(metric_col: str, series: pd.Series) -> list[str]:
    labels = []
    for value in series:
        labels.append(_format_metric_value_for_ui(metric_col, value))
    return labels


def _build_athlete_name_map() -> dict[str, str]:
    athlete_name_map: dict[str, str] = {}
    registry_target_map: dict[str, str] = {}

    registry_df = st.session_state.get("athlete_registry_editor_df")
    if isinstance(registry_df, pd.DataFrame) and not registry_df.empty:
        reg = registry_df.copy()
        if "atleta_id_ficheiro" in reg.columns:
            reg["atleta_id_ficheiro"] = reg["atleta_id_ficheiro"].map(_normalize_athlete_identifier)
        if "atleta_id" in reg.columns:
            reg["atleta_id"] = reg["atleta_id"].map(_normalize_athlete_identifier)
        if "nome" in reg.columns:
            reg["nome"] = reg["nome"].astype(str).str.strip()
        for _, row in reg.iterrows():
            athlete_file_id = str(row.get("atleta_id_ficheiro") or "").strip()
            athlete_db_id = str(row.get("atleta_id") or "").strip()
            athlete_name = str(row.get("nome") or "").strip()
            if athlete_file_id and athlete_db_id:
                registry_target_map[athlete_file_id] = athlete_db_id
            if athlete_file_id and athlete_name and athlete_name.lower() not in {"nan", "none", "<na>"}:
                athlete_name_map[athlete_file_id] = athlete_name

    try:
        athletes_df = read_table("athletes")
    except Exception:
        athletes_df = pd.DataFrame()
    if athletes_df is not None and not athletes_df.empty and {"atleta_id", "nome"}.issubset(athletes_df.columns):
        aux = athletes_df[["atleta_id", "nome"]].copy()
        aux["atleta_id"] = aux["atleta_id"].map(_normalize_athlete_identifier)
        aux["nome"] = aux["nome"].astype(str).str.strip()
        aux = aux[aux["atleta_id"].ne("")]
        db_name_map = {}
        for _, row in aux.iterrows():
            athlete_id = str(row.get("atleta_id") or "").strip()
            athlete_name = str(row.get("nome") or "").strip()
            if athlete_id and athlete_name and athlete_name.lower() not in {"nan", "none", "<na>"}:
                db_name_map[athlete_id] = athlete_name
                athlete_name_map.setdefault(athlete_id, athlete_name)

        for athlete_file_id, athlete_db_id in registry_target_map.items():
            athlete_name = db_name_map.get(athlete_db_id, "")
            if athlete_file_id and athlete_name:
                athlete_name_map[athlete_file_id] = athlete_name

    return athlete_name_map


def _build_athlete_target_map() -> dict[str, str]:
    athlete_target_map: dict[str, str] = {}
    registry_df = st.session_state.get("athlete_registry_editor_df")
    if not isinstance(registry_df, pd.DataFrame) or registry_df.empty:
        return athlete_target_map

    reg = registry_df.copy()
    if "atleta_id_ficheiro" in reg.columns:
        reg["atleta_id_ficheiro"] = reg["atleta_id_ficheiro"].map(_normalize_athlete_identifier)
    if "atleta_id" in reg.columns:
        reg["atleta_id"] = reg["atleta_id"].map(_normalize_athlete_identifier)

    for _, row in reg.iterrows():
        athlete_file_id = str(row.get("atleta_id_ficheiro") or "").strip()
        athlete_db_id = str(row.get("atleta_id") or "").strip()
        if athlete_file_id and athlete_db_id:
            athlete_target_map[athlete_file_id] = athlete_db_id
    return athlete_target_map


def _render_metric_bar_chart(
    source_df: pd.DataFrame,
    metric_col: str,
    athlete_col: str = "atleta_id",
    reference_df: pd.DataFrame | None = None,
    chart_key: str | None = None,
    athlete_name_map: dict[str, str] | None = None,
    athlete_target_map: dict[str, str] | None = None,
    comparison_col: str | None = None,
    yaxis_label: str | None = None,
) -> bool:
    value_col = comparison_col or metric_col
    if source_df is None or source_df.empty or value_col not in source_df.columns or athlete_col not in source_df.columns:
        return False

    chart_df = source_df[[athlete_col, value_col]].copy()
    chart_df[athlete_col] = chart_df[athlete_col].map(_normalize_athlete_identifier)
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df[chart_df[athlete_col].ne("")].copy()
    zero_fill_metrics = {"player_load"}
    if metric_col in zero_fill_metrics:
        chart_df[value_col] = chart_df[value_col].fillna(0.0)
    else:
        chart_df = chart_df[chart_df[value_col].notna()].copy()
    if chart_df.empty:
        return False

    athlete_name_map = athlete_name_map or {}
    athlete_target_map = athlete_target_map or {}
    chart_df["athlete_id_norm"] = chart_df[athlete_col].map(_normalize_athlete_identifier)
    chart_df["athlete_ref_id"] = chart_df["athlete_id_norm"].map(
        lambda athlete_id: athlete_target_map.get(athlete_id, athlete_id)
    )
    chart_df["athlete_label"] = chart_df[athlete_col].map(
        lambda athlete_id: athlete_name_map.get(_normalize_athlete_identifier(athlete_id), _normalize_athlete_identifier(athlete_id))
    )
    chart_df = chart_df.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    fig = go.Figure(
        data=[
            go.Bar(
                name="Sessão Atual",
                x=chart_df["athlete_label"],
                y=chart_df[value_col],
                text=_format_metric_chart_text(metric_col, chart_df[value_col]),
                textposition="outside",
                marker=dict(color="#7fb24d", line=dict(color="#2f3b1f", width=1.0)),
                cliponaxis=False,
            )
        ]
    )

    has_reference = False
    if reference_df is not None and not reference_df.empty and "atleta_id_norm" in reference_df.columns and "reference_value" in reference_df.columns:
        reference_map = (
            reference_df[["atleta_id_norm", "reference_value"]]
            .drop_duplicates(subset=["atleta_id_norm"], keep="last")
            .set_index("atleta_id_norm")["reference_value"]
        )
        ref_chart_df = chart_df[["athlete_ref_id", "athlete_label"]].copy()
        ref_chart_df["reference_value"] = pd.to_numeric(
            ref_chart_df["athlete_ref_id"].map(reference_map),
            errors="coerce",
        )
        if ref_chart_df["reference_value"].notna().any():
            has_reference = True
            fig.add_trace(
                go.Bar(
                    name="Média Individual",
                    x=ref_chart_df["athlete_label"],
                    y=ref_chart_df["reference_value"],
                    text=_format_metric_chart_text(metric_col, ref_chart_df["reference_value"]),
                    textposition="outside",
                    marker=dict(color="#d9dde5", line=dict(color="#6b7280", width=1.0)),
                    cliponaxis=False,
                )
            )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=360,
        xaxis_title="",
        yaxis_title=yaxis_label or _metric_user_label(metric_col),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=has_reference,
        barmode="group",
    )
    fig.update_xaxes(type="category", tickangle=-90, showgrid=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, key=chart_key or f"metric_chart_{metric_col}")
    return True


def _render_phase_metric_bar_chart(
    source_df: pd.DataFrame,
    metric_col: str,
    phase_col: str = "fase",
    reference_df: pd.DataFrame | None = None,
    comparison_col: str | None = None,
    yaxis_label: str | None = None,
    chart_key: str | None = None,
) -> bool:
    value_col = comparison_col or metric_col
    if source_df is None or source_df.empty or value_col not in source_df.columns or phase_col not in source_df.columns:
        return False

    phase_order = ["Warm-Up", "1P", "2P"]
    chart_df = source_df[[phase_col, value_col]].copy()
    chart_df[phase_col] = chart_df[phase_col].astype(str).str.strip()
    chart_df = chart_df[chart_df[phase_col].isin(phase_order)].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce").fillna(0.0)
    if chart_df.empty:
        return False

    chart_df["__fase_ord"] = chart_df[phase_col].map({phase: idx for idx, phase in enumerate(phase_order)})
    chart_df = chart_df.sort_values("__fase_ord").drop(columns="__fase_ord").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Sessão Atual",
            x=chart_df[phase_col],
            y=chart_df[value_col],
            text=_format_metric_chart_text(metric_col, chart_df[value_col]),
            textposition="outside",
            marker=dict(color="#7fb24d", line=dict(color="#2f3b1f", width=1.0)),
            cliponaxis=False,
        )
    )

    has_reference = False
    if reference_df is not None and not reference_df.empty and "reference_value" in reference_df.columns and phase_col in reference_df.columns:
        ref_chart_df = reference_df[[phase_col, "reference_value"]].copy()
        ref_chart_df[phase_col] = ref_chart_df[phase_col].astype(str).str.strip()
        ref_chart_df = ref_chart_df[ref_chart_df[phase_col].isin(phase_order)].copy()
        ref_chart_df["reference_value"] = pd.to_numeric(ref_chart_df["reference_value"], errors="coerce")
        ref_chart_df["__fase_ord"] = ref_chart_df[phase_col].map({phase: idx for idx, phase in enumerate(phase_order)})
        ref_chart_df = ref_chart_df.sort_values("__fase_ord").drop(columns="__fase_ord")
        if not ref_chart_df.empty and ref_chart_df["reference_value"].notna().any():
            has_reference = True
            fig.add_trace(
                go.Bar(
                    name="Média Equipa",
                    x=ref_chart_df[phase_col],
                    y=ref_chart_df["reference_value"],
                    text=_format_metric_chart_text(metric_col, ref_chart_df["reference_value"]),
                    textposition="outside",
                    marker=dict(color="#d9dde5", line=dict(color="#6b7280", width=1.0)),
                    cliponaxis=False,
                )
            )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=320,
        xaxis_title="",
        yaxis_title=yaxis_label or _metric_user_label(metric_col),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=has_reference,
        barmode="group",
    )
    fig.update_xaxes(type="category", showgrid=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, key=chart_key or f"phase_metric_chart_{metric_col}")
    return True


def _order_technical_report_sections(report_sections):
    ordered_titles = [
        "Dados da Sessóo",
        "Validação geográfica",
        "Auditoria de atletas (submissão)",
        "Sincronização",
        "Timeline do Jogo",
        "Qualidade do Sinal GPS",
        "Auditoria de Timestamp",
    ]
    ignored_titles = {
        "Métricas Individuais (GPS-only) — thresholds fixos",
        "Avisos/Problemas (exemplos):",
    }

    section_map = {title: body for title, body in report_sections if title not in ignored_titles}
    ordered_sections = [(title, section_map[title]) for title in ordered_titles if title in section_map]
    remaining_sections = [
        (title, body) for title, body in report_sections
        if title not in ignored_titles and title not in ordered_titles
    ]
    return ordered_sections + remaining_sections


def _render_database_integration_section(f_atleta, genero, selecao, data_sessao, adversario, publish_clicked: bool = False) -> None:
    st.markdown("**Integração na Base de Dados**")

    if st.session_state.get("df_perf") is not None and not st.session_state.df_perf.empty:
        st.info("✅ Dados processados e prontos para serem integrados no Supabase.")
        st.caption("Para publicar, cada atleta do ficheiro tem de ser associado a uma ficha da base de dados.")

        athlete_registry_df = None
        athlete_registry_error = None
        if f_atleta:
            athlete_registry_df, athlete_registry_error = _normalize_athlete_registry_df(
                st.session_state.get("athlete_registry_editor_df"),
                genero,
                selecao,
            )
            if athlete_registry_df is not None:
                st.caption("Associação de atletas reutilizada a partir do passo 2.")
        else:
            athlete_registry_error = "Carrega os ficheiros de atletas para conseguires publicar esta sessão."

        if athlete_registry_error:
            st.warning(athlete_registry_error)

        publish_payload = st.session_state.get("draft_session_payload") or {}
        publish_context = st.session_state.get("draft_context") or {}
        duplicate_sessions_df = _find_potential_duplicate_sessions(
            data_sessao=publish_payload.get("data_sessao", data_sessao),
            selecao=publish_context.get("selecao", selecao),
            genero=publish_context.get("genero", genero),
            contexto=publish_context.get("contexto", contexto),
            jogo=publish_payload.get("jogo", adversario),
        )
        allow_duplicate_publish = False
        if not duplicate_sessions_df.empty:
            data_txt = pd.to_datetime(
                publish_payload.get("data_sessao", data_sessao)
            ).strftime("%d/%m/%Y")
            st.error(
                f"Já existem dados publicados para esta sessão em {data_txt}. "
                "A gravação foi bloqueada para evitar duplicados."
            )
            st.caption("Sessões potencialmente coincidentes já gravadas:")
            st.dataframe(duplicate_sessions_df, use_container_width=True, hide_index=True)
            allow_duplicate_publish = st.checkbox(
                "Permitir gravação mesmo assim",
                key="allow_duplicate_publish",
                help="Usa esta opção apenas se quiseres substituir ou atualizar uma sessão já publicada.",
            )

        if publish_clicked:
            progress_bar = st.progress(0, text="A iniciar transferência para o Supabase...")
            if not duplicate_sessions_df.empty and not allow_duplicate_publish:
                st.error("Publicação interrompida para evitar duplicação da sessão.")
                st.stop()
            progress_text = st.empty()

            def _on_db_progress(event: dict):
                step = max(int(event.get("step", 0)), 0)
                total_steps = max(int(event.get("total_steps", 1)), 1)
                message = str(event.get("message", "A processar..."))
                pct = min(step / total_steps, 1.0)
                progress_bar.progress(pct, text=message)
                progress_text.caption(f"Passo {step}/{total_steps}: {message}")

            try:
                publish_payload = st.session_state.get("draft_session_payload") or {}
                publish_context = st.session_state.get("draft_context") or {}
                df_perf_publish, df_collective_perf_publish, df_qc_publish, df_samples_publish, df_athlete_session_publish = _prepare_publish_payloads(
                    df_perf_draft=st.session_state.df_perf,
                    df_collective_perf_draft=st.session_state.df_collective_perf,
                    df_qc_draft=st.session_state.df_qc,
                    df_samples_draft=st.session_state.df_samples,
                    df_athlete_session_draft=st.session_state.df_athlete_session,
                    session_payload=publish_payload,
                    genero=publish_context.get("genero", genero),
                    selecao=publish_context.get("selecao", selecao),
                    athlete_registry_df=athlete_registry_df,
                    base_dir=CLEANDATA_DIR,
                )
                progress_bar.progress(0.2, text="Dados preparados. A iniciar transferência...")
                progress_text.caption("Passo de preparação concluído.")

                stats = write_session_data(
                    df_perf_publish,
                    df_collective_perf_publish,
                    df_qc_publish,
                    df_samples_publish,
                    df_athlete_session_publish,
                    progress_callback=_on_db_progress,
                )
                session_sk_value = pd.NA
                for candidate_df in [df_perf_publish, df_collective_perf_publish, df_qc_publish, df_samples_publish, df_athlete_session_publish]:
                    if candidate_df is not None and not candidate_df.empty and "session_sk" in candidate_df.columns:
                        candidate_series = pd.to_numeric(candidate_df["session_sk"], errors="coerce").dropna()
                        if not candidate_series.empty:
                            session_sk_value = int(candidate_series.iloc[0])
                            break

                report_pdf_error = None
                try:
                    _generate_session_report_pdfs(
                        session_fingerprint=str(publish_payload.get("session_fingerprint") or ""),
                        selecao=publish_context.get("selecao", selecao),
                        contexto=publish_context.get("contexto", contexto),
                        jogo=publish_payload.get("jogo", adversario),
                        report_txt=st.session_state.get("report_txt", ""),
                        df_metrics=st.session_state.get("df_metrics", pd.DataFrame()),
                        athlete_name_map=_build_athlete_name_map(),
                        athlete_target_map=_build_athlete_target_map(),
                    )
                except Exception as pdf_exc:
                    report_pdf_error = str(pdf_exc)

                report_registry_error = None
                try:
                    save_session_report(
                        {
                            "session_fingerprint": publish_payload.get("session_fingerprint"),
                            "session_sk": session_sk_value,
                            "data": publish_payload.get("data_sessao", data_sessao),
                            "selecao": publish_context.get("selecao", selecao),
                            "genero": publish_context.get("genero", genero),
                            "contexto": publish_context.get("contexto", contexto),
                            "jogo": publish_payload.get("jogo", adversario),
                            "report_title": "Relatorio Tecnico",
                            "report_txt": st.session_state.get("report_txt", ""),
                        }
                    )
                except Exception as report_exc:
                    report_registry_error = str(report_exc)
                progress_bar.progress(1.0, text="Transferência concluída.")
                progress_text.caption("Passo finalizado: todos os envios terminaram.")

                stats_msg = "?? **Resumo da Integração:**\n\n"
                for table, table_stats in stats.items():
                    if table_stats is not None:
                        inserted = table_stats.get('inserted', 0)
                        updated = table_stats.get('updated', 0)
                        stats_msg += f"• **{table}**: {inserted} inseridos, {updated} atualizados\n"

                st.success("✅ Dados gravados com sucesso no Supabase.")
                if report_registry_error:
                    st.warning(
                        "Os dados da sessão foram publicados, mas o registo do relatório técnico não foi guardado. "
                        f"Detalhe: {report_registry_error}"
                    )
                if report_pdf_error:
                    st.warning(
                        "Os dados da sessão foram publicados, mas os PDFs do relatório não foram gerados. "
                        f"Detalhe: {report_pdf_error}"
                    )
                st.markdown(stats_msg)
                st.session_state.publish_success = True
                if not report_pdf_error:
                    st.switch_page("pages/Análise_Performance.py")
            except Exception as e:
                progress_bar.progress(1.0, text="Transferência interrompida.")
                progress_text.caption("A transferência foi interrompida por um erro.")
                st.error(f"❌ Erro ao gravar: {str(e)}")
    else:
        st.empty()


METRIC_INFO = {
    "duracao_min": {"unidade": "min", "definicao": "Duração útil da fase em minutos, calculada a partir dos intervalos temporais válidos.", "calculo": "Soma dos dt válidos convertida para minutos.", "interpretacao": "Representa o tempo efetivo de exposição analisado na fase."},
    "dist_m": {"unidade": "m", "definicao": "Distância total percorrida pelo atleta na fase.", "calculo": "Soma dos deslocamentos ponto a ponto em X_UTM/Y_UTM.", "interpretacao": "Mede o volume locomotor total da fase."},
    "m_min": {"unidade": "m/min", "definicao": "Distância relativa por minuto.", "calculo": "dist_m dividido por duracao_min.", "interpretacao": "Representa a intensidade média locomotora da fase."},
    "vmax_mps": {"unidade": "m/s", "definicao": "Velocidade m�xima instantânea estimada na fase.", "calculo": "M�ximo de dist�ncia por intervalo de tempo entre amostras v�lidas.", "interpretacao": "Representa o pico de velocidade do atleta na fase."},
    "peak_1m_m_min": {"unidade": "m", "definicao": "Maior distância percorrida em qualquer janela contínua de 60 segundos dentro da fase.", "calculo": "Maior distância acumulada em qualquer janela móvel de 60 s.", "interpretacao": "Representa o pico locomotor da fase; não é volume acumulado nem média."},
    "hsr_dist_m": {"unidade": "m", "definicao": "Distância percorrida acima do limiar de high-speed running.", "calculo": f"Soma da distância quando v >= {HSR_MPS:.1f} m/s.", "interpretacao": "Quantifica a exposição a corrida de alta velocidade."},
    "hsr_pct": {"unidade": "%", "definicao": "Percentagem da dist�ncia total realizada em HSR.", "calculo": "hsr_dist_m dividido por dist_m, multiplicado por 100.", "interpretacao": "Representa o peso relativo da alta velocidade no volume total."},
    "sprint_dist_m": {"unidade": "m", "definicao": "Distância percorrida acima do limiar de sprint.", "calculo": f"Soma da distância quando v >= {SPRINT_MPS:.1f} m/s.", "interpretacao": "Quantifica a exposição a corrida de sprint."},
    "n_sprints": {"unidade": "contagem", "definicao": "Número de episódios de sprint.", "calculo": f"Conta bouts consecutivos com v >= {SPRINT_MPS:.1f} m/s e duração mínima de {SPRINT_BOUT_MIN_S:.1f} s.", "interpretacao": "Conta sprints válidos e evita picos isolados como sprint real."},
    "n_acc_2_5": {"unidade": "contagem", "definicao": "Número de instantes com aceleração acima do threshold operacional.", "calculo": f"Conta amostras com aceleração >= {ACC_THR:.1f} m/só.", "interpretacao": "Reflete a exigência de ações de aceleração."},
    "n_dec_3_0": {"unidade": "contagem", "definicao": "Número de instantes com desaceleração abaixo do threshold operacional.", "calculo": f"Conta amostras com desaceleração <= {DEC_THR:.1f} m/só.", "interpretacao": "Reflete a exigência de ações de desaceleração e controlo neuromuscular."},
    "active_time_min": {"unidade": "min", "definicao": "Tempo ativo em movimento durante a fase.", "calculo": "Soma do tempo em que a velocidade estimada é >= 0.5 m/s.", "interpretacao": "Distingue exposição total de tempo efetivamente ativo."},
    "active_pct": {"unidade": "%", "definicao": "Percentagem do tempo da fase em atividade motora.", "calculo": "active_time_min dividido pela duração da fase, multiplicado por 100.", "interpretacao": "Permite comparar fases com diferente tempo de inatividade."},
    "hr_avg_bpm": {"unidade": "bpm", "definicao": "Frequência cardíaca média ao longo da fase.", "calculo": "Média ponderada pelo tempo dos valores HR válidos.", "interpretacao": "Resume a exigência cardiovascular média da fase."},
    "hr_peak_bpm": {"unidade": "bpm", "definicao": "Frequência cardíaca máxima observada na fase.", "calculo": "Máximo dos valores HR válidos.", "interpretacao": "Resume o pico cardiovascular da fase."},
    "hr_time_min": {"unidade": "min", "definicao": "Tempo com sinal de frequência cardíaca válido.", "calculo": "Soma dos intervalos temporais com HR válido.", "interpretacao": "Indica quanta exposição cardíaca entrou no cálculo das m�tricas de carga interna."},
    "hr_time_valid_pct": {"unidade": "%", "definicao": "Percentagem da fase coberta por sinal HR válido.", "calculo": "hr_time_min dividido por duracao_min, multiplicado por 100.", "interpretacao": "Ajuda a perceber a robustez das métricas de carga cardíaca."},
    "beats_total": {"unidade": "batimentos", "definicao": "Número estimado de batimentos acumulados na fase.", "calculo": "Integral da frequência cardíaca ao longo do tempo válido.", "interpretacao": "Proxy direta de carga interna cardiovascular."},
    "dist_per_beat_m": {"unidade": "m/bat", "definicao": "Distância percorrida por batimento cardíaco.", "calculo": "dist_m dividido por beats_total.", "interpretacao": "Proxy de eficiência locomotora por carga interna."},
    "hsr_per_beat_m": {"unidade": "m/bat", "definicao": "Distância HSR por batimento cardíaco.", "calculo": "hsr_dist_m dividido por beats_total.", "interpretacao": "Relaciona exposição a alta velocidade com custo cardíaco."},
    "sprint_per_beat_m": {"unidade": "m/bat", "definicao": "Distância de sprint por batimento cardíaco.", "calculo": "sprint_dist_m dividido por beats_total.", "interpretacao": "Relaciona exposição a sprint com custo cardíaco."},
    "sprints_per_1000_beats": {"unidade": "contagem/1000 bat", "definicao": "Número de sprints por 1000 batimentos.", "calculo": "n_sprints dividido por beats_total, multiplicado por 1000.", "interpretacao": "Expressa a frequência de sprints em função da carga cardíaca."},
    "acc_per_1000_beats": {"unidade": "contagem/1000 bat", "definicao": "Número de acelerações por 1000 batimentos.", "calculo": "n_acc_2_5 dividido por beats_total, multiplicado por 1000.", "interpretacao": "Relaciona ações de aceleração com a carga cardíaca."},
    "dec_per_1000_beats": {"unidade": "contagem/1000 bat", "definicao": "Número de desacelerações por 1000 batimentos.", "calculo": "n_dec_3_0 dividido por beats_total, multiplicado por 1000.", "interpretacao": "Relaciona ações de desaceleração com a carga cardíaca."},
    "external_load_score": {"unidade": "índice", "definicao": "índice proxy de carga externa que combina volume, intensidade, eventos, picos e tempo ativo.", "calculo": "Combinação ponderada de dist_m, m_min, HSR, sprint, eventos, picos e atividade.", "interpretacao": "Serve para comparação relativa de exigência externa entre fases, atletas ou sessões."},
    "total_load_score": {"unidade": "índice", "definicao": "índice proxy de carga total que combina carga externa e frequência cardíaca m�dia.", "calculo": "external_load_score multiplicado por hr_avg_bpm/100.", "interpretacao": "Serve para comparação relativa de exigência total quando existe HR válido."},
    "n_points": {"unidade": "contagem", "definicao": "Número de pontos válidos usados no cálculo das métricas.", "calculo": "Conta linhas com Time, X_UTM e Y_UTM válidos.", "interpretacao": "Quanto maior, mais robusta tende a ser a estimativa."},
    "pct_time_valid": {"unidade": "%", "definicao": "Percentagem de amostras v�lidas na fase.", "calculo": "Proporção de linhas com tempo e coordenadas válidos, multiplicada por 100.", "interpretacao": "Resume a completude do sinal disponível para cálculo."},
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
_render_phase_marker(bool(st.session_state.get("phase3_complete", False)))
with st.expander(
    _phase_title("3. Validação de Dados", bool(st.session_state.get("phase3_complete", False))),
    expanded=not st.session_state.get("phase3_complete", False),
):
    if not f_atleta:
        st.warning("?? Ainda não carregaste ficheiros de atletas. Algumas funcionalidades podem não estar disponíveis.")
    
    saved_fields_df = pd.DataFrame()
    auto_field_match = None
    alat = None
    alon = None
    if f_atleta:
        try:
            saved_fields_df = load_field_reference()
        except Exception:
            saved_fields_df = pd.DataFrame()
        try:
            alat, alon = get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)
        except Exception:
            alat, alon = None, None
        auto_field_match = _find_field_match_from_athletes(saved_fields_df, alat, alon)
    
    use_detected_field = False
    selected_detected_field_name = ""
    
    if auto_field_match:
        match_row = auto_field_match["row"]
        estadio_match = str(match_row.get("estadio") or "").strip() or "Campo sem nome"
        campo_local_match = str(match_row.get("campo_local") or "").strip()
        field_match_label = estadio_match if not campo_local_match else f"{estadio_match} - {campo_local_match}"
        dist_txt = f"{auto_field_match['athletes_m']:.1f} m" if pd.notna(auto_field_match.get("athletes_m")) else "-"
        if auto_field_match.get("match_strength") == "forte":
            st.success(
                f"Campo identificado automaticamente na BD: '{field_match_label}'. "
                f"A referencia geografica dos atletas esta muito proxima (~{dist_txt})."
            )
        else:
            st.info(f"Foi encontrado um campo provável: '{estadio_match}' ({campo_local_match}). A referencia dos atletas sugere este campo (~{dist_txt}).")
        auto_field_decision = st.radio(
            "Como queres continuar?",
            options=[
                "Usar o campo identificado automaticamente",
                "Continuar para identificacao manual do campo",
            ],
            index=0 if auto_field_match.get("match_strength") == "forte" else 1,
            help="Se preferires confirmar ou corrigir o campo, segue para a identificacao manual.",
        )
        use_detected_field = auto_field_decision == "Usar o campo identificado automaticamente"
        selected_detected_field_name = field_match_label
    else:
        st.warning(
            "Nao foi possivel identificar automaticamente um campo na base de dados. "
            "Segue para a identificacao manual."
        )
    
    metodo_campo = "Escolher um campo guardado anteriormente"
    f_campo = None
    if not use_detected_field:
        st.subheader("Identificacao Manual do Campo")
        st.caption("Escolhe uma das opcoes abaixo para identificar ou calibrar o campo manualmente.")
        metodo_campo = st.radio(
            "Como queres definir os 4 cantos?",
            options=["Upload (BL/BR/TL/TR)", "Pick no mapa (clicar 4 cantos)", "Escolher um campo guardado anteriormente"],
            index=0,
            help="Alternativa ao upload: usa um mapa satélite e clica nos 4 cantos do campo.",
        )
        st.caption("Se escolheres 'Pick no mapa', não precisas de carregar os 4 CSVs do campo.")
        f_campo = st.file_uploader(
            "Dados de CAMPO (BL, BR, TL, TR)", accept_multiple_files=True, type=["csv"]
        )
        if (
            metodo_campo == "Pick no mapa (clicar 4 cantos)"
            and st.session_state.last_metodo_campo != metodo_campo
        ):
            st.session_state.pick_corners = []
            st.session_state.pts_gps_picked = None
            st.session_state.pick_last_click_sig = None
        st.session_state.last_metodo_campo = metodo_campo
    else:
        st.session_state.last_metodo_campo = None
    
    have_upload_corners = bool(f_campo)
    have_picked_corners = st.session_state.get("pts_gps_picked") is not None
    
    if metodo_campo == "Upload (BL/BR/TL/TR)" and not have_upload_corners:
        st.info("?? Selecionaste 'Upload', mas ainda não carregaste os 4 CSVs do campo (BL/BR/TL/TR).")
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
    
                # pol�gono final (ret�ngulo ajustado)
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
                    tooltip="Ret�ngulo final (ajustado)",
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
        if use_detected_field and auto_field_match:
            row = auto_field_match["row"]
            pts_gps_recuperado = {
                "BL": [float(row["BL_lat"]), float(row["BL_lon"])],
                "BR": [float(row["BR_lat"]), float(row["BR_lon"])],
                "TL": [float(row["TL_lat"]), float(row["TL_lon"])],
                "TR": [float(row["TR_lat"]), float(row["TR_lon"])],
            }
            pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo_from_pts_gps(
                pts_gps_recuperado, int(row["epsg"]) if pd.notna(row.get("epsg")) else int(epsg_used)
            )
            st.success(f"✅ Campo '{selected_detected_field_name}' carregado automaticamente a partir da BD.")
    
        elif metodo_campo == "Pick no mapa (clicar 4 cantos)":
            pts_gps, (clat, clon), pts_utm, origin, R, angulo_rad, dist_x, dist_y = calibrar_campo_from_pts_gps(
                st.session_state.pts_gps_picked, int(epsg_used)
            )
    
        elif metodo_campo == "Escolher um campo guardado anteriormente":
            df_campos = saved_fields_df if saved_fields_df is not None and not saved_fields_df.empty else load_field_reference()
    
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
    
            # Recalibrar com os cantos recuperados sem ocupar o estado reservado ao pick manual.
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
    
        st.session_state.phase3_complete = True
    except Exception as e:
        st.session_state.phase3_complete = False
        st.error(f"Erro na calibração do campo: {e}")
        st.stop()
    except Exception as e:
        st.session_state.phase3_complete = False
        st.error(f"Erro na calibração do campo: {e}")
        st.stop()
    
    passed_geo, pct_ok, ok_list, fora_list, geo_errors = geo_validacao_por_atleta(
        f_atleta, clat, clon, float(raio_validacao_m), int(
            amostra_geo_n), float(min_pct_atletas_ok)
    )
    n_ok_geo = len({a for a, _ in ok_list})
    n_fora_geo = len({a for a, _ in fora_list})
    n_avaliados_geo = n_ok_geo + n_fora_geo
    n_total_geo = len(set([get_atleta_id(f.name) for f in f_atleta]))
    n_erros_geo = len(geo_errors)

# Validação geográfica permanece ativa em background, sem bloco visual dedicado.
# Campo (usa helper local cacheado, que era o comportamento funcional anterior)
cidade_campo, pais_campo = _reverse_geocode_city_country(clat, clon)

# Atletas (centro estimado)
alat, alon = get_atletas_centroid_latlon(f_atleta, amostra_n=amostra_geo_n)

cidade_atl, pais_atl = None, None
if alat is not None and alon is not None:
    cidade_atl, pais_atl = _reverse_geocode_city_country(alat, alon)

campo_local = ", ".join([p for p in [cidade_campo, pais_campo] if p]) or "—"
atletas_parts = [p for p in [cidade_atl, pais_atl] if p]
atletas_local = ", ".join(atletas_parts) if atletas_parts else "—"

field_save_placeholder = st.container()

athlete_status = {}
atletas_validos = 0
submission_valid = False
audit_data = {}
for f in f_atleta:
    aid = get_atleta_id(f.name)
    audit_data.setdefault(aid, [])
    audit_data[aid].append(infer_fase(f.name))

athlete_status, atletas_validos, submission_valid = _evaluate_athlete_submission(audit_data, contexto)
phase5_complete = bool(submission_valid) and not st.session_state.get("athlete_registry_expanded", True)

_render_phase_marker(phase5_complete)
with st.expander(
    _phase_title("5. Auditoria de Atletas", phase5_complete),
    expanded=(not submission_valid) or st.session_state.get("athlete_registry_expanded", True),
):
    rows = []
    for aid in sorted(
        audit_data.keys(),
        key=lambda x: int(re.search(r"\d+", x).group()
                          ) if re.search(r"\d+", x) else 0,
    ):
        status_info = athlete_status.get(aid, {})
        fases = status_info.get("fases", [])
        is_valid = status_info.get("is_valid", False)
        criterio = status_info.get("criterio", "")
        rows.append(
            {
                "ID Atleta": aid,
                "Ficheiros": len(fases),
                "Critério": criterio,
                "Estado": "OK" if is_valid else "INCOMPLETO",
                "Fases": ", ".join(sorted(set(fases))),
            }
        )
    with st.expander("Auditoria de atletas", expanded=False):
        st.table(pd.DataFrame(rows))
        if contexto == "Jogo":
            st.write(
                f"**Regra de Jogo:** 1P e 2P são obrigatórios por atleta; Warm-Up é opcional. Válidos: {atletas_validos} / {len(audit_data)}"
            )
        else:
            st.write(
                f"**Regra de Treino:** 1 ficheiro por atleta. Válidos: {atletas_validos} / {len(audit_data)}"
            )
    if f_atleta:
        with st.expander("Relação com a Base de Dados", expanded=st.session_state.get("athlete_registry_expanded", True)):
            athlete_registry_df, athlete_registry_error = _build_athlete_registry_editor(
                f_atleta,
                genero,
                selecao,
            )
            if athlete_registry_error:
                st.session_state.athlete_registry_expanded = True
                st.warning(athlete_registry_error)
            else:
                st.session_state.athlete_registry_expanded = False
    if not submission_valid:
        st.warning("A submissão não cumpre as condições mínimas definidas para este contexto.")

if not submission_valid:
    if contexto == "Jogo":
        st.warning(
            "A exportação está desativada porque, em contexto de Jogo, cada atleta tem de ter pelo menos 1P e 2P. O ficheiro de Warm-Up é opcional."
        )
    else:
        st.warning(
            "A exportação está desativada porque, em contexto de Treino, cada atleta tem de ter exatamente 1 ficheiro."
        )
    st.stop()

if not passed_geo:
    st.warning(
        f"A exportação está desativada porque só {n_ok_geo}/{n_avaliados_geo} atletas avaliados ficaram dentro do raio configurado. Os ficheiros sem GPS valido ({n_erros_geo}) não entram nesta percentagem. O mínimo configurado é {min_pct_atletas_ok*100:.0f}%."
    )
    st.stop()

# -- Obter e guardar campo -- #

# Dados do campo
suggested_field_name = _suggest_field_name(clat, clon, cidade_campo, pais_campo)
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

field_reference_df = pd.DataFrame(field_data)
current_field_sig = hashlib.md5(
    (
        f"{round(float(clat), 6)}|{round(float(clon), 6)}|"
        f"{round(float(dist_x), 2)}|{round(float(dist_y), 2)}|{int(epsg_used)}"
    ).encode("utf-8")
).hexdigest()
existing_field_match = None
try:
    saved_fields_df = load_field_reference()
    athletes_center = (alat, alon) if alat is not None and alon is not None else None
    existing_field_match = _find_existing_field_match(field_reference_df, saved_fields_df, athletes_center=athletes_center)
except Exception:
    existing_field_match = None

# Pop Up para Guardar campo na base de dados
if False:
    pass
    """
with st.sidebar.popover('💾 Guardar Campo'):

    existing_name = ""
    existing_obs = ""
    if existing_field_match:
        existing_name = str(existing_field_match["row"].get("estadio") or "").strip()
        existing_obs = str(existing_field_match["row"].get("obs") or "").strip()

    estadio = st.text_input('Adiciona o nome do estadio!', value=existing_name)
    obs = st.text_input('Adiciona uma observação!', value=existing_obs)

    field_data['estadio'] = estadio
    field_data['obs'] = obs
    if st.button("Guardar"):
        campo_df = pd.DataFrame(field_data)
        save_field_reference(campo_df)
        if existing_field_match:
            st.success("Campo guardado/atualizado com sucesso na referencia de campos.")
        else:
            st.success("Campo guardado com sucesso na referencia de campos.")
    """

if st.session_state.field_save_flash_msg:
    st.toast(st.session_state.field_save_flash_msg)
    st.session_state.field_save_flash_msg = None

show_field_save_block = (
    not use_detected_field
    and st.session_state.field_save_completed_sig != current_field_sig
)

if show_field_save_block:
    with field_save_placeholder:
        st.subheader("Guardar Campo")

        existing_name = ""
        if existing_field_match:
            existing_name = str(existing_field_match["row"].get("estadio") or "").strip()

        default_field_name = existing_name or suggested_field_name
        save_col1, save_col2 = st.columns([1.2, 1.0])
        with save_col1:
            estadio = st.text_input("Nome do campo", value=default_field_name)
        with save_col2:
            st.text_input("Sugestao", value=suggested_field_name, disabled=True)

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.text_input("Cidade", value=str(cidade_campo or ""), disabled=True)
        with meta_col2:
            st.text_input("Pais", value=str(pais_campo or ""), disabled=True)

        if existing_field_match:
            st.caption("Foi encontrado um campo semelhante na BD.")

        field_data["estadio"] = estadio
        field_data["obs"] = ""
        if st.button("Guardar Campo na BD"):
            campo_df = pd.DataFrame(field_data)
            save_field_reference(campo_df)
            st.session_state.field_save_completed_sig = current_field_sig
            if existing_field_match:
                st.session_state.field_save_flash_msg = "Campo guardado/atualizado com sucesso na referencia de campos."
            else:
                st.session_state.field_save_flash_msg = "Campo guardado com sucesso na referencia de campos."
            st.rerun()

_render_phase_marker(bool(st.session_state.process_done))
with st.expander(_phase_title("6. Relatório de Jogo | Treino", bool(st.session_state.process_done)), expanded=bool(st.session_state.process_done)):
    btn = st.button("⚙️ Processar e Gerar Relatório", type="primary", key="btn_process_report")
    phase6_content = st.container()

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
                f"Validação geográfica falhou: {n_ok_geo}/{n_avaliados_geo} atletas avaliados dentro do raio configurado (mínimo {min_pct_atletas_ok*100:.0f}%). Ficheiros sem GPS valido: {n_erros_geo}. O processamento foi interrompido."
            )
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
                    "? N�o foi possível gerar ficheiros tempor�rios (verifica colunas Time/Lat/Lon e nomes).")
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
            total_sample_gaps = 0

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
                if "_sample_gaps_corrigidos" in df_sync.columns and df_sync["_sample_gaps_corrigidos"].notna().any():
                    try:
                        total_sample_gaps += int(
                            df_sync["_sample_gaps_corrigidos"].dropna().iloc[0])
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
                        # Fase não jogada / não submetida -> NA (não é falha de qualidade)
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
                met_total["player_load"] = sum(
                    float(fase_mets.get(f, {}).get("player_load", 0.0))
                    if pd.notna(fase_mets.get(f, {}).get("player_load", np.nan))
                    else 0.0
                    for f in fases_target
                )
                met_total["rhie_bouts"] = sum(
                    fase_mets.get(f, {}).get("rhie_bouts", 0) or 0 for f in fases_target
                )
                met_total["rhie_actions"] = sum(
                    fase_mets.get(f, {}).get("rhie_actions", 0) or 0 for f in fases_target
                )
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

                met_total["hr_time_min"] = sum(
                    fase_mets.get(f, {}).get("hr_time_min", 0.0) or 0.0
                    for f in fases_target
                )
                met_total["beats_total"] = sum(
                    fase_mets.get(f, {}).get("beats_total", 0.0)
                    for f in fases_target
                    if pd.notna(fase_mets.get(f, {}).get("beats_total", np.nan))
                )
                met_total["hr_avg_bpm"] = (
                    met_total["beats_total"] / met_total["hr_time_min"]
                    if met_total["hr_time_min"] > 0
                    else np.nan
                )
                met_total["hr_peak_bpm"] = max(
                    (fase_mets.get(f, {}).get("hr_peak_bpm", np.nan) for f in fases_target),
                    default=np.nan,
                )
                met_total["hr_time_valid_pct"] = (
                    met_total["hr_time_min"] / met_total["duracao_min"] * 100.0
                    if met_total["duracao_min"] > 0
                    else np.nan
                )

                for zone_prefix in [
                    "zone1_walk",
                    "zone2_jog",
                    "zone3_run",
                    "zone4_hsr",
                    "zone5_sprint",
                ]:
                    met_total[f"{zone_prefix}_time_min"] = sum(
                        fase_mets.get(f, {}).get(f"{zone_prefix}_time_min", 0.0) or 0.0
                        for f in fases_target
                    )
                    met_total[f"{zone_prefix}_dist_m"] = sum(
                        fase_mets.get(f, {}).get(f"{zone_prefix}_dist_m", 0.0) or 0.0
                        for f in fases_target
                    )

                for peak_metric in [
                    "peak_dist_1m_m", "peak_dist_3m_m", "peak_dist_5m_m",
                    "peak_hsr_1m_m", "peak_hsr_3m_m", "peak_hsr_5m_m",
                    "peak_sprint_1m_m", "peak_sprint_3m_m", "peak_sprint_5m_m",
                    "peak_acc_actions_1m", "peak_acc_actions_3m", "peak_acc_actions_5m",
                    "peak_hi_actions_1m", "peak_hi_actions_3m", "peak_hi_actions_5m",
                    "trimp_banister", "trimp_per_min",
                ]:
                    met_total[peak_metric] = max(
                        (fase_mets.get(f, {}).get(peak_metric, np.nan) for f in fases_target),
                        default=np.nan,
                    )
                met_total.update(derive_load_metrics(met_total))

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
            (
                df_metrics,
                df_perf,
                df_collective_perf,
                df_qc,
                df_samples,
                df_athlete_session,
                df_tracking,
            ) = _build_draft_exports(
                df_metrics,
                out_files,
                session_payload,
                field_data,
            )
            save_draft_outputs(
                df_perf=df_perf,
                df_collective_perf=df_collective_perf,
                df_qc=df_qc,
                df_samples=df_samples,
                df_athlete_session=df_athlete_session,
                df_tracking=df_tracking,
                base_dir=CLEANDATA_DIR,
            )

            df_time_audit = pd.DataFrame(audit_time_rows)
            st.session_state.df_time_audit = df_time_audit
            manual_metricas_txt = _build_manual_metricas_txt()

            status.update(label="Construção do relatório...", state="running")
            # Build report (rotação mantida)
            rot_deg = float(np.degrees(angulo_rad))
            report_lines = []
            report_lines.append(
                "FPF Performance Hub - Relatório de Validação e Normalização")
            report_lines.append("=" * 70)

            report_lines.append("Dados da Sessóo")
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
            report_lines.append(f"Rotação aplicada:    {rot_deg:.2f}° (alinhamento BL->BR com eixo X)")

            report_lines.append("-" * 70)
            report_lines.append("Validação geográfica")
            report_lines.append(
                f"  Raio: {raio_validacao_m:.0f} m | Amostra: { amostra_geo_n} linhas/atleta | % OK: {pct_ok*100:.0f}% "
                f"(mínimo {min_pct_atletas_ok*100:.0f}%)"
            )
            report_lines.append(
                f"  Dentro do raio: {len({a for a, _ in ok_list})} atletas | "
                f"Fora: {len({a for a, _ in fora_list})} atletas | Sem GPS valido: {len(geo_errors)}"
            )

            report_lines.append("-" * 70)
            report_lines.append("Auditoria de atletas (submissão)")
            report_lines.append(
                (
                    f"  Contexto: Jogo | Regra: 1P e 2P obrigatórios; Warm-Up opcional | Atletas válidos: {atletas_validos} / {len(audit_data)}"
                    if contexto == "Jogo"
                    else f"  Contexto: Treino | Regra: 1 ficheiro por atleta | Atletas válidos: {atletas_validos} / {len(audit_data)}"
                )
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

            if df_metrics is not None and not df_metrics.empty:
                report_lines.append("-" * 70)
                report_lines.extend(
                    _build_historical_reference_report_lines(
                        df_metrics=df_metrics,
                        selecao=selecao,
                        contexto=contexto,
                    )
                )

            report_lines.append("-" * 70)
            report_lines.append("Qualidade do Sinal GPS")
            report_lines.append(f"  Micro-gaps corrigidos (≤1 amostra consecutiva): {total_micro_gaps}")
            report_lines.append(f"  Gaps temporais corrigidos no SYNC (≤1 amostra): {total_sample_gaps}")

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
            store_draft_results(
                df_metrics=df_metrics,
                report_txt=report_txt,
                manual_metricas_txt=manual_metricas_txt,
                df_perf=df_perf,
                df_collective_perf=df_collective_perf,
                df_qc=df_qc,
                df_samples=df_samples,
                df_athlete_session=df_athlete_session,
                df_time_audit=df_time_audit,
                draft_session_payload=session_payload,
                draft_context={
                    "genero": genero,
                    "selecao": selecao,
                    "contexto": contexto,
                },
            )
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
    with phase6_content:
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

        # 4. Ordenação por atleta + fase (ordem personalizada)
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

        # 5. Organização vertical por blocos e famílias
        id_cols = [c for c in [col_inicio, "fase"] if c in df_display.columns]
        metric_groups = _build_metric_groups()
        performance_metric_groups = _build_performance_metric_groups()
        technical_metric_groups = _build_technical_metric_groups()

        ordered_metric_cols = []
        for familias in metric_groups.values():
            for cols in familias.values():
                ordered_metric_cols.extend([c for c in cols if c in df_display.columns])

        ordered_metric_cols = list(dict.fromkeys(ordered_metric_cols))

        df_export = df_display[id_cols + ordered_metric_cols].copy()
        df_display_ui = format_metrics_display_dataframe(df_display)
        totals_by_athlete = _build_totals_by_athlete(df_metrics)
        if totals_by_athlete is None:
            totals_by_athlete = pd.DataFrame()
        totals_by_athlete_ui = format_metrics_display_dataframe(totals_by_athlete)
        if totals_by_athlete_ui is None:
            totals_by_athlete_ui = pd.DataFrame()
        athlete_name_map = _build_athlete_name_map()
        athlete_target_map = _build_athlete_target_map()

        st.markdown("**Métricas Coletivas**")
        for idx, (familia, cols) in enumerate(performance_metric_groups.items()):
            collective_family_df = _build_collective_phase_totals(df_metrics, cols)
            collective_cols_presentes = [c for c in cols if c in collective_family_df.columns]
            if not collective_cols_presentes:
                continue
            with st.expander(familia, expanded=(idx == 0)):
                rendered_collective = False
                for metric_col in collective_cols_presentes:
                    metric_spec = COLLECTIVE_PROFILE_METRIC_MAP.get(metric_col, (metric_col, metric_col, _metric_user_label(metric_col)))
                    current_metric_col, _, display_label = metric_spec
                    st.markdown(f"**{display_label}**")
                    reference_phase_df = _load_historical_collective_phase_reference(
                        selecao=selecao,
                        contexto=contexto,
                        metric_col=metric_col,
                    )
                    rendered = _render_phase_metric_bar_chart(
                        collective_family_df,
                        metric_col,
                        phase_col="fase",
                        reference_df=reference_phase_df,
                        comparison_col=current_metric_col,
                        yaxis_label=display_label,
                        chart_key=f"collective_{familia}_{metric_col}",
                    )
                    if not rendered:
                        st.info("Sem dados numéricos disponíveis para esta métrica.")
                    rendered_collective = rendered_collective or rendered
                if not rendered_collective:
                    st.info("Sem dados disponíveis para este grupo.")
                else:
                    st.caption("Verde: sessão atual | Cinza: média equipa da mesma seleção e contexto.")

        st.markdown("**Métricas Individuais**")
        for idx, (familia, cols) in enumerate(performance_metric_groups.items()):
            cols_presentes = [c for c in cols if c in totals_by_athlete.columns]
            if not cols_presentes:
                continue
            with st.expander(familia, expanded=(idx == 0)):
                rendered_any = False
                for metric_col in cols_presentes:
                    metric_spec = INDIVIDUAL_PROFILE_METRIC_MAP.get(metric_col, (metric_col, metric_col, _metric_user_label(metric_col)))
                    current_metric_col, _, display_label = metric_spec
                    st.markdown(f"**{display_label}**")
                    reference_metric_df = _load_historical_individual_metric_reference(
                        selecao=selecao,
                        contexto=contexto,
                        metric_col=metric_col,
                    )
                    rendered = _render_metric_bar_chart(
                        totals_by_athlete,
                        metric_col,
                        athlete_col=col_inicio,
                        reference_df=reference_metric_df,
                        chart_key=f"perf_{familia}_{metric_col}",
                        athlete_name_map=athlete_name_map,
                        athlete_target_map=athlete_target_map,
                        comparison_col=current_metric_col,
                        yaxis_label=display_label,
                    )
                    if not rendered:
                        st.info("Sem dados numéricos disponíveis para esta métrica.")
                    rendered_any = rendered_any or rendered
                if not rendered_any:
                    st.info("Sem dados disponíveis para este grupo.")
                else:
                    st.caption("Verde: sessão atual | Cinza: média individual do próprio atleta.")

        st.empty()

        _render_phase_marker(bool(st.session_state.get("publish_success", False)))
        with st.expander(
            _phase_title("7. Base de Dados | Relatório Técnico", bool(st.session_state.get("publish_success", False))),
            expanded=True,
        ):
            final_action_cols = st.columns(3, gap="medium")
            with final_action_cols[0]:
                publish_action = st.button("Publicar na Base de Dados", key="btn_final_publish", use_container_width=True)
            with final_action_cols[1]:
                report_button_label = "Ocultar Relatório Técnico" if st.session_state.get("show_final_report_section", False) else "Mostrar Relatório Técnico"
                toggle_report = st.button(report_button_label, key="btn_toggle_final_report", use_container_width=True)
            with final_action_cols[2]:
                clear_results = st.button("Limpar Resultados", key="btn_clear_final_results", use_container_width=True)

            if toggle_report:
                st.session_state.show_final_report_section = not st.session_state.get("show_final_report_section", False)
                st.rerun()

            _render_database_integration_section(
                f_atleta,
                genero,
                selecao,
                data_sessao,
                adversario,
                publish_clicked=publish_action,
            )

        if st.session_state.get("show_final_report_section", False) and report_txt:
            with st.expander("Relatório Técnico", expanded=False):
                report_title, report_sections = _parse_report_sections(report_txt)
                report_sections = _order_technical_report_sections(report_sections)
                if report_title:
                    st.caption(report_title)

                if report_sections:
                    for section_title, section_body in report_sections:
                        with st.expander(section_title, expanded=False):
                            st.code(section_body or "Sem dados nesta secção.", language="text")
                else:
                    st.code(report_txt, language="text")

                for section_title, cols in technical_metric_groups.items():
                    cols_presentes = [c for c in cols if c in totals_by_athlete.columns]
                    if not cols_presentes:
                        continue
                    with st.expander(section_title, expanded=False):
                        fallback_cols = []
                        rendered_any = False
                        for metric_col in cols_presentes:
                            numeric_series = pd.to_numeric(totals_by_athlete.get(metric_col), errors="coerce")
                            if numeric_series.notna().any():
                                st.markdown(f"**{_metric_user_label(metric_col)}**")
                                rendered = _render_metric_bar_chart(
                                    totals_by_athlete,
                                    metric_col,
                                    athlete_col=col_inicio,
                                    chart_key=f"tech_{section_title}_{metric_col}",
                                    athlete_name_map=athlete_name_map,
                                    athlete_target_map=athlete_target_map,
                                )
                                rendered_any = rendered_any or rendered
                            else:
                                fallback_cols.append(metric_col)

                        if fallback_cols:
                            st.markdown("**Métricas não numéricas**")
                            fallback_df = totals_by_athlete_ui[[col_inicio] + fallback_cols].copy()
                            fallback_df = fallback_df.rename(columns={col: _metric_user_label(col) for col in fallback_cols})
                            st.dataframe(
                                fallback_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                        if not rendered_any and not fallback_cols:
                            st.info("Sem dados disponíveis para este grupo.")

                if "df_time_audit" in st.session_state and st.session_state.df_time_audit is not None:
                    dfta = st.session_state.df_time_audit
                    if isinstance(dfta, pd.DataFrame) and not dfta.empty:
                        with st.expander("Auditoria de Timestamp", expanded=False):
                            st.dataframe(dfta, use_container_width=True, hide_index=True)
        elif st.session_state.get("show_final_report_section", False):
            st.warning("Sem relatório para mostrar (processa novamente).")

        # (Opcional) botão para limpar resultados
        if clear_results:
            clear_draft_session_state()
            st.rerun()



