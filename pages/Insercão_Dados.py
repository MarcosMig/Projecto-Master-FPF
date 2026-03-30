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
    read_table,
    save_field_to_parquet,
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

    df_existing = read_table("performance_metrics", filters=filters)
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

    return df_metrics_draft, df_perf, df_qc, df_samples, df_athlete_session, df_tracking


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
        return None, f"Faltam associaÃ§Ãµes para os atletas: {', '.join(missing_expected)}."

    return registry_df, None


def _prepare_publish_payloads(
    *,
    df_perf_draft: pd.DataFrame,
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
                f"NÃ£o foi possÃ­vel mapear todos os atletas para publicaÃ§Ã£o: {', '.join(missing_ids)}."
            )
        mapped["session_sk"] = session_sk
        mapped["athlete_sk"] = mapped["atleta_id"].astype(str).map(athlete_map)
        return mapped

    df_perf_publish = _map_for_publish(df_perf_draft)
    df_qc_publish = _map_for_publish(df_qc_draft)
    df_samples_publish = _map_for_publish(df_samples_draft)
    df_athlete_session_publish = _map_for_publish(df_athlete_session_draft)

    df_perf_publish = df_perf_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "phase_id", "fase", "data", "selecao", "genero",
            "contexto", "jogo", "duracao_min", "dist_m", "m_min", "vmax_mps", "peak_1m_m_min",
            "hsr_dist_m", "hsr_pct", "sprint_dist_m", "n_sprints", "n_acc_2_5", "n_dec_3_0",
            "active_time_min", "active_pct",
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
    df_athlete_session_publish = df_athlete_session_publish[
        [
            "session_sk", "athlete_sk", "atleta_id", "participou_warmup", "participou_1p",
            "participou_2p", "fases_disponiveis", "n_samples", "tem_hr", "processado_em",
        ]
    ].copy()

    return df_perf_publish, df_qc_publish, df_samples_publish, df_athlete_session_publish

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="FPF UTM Engine v16", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      div.stButton > button,
      div.stDownloadButton > button {
        white-space: nowrap;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
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
    selecao = st.selectbox("Seleção", options=SELECTION_OPTIONS, index=0 if SELECTION_OPTIONS else None)
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
    st.caption("A associaÃ§Ã£o Ã  base de dados sÃ³ Ã© pedida no fim, se escolheres publicar.")
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

if (
    metodo_campo == "Pick no mapa (clicar 4 cantos)"
    and st.session_state.last_metodo_campo != metodo_campo
):
    st.session_state.pick_corners = []
    st.session_state.pts_gps_picked = None
    st.session_state.pick_last_click_sig = None

st.session_state.last_metodo_campo = metodo_campo

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
        "Picos de Fase": [
            "vmax_mps",
            "peak_1m_m_min",
        ],
        f"HSR ≥ {HSR_MPS:.1f} m/s": [
            "hsr_dist_m",
            "hsr_pct",
        ],
        f"Sprint ≥ {SPRINT_MPS:.1f} m/s": [
            "sprint_dist_m",
            "n_sprints",
        ],
        f"Acc ≥ {ACC_THR:.1f} m/s²": [
            "n_acc_2_5",
        ],
        f"Dec ≤ {DEC_THR:.1f} m/s²": [
            "n_dec_3_0",
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
    ]

    for col in [duration_col] + per90_source_cols:
        if col in totals_df.columns:
            totals_df[col] = pd.to_numeric(totals_df[col], errors="coerce")

    duration = pd.to_numeric(totals_df.get(duration_col), errors="coerce")
    valid_duration = duration.notna() & (duration > 0)

    rename_map = {
        "dist_m": "dist_m_90",
        "hsr_dist_m": "hsr_dist_m_90",
        "sprint_dist_m": "sprint_dist_m_90",
        "active_time_min": "active_time_min_90",
        "n_sprints": "n_sprints_90",
        "n_acc_2_5": "n_acc_2_5_90",
        "n_dec_3_0": "n_dec_3_0_90",
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
        "vmax_mps",
        "peak_1m_m_min",
    ]
    available_cols = [col for col in preferred_cols if col in totals_df.columns]
    totals_df = totals_df[available_cols].copy()
    return round_metrics_dataframe(totals_df)


def _order_technical_report_sections(report_sections):
    ordered_titles = [
        "Dados da Sessão",
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
        df_campos = load_field_reference()

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
n_ok_geo = len({a for a, _ in ok_list})
n_fora_geo = len({a for a, _ in fora_list})
n_avaliados_geo = n_ok_geo + n_fora_geo
n_total_geo = len(set([get_atleta_id(f.name) for f in f_atleta]))
n_erros_geo = len(geo_errors)

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
st.markdown(f"**% Atletas dentro do raio (avaliados):** {pct_ok*100:.0f}%")
st.caption(
    f"{n_ok_geo}/{n_avaliados_geo} atletas avaliados ficaram dentro do raio, {n_fora_geo} fora do raio, {n_erros_geo} com erro de leitura."
)

st.markdown("---")


if passed_geo:
    st.success("✅ Validação geográfica aprovada.")
else:
    st.error(
        f"❌ Validação geográfica falhou: apenas {n_ok_geo}/{n_avaliados_geo} atletas avaliados ficaram dentro do raio configurado."
    )


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

athlete_status, atletas_validos, submission_valid = _evaluate_athlete_submission(audit_data, contexto)

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
if not submission_valid:
    st.warning("A submissão não cumpre as condições mínimas definidas para este contexto.")

st.divider()

# Normalization + export
st.header("Normalização | Calculo Métricas")

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
        f"A exportação está desativada porque só {n_ok_geo}/{n_avaliados_geo} atletas avaliados ficaram dentro do raio configurado. Os ficheiros com erro de leitura ({n_erros_geo}) não entram nesta percentagem. O mínimo configurado é {min_pct_atletas_ok*100:.0f}%."
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

btn_row = st.columns([1.75, 0.8, 0.8, 0.8, 0.8, 1.05])
with btn_row[0]:
    btn = st.button("⚙️ Processar e Gerar Relatório", type="primary")

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
                f"Validação geográfica falhou: {n_ok_geo}/{n_avaliados_geo} atletas avaliados dentro do raio configurado (mínimo {min_pct_atletas_ok*100:.0f}%). Ficheiros com erro de leitura: {n_erros_geo}. O processamento foi interrompido."
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
            (
                df_metrics,
                df_perf,
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
            store_draft_results(
                df_metrics=df_metrics,
                report_txt=report_txt,
                manual_metricas_txt=manual_metricas_txt,
                df_perf=df_perf,
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
    performance_metric_groups = _build_performance_metric_groups()
    technical_metric_groups = _build_technical_metric_groups()

    ordered_metric_cols = []
    for familias in metric_groups.values():
        for cols in familias.values():
            ordered_metric_cols.extend([c for c in cols if c in df_display.columns])

    ordered_metric_cols = list(dict.fromkeys(ordered_metric_cols))

    df_export = df_display[id_cols + ordered_metric_cols].copy()
    df_display_ui = format_metrics_display_dataframe(df_display)
    df_totals_by_athlete = _build_totals_by_athlete(df_display)
    df_totals_by_athlete_ui = format_metrics_display_dataframe(df_totals_by_athlete)

    st.subheader("Métricas Performance")
    for familia, cols in performance_metric_groups.items():
        cols_presentes = [c for c in cols if c in df_display.columns]
        if not cols_presentes:
            continue
        with st.expander(f"Performance | {familia}", expanded=False):
            st.dataframe(
                df_display_ui[id_cols + cols_presentes],
                use_container_width=True,
                hide_index=True,
            )

    # 6️⃣ Downloads
    action_cols = st.columns([1.15, 1.05, 1.8])
    with action_cols[0]:
        st.download_button(
            "⬇️ Download Métricas (.csv)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="metricas_individuais_FPF.csv",
            mime="text/csv",
        )

    if not df_totals_by_athlete.empty:
        st.subheader("Totais por Atleta")
        st.caption("Resumo da fase Total com metricas absolutas e normalizadas por 90 minutos.")
        st.dataframe(
            df_totals_by_athlete_ui,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Totais por Atleta (.csv)",
            data=df_totals_by_athlete.to_csv(index=False).encode("utf-8"),
            file_name="totais_por_atleta_90.csv",
            mime="text/csv",
            key="download_totals_by_athlete",
        )


    st.subheader("Integração na Base de Dados")

    if st.session_state.get("df_perf") is not None and not st.session_state.df_perf.empty:
        st.info("Os dados jÃ¡ estÃ£o em modo draft. Podes terminar em visualizaÃ§Ã£o/download ou avanÃ§ar para publicaÃ§Ã£o.")
        publish_mode = st.radio(
            "Destino final desta sessÃ£o",
            options=["Visualizar / Download", "Publicar na base de dados"],
            horizontal=True,
            key="publish_mode",
        )
        if publish_mode == "Visualizar / Download":
            st.info("✅ Dados processados e prontos para serem integrados no Supabase.")
        
        else:
            st.caption("Para publicar, cada atleta do ficheiro tem de ser associado a uma ficha da base de dados.")
            athlete_registry_df = None
            athlete_registry_error = None

            if f_atleta:
                with st.expander("Ficha de Atletas para PublicaÃ§Ã£o", expanded=True):
                    athlete_registry_df, athlete_registry_error = _build_athlete_registry_editor(
                        f_atleta,
                        genero,
                        selecao,
                    )
            else:
                athlete_registry_error = "Carrega os ficheiros de atletas para conseguires publicar esta sessÃ£o."

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
            if st.button("💾 Gravar na Base", key="btn_save_duckdb"):
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
                    df_perf_publish, df_qc_publish, df_samples_publish, df_athlete_session_publish = _prepare_publish_payloads(
                        df_perf_draft=st.session_state.df_perf,
                        df_qc_draft=st.session_state.df_qc,
                        df_samples_draft=st.session_state.df_samples,
                        df_athlete_session_draft=st.session_state.df_athlete_session,
                        session_payload=publish_payload,
                        genero=publish_context.get("genero", genero),
                        selecao=publish_context.get("selecao", selecao),
                        athlete_registry_df=athlete_registry_df,
                        base_dir=CLEANDATA_DIR,
                    )
                    progress_bar.progress(0.2, text="Dados preparados. A iniciar transferÃªncia...")
                    progress_text.caption("Passo de preparaÃ§Ã£o concluÃ­do.")

                    stats = write_session_data(
                        df_perf_publish,
                        df_qc_publish,
                        df_samples_publish,
                        df_athlete_session_publish,
                        progress_callback=_on_db_progress,
                    )
                    progress_bar.progress(1.0, text="Transferência concluída.")
                    progress_text.caption("Passo finalizado: todos os envios terminaram.")
                    
                    # Build stats message
                    stats_msg = "📊 **Resumo da Integração:**\n\n"
                    for table, table_stats in stats.items():
                        if table_stats is not None:
                            inserted = table_stats.get('inserted', 0)
                            updated = table_stats.get('updated', 0)
                            stats_msg += f"• **{table}**: {inserted} inseridos, {updated} atualizados\n"
                    
                    st.success("✅ Dados gravados com sucesso no Supabase.")
                    st.markdown(stats_msg)
                except Exception as e:
                    progress_bar.progress(1.0, text="Transferência interrompida.")
                    progress_text.caption("A transferência foi interrompida por um erro.")
                    st.error(f"❌ Erro ao gravar: {str(e)}")
    else:
        st.warning("📊 Processa a sessão primeiro para gravar os dados.")

    st.subheader("Relatório Técnico")
    if report_txt:
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
            cols_presentes = [c for c in cols if c in df_display.columns]
            if not cols_presentes:
                continue
            with st.expander(section_title, expanded=False):
                st.dataframe(
                    df_display_ui[id_cols + cols_presentes],
                    use_container_width=True,
                    hide_index=True,
                )

        if "df_time_audit" in st.session_state and st.session_state.df_time_audit is not None:
            dfta = st.session_state.df_time_audit
            if isinstance(dfta, pd.DataFrame) and not dfta.empty:
                with st.expander("Auditoria de Timestamp", expanded=False):
                    st.dataframe(dfta, use_container_width=True, hide_index=True)

        report_action_cols = st.columns([1.35, 1.0, 3.65])
        with report_action_cols[0]:
            st.download_button(
                "⬇️ Download Relatório (.txt)",
                data=report_txt.encode("utf-8"),
                file_name="relatorio_FPF.txt",
                mime="text/plain",
            )
        with report_action_cols[1]:
            clear_results = st.button("🧹 Limpar resultados")
    else:
        st.warning("Sem relatório para mostrar (processa novamente).")
        report_action_cols = st.columns([1.0, 5.0])
        with report_action_cols[0]:
            clear_results = st.button("🧹 Limpar resultados")

    # (Opcional) botão para limpar resultados
    if clear_results:
        clear_draft_session_state()
        st.rerun()
