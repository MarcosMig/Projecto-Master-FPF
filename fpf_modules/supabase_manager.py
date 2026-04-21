"""
Supabase manager module for FPF analytics.

Replaces DuckDB with Supabase (PostgreSQL) for persistent cloud storage.
Maintains the same interface as the previous duckdb_utils/data_manager modules.
"""

import os
import mimetypes
import hashlib
import pandas as pd
import requests
import streamlit as st
from typing import Optional, List, Dict, Callable
from datetime import date, datetime
import math
import time
from urllib.parse import urlparse
from .constants import CLEANDATA_DIR, SELECOES_OPCOES

try:
    import numpy as np
except ImportError:
    np = None

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    SUPABASE_IMPORT_ERROR = None
except ImportError as exc:
    SUPABASE_AVAILABLE = False
    Client = None
    SUPABASE_IMPORT_ERROR = exc


_SCHEMA_INITIALIZED = False


def get_supabase_client() -> Client:
    """Get or create a Supabase client using credentials from Streamlit secrets."""
    if not SUPABASE_AVAILABLE:
        details = f" Import error: {SUPABASE_IMPORT_ERROR}" if SUPABASE_IMPORT_ERROR else ""
        raise ImportError(
            "Supabase could not be imported. Install or repair it with: "
            "pip install --force-reinstall supabase cryptography"
            f"{details}"
        )
    
    # Get credentials from Streamlit secrets
    supabase_url = st.secrets.get("supabase", {}).get("url")
    supabase_key = st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials not found in .streamlit/secrets.toml\n"
            "Please add:\n"
            "[supabase]\n"
            'url = "your-project-url"\n'
            'key = "your-anon-key"'
        )
    
    return create_client(supabase_url, supabase_key)


def initialize_schema(force: bool = False) -> None:
    """Create all necessary tables in Supabase if they don't exist."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED and not force:
        return

    client = get_supabase_client()
    
    tables_sql = """
    -- Selections dimension
    CREATE TABLE IF NOT EXISTS selecoes (
        selection_sk SERIAL PRIMARY KEY,
        codigo TEXT UNIQUE NOT NULL,
        escalao TEXT,
        genero TEXT,
        ativo BOOLEAN DEFAULT true,
        sort_order INTEGER,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS escalao TEXT;
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS genero TEXT;
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS ativo BOOLEAN DEFAULT true;
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS sort_order INTEGER;
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
    ALTER TABLE selecoes ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

    -- Athletes dimension
    CREATE TABLE IF NOT EXISTS athletes (
        athlete_sk SERIAL PRIMARY KEY,
        atleta_id TEXT UNIQUE NOT NULL,
        nome TEXT,
        data_nascimento DATE,
        posicao TEXT,
        pe_preferencial TEXT,
        altura_cm DOUBLE PRECISION,
        peso_kg DOUBLE PRECISION,
        numero_camisola INTEGER,
        escalao TEXT,
        selecao TEXT,
        genero TEXT,
        hr_max_bpm DOUBLE PRECISION,
        hr_rest_bpm DOUBLE PRECISION,
        ativo BOOLEAN DEFAULT true,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS nome TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS data_nascimento DATE;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS posicao TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS pe_preferencial TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS altura_cm DOUBLE PRECISION;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS peso_kg DOUBLE PRECISION;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS numero_camisola INTEGER;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS escalao TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS selecao TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS foto_url TEXT;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS hr_max_bpm DOUBLE PRECISION;
    ALTER TABLE athletes ADD COLUMN IF NOT EXISTS hr_rest_bpm DOUBLE PRECISION;

    -- Fields dimension
    CREATE TABLE IF NOT EXISTS fields (
        field_sk SERIAL PRIMARY KEY,
        field_fingerprint TEXT UNIQUE NOT NULL,
        estadio TEXT,
        campo_local TEXT,
        cidade TEXT,
        pais TEXT,
        clat DOUBLE PRECISION,
        clon DOUBLE PRECISION,
        dist_x DOUBLE PRECISION,
        dist_y DOUBLE PRECISION,
        rotation DOUBLE PRECISION,
        epsg INTEGER,
        bl_lat DOUBLE PRECISION,
        bl_lon DOUBLE PRECISION,
        br_lat DOUBLE PRECISION,
        br_lon DOUBLE PRECISION,
        tl_lat DOUBLE PRECISION,
        tl_lon DOUBLE PRECISION,
        tr_lat DOUBLE PRECISION,
        tr_lon DOUBLE PRECISION,
        obs TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS estadio TEXT;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS campo_local TEXT;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS cidade TEXT;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS pais TEXT;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS clat DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS clon DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS dist_x DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS dist_y DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS rotation DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS epsg INTEGER;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS bl_lat DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS bl_lon DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS br_lat DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS br_lon DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS tl_lat DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS tl_lon DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS tr_lat DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS tr_lon DOUBLE PRECISION;
    ALTER TABLE fields ADD COLUMN IF NOT EXISTS obs TEXT;

    -- Sessions dimension
    CREATE TABLE IF NOT EXISTS sessions (
        session_sk SERIAL PRIMARY KEY,
        session_fingerprint TEXT UNIQUE NOT NULL,
        started_at TIMESTAMPTZ,
        device TEXT
    );

    -- Games dimension
    CREATE TABLE IF NOT EXISTS games (
        game_sk SERIAL PRIMARY KEY,
        game_date DATE,
        opponent TEXT,
        location TEXT,
        competition TEXT
    );

    -- Metrics fact table (legacy)
    CREATE TABLE IF NOT EXISTS metrics (
        athlete_sk INTEGER REFERENCES athletes(athlete_sk),
        session_sk INTEGER REFERENCES sessions(session_sk),
        game_sk INTEGER REFERENCES games(game_sk),
        timestamp TIMESTAMPTZ,
        metric_name TEXT,
        metric_value DOUBLE PRECISION,
        PRIMARY KEY (athlete_sk, session_sk, game_sk, timestamp, metric_name)
    );

    -- Performance metrics
    CREATE TABLE IF NOT EXISTS performance_metrics (
        session_sk INTEGER REFERENCES sessions(session_sk),
        athlete_sk INTEGER REFERENCES athletes(athlete_sk),
        atleta_id TEXT,
        phase_id INTEGER,
        fase TEXT,
        data DATE,
        selecao TEXT,
        genero TEXT,
        contexto TEXT,
        jogo TEXT,
        duracao_min DOUBLE PRECISION,
        dist_m DOUBLE PRECISION,
        m_min DOUBLE PRECISION,
        vmax_mps DOUBLE PRECISION,
        peak_1m_m_min DOUBLE PRECISION,
        hsr_dist_m DOUBLE PRECISION,
        hsr_pct DOUBLE PRECISION,
        sprint_dist_m DOUBLE PRECISION,
        n_sprints INTEGER,
        n_acc_2_5 INTEGER,
        n_dec_3_0 INTEGER,
        active_time_min DOUBLE PRECISION,
        active_pct DOUBLE PRECISION,
        hr_avg_bpm DOUBLE PRECISION,
        hr_peak_bpm DOUBLE PRECISION,
        hr_time_min DOUBLE PRECISION,
        hr_time_valid_pct DOUBLE PRECISION,
        beats_total DOUBLE PRECISION,
        dist_per_beat_m DOUBLE PRECISION,
        hsr_per_beat_m DOUBLE PRECISION,
        sprint_per_beat_m DOUBLE PRECISION,
        sprints_per_1000_beats DOUBLE PRECISION,
        acc_per_1000_beats DOUBLE PRECISION,
        dec_per_1000_beats DOUBLE PRECISION,
        external_load_score DOUBLE PRECISION,
        total_load_score DOUBLE PRECISION,
        player_load DOUBLE PRECISION,
        rhie_bouts INTEGER,
        rhie_actions INTEGER,
        zone1_walk_time_min DOUBLE PRECISION,
        zone1_walk_dist_m DOUBLE PRECISION,
        zone2_jog_time_min DOUBLE PRECISION,
        zone2_jog_dist_m DOUBLE PRECISION,
        zone3_run_time_min DOUBLE PRECISION,
        zone3_run_dist_m DOUBLE PRECISION,
        zone4_hsr_time_min DOUBLE PRECISION,
        zone4_hsr_dist_m DOUBLE PRECISION,
        zone5_sprint_time_min DOUBLE PRECISION,
        zone5_sprint_dist_m DOUBLE PRECISION,
        peak_dist_1m_m DOUBLE PRECISION,
        peak_dist_3m_m DOUBLE PRECISION,
        peak_dist_5m_m DOUBLE PRECISION,
        peak_hsr_1m_m DOUBLE PRECISION,
        peak_hsr_3m_m DOUBLE PRECISION,
        peak_hsr_5m_m DOUBLE PRECISION,
        peak_sprint_1m_m DOUBLE PRECISION,
        peak_sprint_3m_m DOUBLE PRECISION,
        peak_sprint_5m_m DOUBLE PRECISION,
        peak_acc_actions_1m DOUBLE PRECISION,
        peak_acc_actions_3m DOUBLE PRECISION,
        peak_acc_actions_5m DOUBLE PRECISION,
        peak_hi_actions_1m DOUBLE PRECISION,
        peak_hi_actions_3m DOUBLE PRECISION,
        peak_hi_actions_5m DOUBLE PRECISION,
        trimp_banister DOUBLE PRECISION,
        trimp_per_min DOUBLE PRECISION,
        PRIMARY KEY (session_sk, athlete_sk, phase_id)
    );

    CREATE TABLE IF NOT EXISTS collective_performance_metrics (
        session_sk INTEGER REFERENCES sessions(session_sk),
        phase_id INTEGER,
        fase TEXT,
        data DATE,
        selecao TEXT,
        genero TEXT,
        contexto TEXT,
        jogo TEXT,
        duracao_min_total DOUBLE PRECISION,
        dist_m_total DOUBLE PRECISION,
        hsr_dist_m_total DOUBLE PRECISION,
        sprint_dist_m_total DOUBLE PRECISION,
        active_time_min_total DOUBLE PRECISION,
        m_min_avg DOUBLE PRECISION,
        hsr_pct_avg DOUBLE PRECISION,
        active_pct_avg DOUBLE PRECISION,
        n_sprints_total INTEGER,
        n_acc_2_5_total INTEGER,
        n_dec_3_0_total INTEGER,
        vmax_mps_max DOUBLE PRECISION,
        peak_1m_m_min_max DOUBLE PRECISION,
        hr_avg_bpm_avg DOUBLE PRECISION,
        external_load_score_total DOUBLE PRECISION,
        total_load_score_total DOUBLE PRECISION,
        player_load_total DOUBLE PRECISION,
        rhie_bouts_total INTEGER,
        rhie_actions_total INTEGER,
        trimp_banister_total DOUBLE PRECISION,
        PRIMARY KEY (session_sk, phase_id)
    );

    -- Quality metrics
    CREATE TABLE IF NOT EXISTS quality_metrics (
        session_sk INTEGER REFERENCES sessions(session_sk),
        athlete_sk INTEGER REFERENCES athletes(athlete_sk),
        atleta_id TEXT,
        phase_id INTEGER,
        fase TEXT,
        n_points INTEGER,
        pct_time_valid DOUBLE PRECISION,
        n_gaps_gt2s INTEGER,
        qc_grade TEXT,
        qc_flags TEXT,
        vmax_mps_qc DOUBLE PRECISION,
        n_jumps_gt15m INTEGER,
        n_gaps_gt2s_qc INTEGER,
        PRIMARY KEY (session_sk, athlete_sk, phase_id)
    );

    -- Samples (raw tracking data)
    CREATE TABLE IF NOT EXISTS samples (
        session_sk INTEGER REFERENCES sessions(session_sk),
        athlete_sk INTEGER REFERENCES athletes(athlete_sk),
        atleta_id TEXT,
        fase TEXT,
        time TIMESTAMPTZ,
        time_evento_s DOUBLE PRECISION,
        time_evento TEXT,
        periodo_jogo TEXT,
        minuto_jogo INTEGER,
        lat DOUBLE PRECISION,
        lon DOUBLE PRECISION,
        x_utm DOUBLE PRECISION,
        y_utm DOUBLE PRECISION,
        x_norm DOUBLE PRECISION,
        y_norm DOUBLE PRECISION,
        speed_mps DOUBLE PRECISION,
        acc_mps2 DOUBLE PRECISION,
        hr_bpm DOUBLE PRECISION,
        phase_id INTEGER,
        PRIMARY KEY (session_sk, athlete_sk, phase_id, time)
    );
    ALTER TABLE samples ADD COLUMN IF NOT EXISTS x_norm DOUBLE PRECISION;
    ALTER TABLE samples ADD COLUMN IF NOT EXISTS y_norm DOUBLE PRECISION;
    ALTER TABLE samples ADD COLUMN IF NOT EXISTS speed_mps DOUBLE PRECISION;
    ALTER TABLE samples ADD COLUMN IF NOT EXISTS acc_mps2 DOUBLE PRECISION;
    CREATE INDEX IF NOT EXISTS idx_samples_session_phase_time ON samples(session_sk, phase_id, time_evento_s);
    CREATE INDEX IF NOT EXISTS idx_samples_session_athlete_phase_timeevento
        ON samples(session_sk, athlete_sk, phase_id, time_evento_s);

    -- Athlete session participation
    CREATE TABLE IF NOT EXISTS athlete_session (
        session_sk INTEGER REFERENCES sessions(session_sk),
        athlete_sk INTEGER REFERENCES athletes(athlete_sk),
        atleta_id TEXT,
        participou_warmup BOOLEAN,
        participou_1p BOOLEAN,
        participou_2p BOOLEAN,
        fases_disponiveis TEXT,
        n_samples INTEGER,
        tem_hr BOOLEAN,
        processado_em TIMESTAMPTZ,
        PRIMARY KEY (session_sk, athlete_sk)
    );
    """
    
    # Execute via RPC or direct SQL (if your Supabase tier supports it)
    # For now, we assume the schema was created beforehand in Supabase.
    _SCHEMA_INITIALIZED = True


def upload_image_to_storage(
    bucket_name: str,
    object_path: str,
    file_bytes: bytes,
    content_type: Optional[str] = None,
) -> str:
    """Upload an image to Supabase Storage and return its public URL."""
    client = get_supabase_client()
    storage = client.storage.from_(bucket_name)
    resolved_content_type = content_type or mimetypes.guess_type(object_path)[0] or "application/octet-stream"

    try:
        storage.upload(
            object_path,
            file_bytes,
            {"content-type": resolved_content_type, "upsert": "true"},
        )
    except Exception as exc:
        message = str(exc)
        if "Bucket not found" in message:
            raise RuntimeError(
                f"O bucket '{bucket_name}' nao existe no Supabase Storage. "
                f"Cria esse bucket primeiro para permitir o upload de fotos."
            ) from exc
        raise RuntimeError(f"Falha ao enviar imagem para o Supabase Storage: {exc}") from exc

    public_url = storage.get_public_url(object_path)
    if isinstance(public_url, dict):
        resolved_url = public_url.get("publicURL") or public_url.get("publicUrl") or public_url.get("data", {}).get("publicUrl") or ""
    elif hasattr(public_url, "get"):
        resolved_url = public_url.get("publicURL") or public_url.get("publicUrl") or ""
    else:
        resolved_url = str(public_url)
    if not resolved_url:
        raise RuntimeError("A imagem foi enviada, mas não foi possível obter o URL público.")
    return resolved_url


def build_public_storage_url(bucket_name: str, object_path: str) -> str:
    """Build a public URL for a storage object without uploading it."""
    client = get_supabase_client()
    storage = client.storage.from_(bucket_name)
    public_url = storage.get_public_url(object_path)
    if isinstance(public_url, dict):
        return public_url.get("publicURL") or public_url.get("publicUrl") or public_url.get("data", {}).get("publicUrl") or ""
    if hasattr(public_url, "get"):
        return public_url.get("publicURL") or public_url.get("publicUrl") or ""
    return str(public_url) or ""


def public_url_exists(url: str, timeout_seconds: int = 5) -> bool:
    """Check whether a public URL is reachable."""
    if not url:
        return False
    try:
        response = requests.get(url, timeout=timeout_seconds, stream=True)
        return response.status_code == 200
    except Exception:
        return False


def insert_or_update_table(
    table_name: str,
    df: pd.DataFrame,
    pk_columns: List[str] = None,
    batch_size: Optional[int] = None,
    batch_progress_callback: Optional[Callable] = None,
    max_retries: int = 0,
) -> Dict:
    """Insert or update DataFrame into Supabase table (upsert).
    
    Args:
        table_name: name of the Supabase table
        df: DataFrame to insert/update
        pk_columns: list of column names forming the primary key
    
    Returns:
        dict with stats: {'inserted': int, 'updated': int, 'success': bool, 'error': str | None}
    """
    if df.empty:
        return {'inserted': 0, 'updated': 0, 'success': True, 'error': None}
    
    client = get_supabase_client()
    stats = {'inserted': 0, 'updated': 0, 'success': True, 'error': None}

    df_to_upload = df.copy()
    if pk_columns:
        pk_present = [col for col in pk_columns if col in df_to_upload.columns]
        if pk_present:
            df_to_upload = df_to_upload.drop_duplicates(subset=pk_present, keep="last")

    records = _dataframe_to_supabase_records(df_to_upload)

    if batch_size is None or batch_size <= 0:
        batches = [records]
    else:
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]

    try:
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            if batch_progress_callback is not None and total_batches > 1:
                batch_progress_callback(batch_index, total_batches, len(batch))

            attempt = 0
            while True:
                try:
                    if pk_columns is None or len(pk_columns) == 0:
                        response = client.table(table_name).insert(
                            batch,
                            returning="minimal",
                        ).execute()
                        stats['inserted'] += len(response.data) if response.data else len(batch)
                    else:
                        response = client.table(table_name).upsert(
                            batch,
                            on_conflict=",".join(pk_columns),
                            ignore_duplicates=False,
                            returning="minimal",
                        ).execute()
                        stats['inserted'] += len(response.data) if response.data else len(batch)
                    break
                except Exception as e:
                    if attempt >= max_retries or not _is_transient_supabase_error(e):
                        raise

                    attempt += 1
                    wait_seconds = min(2 ** attempt, 10)
                    if batch_progress_callback is not None:
                        batch_progress_callback(
                            batch_index,
                            total_batches,
                            len(batch),
                            retry_attempt=attempt,
                            retry_wait_seconds=wait_seconds,
                        )
                    time.sleep(wait_seconds)
    except Exception as e:
        stats['success'] = False
        stats['error'] = str(e)
        st.error(f"Error writing to {table_name}: {str(e)}")
        return stats

    return stats
def delete_table_rows(table_name: str, filters: Dict) -> Dict:
    """Delete rows from a Supabase table using equality filters."""
    client = get_supabase_client()
    stats = {"deleted": 0, "success": True, "error": None}

    try:
        query = client.table(table_name).delete()
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        response = query.execute()
        stats["deleted"] = len(response.data) if getattr(response, "data", None) else 0
    except Exception as exc:
        stats["success"] = False
        stats["error"] = str(exc)
        st.error(f"Error deleting from {table_name}: {exc}")
    return stats


def _table_has_session_rows(table_name: str, session_sk: int) -> bool:
    """Check whether a table already contains rows for a given session."""
    if session_sk is None:
        return False

    df = read_table(table_name, {"session_sk": session_sk}, columns="session_sk", limit=1)
    return df is not None and not df.empty


def _get_session_athlete_sks(session_sk: int) -> List[int]:
    """Collect athlete_sks linked to a session from lightweight fact tables."""
    athlete_sks = set()
    for table_name in ["athlete_session", "performance_metrics", "quality_metrics"]:
        df = read_table(table_name, {"session_sk": session_sk}, columns="athlete_sk")
        if df is None or df.empty or "athlete_sk" not in df.columns:
            continue
        athlete_sks.update(
            pd.to_numeric(df["athlete_sk"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
    return sorted(athlete_sks)


def _delete_samples_for_session(session_sk: int) -> Dict:
    """Delete samples in small indexed slices to avoid Supabase timeouts."""
    stats = {"deleted": 0, "success": True, "error": None}
    athlete_sks = _get_session_athlete_sks(session_sk)
    phase_ids = [0, 1, 2]

    try:
        if athlete_sks:
            for athlete_sk in athlete_sks:
                for phase_id in phase_ids:
                    result = delete_table_rows(
                        "samples",
                        {
                            "session_sk": session_sk,
                            "athlete_sk": athlete_sk,
                            "phase_id": phase_id,
                        },
                    )
                    stats["deleted"] += int(result.get("deleted", 0) or 0)
                    if not result.get("success", False):
                        stats["success"] = False
                        stats["error"] = result.get("error")
        else:
            # Fallback for partial/orphan uploads where athlete_session was not written.
            for phase_id in phase_ids:
                result = delete_table_rows(
                    "samples",
                    {"session_sk": session_sk, "phase_id": phase_id},
                )
                stats["deleted"] += int(result.get("deleted", 0) or 0)
                if not result.get("success", False):
                    stats["success"] = False
                    stats["error"] = result.get("error")
    except Exception as exc:
        stats["success"] = False
        stats["error"] = str(exc)

    return stats


def cleanup_session_upload(session_sk: int, delete_session_row: bool = True) -> Dict:
    """Best-effort cleanup for a partially uploaded session."""
    stats = {"success": True, "deleted": {}, "errors": {}}

    if session_sk is None:
        return stats

    table_order = [
        "session_reports",
        "quality_metrics",
        "collective_performance_metrics",
        "performance_metrics",
        "athlete_session",
    ]

    samples_result = _delete_samples_for_session(session_sk)
    stats["deleted"]["samples"] = samples_result.get("deleted", 0)
    if not samples_result.get("success", False):
        stats["success"] = False
        stats["errors"]["samples"] = samples_result.get("error")

    for table_name in table_order:
        result = delete_table_rows(table_name, {"session_sk": session_sk})
        stats["deleted"][table_name] = result.get("deleted", 0)
        if not result.get("success", False):
            stats["success"] = False
            stats["errors"][table_name] = result.get("error")

    if delete_session_row:
        remaining_refs = any(
            _table_has_session_rows(table_name, session_sk)
            for table_name in ["samples", *table_order]
        )
        if not remaining_refs:
            result = delete_table_rows("sessions", {"session_sk": session_sk})
            stats["deleted"]["sessions"] = result.get("deleted", 0)
            if not result.get("success", False):
                stats["success"] = False
                stats["errors"]["sessions"] = result.get("error")

    return stats


def delete_public_storage_url(bucket_name: str, public_url: str) -> None:
    """Delete a storage object from its public URL when possible."""
    resolved_url = str(public_url or "").strip()
    if not resolved_url:
        return

    parsed = urlparse(resolved_url)
    marker = f"/storage/v1/object/public/{bucket_name}/"
    if marker not in parsed.path:
        return

    object_path = parsed.path.split(marker, 1)[1]
    if not object_path:
        return

    client = get_supabase_client()
    storage = client.storage.from_(bucket_name)
    try:
        storage.remove([object_path])
    except Exception:
        pass


def _is_transient_supabase_error(error: Exception) -> bool:
    """Detect transient API/proxy failures worth retrying."""
    message = str(error).lower()
    transient_markers = [
        "'code': 520",
        '"code": 520',
        "error code 520",
        "cloudflare",
        "timed out",
        "timeout",
        "connection reset",
        "temporarily unavailable",
        "server is returning an unknown error",
    ]
    return any(marker in message for marker in transient_markers)


def _json_safe_value(value):
    """Convert pandas/numpy/date values to JSON-serializable Python primitives."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, str) and value.strip() in {"NaT", "nan", "None", "<NA>", ""}:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _dataframe_to_supabase_records(df: pd.DataFrame) -> List[Dict]:
    """Serialize a DataFrame into Supabase-safe records."""
    records = []
    for record in df.to_dict("records"):
        serialized = {key: _json_safe_value(value) for key, value in record.items()}
        records.append(serialized)
    return records


def _sync_dimension_tables(session_sks: List[int], athlete_sks: List[int], base_dir=None) -> None:
    """Ensure referenced dimensions already exist in Supabase before fact inserts."""
    if session_sks:
        session_df = read_table("sessions")
        known_session_sks = set()
        if session_df is not None and not session_df.empty and "session_sk" in session_df.columns:
            known_session_sks = set(pd.to_numeric(session_df["session_sk"], errors="coerce").dropna().astype(int).tolist())
        missing_sessions = [session_sk for session_sk in session_sks if session_sk not in known_session_sks]
        if missing_sessions:
            raise RuntimeError(
                f"Faltam sessões de referência no Supabase para os session_sk: {', '.join(map(str, missing_sessions))}."
            )

    if athlete_sks:
        athlete_df = read_table("athletes", columns="athlete_sk")
        known_athlete_sks = set()
        if athlete_df is not None and not athlete_df.empty and "athlete_sk" in athlete_df.columns:
            known_athlete_sks = set(pd.to_numeric(athlete_df["athlete_sk"], errors="coerce").dropna().astype(int).tolist())
        missing_athletes = [athlete_sk for athlete_sk in athlete_sks if athlete_sk not in known_athlete_sks]
        if missing_athletes:
            raise RuntimeError(
                f"Faltam atletas de referência no Supabase para os athlete_sk: {', '.join(map(str, missing_athletes[:20]))}."
            )


def _emit_progress(progress_callback: Optional[Callable], step: int, total_steps: int, message: str, table: Optional[str] = None) -> None:
    """Send progress updates to the UI when a callback is provided."""
    if progress_callback is None:
        return

    progress_callback({
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "table": table,
    })


def read_table(table_name: str, filters: Dict = None, columns: str = "*", limit: Optional[int] = None) -> pd.DataFrame:
    """Read data from Supabase table.
    
    Args:
        table_name: name of the Supabase table
        filters: optional dict of {column: value} for filtering
        columns: comma-separated columns to select
        limit: optional row limit
    
    Returns:
        DataFrame with query results
    """
    client = get_supabase_client()
    
    def _build_query():
        query = client.table(table_name).select(columns)
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        return query

    try:
        if limit is not None and limit > 0:
            response = _build_query().limit(limit).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()

        page_size = 1000
        offset = 0
        rows = []
        while True:
            response = _build_query().range(offset, offset + page_size - 1).execute()
            batch = response.data or []
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error reading from {table_name}: {str(e)}")
        return pd.DataFrame()


def save_session_report(report_payload: Dict) -> Dict:
    """Persist a technical session report for a game or training session."""
    initialize_schema()

    if not report_payload:
        raise ValueError("Session report payload is empty.")

    now_ts = pd.Timestamp.utcnow()
    row = {
        "session_fingerprint": str(report_payload.get("session_fingerprint") or "").strip(),
        "session_sk": report_payload.get("session_sk"),
        "data": report_payload.get("data"),
        "selecao": str(report_payload.get("selecao") or "").strip(),
        "genero": str(report_payload.get("genero") or "").strip(),
        "contexto": str(report_payload.get("contexto") or "").strip(),
        "jogo": str(report_payload.get("jogo") or "").strip(),
        "report_title": str(report_payload.get("report_title") or "").strip(),
        "report_txt": str(report_payload.get("report_txt") or "").strip(),
        "updated_at": now_ts,
    }
    if report_payload.get("created_at") is not None:
        row["created_at"] = report_payload.get("created_at")
    else:
        row["created_at"] = now_ts

    if not row["session_fingerprint"]:
        raise ValueError("Session report requires session_fingerprint.")
    if not row["selecao"]:
        raise ValueError("Session report requires selecao.")
    if not row["contexto"]:
        raise ValueError("Session report requires contexto.")
    if not row["report_txt"]:
        raise ValueError("Session report requires report_txt.")

    df = pd.DataFrame([row])
    if "session_sk" in df.columns:
        df["session_sk"] = pd.to_numeric(df["session_sk"], errors="coerce").astype("Int64")
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date

    stats = insert_or_update_table("session_reports", df, pk_columns=["session_fingerprint"])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao guardar relatorio tecnico.")
    return stats


# ==================== High-level API (matches old data_manager interface) ====================

def write_session_data(
    df_perf, df_collective_perf, df_qc, df_samples, df_athlete_session,
    db_file=None,  # ignored in Supabase mode, kept for compatibility
    progress_callback: Optional[Callable] = None,
):
    """Persist performance, collective performance, quality, samples, and athlete_session DataFrames to Supabase.
    
    This is the main integration point for the analytic pipeline.
    Uses upsert logic to avoid duplicates.
    
    Args:
        df_perf: performance metrics DataFrame
        df_collective_perf: collective performance metrics DataFrame
        df_qc: quality metrics DataFrame
        df_samples: samples (tracking) DataFrame
        df_athlete_session: athlete session participation DataFrame
        db_file: ignored (for compatibility with old DuckDB interface)
    
    Returns:
        dict with integration stats
    """
    total_steps = 2 + sum(
        1 for df in [df_perf, df_collective_perf, df_qc, df_samples, df_athlete_session]
        if df is not None and not df.empty
    )

    current_step = 1
    _emit_progress(progress_callback, current_step, total_steps, "A validar esquema e preparar ligação ao Supabase...")
    initialize_schema()
    
    stats = {
        'performance_metrics': None,
        'collective_performance_metrics': None,
        'quality_metrics': None,
        'samples': None,
        'athlete_session': None,
    }

    session_sks = set()
    athlete_sks = set()
    for df in [df_perf, df_collective_perf, df_qc, df_samples, df_athlete_session]:
        if df is None or df.empty:
            continue
        if "session_sk" in df.columns:
            session_sks.update(pd.to_numeric(df["session_sk"], errors="coerce").dropna().astype(int).tolist())
        if "athlete_sk" in df.columns:
            athlete_sks.update(pd.to_numeric(df["athlete_sk"], errors="coerce").dropna().astype(int).tolist())

    current_step += 1
    _emit_progress(progress_callback, current_step, total_steps, "A sincronizar atletas e sessões de referência...")
    _sync_dimension_tables(
        session_sks=sorted(session_sks),
        athlete_sks=sorted(athlete_sks),
        base_dir=CLEANDATA_DIR,
    )

    rollback_session_sk = None
    rollback_enabled = False
    if len(session_sks) == 1:
        rollback_session_sk = next(iter(session_sks))
        tracked_tables = [
            "performance_metrics",
            "collective_performance_metrics",
            "quality_metrics",
            "samples",
            "athlete_session",
            "session_reports",
        ]
        rollback_enabled = not any(
            _table_has_session_rows(table_name, rollback_session_sk)
            for table_name in tracked_tables
        )

    def _raise_with_optional_cleanup(error_message: str) -> None:
        if rollback_enabled and rollback_session_sk is not None:
            cleanup_stats = cleanup_session_upload(rollback_session_sk, delete_session_row=True)
            if cleanup_stats.get("success", False):
                raise RuntimeError(
                    f"{error_message} A limpeza automática dos dados parciais da sessão {rollback_session_sk} foi concluída."
                )
            raise RuntimeError(
                f"{error_message} A limpeza automática da sessão {rollback_session_sk} falhou parcialmente: {cleanup_stats.get('errors')}."
            )
        raise RuntimeError(error_message)
    
    if df_perf is not None and not df_perf.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar performance_metrics ({len(df_perf)} linhas)...", "performance_metrics")
        stats['performance_metrics'] = insert_or_update_table(
            "performance_metrics", df_perf,
            pk_columns=["session_sk", "athlete_sk", "phase_id"],
        )
        if not stats['performance_metrics'].get("success", False):
            _raise_with_optional_cleanup(
                f"Falha ao gravar performance_metrics: {stats['performance_metrics'].get('error')}"
            )

    if df_collective_perf is not None and not df_collective_perf.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar collective_performance_metrics ({len(df_collective_perf)} linhas)...", "collective_performance_metrics")
        stats['collective_performance_metrics'] = insert_or_update_table(
            "collective_performance_metrics", df_collective_perf,
            pk_columns=["session_sk", "phase_id"],
        )
        if not stats['collective_performance_metrics'].get("success", False):
            _raise_with_optional_cleanup(
                f"Falha ao gravar collective_performance_metrics: {stats['collective_performance_metrics'].get('error')}"
            )
    
    if df_qc is not None and not df_qc.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar quality_metrics ({len(df_qc)} linhas)...", "quality_metrics")
        stats['quality_metrics'] = insert_or_update_table(
            "quality_metrics", df_qc,
            pk_columns=["session_sk", "athlete_sk", "phase_id"],
        )
        if not stats['quality_metrics'].get("success", False):
            _raise_with_optional_cleanup(
                f"Falha ao gravar quality_metrics: {stats['quality_metrics'].get('error')}"
            )
    
    if df_samples is not None and not df_samples.empty:
        df_samples = df_samples.copy()
        if "time" in df_samples.columns:
            df_samples["time"] = pd.to_datetime(df_samples["time"], errors="coerce")
            df_samples = df_samples[df_samples["time"].notna()].copy()
        if df_samples.empty:
            stats['samples'] = {'inserted': 0, 'updated': 0, 'success': True, 'error': None}
        else:
            current_step += 1
            _emit_progress(progress_callback, current_step, total_steps, f"A gravar samples ({len(df_samples)} linhas)...", "samples")
            stats['samples'] = insert_or_update_table(
                "samples", df_samples,
                pk_columns=["session_sk", "athlete_sk", "phase_id", "time"],
                batch_size=5000,
                max_retries=3,
                batch_progress_callback=lambda idx, total, size, retry_attempt=0, retry_wait_seconds=0: _emit_progress(
                    progress_callback,
                    current_step,
                    total_steps,
                    (
                        f"A gravar samples: lote {idx}/{total} ({size} linhas)..."
                        if retry_attempt == 0
                        else f"A repetir samples: lote {idx}/{total}, tentativa {retry_attempt}/3 em {retry_wait_seconds}s..."
                    ),
                    "samples",
                ),
            )
            if not stats['samples'].get("success", False):
                _raise_with_optional_cleanup(
                    f"Falha ao gravar samples: {stats['samples'].get('error')}"
                )
    
    if df_athlete_session is not None and not df_athlete_session.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar athlete_session ({len(df_athlete_session)} linhas)...", "athlete_session")
        stats['athlete_session'] = insert_or_update_table(
            "athlete_session", df_athlete_session,
            pk_columns=["session_sk", "athlete_sk"]
        )
        if not stats['athlete_session'].get("success", False):
            _raise_with_optional_cleanup(
                f"Falha ao gravar athlete_session: {stats['athlete_session'].get('error')}"
            )

    _emit_progress(progress_callback, total_steps, total_steps, "Transferência concluída com sucesso.")
    
    return stats


def write_metrics(metrics_df, db_file=None):
    """Persist a metrics dataframe into the Supabase metrics table (legacy).
    
    Args:
        metrics_df: DataFrame with columns ['athlete_sk', 'session_sk', 'game_sk', 'timestamp', 'metric_name', 'metric_value']
        db_file: ignored (for compatibility with old DuckDB interface)
    """
    if metrics_df is None or metrics_df.empty:
        return
    
    initialize_schema()
    insert_or_update_table(
        "metrics", metrics_df,
        pk_columns=["athlete_sk", "session_sk", "game_sk", "timestamp", "metric_name"]
    )


# ==================== Parquet utilities (maintained for field/athlete/session dimensions) ====================

FIELD_REFERENCE_COLUMNS = [
    "field_fingerprint",
    "estadio",
    "campo_local",
    "cidade",
    "pais",
    "clat",
    "clon",
    "dist_x",
    "dist_y",
    "rotation",
    "epsg",
    "BL_lat",
    "BL_lon",
    "BR_lat",
    "BR_lon",
    "TL_lat",
    "TL_lon",
    "TR_lat",
    "TR_lon",
    "obs",
]

SELECTION_REFERENCE_COLUMNS = [
    "selection_sk",
    "codigo",
    "escalao",
    "genero",
    "ativo",
    "sort_order",
    "created_at",
    "updated_at",
]

SESSION_REPORT_COLUMNS = [
    "report_sk",
    "session_fingerprint",
    "session_sk",
    "data",
    "selecao",
    "genero",
    "contexto",
    "jogo",
    "report_title",
    "report_txt",
    "created_at",
    "updated_at",
]


def _parse_selection_label(label: str) -> tuple[str, str]:
    value = str(label or "").strip()
    if not value:
        return "", ""
    parts = value.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] in {"M", "F"}:
        return parts[0], parts[1]
    return value, ""


def _default_selections_reference_df() -> pd.DataFrame:
    now_ts = pd.Timestamp.utcnow()
    rows = []
    for idx, codigo in enumerate(SELECOES_OPCOES, start=1):
        escalao, genero = _parse_selection_label(codigo)
        rows.append(
            {
                "selection_sk": idx,
                "codigo": codigo,
                "escalao": escalao,
                "genero": genero,
                "ativo": True,
                "sort_order": idx,
                "created_at": now_ts,
                "updated_at": now_ts,
            }
        )
    return pd.DataFrame(rows, columns=SELECTION_REFERENCE_COLUMNS)


def _prepare_selections_reference_df(df_new: pd.DataFrame) -> pd.DataFrame:
    if df_new is None or df_new.empty:
        return _default_selections_reference_df()

    df = df_new.copy()
    for col in SELECTION_REFERENCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[SELECTION_REFERENCE_COLUMNS].copy()
    df["codigo"] = df["codigo"].astype(str).str.strip()
    df = df[df["codigo"].ne("")].drop_duplicates(subset=["codigo"], keep="last")

    parsed_pairs = df["codigo"].map(_parse_selection_label)
    parsed_df = pd.DataFrame(parsed_pairs.tolist(), columns=["parsed_escalao", "parsed_genero"], index=df.index)

    df["escalao"] = df["escalao"].astype("string").fillna("").str.strip()
    df["genero"] = df["genero"].astype("string").fillna("").str.strip()
    df.loc[df["escalao"].eq(""), "escalao"] = parsed_df["parsed_escalao"]
    df.loc[df["genero"].eq(""), "genero"] = parsed_df["parsed_genero"]

    df["ativo"] = df["ativo"].fillna(True).astype(bool)
    df["selection_sk"] = pd.to_numeric(df["selection_sk"], errors="coerce")
    if df["selection_sk"].isna().any():
        df["selection_sk"] = range(1, len(df) + 1)
    df["selection_sk"] = df["selection_sk"].astype(int)

    df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce")
    if df["sort_order"].isna().any():
        df["sort_order"] = range(1, len(df) + 1)
    df["sort_order"] = df["sort_order"].astype(int)

    now_ts = pd.Timestamp.utcnow()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").fillna(now_ts)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce").fillna(now_ts)
    return df.sort_values(["sort_order", "codigo"], na_position="last").reset_index(drop=True)


def save_selections_reference(df_new: pd.DataFrame) -> pd.DataFrame:
    """Persist selections reference data to Supabase."""
    df_prepared = _prepare_selections_reference_df(df_new)
    if df_prepared.empty:
        return pd.DataFrame(columns=SELECTION_REFERENCE_COLUMNS)

    initialize_schema()
    # Preserve the database-managed surrogate key when upserting by codigo.
    upload_columns = [
        "codigo",
        "escalao",
        "genero",
        "ativo",
        "sort_order",
        "updated_at",
    ]
    if "created_at" in df_prepared.columns:
        upload_columns.append("created_at")
    stats = insert_or_update_table("selecoes", df_prepared[upload_columns], pk_columns=["codigo"])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao guardar selecoes no Supabase.")
    return df_prepared


def read_selections_reference(active_only: bool = False) -> pd.DataFrame:
    """Read selections reference data from Supabase."""
    initialize_schema()
    df_db = read_table("selecoes")
    if df_db is None or df_db.empty:
        df_db = _default_selections_reference_df()

    df = _prepare_selections_reference_df(df_db)
    if active_only:
        df = df[df["ativo"]].copy()
    return df.sort_values(["sort_order", "codigo"], na_position="last").reset_index(drop=True)


def sync_selections_reference(include_default: bool = True) -> pd.DataFrame:
    """Ensure Supabase contains the known selections plus any athlete-linked selections."""
    frames = []
    try:
        current_df = read_table("selecoes")
    except Exception:
        current_df = pd.DataFrame()

    if current_df is not None and not current_df.empty:
        frames.append(current_df)

    if include_default:
        frames.append(_default_selections_reference_df())

    try:
        athletes_df = read_table("athletes")
    except Exception:
        athletes_df = pd.DataFrame()

    if athletes_df is not None and not athletes_df.empty and "selecao" in athletes_df.columns:
        selecao_series = athletes_df["selecao"].astype("string").fillna("").str.strip()
        athlete_codes = [code for code in selecao_series.unique().tolist() if code]
        if athlete_codes:
            rows = []
            now_ts = pd.Timestamp.utcnow()
            for idx, codigo in enumerate(athlete_codes, start=1):
                escalao, genero = _parse_selection_label(codigo)
                rows.append(
                    {
                        "selection_sk": idx,
                        "codigo": codigo,
                        "escalao": escalao,
                        "genero": genero,
                        "ativo": True,
                        "sort_order": idx,
                        "created_at": now_ts,
                        "updated_at": now_ts,
                    }
                )
            frames.append(pd.DataFrame(rows, columns=SELECTION_REFERENCE_COLUMNS))

    if not frames:
        return save_selections_reference(_default_selections_reference_df())

    merged = pd.concat(frames, ignore_index=True)
    merged = _prepare_selections_reference_df(merged)
    return save_selections_reference(merged)


def _field_fingerprint_from_row(row: pd.Series) -> str:
    parts = [
        f"{float(row.get('BL_lat', 0.0)):.6f}",
        f"{float(row.get('BL_lon', 0.0)):.6f}",
        f"{float(row.get('BR_lat', 0.0)):.6f}",
        f"{float(row.get('BR_lon', 0.0)):.6f}",
        f"{float(row.get('TL_lat', 0.0)):.6f}",
        f"{float(row.get('TL_lon', 0.0)):.6f}",
        f"{float(row.get('TR_lat', 0.0)):.6f}",
        f"{float(row.get('TR_lon', 0.0)):.6f}",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _prepare_field_reference_df(df_new: pd.DataFrame) -> pd.DataFrame:
    if df_new is None or df_new.empty:
        return pd.DataFrame(columns=FIELD_REFERENCE_COLUMNS)

    df = df_new.copy()
    rename_map = {
        "bl_lat": "BL_lat",
        "bl_lon": "BL_lon",
        "br_lat": "BR_lat",
        "br_lon": "BR_lon",
        "tl_lat": "TL_lat",
        "tl_lon": "TL_lon",
        "tr_lat": "TR_lat",
        "tr_lon": "TR_lon",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in FIELD_REFERENCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["estadio", "campo_local", "cidade", "pais", "obs"]:
        df[col] = df[col].astype("string").fillna("").str.strip()

    for col in [
        "clat", "clon", "dist_x", "dist_y", "rotation",
        "BL_lat", "BL_lon", "BR_lat", "BR_lon", "TL_lat", "TL_lon", "TR_lat", "TR_lon",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "epsg" in df.columns:
        df["epsg"] = pd.to_numeric(df["epsg"], errors="coerce").astype("Int64")

    if "field_fingerprint" not in df.columns or df["field_fingerprint"].isna().any():
        df["field_fingerprint"] = df.apply(_field_fingerprint_from_row, axis=1)

    return df[FIELD_REFERENCE_COLUMNS].drop_duplicates(subset=["field_fingerprint"], keep="last")


def save_field_reference(df_new):
    """
    Persist field data to Supabase.

    Persist field data to Supabase.
    """
    df_prepared = _prepare_field_reference_df(df_new)
    if df_prepared.empty:
        return pd.DataFrame()

    initialize_schema()
    df_db = df_prepared.rename(
        columns={
            "BL_lat": "bl_lat",
            "BL_lon": "bl_lon",
            "BR_lat": "br_lat",
            "BR_lon": "br_lon",
            "TL_lat": "tl_lat",
            "TL_lon": "tl_lon",
            "TR_lat": "tr_lat",
            "TR_lon": "tr_lon",
        }
    )
    stats = insert_or_update_table("fields", df_db, pk_columns=["field_fingerprint"])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao guardar campo no Supabase.")
    return df_prepared


def read_field_reference():
    """
    Read field data from Supabase.
    """
    initialize_schema()
    df_db = read_table("fields")
    if df_db is None or df_db.empty:
        return pd.DataFrame(columns=FIELD_REFERENCE_COLUMNS)
    return _prepare_field_reference_df(df_db)


def migrate_field_parquet_to_supabase(parquet_path: str) -> Dict:
    """One-time migration helper for legacy local field references."""
    stats = {"migrated": 0, "success": True, "error": None}
    try:
        if not parquet_path or not os.path.exists(parquet_path):
            stats["error"] = "Parquet file not found."
            stats["success"] = False
            return stats

        df_legacy = pd.read_parquet(parquet_path)
        df_prepared = _prepare_field_reference_df(df_legacy)
        if df_prepared.empty:
            return stats

        save_field_reference(df_prepared)
        stats["migrated"] = int(len(df_prepared))
        return stats
    except Exception as exc:
        stats["success"] = False
        stats["error"] = str(exc)
        return stats


def append_dedup_parquet(df_new, filename, subset_keys):
    """Append to parquet with deduplication by composite key."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if df_new is None or df_new.empty:
        return pd.DataFrame()

    if os.path.exists(filename):
        df_old = pd.read_parquet(filename)
        # Ensure consistent dtypes
        for col in df_new.columns:
            if col in df_old.columns:
                if df_old[col].dtype != df_new[col].dtype:
                    if df_old[col].dtype in ['datetime64[ns, UTC]', 'datetime64[ns]']:
                        df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                    elif df_old[col].dtype == 'object':
                        df_new[col] = df_new[col].astype(str)
                    elif df_old[col].dtype == 'int64':
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').astype('Int64')
                    elif df_old[col].dtype == 'float64':
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    elif df_old[col].dtype == 'bool':
                        df_new[col] = df_new[col].astype(bool)
        
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=subset_keys, keep="last")
    else:
        df_final = df_new.copy()

    df_final.to_parquet(filename, index=False)
    return df_final


def resolve_athlete_sk(df_metrics, base_dir=CLEANDATA_DIR, genero=None, athlete_profiles=None, selecao=None):
    """Resolve persistent athlete_sk from atleta_id using Supabase as the source of truth."""
    initialize_schema()

    atletas = pd.Series(dtype="object")
    if df_metrics is not None and not df_metrics.empty and "atleta_id" in df_metrics.columns:
        atletas = pd.Series(df_metrics["atleta_id"].dropna().astype(str).str.strip().unique())

    try:
        df_dim = read_table("athletes")
    except Exception:
        df_dim = pd.DataFrame()

    if df_dim is None or df_dim.empty:
        df_dim = pd.DataFrame(columns=[
            "athlete_sk", "atleta_id", "nome", "foto_url", "data_nascimento", "posicao",
            "genero", "ativo"
        ])

    if not df_dim.empty:
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str).str.strip()
    existing_rows = (
        df_dim.drop_duplicates(subset=["atleta_id"], keep="last").set_index("atleta_id").to_dict(orient="index")
        if not df_dim.empty and "atleta_id" in df_dim.columns
        else {}
    )

    profiles_map = {}
    if athlete_profiles is not None and not athlete_profiles.empty and "atleta_id" in athlete_profiles.columns:
        df_profiles = athlete_profiles.copy()
        df_profiles["atleta_id"] = df_profiles["atleta_id"].astype(str).str.strip()
        profiles_map = df_profiles.set_index("atleta_id").to_dict(orient="index")

    existing = dict(
        zip(
            df_dim.get("atleta_id", pd.Series(dtype="object")),
            pd.to_numeric(df_dim.get("athlete_sk", pd.Series(dtype="float64")), errors="coerce"),
        )
    )

    upload_rows = []
    now_ts = pd.Timestamp.utcnow()
    for aid in atletas.tolist():
        if not aid:
            continue
        if aid not in existing or pd.isna(existing.get(aid)):
            raise ValueError(
                f"O atleta '{aid}' nÃ£o existe na base de dados. Cria-o primeiro antes de publicar."
            )

        profile = profiles_map.get(aid, {})
        if profile:
            existing_row = existing_rows.get(aid, {})
            upload_rows.append({
                "atleta_id": aid,
                "nome": profile.get("nome") or existing_row.get("nome"),
                "foto_url": profile.get("foto_url") or existing_row.get("foto_url"),
                "data_nascimento": profile.get("data_nascimento") or existing_row.get("data_nascimento"),
                "posicao": profile.get("posicao") or existing_row.get("posicao"),
                "genero": profile.get("genero") or existing_row.get("genero") or genero,
                "ativo": existing_row.get("ativo", True),
            })

    if upload_rows:
        df_upload = pd.DataFrame(upload_rows).drop_duplicates(subset=["atleta_id"], keep="last")
        allowed_cols = ["atleta_id", "nome", "foto_url", "data_nascimento", "posicao", "genero", "ativo"]
        df_upload = df_upload[[c for c in allowed_cols if c in df_upload.columns]]
        insert_or_update_table("athletes", df_upload, pk_columns=["atleta_id"])

        df_dim = read_table("athletes")
        if df_dim is None or df_dim.empty:
            raise RuntimeError("Falha ao refrescar os atletas a partir da base de dados.")
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str).str.strip()
        existing = dict(
            zip(
                df_dim.get("atleta_id", pd.Series(dtype="object")),
                pd.to_numeric(df_dim.get("athlete_sk", pd.Series(dtype="float64")), errors="coerce"),
            )
        )

    return {aid: int(sk) for aid, sk in existing.items() if pd.notna(sk)}


def resolve_session_sk(session_fingerprint, session_payload, base_dir=CLEANDATA_DIR):
    """Resolve persistent session_sk from session fingerprint using Supabase."""
    initialize_schema()
    session_fingerprint = str(session_fingerprint or session_payload.get("session_fingerprint") or "").strip()
    if not session_fingerprint:
        raise ValueError("session_fingerprint is required to resolve session_sk.")

    existing_df = read_table("sessions", {"session_fingerprint": session_fingerprint})
    if existing_df is not None and not existing_df.empty and "session_sk" in existing_df.columns:
        session_sk = pd.to_numeric(existing_df["session_sk"], errors="coerce").dropna()
        if not session_sk.empty:
            return int(session_sk.iloc[0])

    started_at = (
        session_payload.get("started_at")
        or session_payload.get("data")
        or session_payload.get("data_sessao")
    )
    sessions_df = read_table("sessions", columns="session_sk")
    if sessions_df is not None and not sessions_df.empty and "session_sk" in sessions_df.columns:
        existing_session_sks = pd.to_numeric(sessions_df["session_sk"], errors="coerce").dropna()
        next_session_sk = int(existing_session_sks.max()) + 1 if not existing_session_sks.empty else 1
    else:
        next_session_sk = 1

    payload = pd.DataFrame([{
        "session_sk": next_session_sk,
        "session_fingerprint": session_fingerprint,
        "started_at": pd.to_datetime(started_at, errors="coerce"),
        "device": session_payload.get("device"),
    }])
    stats = insert_or_update_table("sessions", payload, pk_columns=["session_fingerprint"])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao criar sessão no Supabase.")

    refreshed_df = read_table("sessions", {"session_fingerprint": session_fingerprint})
    if refreshed_df is None or refreshed_df.empty or "session_sk" not in refreshed_df.columns:
        raise RuntimeError("Não foi possível recuperar session_sk a partir do Supabase.")

    session_sk = pd.to_numeric(refreshed_df["session_sk"], errors="coerce").dropna()
    if session_sk.empty:
        raise RuntimeError("session_sk inválido devolvido pelo Supabase.")
    return int(session_sk.iloc[0])


def resolve_game_sk(game_payload, base_dir=CLEANDATA_DIR):
    """Resolve persistent game_sk from game metadata using Supabase."""
    initialize_schema()
    filters = {}
    for key in ["game_date", "opponent", "location", "competition"]:
        value = game_payload.get(key) if isinstance(game_payload, dict) else None
        if value is not None and str(value).strip() != "" and not pd.isna(value):
            filters[key] = value

    if filters:
        existing_df = read_table("games", filters)
        if existing_df is not None and not existing_df.empty and "game_sk" in existing_df.columns:
            game_sk = pd.to_numeric(existing_df["game_sk"], errors="coerce").dropna()
            if not game_sk.empty:
                return int(game_sk.iloc[0])

    payload = pd.DataFrame([{
        "game_date": pd.to_datetime(game_payload.get("game_date"), errors="coerce").date() if isinstance(game_payload, dict) and game_payload.get("game_date") is not None and str(game_payload.get("game_date")).strip() != "" else None,
        "opponent": game_payload.get("opponent") if isinstance(game_payload, dict) else None,
        "location": game_payload.get("location") if isinstance(game_payload, dict) else None,
        "competition": game_payload.get("competition") if isinstance(game_payload, dict) else None,
    }])
    stats = insert_or_update_table("games", payload, pk_columns=[])
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao criar jogo no Supabase.")

    refreshed_df = read_table("games", filters or None)
    if refreshed_df is None or refreshed_df.empty or "game_sk" not in refreshed_df.columns:
        raise RuntimeError("Não foi possível recuperar game_sk a partir do Supabase.")

    refreshed_df["game_sk"] = pd.to_numeric(refreshed_df["game_sk"], errors="coerce")
    refreshed_df = refreshed_df.dropna(subset=["game_sk"]).sort_values("game_sk", ascending=False)
    if refreshed_df.empty:
        raise RuntimeError("game_sk inválido devolvido pelo Supabase.")
    return int(refreshed_df["game_sk"].iloc[0])
