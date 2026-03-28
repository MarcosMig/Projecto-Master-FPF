"""
Supabase manager module for FPF analytics.

Replaces DuckDB with Supabase (PostgreSQL) for persistent cloud storage.
Maintains the same interface as the previous duckdb_utils/data_manager modules.
"""

import os
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Callable
from datetime import date, datetime
import math
import time
from .constants import CAMPOS_DIR, CLEANDATA_DIR

try:
    import numpy as np
except ImportError:
    np = None

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None


def get_supabase_client() -> Client:
    """Get or create a Supabase client using credentials from Streamlit secrets."""
    if not SUPABASE_AVAILABLE:
        raise ImportError(
            "Supabase is not installed. Install it with: pip install supabase"
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


def initialize_schema() -> None:
    """Create all necessary tables in Supabase if they don't exist."""
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
        PRIMARY KEY (session_sk, athlete_sk, phase_id)
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
    """Ensure referenced sessions and athletes exist in Supabase before fact inserts."""
    base_dir = base_dir or CLEANDATA_DIR

    if athlete_sks:
        athletes_path = os.path.join(base_dir, "athletes.parquet")
        if os.path.exists(athletes_path):
            df_athletes = pd.read_parquet(athletes_path)
            if not df_athletes.empty and "athlete_sk" in df_athletes.columns:
                df_athletes = df_athletes[df_athletes["athlete_sk"].isin(athlete_sks)].copy()
                allowed_cols = [
                    "athlete_sk", "atleta_id", "nome", "data_nascimento", "posicao",
                    "numero_camisola", "pe_preferencial", "altura_cm", "peso_kg", "escalao", "selecao",
                    "genero", "ativo", "created_at", "updated_at"
                ]
                df_athletes = df_athletes[[c for c in allowed_cols if c in df_athletes.columns]]
                if not df_athletes.empty:
                    insert_or_update_table(
                        "athletes",
                        df_athletes,
                        pk_columns=["athlete_sk"],
                    )

    if session_sks:
        sessions_path = os.path.join(base_dir, "sessions.parquet")
        if os.path.exists(sessions_path):
            df_sessions = pd.read_parquet(sessions_path)
            if not df_sessions.empty and "session_sk" in df_sessions.columns:
                df_sessions = df_sessions[df_sessions["session_sk"].isin(session_sks)].copy()
                if "session_fingerprint" not in df_sessions.columns:
                    return

                if "started_at" not in df_sessions.columns:
                    if "data" in df_sessions.columns:
                        df_sessions["started_at"] = pd.to_datetime(df_sessions["data"], errors="coerce")
                    else:
                        df_sessions["started_at"] = pd.NaT

                if "device" not in df_sessions.columns:
                    df_sessions["device"] = None

                allowed_cols = ["session_sk", "session_fingerprint", "started_at", "device"]
                df_sessions = df_sessions[[c for c in allowed_cols if c in df_sessions.columns]]
                if not df_sessions.empty:
                    insert_or_update_table(
                        "sessions",
                        df_sessions,
                        pk_columns=["session_sk"],
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


def read_table(table_name: str, filters: Dict = None) -> pd.DataFrame:
    """Read data from Supabase table.
    
    Args:
        table_name: name of the Supabase table
        filters: optional dict of {column: value} for filtering
    
    Returns:
        DataFrame with query results
    """
    client = get_supabase_client()
    
    try:
        query = client.table(table_name).select("*")
        
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        
        response = query.execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error reading from {table_name}: {str(e)}")
        return pd.DataFrame()


# ==================== High-level API (matches old data_manager interface) ====================

def write_session_data(
    df_perf, df_qc, df_samples, df_athlete_session,
    db_file=None,  # ignored in Supabase mode, kept for compatibility
    progress_callback: Optional[Callable] = None,
):
    """Persist performance, quality, samples, and athlete_session DataFrames to Supabase.
    
    This is the main integration point for the analytic pipeline.
    Uses upsert logic to avoid duplicates.
    
    Args:
        df_perf: performance metrics DataFrame
        df_qc: quality metrics DataFrame
        df_samples: samples (tracking) DataFrame
        df_athlete_session: athlete session participation DataFrame
        db_file: ignored (for compatibility with old DuckDB interface)
    
    Returns:
        dict with integration stats
    """
    total_steps = 2 + sum(
        1 for df in [df_perf, df_qc, df_samples, df_athlete_session]
        if df is not None and not df.empty
    )

    current_step = 1
    _emit_progress(progress_callback, current_step, total_steps, "A validar esquema e preparar ligação ao Supabase...")
    initialize_schema()
    
    stats = {
        'performance_metrics': None,
        'quality_metrics': None,
        'samples': None,
        'athlete_session': None,
    }

    session_sks = set()
    athlete_sks = set()
    for df in [df_perf, df_qc, df_samples, df_athlete_session]:
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
    
    if df_perf is not None and not df_perf.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar performance_metrics ({len(df_perf)} linhas)...", "performance_metrics")
        stats['performance_metrics'] = insert_or_update_table(
            "performance_metrics", df_perf,
            pk_columns=["session_sk", "athlete_sk", "phase_id"],
        )
        if not stats['performance_metrics'].get("success", False):
            raise RuntimeError(f"Falha ao gravar performance_metrics: {stats['performance_metrics'].get('error')}")
    
    if df_qc is not None and not df_qc.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar quality_metrics ({len(df_qc)} linhas)...", "quality_metrics")
        stats['quality_metrics'] = insert_or_update_table(
            "quality_metrics", df_qc,
            pk_columns=["session_sk", "athlete_sk", "phase_id"],
        )
        if not stats['quality_metrics'].get("success", False):
            raise RuntimeError(f"Falha ao gravar quality_metrics: {stats['quality_metrics'].get('error')}")
    
    if df_samples is not None and not df_samples.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar samples ({len(df_samples)} linhas)...", "samples")
        stats['samples'] = insert_or_update_table(
            "samples", df_samples,
            pk_columns=["session_sk", "athlete_sk", "phase_id", "time"],
            batch_size=1500,
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
            raise RuntimeError(f"Falha ao gravar samples: {stats['samples'].get('error')}")
    
    if df_athlete_session is not None and not df_athlete_session.empty:
        current_step += 1
        _emit_progress(progress_callback, current_step, total_steps, f"A gravar athlete_session ({len(df_athlete_session)} linhas)...", "athlete_session")
        stats['athlete_session'] = insert_or_update_table(
            "athlete_session", df_athlete_session,
            pk_columns=["session_sk", "athlete_sk"]
        )
        if not stats['athlete_session'].get("success", False):
            raise RuntimeError(f"Falha ao gravar athlete_session: {stats['athlete_session'].get('error')}")

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

def save_field_to_parquet(df_new, filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """
    Appends new field data to the parquet database.
    Deduplicates by field corner coordinates to avoid repeated entries.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        df_old = pd.read_parquet(filename)
        df_final = pd.concat([df_old, df_new]).drop_duplicates(
            subset=['BL_lat', 'BR_lat', 'TL_lat', 'TR_lat'], keep='last'
        )
    else:
        df_final = df_new

    df_final.to_parquet(filename, index=False)


def read_field_from_parquet(filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """Read field data from parquet database."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        return pd.read_parquet(filename)
    else:
        return pd.DataFrame()


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
    """Resolve persistent athlete_sk from atleta_id (using parquet cache)."""
    path = os.path.join(base_dir, "athletes.parquet")
    os.makedirs(base_dir, exist_ok=True)

    atletas = pd.Series(dtype="object")
    if df_metrics is not None and not df_metrics.empty and "atleta_id" in df_metrics.columns:
        atletas = pd.Series(df_metrics["atleta_id"].dropna().astype(str).unique())

    if os.path.exists(path):
        df_dim = pd.read_parquet(path)
    else:
        df_dim = pd.DataFrame(columns=[
            "athlete_sk", "atleta_id", "nome", "data_nascimento", "posicao",
            "numero_camisola", "pe_preferencial", "altura_cm", "peso_kg", "escalao", "selecao",
            "genero", "ativo", "created_at", "updated_at"
        ])

    if not df_dim.empty:
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str)

    profiles_map = {}
    if athlete_profiles is not None and not athlete_profiles.empty and "atleta_id" in athlete_profiles.columns:
        df_profiles = athlete_profiles.copy()
        df_profiles["atleta_id"] = df_profiles["atleta_id"].astype(str)
        profiles_map = df_profiles.set_index("atleta_id").to_dict(orient="index")

    existing = dict(zip(df_dim.get("atleta_id", pd.Series(dtype="object")), df_dim.get("athlete_sk", pd.Series(dtype="int64"))))
    next_id = 1 if df_dim.empty else int(pd.to_numeric(df_dim["athlete_sk"], errors="coerce").max()) + 1

    now_ts = pd.Timestamp.utcnow()
    new_rows = []
    for aid in atletas.tolist():
        profile = profiles_map.get(aid, {})
        if aid not in existing:
            existing[aid] = next_id
            new_rows.append({
                "athlete_sk": next_id,
                "atleta_id": aid,
                "nome": profile.get("nome"),
                "data_nascimento": profile.get("data_nascimento"),
                "posicao": profile.get("posicao"),
                "numero_camisola": profile.get("numero_camisola"),
                "pe_preferencial": profile.get("pe_preferencial"),
                "altura_cm": profile.get("altura_cm"),
                "peso_kg": profile.get("peso_kg"),
                "escalao": profile.get("escalao"),
                "selecao": profile.get("selecao", selecao),
                "genero": profile.get("genero", genero),
                "ativo": True,
                "created_at": now_ts,
                "updated_at": now_ts,
            })
            next_id += 1
        elif profile:
            idx = df_dim["atleta_id"].astype(str) == aid
            for col in ["nome", "data_nascimento", "posicao", "numero_camisola", "pe_preferencial", "altura_cm", "peso_kg", "escalao", "selecao"]:
                if col in df_dim.columns and profile.get(col) is not None and not pd.isna(profile.get(col)):
                    df_dim.loc[idx, col] = profile.get(col)
            if "genero" in df_dim.columns and genero is not None:
                df_dim.loc[idx, "genero"] = df_dim.loc[idx, "genero"].fillna(profile.get("genero", genero))
            if "updated_at" in df_dim.columns:
                df_dim.loc[idx, "updated_at"] = now_ts

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_dim = pd.concat([df_dim, df_new], ignore_index=True)
    if new_rows or profiles_map:
        df_dim.to_parquet(path, index=False)

    return existing


def resolve_session_sk(session_fingerprint, session_payload, base_dir=CLEANDATA_DIR):
    """Resolve persistent session_sk from session fingerprint (using parquet cache)."""
    path = os.path.join(base_dir, "sessions.parquet")
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame()

    if not df.empty and "session_fingerprint" in df.columns:
        mask = df["session_fingerprint"].astype(str) == str(session_fingerprint)
        if mask.any():
            return int(df.loc[mask, "session_sk"].iloc[0])

    next_id = 1 if df.empty else int(pd.to_numeric(df["session_sk"], errors="coerce").max()) + 1

    payload = dict(session_payload)
    payload["session_sk"] = next_id
    if "created_at" not in payload:
        payload["created_at"] = pd.Timestamp.utcnow()

    df_new = pd.DataFrame([payload])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_parquet(path, index=False)

    return next_id


def resolve_game_sk(game_payload, base_dir=CLEANDATA_DIR):
    """Resolve persistent game_sk from game metadata (using parquet cache)."""
    path = os.path.join(base_dir, "games.parquet")
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame()

    if not df.empty:
        mask = pd.Series([True] * len(df))
        for k, v in game_payload.items():
            if k in df.columns:
                mask &= df[k] == v
        if mask.any():
            return int(df.loc[mask, "game_sk"].iloc[0])

    next_id = 1 if df.empty else int(pd.to_numeric(df["game_sk"], errors="coerce").max()) + 1

    rec = dict(game_payload)
    rec["game_sk"] = next_id
    if "game_date" in rec and pd.isna(rec.get("game_date")):
        rec["game_date"] = pd.NaT

    df_new = pd.DataFrame([rec])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_parquet(path, index=False)

    return next_id
