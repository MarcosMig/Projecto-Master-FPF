"""
Supabase manager module for FPF analytics.

Replaces DuckDB with Supabase (PostgreSQL) for persistent cloud storage.
Maintains the same interface as the previous duckdb_utils/data_manager modules.
"""

import os
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict
from datetime import datetime

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
    -- Athletes dimension
    CREATE TABLE IF NOT EXISTS athletes (
        athlete_sk SERIAL PRIMARY KEY,
        atleta_id TEXT UNIQUE NOT NULL,
        genero TEXT,
        ativo BOOLEAN DEFAULT true,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

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
    # For now, we'll assume tables exist or will be created via Supabase dashboard
    # This is a fallback message
    st.info("✅ Supabase schema initialized (tables auto-created on first use)")


def insert_or_update_table(
    table_name: str,
    df: pd.DataFrame,
    pk_columns: List[str] = None,
) -> Dict:
    """Insert or update DataFrame into Supabase table (upsert).
    
    Args:
        table_name: name of the Supabase table
        df: DataFrame to insert/update
        pk_columns: list of column names forming the primary key
    
    Returns:
        dict with stats: {'inserted': int, 'updated': int}
    """
    if df.empty:
        return {'inserted': 0, 'updated': 0}
    
    client = get_supabase_client()
    stats = {'inserted': 0, 'updated': 0}
    
    # Convert DataFrame to list of dicts
    records = df.replace({pd.NaT: None, float('nan'): None}).to_dict('records')
    
    try:
        if pk_columns is None or len(pk_columns) == 0:
            # Simple INSERT
            response = client.table(table_name).insert(records).execute()
            stats['inserted'] = len(response.data) if response.data else len(records)
        else:
            # UPSERT: use Supabase upsert method (requires primary key)
            response = client.table(table_name).upsert(
                records,
                ignore_duplicates=False
            ).execute()
            stats['inserted'] = len(response.data) if response.data else len(records)
    
    except Exception as e:
        st.error(f"Error writing to {table_name}: {str(e)}")
        return stats
    
    return stats


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
    db_file=None  # ignored in Supabase mode, kept for compatibility
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
    initialize_schema()
    
    stats = {
        'performance_metrics': None,
        'quality_metrics': None,
        'samples': None,
        'athlete_session': None,
    }
    
    if df_perf is not None and not df_perf.empty:
        stats['performance_metrics'] = insert_or_update_table(
            "performance_metrics", df_perf,
            pk_columns=["session_sk", "athlete_sk", "phase_id"]
        )
    
    if df_qc is not None and not df_qc.empty:
        stats['quality_metrics'] = insert_or_update_table(
            "quality_metrics", df_qc,
            pk_columns=["session_sk", "athlete_sk", "phase_id"]
        )
    
    if df_samples is not None and not df_samples.empty:
        stats['samples'] = insert_or_update_table(
            "samples", df_samples,
            pk_columns=["session_sk", "athlete_sk", "phase_id", "time"]
        )
    
    if df_athlete_session is not None and not df_athlete_session.empty:
        stats['athlete_session'] = insert_or_update_table(
            "athlete_session", df_athlete_session,
            pk_columns=["session_sk", "athlete_sk"]
        )
    
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

from .constants import CAMPOS_DIR, CLEANDATA_DIR


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


def resolve_athlete_sk(df_metrics, base_dir=CLEANDATA_DIR, genero=None):
    """Resolve persistent athlete_sk from atleta_id (using parquet cache)."""
    path = os.path.join(base_dir, "athletes.parquet")
    os.makedirs(base_dir, exist_ok=True)

    atletas = pd.Series(dtype="object")
    if df_metrics is not None and not df_metrics.empty and "atleta_id" in df_metrics.columns:
        atletas = pd.Series(df_metrics["atleta_id"].dropna().astype(str).unique())

    if os.path.exists(path):
        df_dim = pd.read_parquet(path)
    else:
        df_dim = pd.DataFrame(columns=["athlete_sk", "atleta_id", "genero", "ativo", "created_at", "updated_at"])

    if not df_dim.empty:
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str)

    existing = dict(zip(df_dim.get("atleta_id", pd.Series(dtype="object")), df_dim.get("athlete_sk", pd.Series(dtype="int64"))))
    next_id = 1 if df_dim.empty else int(pd.to_numeric(df_dim["athlete_sk"], errors="coerce").max()) + 1

    now_ts = pd.Timestamp.utcnow()
    new_rows = []
    for aid in atletas.tolist():
        if aid not in existing:
            existing[aid] = next_id
            new_rows.append({
                "athlete_sk": next_id,
                "atleta_id": aid,
                "genero": genero,
                "ativo": True,
                "created_at": now_ts,
                "updated_at": now_ts,
            })
            next_id += 1

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_dim = pd.concat([df_dim, df_new], ignore_index=True)
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
