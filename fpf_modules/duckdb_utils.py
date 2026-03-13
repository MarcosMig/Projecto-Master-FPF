"""
Utilities for working with DuckDB and parquet files.

DuckDB can query parquet files directly without needing to convert them,
but it's often convenient to create a persistent DuckDB database and
attach parquet data as tables for fast analytical queries.
"""

import os
from typing import List, Optional
import pandas as pd

# Try to import duckdb; if not available, provide helpful error on first use
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


def connect(db_path: str = ":memory:"):
    """Open a connection to a DuckDB database file.

    Args:
        db_path: path to the .duckdb file or ":memory:" for an in-memory database.

    Returns:
        A DuckDB connection object.
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError(
            "DuckDB is not installed. Install it with: pip install duckdb"
        )
    
    # make sure directory exists if using a file-backed database
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path)


def register_parquet(
    con,
    parquet_path: str,
    table_name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Register a parquet file as a DuckDB table.

    The table is created inside the given connection; subsequent SQL
    queries can reference it directly.

    Args:
        con: open DuckDB connection.
        parquet_path: filesystem path to the parquet file.
        table_name: name of the table in DuckDB.  If not provided the
            filename (without extension) is used.
        overwrite: if True and the table already exists, it will be
            dropped first.

    Returns:
        The name of the table created.
    """
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(parquet_path))[0]
    if overwrite:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
    # DuckDB can directly read parquet files via parquet_scan / parquet_read
    con.execute(
        f"CREATE TABLE {table_name} AS SELECT * FROM parquet_scan('{parquet_path}')"
    )
    return table_name


def register_directory(
    con,
    directory: str,
    suffix: str = ".parquet",
    overwrite: bool = False,
) -> List[str]:
    """Walk a directory and register every parquet file it contains.

    Args:
        con: open DuckDB connection.
        directory: base path to search for parquet files.
        suffix: file name suffix to include; defaults to ".parquet".
        overwrite: see :func:`register_parquet`.

    Returns:
        A list of table names that were registered.
    """
    tables: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith(suffix.lower()):
                full = os.path.join(root, fn)
                name = os.path.splitext(fn)[0]
                register_parquet(con, full, table_name=name, overwrite=overwrite)
                tables.append(name)
    return tables


def query(con, sql: str, **params):
    """Run an SQL query and return a pandas DataFrame."""
    return con.execute(sql, params).fetchdf()


def initialize_schema(con) -> None:
    """Create the analytic schema (dimensions + fact table) if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS athletes (
        athlete_sk   INTEGER PRIMARY KEY,
        atleta_id    TEXT UNIQUE,
        genero       TEXT,
        ativo        BOOLEAN,
        created_at   TIMESTAMP,
        updated_at   TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_sk          INTEGER PRIMARY KEY,
        session_fingerprint TEXT UNIQUE,
        started_at          TIMESTAMP,
        device              TEXT
    );

    CREATE TABLE IF NOT EXISTS games (
        game_sk     INTEGER PRIMARY KEY,
        game_date   DATE,
        opponent    TEXT,
        location    TEXT,
        competition TEXT
    );

    CREATE TABLE IF NOT EXISTS metrics (
        athlete_sk   INTEGER REFERENCES athletes(athlete_sk),
        session_sk   INTEGER REFERENCES sessions(session_sk),
        game_sk      INTEGER REFERENCES games(game_sk),
        timestamp    TIMESTAMP,
        metric_name  TEXT,
        metric_value DOUBLE,
        PRIMARY KEY (athlete_sk, session_sk, game_sk, timestamp, metric_name)
    );
    """
    con.execute(ddl)


def insert_metrics(con, df_metrics) -> None:
    """Append a DataFrame of metrics into the metrics fact table.

    The dataframe should contain at least the columns:
    ['athlete_sk','session_sk','game_sk','timestamp','metric_name','metric_value']
    """
    # duckdb can ingest pandas directly
    con.register("_tmp_metrics", df_metrics)
    con.execute(
        "INSERT INTO metrics SELECT * FROM _tmp_metrics;"
    )
    con.unregister("_tmp_metrics")


def insert_table(con, table_name: str, df, pk_columns: list = None) -> dict:
    """Insert or update DataFrame into a DuckDB table (upsert based on primary key).
    
    Args:
        con: DuckDB connection
        table_name: name of the target table
        df: DataFrame to insert
        pk_columns: list of column names that form the primary key. If None, simple INSERT.
    
    Returns:
        dict with stats: {'inserted': int, 'updated': int, 'skipped': int}
    """
    if df.empty:
        return {'inserted': 0, 'updated': 0, 'skipped': 0}
    
    stats = {'inserted': 0, 'updated': 0, 'skipped': 0}
    
    # Pre-process timestamp columns to ensure proper format
    df_processed = df.copy()
    for col in df_processed.columns:
        if col.lower() == 'time' and df_processed[col].dtype == 'object':
            # Special handling for 'time' column - convert relative time to full timestamp
            try:
                time_strs = df_processed[col].astype(str)
                # Check if strings contain date separators
                has_dates = time_strs.str.contains('-', na=False)
                if has_dates.any():
                    # Some values have dates, try full timestamp format first
                    df_processed[col] = pd.to_datetime(df_processed[col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    # Fill any NaT values with time-only parsing
                    still_nat = df_processed[col].isna()
                    if still_nat.any():
                        time_only = pd.to_datetime(df_processed[col].astype(str), format='%H:%M:%S.%f', errors='coerce')
                        base_date = pd.Timestamp('2023-01-01')
                        df_processed.loc[still_nat, col] = base_date + (time_only.loc[still_nat] - time_only.loc[still_nat].dt.normalize())
                else:
                    # Handle relative time format by prepending base date
                    full_timestamps = '2023-01-01 ' + time_strs
                    df_processed[col] = pd.to_datetime(full_timestamps, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                # Keep as datetime for DuckDB TIMESTAMP compatibility
            except Exception:
                # Fallback conversion
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
        elif ('timestamp' in col.lower() or 'processado_em' in col.lower()) and df_processed[col].dtype in ['object', 'datetime64[ns]']:
            # Handle other timestamp columns - keep as datetime for DuckDB compatibility
            try:
                # Try full timestamp format first
                df_processed[col] = pd.to_datetime(df_processed[col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                # If many values are still NaT, try without format
                if df_processed[col].isna().mean() > 0.5:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except Exception:
                # Fallback conversion
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
    
    con.register(f"_tmp_{table_name}", df_processed)
    
    # Validate that required columns don't have NULL values
    if table_name == 'samples' and 'time' in df_processed.columns:
        null_times = df_processed['time'].isna().sum()
        if null_times > 0:
            raise ValueError(f"Cannot insert {null_times} rows with NULL time values into samples table. Time column is NOT NULL.")
    
    if pk_columns is None or len(pk_columns) == 0:
        # Simple INSERT if no PK specified
        con.execute(f"INSERT INTO {table_name} SELECT * FROM _tmp_{table_name};")
        stats['inserted'] = len(df)
    else:
        # UPSERT: Delete existing rows with same PK, then insert all
        # Build WHERE clause with proper type casting for timestamps
        pk_conditions = []
        for col in pk_columns:
            pk_conditions.append(f"{table_name}.{col} = _tmp_{table_name}.{col}")
        pk_where_clause = " AND ".join(pk_conditions)
        
        # Build SELECT clause for subquery (data is already pre-processed)
        select_clause = ", ".join(pk_columns)
        
        # Count existing rows before delete
        count_existing = con.execute(
            f"SELECT COUNT(*) as cnt FROM {table_name} WHERE ({', '.join(pk_columns)}) IN (SELECT {select_clause} FROM _tmp_{table_name})"
        ).fetchall()[0][0]
        
        # Delete existing rows with same PK
        con.execute(f"DELETE FROM {table_name} WHERE ({', '.join(pk_columns)}) IN (SELECT {select_clause} FROM _tmp_{table_name});")
        
        # Insert all new rows
        con.execute(f"INSERT INTO {table_name} SELECT * FROM _tmp_{table_name};")
        
        stats['updated'] = count_existing
        stats['inserted'] = len(df) - count_existing
    
    con.unregister(f"_tmp_{table_name}")
    return stats



def create_analytics_tables(con) -> None:
    """Create additional analytics tables (performance, quality, samples, athlete_session)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS performance_metrics (
        session_sk      INTEGER,
        athlete_sk      INTEGER,
        atleta_id       TEXT,
        phase_id        INTEGER,
        fase            TEXT,
        data            DATE,
        selecao         TEXT,
        genero          TEXT,
        contexto        TEXT,
        jogo            TEXT,
        duracao_min     DOUBLE,
        dist_m          DOUBLE,
        m_min           DOUBLE,
        vmax_mps        DOUBLE,
        peak_1m_m_min   DOUBLE,
        hsr_dist_m      DOUBLE,
        hsr_pct         DOUBLE,
        sprint_dist_m   DOUBLE,
        n_sprints       INTEGER,
        n_acc_2_5       INTEGER,
        n_dec_3_0       INTEGER,
        active_time_min DOUBLE,
        active_pct      DOUBLE,
        PRIMARY KEY (session_sk, athlete_sk, phase_id)
    );

    CREATE TABLE IF NOT EXISTS quality_metrics (
        session_sk      INTEGER,
        athlete_sk      INTEGER,
        atleta_id       TEXT,
        phase_id        INTEGER,
        fase            TEXT,
        n_points        INTEGER,
        pct_time_valid  DOUBLE,
        n_gaps_gt2s     INTEGER,
        qc_grade        TEXT,
        qc_flags        TEXT,
        vmax_mps_qc     DOUBLE,
        n_jumps_gt15m   INTEGER,
        n_gaps_gt2s_qc  INTEGER,
        PRIMARY KEY (session_sk, athlete_sk, phase_id)
    );

    CREATE TABLE IF NOT EXISTS samples (
        session_sk      INTEGER,
        athlete_sk      INTEGER,
        atleta_id       TEXT,
        fase            TEXT,
        time            TIMESTAMP,
        time_evento_s   DOUBLE,
        time_evento     TEXT,
        periodo_jogo    TEXT,
        minuto_jogo     INTEGER,
        lat             DOUBLE,
        lon             DOUBLE,
        x_utm           DOUBLE,
        y_utm           DOUBLE,
        x_norm          DOUBLE,
        y_norm          DOUBLE,
        speed_mps       DOUBLE,
        acc_mps2        DOUBLE,
        hr_bpm          DOUBLE,
        phase_id        INTEGER,
        PRIMARY KEY (session_sk, athlete_sk, phase_id, time)
    );

    CREATE TABLE IF NOT EXISTS athlete_session (
        session_sk              INTEGER,
        athlete_sk              INTEGER,
        atleta_id               TEXT,
        participou_warmup       BOOLEAN,
        participou_1p           BOOLEAN,
        participou_2p           BOOLEAN,
        fases_disponiveis       TEXT,
        n_samples               INTEGER,
        tem_hr                  BOOLEAN,
        processado_em           TIMESTAMP,
        PRIMARY KEY (session_sk, athlete_sk)
    );
    """
    con.execute(ddl)


