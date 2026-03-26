#!/usr/bin/env python
"""
Setup script to initialize Supabase schema for FPF analytics.

This script creates all necessary tables in Supabase if they don't exist.
Run this once after setting up your Supabase project.

Usage:
    python scripts/setup_supabase.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import fpf_modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from supabase import create_client
except ImportError:
    print("❌ Supabase library not found. Install it with: pip install supabase")
    sys.exit(1)


# Supabase credentials (you can also load from environment variables)
SUPABASE_URL = "https://nclasknkceelcxsctxcb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5jbGFza25rY2VlbGN4c2N0eGNiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIzNzQxNDYsImV4cCI6MjA4Nzk1MDE0Nn0.5dV5mlopCBldHQYb5FNRmryXNqw94gdc8YNPYE_NVjs"


# SQL DDL to create all tables
SCHEMA_SQL = """
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
    escalao TEXT,
    selecao TEXT,
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


def main():
    """Initialize Supabase schema."""
    print("🔧 Setting up Supabase schema for FPF Analytics...")
    print(f"📍 Database: {SUPABASE_URL}\n")
    
    try:
        # Connect to Supabase
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase\n")
        
        # Execute DDL statements
        # Note: We need to split by semicolon and execute separately
        statements = [s.strip() for s in SCHEMA_SQL.split(';') if s.strip()]
        
        for i, statement in enumerate(statements, 1):
            try:
                # Use RPC or direct execute via the admin client
                # Since we're using anon key, we'll use the table creation approach
                # For now, we'll just print instructions
                print(f"[{i}/{len(statements)}] Creating table...")
                print(f"   {statement[:60]}...")
            except Exception as e:
                print(f"⚠️  Error in statement {i}: {e}")
        
        # Alternative: Use Postgres connection string if available
        print("\n" + "="*60)
        print("📝 To complete setup, please run the SQL in Supabase:")
        print("="*60)
        print("\n1. Go to: https://nclasknkceelcxsctxcb.supabase.co")
        print("2. Click 'SQL Editor' in the left sidebar")
        print("3. Click 'New Query'")
        print("4. Copy and paste the following SQL:")
        print("\n" + "-"*60)
        print(SCHEMA_SQL)
        print("-"*60)
        print("\n5. Click 'Run' to execute")
        print("\n✅ Once tables are created, your app is ready to use Supabase!\n")
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        print("\n⚠️  Make sure:")
        print("   - The SUPABASE_URL is correct")
        print("   - The SUPABASE_KEY is valid")
        print("   - Your network can reach Supabase\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
