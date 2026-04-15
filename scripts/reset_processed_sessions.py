#!/usr/bin/env python
"""
Reset processed GPS/game-session data in Supabase PostgreSQL.

This preserves reference tables:
  athletes, fields, selecoes

It clears processed/session tables:
  session_reports, samples, quality_metrics, collective_performance_metrics,
  performance_metrics, athlete_session, metrics, sessions, games

Usage:
  $env:DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres"
  python scripts/reset_processed_sessions.py

Optional:
  python scripts/reset_processed_sessions.py --yes
"""

from __future__ import annotations

import argparse
import os
import sys


RESET_SQL = """
BEGIN;

TRUNCATE TABLE
    session_reports,
    samples,
    quality_metrics,
    collective_performance_metrics,
    performance_metrics,
    athlete_session,
    metrics,
    sessions,
    games
RESTART IDENTITY;

ALTER TABLE samples ADD COLUMN IF NOT EXISTS x_norm DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS y_norm DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS speed_mps DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS acc_mps2 DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_samples_session_phase_time
ON samples(session_sk, phase_id, time_evento_s);

COMMIT;
"""


VERIFY_SQL = """
SELECT 'session_reports' AS table_name, COUNT(*) AS rows_remaining FROM session_reports
UNION ALL SELECT 'samples', COUNT(*) FROM samples
UNION ALL SELECT 'quality_metrics', COUNT(*) FROM quality_metrics
UNION ALL SELECT 'collective_performance_metrics', COUNT(*) FROM collective_performance_metrics
UNION ALL SELECT 'performance_metrics', COUNT(*) FROM performance_metrics
UNION ALL SELECT 'athlete_session', COUNT(*) FROM athlete_session
UNION ALL SELECT 'metrics', COUNT(*) FROM metrics
UNION ALL SELECT 'sessions', COUNT(*) FROM sessions
UNION ALL SELECT 'games', COUNT(*) FROM games
ORDER BY table_name;
"""


def _load_psycopg():
    try:
        import psycopg
    except ImportError:
        print("Missing dependency: psycopg")
        print('Install it with: python -m pip install "psycopg[binary]"')
        sys.exit(1)
    return psycopg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is not set.")
        print('Example:')
        print('$env:DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres"')
        return 1

    print("This will permanently delete processed sessions from Supabase.")
    print("Preserved tables: athletes, fields, selecoes")
    print("Cleared tables: session_reports, samples, quality_metrics, collective_performance_metrics,")
    print("                performance_metrics, athlete_session, metrics, sessions, games")
    print()

    if not args.yes:
        confirmation = input("Type RESET_PROCESSED_SESSIONS to continue: ").strip()
        if confirmation != "RESET_PROCESSED_SESSIONS":
            print("Cancelled. No database changes were made.")
            return 0

    psycopg = _load_psycopg()
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(RESET_SQL)
        with conn.cursor() as cur:
            cur.execute(VERIFY_SQL)
            rows = cur.fetchall()

    print()
    print("Rows remaining after reset:")
    for table_name, rows_remaining in rows:
        print(f"  {table_name}: {rows_remaining}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
