#!/usr/bin/env python
"""
Reset processed GPS/game-session data through the Supabase HTTP API.

Use this when direct PostgreSQL access via psql/psycopg cannot resolve the
db.<project-ref>.supabase.co host.

This preserves reference tables:
  athletes, fields, selecoes

It clears processed/session tables:
  session_reports, samples, quality_metrics, collective_performance_metrics,
  performance_metrics, athlete_session, metrics, sessions, games

Usage:
  python scripts/reset_processed_sessions_supabase_api.py
  python scripts/reset_processed_sessions_supabase_api.py --yes
  python scripts/reset_processed_sessions_supabase_api.py --session-sk 1

Credentials are loaded from environment variables first:
  SUPABASE_URL
  SUPABASE_KEY

If they are missing, the script tries .streamlit/secrets.toml.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

try:
    from supabase import create_client
except ImportError:
    print("Missing dependency: supabase")
    print("Install it with: python -m pip install supabase")
    sys.exit(1)


SESSION_TABLES = [
    "session_reports",
    "quality_metrics",
    "collective_performance_metrics",
    "performance_metrics",
    "athlete_session",
]

ALL_ROW_TABLES = [
    "metrics",
    "games",
]

PHASE_IDS = [0, 1, 2]


def _load_credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if url and key:
        return url, key

    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as fh:
            secrets = tomllib.load(fh)
        supabase = secrets.get("supabase", {})
        url = str(supabase.get("url", "")).strip()
        key = str(supabase.get("key", "")).strip()
        if url and key:
            return url, key

    raise RuntimeError(
        "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY, "
        "or add them to .streamlit/secrets.toml."
    )


def _execute_with_retry(label: str, fn, retries: int = 3) -> Any:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait_s = min(2**attempt, 8)
            print(f"  retry {attempt}/{retries - 1}: {label} failed: {exc}; waiting {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last_exc}") from last_exc


def _read_all(client, table: str, columns: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        query = client.table(table).select(columns)
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        response = _execute_with_retry(
            f"read {table}",
            lambda query=query, offset=offset: query.range(offset, offset + page_size - 1).execute(),
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _has_any_rows(client, table: str) -> bool:
    response = _execute_with_retry(
        f"check {table}",
        lambda: client.table(table).select("*").limit(1).execute(),
    )
    return bool(response.data)


def _delete_query(query, label: str) -> int:
    response = _execute_with_retry(label, lambda: query.execute())
    return len(response.data or [])


def _delete_by_session(client, table: str, session_sk: int) -> int:
    return _delete_query(
        client.table(table).delete().eq("session_sk", session_sk),
        f"delete {table} session_sk={session_sk}",
    )


def _delete_samples_for_session(client, session_sk: int) -> None:
    athlete_rows = _read_all(
        client,
        "athlete_session",
        columns="athlete_sk",
        filters={"session_sk": session_sk},
    )
    athlete_sks = sorted(
        {
            int(row["athlete_sk"])
            for row in athlete_rows
            if row.get("athlete_sk") is not None
        }
    )

    if not athlete_sks:
        print(f"  samples: no athlete_session rows found; deleting by phase for session {session_sk}")
        for phase_id in PHASE_IDS:
            _delete_query(
                client.table("samples").delete().eq("session_sk", session_sk).eq("phase_id", phase_id),
                f"delete samples session={session_sk} phase={phase_id}",
            )
        return

    print(f"  samples: deleting session {session_sk} for {len(athlete_sks)} athletes")
    for athlete_sk in athlete_sks:
        for phase_id in PHASE_IDS:
            _delete_query(
                client.table("samples")
                .delete()
                .eq("session_sk", session_sk)
                .eq("athlete_sk", athlete_sk)
                .eq("phase_id", phase_id),
                f"delete samples session={session_sk} athlete={athlete_sk} phase={phase_id}",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--session-sk", type=int, default=None, help="Delete only this session_sk.")
    args = parser.parse_args()

    url, key = _load_credentials()
    client = create_client(url, key)

    if args.session_sk is None:
        print("This will permanently delete all processed sessions through the Supabase API.")
    else:
        print(f"This will permanently delete processed data for session_sk={args.session_sk}.")
    print("Preserved tables: athletes, fields, selecoes")
    print("Cleared tables: session_reports, samples, quality_metrics, collective_performance_metrics,")
    print("                performance_metrics, athlete_session, metrics, sessions, games")
    print()
    if not args.yes:
        expected = "RESET_PROCESSED_SESSIONS" if args.session_sk is None else f"DELETE_SESSION_{args.session_sk}"
        confirmation = input(f"Type {expected} to continue: ").strip()
        if confirmation != expected:
            print("Cancelled. No database changes were made.")
            return 0

    sessions = _read_all(client, "sessions", columns="session_sk")
    session_sks = sorted(
        {
            int(row["session_sk"])
            for row in sessions
            if row.get("session_sk") is not None
        }
    )
    if args.session_sk is not None:
        session_sks = [sk for sk in session_sks if sk == int(args.session_sk)]

    print(f"Found {len(session_sks)} sessions.")
    for session_sk in session_sks:
        print(f"Deleting session {session_sk}")
        _delete_samples_for_session(client, session_sk)
        for table in SESSION_TABLES:
            _delete_by_session(client, table, session_sk)
        _delete_by_session(client, "sessions", session_sk)

    if args.session_sk is None:
        for table in ALL_ROW_TABLES:
            print(f"Deleting all rows from {table}")
            if table == "games":
                query = client.table(table).delete().neq("game_sk", -1)
            else:
                query = client.table(table).delete().neq("metric_name", "__never__")
            _delete_query(query, f"delete all {table}")

    print()
    print("Remaining processed rows:")
    for table in ["session_reports", "samples", "quality_metrics", "collective_performance_metrics", "performance_metrics", "athlete_session", "sessions"]:
        print(f"  {table}: {'still has rows' if _has_any_rows(client, table) else '0'}")

    print()
    print("Done. If samples is still non-zero, there are orphan rows without a matching sessions row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
