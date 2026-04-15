#!/usr/bin/env python
"""
Backfill samples.x_norm/y_norm through the Supabase HTTP API.

This recalculates normalized pitch coordinates from x_utm/y_utm by session:
  x_norm = (x_utm - min_session_x_utm) / range_session_x_utm * 120
  y_norm = (y_utm - min_session_y_utm) / range_session_y_utm * 80

Use after processed data exists but normalized coordinates were generated
without rebasing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

try:
    from supabase import create_client
except ImportError:
    print("Missing dependency: supabase")
    sys.exit(1)


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
    raise RuntimeError("Supabase credentials not found.")


def _run(label: str, fn, retries: int = 3):
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
    raise RuntimeError(f"{label} failed: {last_exc}") from last_exc


def _read_all(client, table: str, columns: str, filters: dict | None = None) -> pd.DataFrame:
    rows = []
    offset = 0
    page_size = 1000
    while True:
        query = client.table(table).select(columns)
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        response = _run(
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
    return pd.DataFrame(rows)


def _upsert_batches(client, table: str, df: pd.DataFrame, batch_size: int = 1000) -> None:
    records = df.to_dict(orient="records")
    total = (len(records) + batch_size - 1) // batch_size
    for idx in range(total):
        batch = records[idx * batch_size : (idx + 1) * batch_size]
        print(f"  upsert {table}: batch {idx + 1}/{total} ({len(batch)} rows)")
        _run(
            f"upsert {table}",
            lambda batch=batch: client.table(table)
            .upsert(batch, on_conflict="session_sk,athlete_sk,phase_id,time")
            .execute(),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-sk", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    url, key = _load_credentials()
    client = create_client(url, key)

    if not args.yes:
        confirmation = input("Type BACKFILL_XY_NORM to continue: ").strip()
        if confirmation != "BACKFILL_XY_NORM":
            print("Cancelled.")
            return 0

    sessions_df = _read_all(client, "sessions", "session_sk")
    if sessions_df.empty:
        print("No sessions found.")
        return 0

    session_sks = pd.to_numeric(sessions_df["session_sk"], errors="coerce").dropna().astype(int).tolist()
    if args.session_sk is not None:
        session_sks = [sk for sk in session_sks if sk == int(args.session_sk)]

    for session_sk in session_sks:
        print(f"Backfilling session {session_sk}")
        samples = _read_all(
            client,
            "samples",
            columns="session_sk,athlete_sk,phase_id,time,x_utm,y_utm,x_norm,y_norm",
            filters={"session_sk": int(session_sk)},
        )
        if samples.empty:
            print("  no samples")
            continue

        for col in ["x_utm", "y_utm"]:
            samples[col] = pd.to_numeric(samples[col], errors="coerce")
        valid = samples["x_utm"].notna() & samples["y_utm"].notna()
        if not valid.any():
            print("  no valid x_utm/y_utm")
            continue

        x = samples.loc[valid, "x_utm"]
        y = samples.loc[valid, "y_utm"]
        x_range = float(x.max() - x.min())
        y_range = float(y.max() - y.min())
        if x_range <= 0 or y_range <= 0:
            print("  invalid coordinate range")
            continue

        samples.loc[valid, "x_norm"] = ((x - float(x.min())) / x_range * 120.0).clip(0.0, 120.0)
        samples.loc[valid, "y_norm"] = ((y - float(y.min())) / y_range * 80.0).clip(0.0, 80.0)
        upload = samples.loc[valid, ["session_sk", "athlete_sk", "phase_id", "time", "x_norm", "y_norm"]].copy()
        _upsert_batches(client, "samples", upload)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
