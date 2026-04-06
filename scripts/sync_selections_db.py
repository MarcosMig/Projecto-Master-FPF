#!/usr/bin/env python
"""Sync selections reference data into Supabase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fpf_modules.supabase_manager import read_selections_reference, sync_selections_reference


def main() -> int:
    df = sync_selections_reference(include_default=True)
    print(f"Selecoes sincronizadas: {len(df)}")

    try:
        current_df = read_selections_reference(active_only=False)
        print("Codigos atuais:")
        for codigo in current_df["codigo"].astype(str).tolist():
            print(f" - {codigo}")
    except Exception as exc:
        print(f"Nao foi possivel reler as selecoes: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
