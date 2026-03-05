# -*- coding: utf-8 -*-
"""
Admin de Utilizadores — FPF Performance Hub (Parquet)

IMPORTANTE:
- Para escrever/ler Parquet precisas de um engine: pyarrow OU fastparquet.
  pip install pyarrow bcrypt pandas
  (ou) pip install fastparquet bcrypt pandas

Uso:
  python admin_users.py init
  python admin_users.py add --email "x@y.com" --password "..." --nome "..." --entidade "..." --role admin
  python admin_users.py deactivate --email "x@y.com"
  python admin_users.py list
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import bcrypt

DEFAULT_USERS_PATH = Path("data/users.parquet")


def _hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw, salt).decode("utf-8")


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "email", "password_hash", "nome", "entidade",
        "is_active", "role", "created_at"
    ])


def _load(users_path: Path) -> pd.DataFrame:
    if users_path.exists():
        df = pd.read_parquet(users_path)
    else:
        df = _empty_df()

    if "email" in df.columns and not df.empty:
        df["email"] = df["email"].astype(str).str.strip().str.lower()
    if "is_active" in df.columns and not df.empty:
        df["is_active"] = df["is_active"].astype(bool)
    return df


def _save(df: pd.DataFrame, users_path: Path) -> None:
    users_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(users_path, index=False)


def init(users_path: Path) -> None:
    df = _load(users_path)
    _save(df, users_path)
    print(f"OK: {users_path} pronto ({len(df)} utilizadores).")


def add_user(users_path: Path, email: str, password: str, nome: str, entidade: str,
             role: str = "user", is_active: bool = True) -> None:
    df = _load(users_path)
    email_norm = email.strip().lower()

    if not df.empty and (df["email"] == email_norm).any():
        raise ValueError("Email já existe.")

    row = {
        "email": email_norm,
        "password_hash": _hash_password(password),
        "nome": nome.strip(),
        "entidade": entidade.strip(),
        "is_active": bool(is_active),
        "role": role.strip(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df, users_path)
    print(f"OK: utilizador criado ({email_norm}).")


def deactivate(users_path: Path, email: str) -> None:
    df = _load(users_path)
    if df.empty:
        raise ValueError("Não há utilizadores.")
    email_norm = email.strip().lower()
    m = df["email"] == email_norm
    if not m.any():
        raise ValueError("Email não encontrado.")
    df.loc[m, "is_active"] = False
    _save(df, users_path)
    print(f"OK: utilizador desativado ({email_norm}).")


def list_users(users_path: Path) -> None:
    df = _load(users_path)
    if df.empty:
        print("(vazio)")
        return
    cols = ["email", "nome", "entidade", "role", "is_active", "created_at"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values(["entidade", "nome", "email"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users-path", default=str(DEFAULT_USERS_PATH), help="Caminho para users.parquet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    p_add = sub.add_parser("add")
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--password", required=True)
    p_add.add_argument("--nome", required=True)
    p_add.add_argument("--entidade", required=True)
    p_add.add_argument("--role", default="user")
    p_add.add_argument("--inactive", action="store_true")

    p_deact = sub.add_parser("deactivate")
    p_deact.add_argument("--email", required=True)

    sub.add_parser("list")

    args = ap.parse_args()
    users_path = Path(args.users_path)

    if args.cmd == "init":
        init(users_path)
    elif args.cmd == "add":
        add_user(
            users_path,
            email=args.email,
            password=args.password,
            nome=args.nome,
            entidade=args.entidade,
            role=args.role,
            is_active=not args.inactive,
        )
    elif args.cmd == "deactivate":
        deactivate(users_path, args.email)
    elif args.cmd == "list":
        list_users(users_path)


if __name__ == "__main__":
    main()
