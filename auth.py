# -*- coding: utf-8 -*-
"""
auth.py — Autenticação via Parquet (FPF Performance Hub)

Dependências:
  pip install pandas bcrypt pyarrow

Ficheiro de utilizadores:
  data/users.parquet

Schema esperado (colunas mínimas):
  email, password_hash, nome, entidade, is_active, role
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import bcrypt

USERS_PATH = Path("data/users.parquet")


def load_users() -> pd.DataFrame:
    if not USERS_PATH.exists():
        return pd.DataFrame(columns=[
            "email", "password_hash", "nome", "entidade",
            "is_active", "role", "created_at"
        ])

    df = pd.read_parquet(USERS_PATH)

    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.strip().str.lower()

    if "is_active" not in df.columns:
        df["is_active"] = True
    else:
        df["is_active"] = df["is_active"].astype(bool)

    if "role" not in df.columns:
        df["role"] = "user"

    for col in ["nome", "entidade", "password_hash"]:
        if col not in df.columns:
            df[col] = ""

    return df


def verify_login(email: str, password: str) -> Optional[Dict[str, Any]]:
    df = load_users()
    if df.empty:
        return None

    email_norm = (email or "").strip().lower()
    if not email_norm:
        return None

    user = df[df["email"] == email_norm]
    if user.empty:
        return None

    r = user.iloc[0]
    if not bool(r.get("is_active", True)):
        return None

    ph = str(r.get("password_hash", "") or "")
    if not ph:
        return None

    ok = bcrypt.checkpw(password.encode("utf-8"), ph.encode("utf-8"))
    if not ok:
        return None

    return {
        "email": email_norm,
        "nome": str(r.get("nome", "") or ""),
        "entidade": str(r.get("entidade", "") or ""),
        "role": str(r.get("role", "user") or "user"),
    }


def _ensure_auth_state():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    for k in ["user_nome", "user_entidade", "user_email", "user_role"]:
        if k not in st.session_state:
            st.session_state[k] = ""


def login_screen(title: str = "FPF Performance Hub"):
    _ensure_auth_state()

    st.title(title)

    email = st.text_input("Email", value="", placeholder="nome@entidade.pt")
    password = st.text_input("Password", value="", type="password")

    if st.button("Entrar", type="primary", use_container_width=True):
        user = verify_login(email, password)
        if user is None:
            st.error("Credenciais inválidas ou utilizador inativo.")
            st.stop()

        st.session_state.auth = True
        st.session_state.user_nome = user["nome"]
        st.session_state.user_entidade = user["entidade"]
        st.session_state.user_email = user["email"]
        st.session_state.user_role = user["role"]
        st.rerun()


def require_login(title: str = "FPF Performance Hub"):
    _ensure_auth_state()
    if not st.session_state.auth:
        login_screen(title=title)
        st.stop()


def render_user_header():
    """
    Header no formato:
      FPF
      Marcos Cardoso

    Chamar imediatamente antes de `st.subheader("Dados da Sessão")`.
    """
    _ensure_auth_state()
    entidade = (st.session_state.user_entidade or "").strip()
    nome = (st.session_state.user_nome or "").strip()

    st.markdown(
        f"""
<div style="line-height:1.3;margin-bottom:15px;">
  <div style="font-weight:600;font-size:16px;">{entidade}</div>
  <div style="font-size:14px;color:#9aa0a6;">{nome}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_logout(location: str = "sidebar"):
    def _do_logout():
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if location == "sidebar":
        if st.sidebar.button("Logout", use_container_width=True):
            _do_logout()
    else:
        if st.button("Logout", use_container_width=True):
            _do_logout()
