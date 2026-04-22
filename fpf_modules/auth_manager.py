import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


USERS_FILE = Path(__file__).resolve().parents[1] / "Data" / "admin" / "app_users.json"
PBKDF2_ITERATIONS = 200_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _ensure_store() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("[]", encoding="utf-8")


def _load_users() -> list[dict]:
    _ensure_store()
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = []
    return data if isinstance(data, list) else []


def _save_users(users: list[dict]) -> None:
    _ensure_store()
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), password_hash.hex()


def _verify_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    _, candidate_hash = _hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(candidate_hash, str(password_hash_hex or ""))


def get_secrets_auth() -> tuple[str | None, str | None]:
    try:
        auth = st.secrets["auth"]
        username = str(auth.get("username") or "").strip()
        password = str(auth.get("password") or "")
        return (username or None), (password or None)
    except Exception:
        return None, None


def ensure_auth_state() -> None:
    defaults = {
        "auth": False,
        "login_user": "",
        "login_role": "",
        "login_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def login_user(user: dict) -> None:
    st.session_state.auth = True
    st.session_state.login_user = str(user.get("username", ""))
    st.session_state.login_role = str(user.get("role", "viewer"))
    st.session_state.login_source = str(user.get("source", ""))


def logout_user() -> None:
    for key in (
        "auth",
        "login_user",
        "login_role",
        "login_source",
        "admin_auth",
        "admin_user",
        "selected_modality",
    ):
        st.session_state.pop(key, None)
    ensure_auth_state()


def render_login_page() -> None:
    ensure_auth_state()
    st.markdown(
        """
        <style>
          .stApp { background-color: #0e1117; }
          header, footer {visibility: hidden;}
          [data-testid="stSidebar"] {display: none;}
          .main .block-container {
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
            padding-top: 10vh !important;
            max-width: 430px !important;
            margin-left: auto !important;
            margin-right: auto !important;
          }
          [data-testid="stVerticalBlock"] {
            gap: 0.85rem;
          }
          .login-title{
            text-align: center;
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 1rem 0;
            letter-spacing: 0;
          }
          .stTextInput input{
            background: #0e1117 !important;
            border: 1px solid #30363d !important;
            border-radius: 10px !important;
            height: 2.75rem;
          }
          .stButton > button{
            width: 100%;
            background: #E30613 !important;
            color: #fff !important;
            font-weight: 800;
            border: 0;
            height: 2.8rem;
            border-radius: 10px;
            margin-top: 0.35rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.35, 1])
    with center:
        st.markdown('<div class="login-title">FPF Performance Hub</div>', unsafe_allow_html=True)

        username = st.text_input("Utilizador", key="global_login_user")
        password = st.text_input("Password", type="password", key="global_login_password")

        if st.button("Entrar", type="primary"):
            auth_result = authenticate_user(username, password)
            if auth_result.get("success"):
                login_user(auth_result.get("user", {}))
                st.rerun()
            st.error("Credenciais invalidas.")


def require_login() -> None:
    ensure_auth_state()
    if not st.session_state.auth:
        render_login_page()
        st.stop()


def require_admin() -> None:
    require_login()
    if st.session_state.get("login_role") != "admin":
        st.error("Acesso reservado a administradores.")
        st.stop()


def list_users() -> pd.DataFrame:
    users = _load_users()
    rows = []
    for user in users:
        rows.append(
            {
                "username": user.get("username", ""),
                "role": user.get("role", "viewer"),
                "active": bool(user.get("active", True)),
                "created_at": user.get("created_at"),
                "created_by": user.get("created_by", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["username", "role", "active", "created_at", "created_by"])
    return pd.DataFrame(rows).sort_values(["role", "username"], na_position="last").reset_index(drop=True)


def authenticate_user(username: str, password: str) -> dict:
    normalized = _normalize_username(username)
    raw_password = str(password or "")
    if not normalized or not raw_password:
        return {"success": False, "reason": "missing_credentials"}

    for user in _load_users():
        if _normalize_username(user.get("username")) != normalized:
            continue
        if not bool(user.get("active", True)):
            return {"success": False, "reason": "inactive"}
        if _verify_password(raw_password, str(user.get("salt", "")), str(user.get("password_hash", ""))):
            return {
                "success": True,
                "user": {
                    "username": str(user.get("username", "")),
                    "role": str(user.get("role", "viewer")),
                    "source": "file",
                },
            }
        return {"success": False, "reason": "invalid_password"}

    secrets_user, secrets_pass = get_secrets_auth()
    if secrets_user and secrets_pass and username == secrets_user and raw_password == secrets_pass:
        return {
            "success": True,
            "user": {
                "username": secrets_user,
                "role": "admin",
                "source": "secrets",
            },
        }

    return {"success": False, "reason": "not_found"}


def create_user(username: str, password: str, role: str = "viewer", created_by: str = "") -> dict:
    normalized = _normalize_username(username)
    if len(normalized) < 3:
        return {"success": False, "error": "O utilizador tem de ter pelo menos 3 caracteres."}
    if len(str(password or "")) < 8:
        return {"success": False, "error": "A password tem de ter pelo menos 8 caracteres."}
    if role not in {"admin", "editor", "viewer"}:
        return {"success": False, "error": "Role inválido."}

    users = _load_users()
    if any(_normalize_username(user.get("username")) == normalized for user in users):
        return {"success": False, "error": "Esse utilizador já existe."}

    secrets_user, _ = get_secrets_auth()
    if secrets_user and normalized == _normalize_username(secrets_user):
        return {"success": False, "error": "Esse utilizador já está reservado nas credenciais base da aplicação."}

    salt_hex, password_hash_hex = _hash_password(password)
    users.append(
        {
            "username": normalized,
            "salt": salt_hex,
            "password_hash": password_hash_hex,
            "role": role,
            "active": True,
            "created_at": _utc_now_iso(),
            "created_by": str(created_by or "").strip(),
        }
    )
    _save_users(users)
    return {"success": True}
