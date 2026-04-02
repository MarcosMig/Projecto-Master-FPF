import streamlit as st

from fpf_modules.auth_manager import authenticate_user, create_user, list_users


st.set_page_config(page_title="Administração | Utilizadores", layout="wide")


def _ensure_admin_auth() -> None:
    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False
    if "admin_user" not in st.session_state:
        st.session_state.admin_user = ""


_ensure_admin_auth()

st.title("Administração")
st.caption("Criar e consultar utilizadores da aplicação.")

if not st.session_state.admin_auth:
    st.subheader("Autenticação de Administrador")
    username = st.text_input("Utilizador administrador", key="admin_username")
    password = st.text_input("Password", type="password", key="admin_password")

    if st.button("Entrar", type="primary"):
        auth_result = authenticate_user(username, password)
        user = auth_result.get("user", {})
        if auth_result.get("success") and user.get("role") == "admin":
            st.session_state.admin_auth = True
            st.session_state.admin_user = user.get("username", username)
            st.rerun()
        st.error("Credenciais inválidas ou sem permissões de administrador.")

    st.info("Podes entrar com o utilizador administrador definido em `st.secrets` ou com um utilizador com role `admin` já criado.")
    st.stop()

toolbar_left, toolbar_right = st.columns([1, 1])
with toolbar_left:
    st.success(f"Administrador autenticado: {st.session_state.admin_user}")
with toolbar_right:
    if st.button("Terminar sessão"):
        st.session_state.admin_auth = False
        st.session_state.admin_user = ""
        st.rerun()

st.subheader("Novo Utilizador")
with st.form("create_user_form", clear_on_submit=True):
    new_username = st.text_input("Utilizador")
    new_password = st.text_input("Password", type="password")
    new_role = st.selectbox("Perfil", options=["editor", "viewer", "admin"], index=0)
    submitted = st.form_submit_button("Criar utilizador", type="primary")

if submitted:
    result = create_user(
        username=new_username,
        password=new_password,
        role=new_role,
        created_by=st.session_state.admin_user,
    )
    if result.get("success"):
        st.success(f"Utilizador `{new_username.strip().lower()}` criado com sucesso.")
    else:
        st.error(result.get("error", "Não foi possível criar o utilizador."))

st.subheader("Utilizadores Registados")
users_df = list_users()
if users_df.empty:
    st.info("Ainda não existem utilizadores criados.")
else:
    st.dataframe(users_df, use_container_width=True, hide_index=True)
