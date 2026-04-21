import streamlit as st

from fpf_modules.auth_manager import create_user, list_users, logout_user, require_login


st.set_page_config(page_title="Administracao | Utilizadores", layout="wide")

require_login()

st.title("Administracao")
st.caption("Criar e consultar utilizadores da aplicacao.")

toolbar_left, toolbar_right = st.columns([1, 1])
with toolbar_left:
    st.success(f"Utilizador autenticado: {st.session_state.login_user}")
with toolbar_right:
    if st.button("Terminar sessao"):
        logout_user()
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
        created_by=st.session_state.login_user,
    )
    if result.get("success"):
        st.success(f"Utilizador `{new_username.strip().lower()}` criado com sucesso.")
    else:
        st.error(result.get("error", "Nao foi possivel criar o utilizador."))

st.subheader("Utilizadores Registados")
users_df = list_users()
if users_df.empty:
    st.info("Ainda nao existem utilizadores criados.")
else:
    st.dataframe(users_df, use_container_width=True, hide_index=True)
