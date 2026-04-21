import streamlit as st

from fpf_modules.auth_manager import ensure_auth_state, logout_user, render_login_page


st.set_page_config(page_title="FPF Performance Hub", layout="wide")

ensure_auth_state()

if not st.session_state.auth:
    render_login_page()
    st.stop()

with st.sidebar:
    st.caption(f"Utilizador: {st.session_state.login_user}")
    st.caption(f"Perfil: {st.session_state.login_role}")
    if st.button("Terminar sessao"):
        logout_user()
        st.rerun()

pages = {
    "Performance Hub": [
        st.Page("pages/Inicio.py", title="Inicio", default=True, url_path="inicio"),
        st.Page("pages/Insercão_Dados.py", title="Upload de Dados"),
    ],
    "Base de Dados": [
        st.Page("pages/Seleções.py", title="Selecoes", url_path="selecoes"),
        st.Page("pages/Atletas.py", title="Atletas", url_path="atletas"),
        st.Page("pages/Campos.py", title="Campos", url_path="campos"),
        st.Page("pages/Registos.py", title="Jogos | Treinos", url_path="registos"),
    ],
    "Analise": [
        st.Page("pages/Análise_Performance.py", title="Comparar Perfis", url_path="comparar-perfis"),
        st.Page("pages/Análise_Temporal.py", title="Perfil Temporal", url_path="perfil-temporal"),
        st.Page("pages/Análise_Performance_Dev1.py", title="Analise Posicional", url_path="analise-posicional"),
    ],
    "Administracao": [
        st.Page("pages/Admin_Utilizadores.py", title="Utilizadores", url_path="admin-utilizadores"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
