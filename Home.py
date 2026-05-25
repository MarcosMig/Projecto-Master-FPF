import streamlit as st

from fpf_modules.auth_manager import ensure_auth_state, logout_user, render_login_page
from fpf_modules.modalities import (
    clear_modality,
    current_modality_label,
    ensure_modality_state,
    render_modality_selector,
    render_preview_modality,
)


st.set_page_config(page_title="FPF Performance Hub", layout="wide")

ensure_auth_state()
ensure_modality_state()

if not st.session_state.auth:
    render_login_page()
    st.stop()

if not st.session_state.selected_modality:
    render_modality_selector()
    st.stop()

with st.sidebar:
    st.caption(f"Utilizador: {st.session_state.login_user}")
    st.caption(f"Perfil: {st.session_state.login_role}")
    st.caption(f"Modalidade: {current_modality_label()}")
    if st.button("Trocar modalidade"):
        clear_modality()
        st.rerun()
    if st.button("Terminar sessao"):
        logout_user()
        st.rerun()

if st.session_state.selected_modality == "futebol":
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
elif st.session_state.selected_modality == "futsal":
    pages = {
        "Futsal Hub": [
            st.Page("pages/Futsal_Inicio.py", title="Inicio", default=True, url_path="futsal-inicio"),
            st.Page("pages/Futsal_Atletas.py", title="Atletas", url_path="futsal-atletas"),
            st.Page("pages/Futsal_Criar_Atleta.py", title="Criar Atleta", url_path="futsal-criar-atleta"),
            st.Page("pages/Futsal_Insercao_Dados.py", title="Inserção de Dados", url_path="futsal-insercao-dados"),
            st.Page("pages/Futsal_Exercicios.py", title="Exercicios", url_path="futsal-exercicios"),
            st.Page("pages/Futsal_Referenciais.py", title="Referenciais", url_path="futsal-referenciais"),
            st.Page("pages/Futsal_Comparacao_Perfis.py", title="Comparacao de Perfis", url_path="futsal-comparacao-perfis"),
        ],
    }

    navigation = st.navigation(pages)
    navigation.run()
else:
    render_preview_modality()
