import streamlit as st


st.set_page_config(page_title="FPF Analytics Hub", layout="wide")


navigation = st.navigation(
    {
        "Hub": [
            st.Page("pages/Inicio.py", title="Inicio", default=True, url_path="inicio"),
            st.Page("pages/Insercão_Dados.py", title="Insercao de Dados"),
        ],
        "Base de Dados": [
            st.Page("pages/Seleções.py", title="Selecoes", url_path="selecoes"),
            st.Page("pages/Atletas.py", title="Atletas", url_path="atletas"),
            st.Page("pages/Campos.py", title="Campos", url_path="campos"),
            st.Page("pages/Registos.py", title="Jogos | Treinos", url_path="jogos-treinos"),
        ],
        "Analise": [
            st.Page("pages/Análise_Performance.py", title="Comparar Perfis", url_path="comparar-perfis"),
            st.Page("pages/Análise_Temporal.py", title="Analise Temporal", url_path="analise-temporal"),
            st.Page("pages/Análise_Performance_Dev1.py", title="Analise Performance Dev1", url_path="analise-performance-dev1"),
            st.Page("pages/Análise_Performance copy.py", title="Analise Performance Copy", url_path="analise-performance-copy"),
            st.Page("pages/Análise_Performance copy 2.py", title="Analise Performance Copy 2", url_path="analise-performance-copy-2"),
        ],
        "Administracao": [
            st.Page("pages/Admin_Utilizadores.py", title="Utilizadores", url_path="admin-utilizadores"),
        ],
    }
)

navigation.run()
