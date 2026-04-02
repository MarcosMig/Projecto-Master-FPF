import streamlit as st


st.set_page_config(page_title="FPF Analytics Hub", layout="wide")


def _render_home() -> None:
    st.title("FPF Analytics Hub")
    st.caption("Centro de navegacao da aplicacao.")


navigation = st.navigation(
    {
        "Hub": [
            st.Page(_render_home, title="Início", default=True),
            st.Page("pages/Insercão_Dados.py", title="Inserção de Dados"),
        ],
        "Base de Dados": [
            st.Page("pages/Atletas.py", title="Atletas", url_path="atletas"),
            st.Page("pages/Seleções.py", title="Seleções", url_path="selecoes"),
        ],
        "Análise": [
            st.Page("pages/Análise_Performance.py", title="Comparar Perfis", url_path="comparar-perfis"),
            st.Page("pages/Análise_Temporal.py", title="Análise Temporal", url_path="analise-temporal"),
            st.Page("pages/Análise_Performance_Dev1.py", title="Análise Performance Dev1", url_path="analise-performance-dev1"),
            st.Page("pages/Análise_Performance copy.py", title="Análise Performance Copy", url_path="analise-performance-copy"),
            st.Page("pages/Análise_Performance copy 2.py", title="Análise Performance Copy 2", url_path="analise-performance-copy-2"),
        ],
        "Administração": [
            st.Page("pages/Admin_Utilizadores.py", title="Utilizadores", url_path="admin-utilizadores"),
        ],
    }
)

navigation.run()
