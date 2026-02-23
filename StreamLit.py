import streamlit as st

# OBRIGATÓRIO: Sem nada antes destas linhas
st.set_page_config(page_title="FPF UTM Engine", layout="wide")

import pandas as pd
import numpy as np

# --- LOGIN SIMPLES PARA TESTE ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("⚽ FPF Login")
    user = st.text_input("Utilizador")
    pw = st.text_input("Password", type="password")
    if st.button("Entrar"):
        if user == "miguel.cardoso" and pw == "fpf2026":
            st.session_state.auth = True
            st.rerun()
else:
    st.success("Logado com sucesso!")
    st.write("O erro desapareceu. Agora podemos carregar os dados.")
