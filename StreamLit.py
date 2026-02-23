# -*- coding: utf-8 -*-
import streamlit as st

# Tenta isto para travar o crash de renderização
if 'auth' not in st.session_state:
    st.session_state.auth = False

# Layout simples sem config inicial para evitar o erro do Edge
st.title("⚽ FPF UTM Engine")

if not st.session_state.auth:
    # Interface de Login em bloco único
    with st.container():
        st.write("---")
        user = st.text_input("Utilizador", key="user")
        pw = st.text_input("Password", type="password", key="pass")
        if st.button("Entrar"):
            if user == "miguel.cardoso" and pw == "fpf2026":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Dados inválidos")
else:
    st.success("Sessão Iniciada")
    st.write("### 1. Upload de Dados")
    files = st.file_uploader("Arraste os ficheiros", accept_multiple_files=True)
    if st.button("Sair"):
        st.session_state.auth = False
        st.rerun()
