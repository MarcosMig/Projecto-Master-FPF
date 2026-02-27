import streamlit as st
import pandas as pd
import time
from engine_v9 import run_utm_v9_logic # Importa a lógica real

# --- LOG IN (Mantém o teu código original do Estavel 2) ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'engine_run' not in st.session_state: st.session_state.engine_run = False

# [Aqui entra o teu apply_login_style() e lógica de login do FPF Estavel 2.py]

if not st.session_state.auth:
    # Mostra login...
    st.stop()

# --- VALIDAÇÃO (Bloco 2 do Estavel 2) ---
st.title("Validação de Dados e Engine V9")
f_campo = st.file_uploader("Dados de CAMPO", accept_multiple_files=True)
f_atleta = st.file_uploader("Dados de ATLETAS", accept_multiple_files=True)

# [Toda a tua lógica de is_geo_valid e quorum_ok aqui]

if f_campo and f_atleta:
    # Se passar na validação do Estavel 2:
    st.divider()
    
    if not st.session_state.engine_run:
        # A PERGUNTA QUE PEDISTE
        st.subheader("⚙️ Pipeline Final")
        if st.button("Executamos UTM Engine V9... ?", type="primary", use_container_width=True):
            with st.status("A executar UTM Engine V9.2 - Pipeline Final Otimizado...", expanded=True) as status:
                st.write("1. Calibração de Campo...")
                # Corre a lógica real do script v9.2
                result = run_utm_v9_logic(f_campo, f_atleta)
                st.session_state.res_v9 = result
                
                st.write("2. Processamento Temporário...")
                time.sleep(1)
                st.write("3. Sincronização Cronológica...")
                time.sleep(1)
                st.write("4. Limpeza e Relatórios...")
                
                st.session_state.engine_run = True
                status.update(label="✅ Processamento V9.2 Concluído!", state="complete")
            st.rerun()
            
    else:
        # RELATÓRIO DO PROCESSAMENTO (Fase 4 do script v9.2)
        r = st.session_state.res_v9
        st.header("📊 Relatório de Geometria (v9.2)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Comprimento (X)", f"{r['x']:.2f} m")
        c2.metric("Largura (Y)", f"{r['y']:.2f} m")
        c3.metric("Rotação", f"{r['rot']:.2f}°")
        
        st.success(f"Foram gerados {r['count']} ficheiros SYNC com sucesso.")
        
        if st.button("📤 Submeter para Base de Dados", type="primary"):
            st.balloons()