import pandas as pd
import streamlit as st

from fpf_modules.constants import CAMPOS_DIR, CLEANDATA_DIR


st.header('Teste se data esta a ser guardade em parquet!')
st.write('teste produção!')


teste = pd.read_parquet(CLEANDATA_DIR + '/performance_metrics.parquet')

n_linhas = teste.shape[0]

st.write(n_linhas)
