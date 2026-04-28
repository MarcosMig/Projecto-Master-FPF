import streamlit as st


st.title("Futsal Performance Hub")
st.caption("Base de dados de atletas e acompanhamento multidimensional da modalidade.")

st.write(
    """
Este espaço de futsal fica orientado para centralizar a ficha individual de cada atleta.
Aqui podemos registar dados base, indicadores antropométricos e avaliações físicas,
técnicas e psicológicas num único local.
"""
)

st.markdown(
    """
**Objetivo desta área**

- Construir uma base de dados própria para o futsal.
- Acompanhar a evolução individual de cada atleta ao longo do tempo.
- Reunir informação transversal de observação, teste e contexto competitivo.
- Preparar a ligação futura entre avaliações e análise de performance.
"""
)

st.info(
    "A página `Atletas` já permite registar a ficha completa de futsal. "
    "Os dados ficam separados da modalidade de futebol."
)
