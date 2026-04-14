import streamlit as st


st.title("Federação Portuguesa de Futebol")
st.header("Bem-vindo!")
st.write(
    """
O Performance Hub foi desenvolvido para analisar o desempenho de atletas com base em dados GPS.

Carrega os ficheiros, valida os registos e obtém automaticamente métricas claras e organizadas.

Tudo num único local, acessível em qualquer local e momento, permitindo-nos focar no que realmente importa: interpretar os dados e tomar decisões no treino e na competição.
    """
)

st.subheader("Guia para upload de dados GPS")
st.markdown(
    """
Para garantir uma leitura correta, os ficheiros carregados no módulo **Upload de Dados** devem seguir a estrutura abaixo.

**Formato do ficheiro**

- Ficheiros em formato CSV.
- O separador pode ser vírgula ou ponto e vírgula.
- A ordem das colunas pode variar, mas os nomes das colunas obrigatórias devem ser respeitados.

**Colunas obrigatórias**

- `Time`: instante temporal da amostra.
- `Lat`: latitude em coordenadas GPS.
- `Lon`: longitude em coordenadas GPS.

**Colunas opcionais reconhecidas**

- Frequência cardíaca: `HR_bpm`, `HR`, `HeartRate`, `Heart Rate`, `Heart_Rate`, `BPM` ou `Pulse`.

**Nome dos ficheiros dos atletas**

- O ID do atleta deve estar no nome do ficheiro, preferencialmente no formato `Player-<id>`.
- A fase deve estar identificada no nome do ficheiro:
  - aquecimento: `Warm` ou `WUP`;
  - primeira parte: `1P`, `Primeira` ou `First`;
  - segunda parte: `2P`, `Segunda` ou `Second`.
- Exemplo de nomes completos: `Player-10-1P.csv`, `Player-10-2P.csv` ou `Player-10-Warm.csv`.

Ficheiros de outros provedores podem exigir um mapeamento prévio de colunas, por exemplo quando usam nomes como `Latitude`, `Longitude`, `Timestamp` ou coordenadas já convertidas em metros.
    """
)
