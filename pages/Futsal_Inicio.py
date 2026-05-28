import streamlit as st


st.title("Futsal Performance Hub")
st.caption("Base de dados, referenciais automáticos e enquadramento multidimensional da atleta.")

st.write(
    """
O `Futsal Performance Hub` foi desenvolvido para centralizar, estruturar e interpretar a
informação de acompanhamento da atleta ao longo do tempo. A plataforma organiza a
observação e a avaliação em quatro grandes vertentes: `Antropometria`, `Fisico`,
`Tecnico | Tatica` e `Psicologico`, permitindo que cada registo fique associado ao
histórico individual da atleta e ao contexto da seleção.
"""
)

st.markdown(
    """
**Como está organizado o sistema**

- `Base de dados principal`: reúne todas as atletas, avaliações, anos, seleções e testes.
- `Histórico acumulado`: cada nova inserção passa a integrar a evolução longitudinal da atleta.
- `Referenciais automáticos`: percentis, medianas, z-scores e distribuições são calculados a partir da própria base.
- `Comparação de perfis`: permite comparar atletas entre si e identificar perfis semelhantes.
"""
)

st.markdown(
    """
**Lógica de leitura dos resultados**

- O `resultado da atleta` é sempre apresentado no seu `valor absoluto`.
  Exemplos: `1.82 s` no Sprint 10m, `29.4 cm` no CMJ, `5` numa escala técnica de `1-7`.
- O `enquadramento` é comparativo e usa o `histórico acumulado` para interpretar esse resultado dentro de um grupo comparável.
- Esse grupo comparável é construído, sempre que possível, a partir de variáveis como `género`, `escalão`, `seleção`, `posição` e, quando aplicável, `maturação`.
"""
)

st.markdown(
    """
**Como são lidos os referenciais**

- `P25`, `P50`, `P75` e `P90` representam `percentis` da distribuição histórica.
- `P50` corresponde à `mediana`: metade dos registos está abaixo e metade acima.
- `P25` e `P75` ajudam a definir a `zona central` do grupo, enquanto `P90` identifica valores de referência mais elevados.
- Em métricas onde `menos é melhor`, como `sprints` ou `505`, a interpretação é invertida internamente para que um enquadramento melhor continue a ser lido como desempenho superior.
"""
)

st.markdown(
    """
**Como é lido o z-score**

- O `z-score` mede a `distância` do resultado da atleta em relação à média do grupo, expressa em `desvios-padrão`.
- Um `z-score = 0` significa que a atleta está exatamente na média do grupo.
- Um `z-score positivo` indica desempenho acima da média.
- Um `z-score negativo` indica desempenho abaixo da média.
- Quanto maior o afastamento de `0`, maior é a diferença relativa face ao grupo de referência.
"""
)

st.markdown(
    """
**Como são construídos os índices ANT | FIS | TEC | TAT | PSI**

- Os indicadores `ANT | FIS | TEC | TAT | PSI` representam `índices sintéticos de perfil` em escala `0-100`.
- Estes índices `não substituem` o valor bruto de cada teste; resumem a `posição relativa` da atleta em cada grande dimensão.
- Cada índice resulta da `média dos percentis` das métricas que pertencem a esse bloco.
- Exemplo:
  `FIS` pode integrar sprints, agilidade, salto, reatividade e resistência;
  `TEC` integra as métricas técnicas;
  `TAT` integra as métricas táticas;
  `PSI` integra os fatores psicológicos;
  `ANT` integra as variáveis antropométricas e de maturação.
- Assim, um `índice 75` significa que, em média, a atleta se posiciona acima de grande parte do grupo nessa dimensão, enquanto um `índice 50` a coloca perto da mediana histórica.
"""
)

st.markdown(
    """
**Objetivo funcional da plataforma**

- apoiar o registo estruturado da informação da atleta;
- acompanhar evolução individual ao longo do tempo;
- criar referenciais dinâmicos por seleção, escalão, posição e maturação;
- facilitar leitura técnica para decisão, comparação e relatório.
"""
)

st.info(
    "Utilize `Atletas` para consultar fichas e histórico, `Criar Atleta` para novos registos, "
    "`Inserção de Dados` para cargas em lote, `Referenciais` para enquadramento estatístico e "
    "`Comparacao de Perfis` para análise entre atletas."
)
