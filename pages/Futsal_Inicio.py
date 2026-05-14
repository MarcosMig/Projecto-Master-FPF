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
- `Referenciais automáticos`: percentis, medianas e distribuições são calculados a partir da própria base.
- `Comparação de perfis`: permite comparar atletas entre si e identificar perfis semelhantes.
"""
)

st.markdown(
    """
**Lógica de leitura dos resultados**

- O `resultado da atleta` é sempre apresentado no seu valor absoluto.
- O `enquadramento` é comparativo e usa o histórico acumulado para interpretar esse resultado.
- Os indicadores `ANT | FIS | TEC | TAT | PSI` representam índices sintéticos de perfil em escala `0-100`.
- Estes índices não substituem o valor bruto; ajudam a resumir a posição relativa da atleta no grupo comparável.
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
