# Projecto-Master-FPF
Plataforma Analise Posicional Multi Seleção

## Upload de ficheiros e identificação de campos

Atualmente este repositório **não contém implementação de código** para o fluxo de file upload.

Para contemplar identificação de campo no upload, o backend deve receber:

- `field_id`: identificador único do campo no formulário.
- `field_name`: nome lógico do campo (ex.: `mapa_calor`, `dados_jogadores`).
- `file`: conteúdo binário do ficheiro.
- `metadata` (opcional): contexto adicional (utilizador, versão, origem).

### Exemplo de payload (`multipart/form-data`)

```text
field_id=position-analysis-csv
field_name=analise_posicional
file=<arquivo.csv>
metadata={"source":"web","season":"2025"}
```

### Validações mínimas recomendadas

1. Validar se `field_id` pertence à lista de campos permitidos.
2. Validar extensão e tipo MIME do ficheiro.
3. Garantir tamanho máximo por campo.
4. Persistir log com `field_id` + nome do ficheiro + timestamp.

Com isto, o upload passa a ter rastreabilidade por campo e suporte para regras específicas por tipo de dado.
