# Análise de Estrutura e Calls entre Módulos (Projecto-Master-FPF)

## 1) Estrutura funcional

- **`Main.py`**: aplicação Streamlit principal (ingestão CSV, calibração de campo, sincronização, métricas, QC e exportação parquet).
- **`pages/Análise_Performance.py`**: página analítica de visualização (compactação + snapshot posicional).
- **`fpf_modules/constants.py`**: constantes globais, paths e thresholds.
- **`fpf_modules/io_utils.py`**: leitura/normalização básica de CSV, extração de atleta/fase, hash de sessão.
- **`fpf_modules/pipeline.py`**: pipeline de transformação temporal e sincronização por atleta.
- **`fpf_modules/metrics.py`**: cálculo de métricas de performance por atleta/fase.
- **`fpf_modules/qc.py`**: indicadores de qualidade de GPS.
- **`fpf_modules/geo.py`**: calibração/retangularização de campo e validações geoespaciais.
- **`fpf_modules/normalize.py`**: normalização de coordenadas e direção de ataque.
- **`fpf_modules/data_manager.py`**: persistência parquet, resolução de chaves substitutas (`session_sk`, `athlete_sk`).
- **`fpf_modules/utils.py`**: utilitários de arredondamento e bytes para download.

## 2) Fluxo principal de dados (alto nível)

1. Upload de ficheiros e metadados em `Main.py`.
2. Leitura/limpeza via `io_utils.read_csv_upload` + `clean_cols`.
3. Calibração de campo (`geo.calibrar_campo*`) e transformação para coordenadas de jogo.
4. Sincronização por atleta (`pipeline.processar_atletas_para_temp` + `pipeline.sincronizar`).
5. Cálculo de métricas (`metrics.compute_metrics_for_df`) e QC (`qc.qc_gps_df`).
6. Normalização de tracking (`normalize.normalize_tracking_data`).
7. Persistência em parquet (`data_manager.append_dedup_parquet`, `resolve_*_sk`).
8. Consumo analítico em `pages/Análise_Performance.py`.

## 3) Dependências/calls entre módulos

### `Main.py` chama
- `fpf_modules.constants`: thresholds e nomes de colunas.
- `fpf_modules.io_utils`: leitura e identificação de ficheiros.
- `fpf_modules.geo`: calibração e validações geográficas.
- `fpf_modules.pipeline`: processamento/sincronização temporal.
- `fpf_modules.metrics`: auditoria temporal e métricas.
- `fpf_modules.qc`: validação de qualidade GPS.
- `fpf_modules.normalize`: normalização de tracking.
- `fpf_modules.data_manager`: persistência analítica.
- `fpf_modules.utils`: apoio a exportação/apresentação.

### Dependências internas dos módulos
- `pipeline.py` depende de: `io_utils`, `metrics.time_to_seconds`, `constants`.
- `qc.py` depende de: `metrics.time_to_seconds`.
- `geo.py` depende de: `io_utils` e `constants`.
- `data_manager.py` depende de: `constants`.
- `normalize.py` é quase autónomo (numpy/pandas).

## 4) Pontos de risco encontrados

1. **Sensibilidade a nomes de colunas em métricas**: `compute_metrics_for_df` estava rígida em `X_UTM`/`Y_UTM` e podia falhar com `x_utm`/`y_utm` (schema usado em `tracking.parquet`).
2. **Acoplamento forte ao schema parquet** na página analítica e em merges; qualquer drift de colunas quebra a visualização.
3. **Paths baseados em `os.getcwd()`** em `constants.py` podem variar se a app for lançada fora da raiz do projeto.
4. **Página `Análise_Performance.py`** ainda contém TODOs funcionais (tratamento de treino e convex hull incompleto).

## 5) Correção aplicada nesta iteração

- Tornada a função `compute_metrics_for_df` resiliente a variações de schema:
  - identifica dinamicamente coluna temporal (`Time`, `time`) e espaciais (`X_UTM`/`x_utm`, `Y_UTM`/`y_utm`),
  - devolve métricas default quando colunas mínimas não existem, evitando `KeyError`.

## 6) Recomendações (próximos passos)

- Introduzir um **schema contract** central (lista de colunas obrigatórias por etapa) com validação explícita.
- Criar **testes unitários** para:
  - `metrics.compute_metrics_for_df` (schemas alternativos),
  - `pipeline.sincronizar` (timestamps incompletos),
  - `pages/Análise_Performance.py` (casos sem dados por filtro).
- Desacoplar paths de `cwd` (usar `Path(__file__).resolve().parents[...]`).
- Consolidar nomenclatura de colunas (snake_case vs PascalCase) para reduzir conversões.
