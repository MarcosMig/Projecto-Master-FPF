# Athlete Profile Design

## Objetivo

Construir uma camada analítica em cima das tabelas já publicadas para:

- acumular métricas por atleta ao longo do tempo
- recalcular métricas normalizadas por `90'`
- definir um `perfil de atleta`
- comparar uma sessão/jogo com o perfil histórico do próprio atleta

## Fonte de verdade

As tabelas base continuam a ser:

- `public.athletes`
- `public.sessions`
- `public.performance_metrics`
- `public.quality_metrics`
- `public.samples`
- `public.athlete_session`

Nada do perfil deve substituir estas tabelas. O perfil é sempre uma camada derivada.

## Regra de modelação

O perfil deve ser calculado a partir de linhas:

- com `fase = 'Total'`
- com `duracao_min > 0`
- idealmente com qualidade válida

## Métricas por tipo de agregação

### Somar

- `duracao_min`
- `dist_m`
- `hsr_dist_m`
- `sprint_dist_m`
- `n_sprints`
- `n_acc_2_5`
- `n_dec_3_0`
- `active_time_min`

### Recalcular a partir dos totais

- `dist_m_90 = sum(dist_m) / sum(duracao_min) * 90`
- `hsr_dist_m_90 = sum(hsr_dist_m) / sum(duracao_min) * 90`
- `sprint_dist_m_90 = sum(sprint_dist_m) / sum(duracao_min) * 90`
- `n_sprints_90 = sum(n_sprints) / sum(duracao_min) * 90`
- `n_acc_2_5_90 = sum(n_acc_2_5) / sum(duracao_min) * 90`
- `n_dec_3_0_90 = sum(n_dec_3_0) / sum(duracao_min) * 90`
- `m_min = sum(dist_m) / sum(duracao_min)`
- `hsr_pct = sum(hsr_dist_m) / sum(dist_m) * 100`
- `active_pct = sum(active_time_min) / sum(duracao_min) * 100`

### Picos

- `vmax_mps`: usar `max`
- `peak_1m_m_min`: usar `max`

## Perfis recomendados

### Perfil Jogo

Base:

- `contexto = 'Jogo'`
- `fase = 'Total'`

Versões:

- global
- últimos `5` jogos
- últimos `10` jogos
- época atual

### Perfil Treino

Base:

- `contexto = 'Treino'`
- `fase = 'Total'`

Versões:

- global
- últimos `5` treinos
- últimos `10` treinos
- bloco temporal selecionado

## Camadas propostas

### 1. View base por sessão total

Nome sugerido:

- `public.vw_perf_total_session`

Função:

- filtrar `performance_metrics` para `fase = 'Total'`
- ligar com `sessions`
- expor `contexto`, `data`, `selecao`, `genero`

### 2. View de acumulado por atleta

Nome sugerido:

- `public.vw_perf_athlete_aggregate`

Função:

- acumular por atleta e por contexto
- recalcular métricas derivadas e por `90'`

### 3. View de perfil jogo

Nome sugerido:

- `public.vw_profile_game`

Função:

- agregar apenas jogos
- produzir baseline para comparação

### 4. View de perfil treino

Nome sugerido:

- `public.vw_profile_training`

Função:

- agregar apenas treinos
- produzir baseline para comparação

## Comparação sessão vs perfil

Para cada atleta, comparar:

- valor da sessão
- valor do perfil
- diferença absoluta
- diferença percentual

Exemplo:

- `dist_m_90_sessao`
- `dist_m_90_perfil`
- `dist_m_90_delta`
- `dist_m_90_delta_pct`

## Fases de implementação

### Fase 1

- criar `vw_perf_total_session`
- criar `vw_perf_athlete_aggregate`
- usar estas views em páginas de análise

### Fase 2

- criar `vw_profile_game`
- criar `vw_profile_training`
- comparação pós-jogo e pós-treino contra perfil individual

### Fase 3

- filtros por época
- filtros por adversário
- filtros por posição
- tendências recentes
