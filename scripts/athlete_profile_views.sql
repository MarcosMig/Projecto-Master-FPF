-- Base total-session view
create or replace view public.vw_perf_total_session as
select
    pm.session_sk,
    pm.athlete_sk,
    pm.atleta_id,
    pm.data,
    pm.selecao,
    pm.genero,
    pm.contexto,
    pm.jogo,
    pm.fase,
    pm.duracao_min,
    pm.dist_m,
    pm.m_min,
    pm.vmax_mps,
    pm.peak_1m_m_min,
    pm.hsr_dist_m,
    pm.hsr_pct,
    pm.sprint_dist_m,
    pm.n_sprints,
    pm.n_acc_2_5,
    pm.n_dec_3_0,
    pm.active_time_min,
    pm.active_pct
from public.performance_metrics pm
where pm.fase = 'Total'
  and coalesce(pm.duracao_min, 0) > 0;


-- Aggregate by athlete and context
create or replace view public.vw_perf_athlete_aggregate as
select
    atleta_id,
    athlete_sk,
    contexto,
    count(distinct session_sk) as n_sessoes,
    min(data) as primeira_data,
    max(data) as ultima_data,
    sum(duracao_min) as duracao_min_total,
    sum(dist_m) as dist_m_total,
    sum(hsr_dist_m) as hsr_dist_m_total,
    sum(sprint_dist_m) as sprint_dist_m_total,
    sum(n_sprints) as n_sprints_total,
    sum(n_acc_2_5) as n_acc_2_5_total,
    sum(n_dec_3_0) as n_dec_3_0_total,
    sum(active_time_min) as active_time_min_total,
    case when sum(duracao_min) > 0 then sum(dist_m) / sum(duracao_min) end as m_min,
    case when sum(dist_m) > 0 then sum(hsr_dist_m) / sum(dist_m) * 100 end as hsr_pct,
    case when sum(duracao_min) > 0 then sum(active_time_min) / sum(duracao_min) * 100 end as active_pct,
    case when sum(duracao_min) > 0 then sum(dist_m) / sum(duracao_min) * 90 end as dist_m_90,
    case when sum(duracao_min) > 0 then sum(hsr_dist_m) / sum(duracao_min) * 90 end as hsr_dist_m_90,
    case when sum(duracao_min) > 0 then sum(sprint_dist_m) / sum(duracao_min) * 90 end as sprint_dist_m_90,
    case when sum(duracao_min) > 0 then sum(n_sprints) / sum(duracao_min) * 90 end as n_sprints_90,
    case when sum(duracao_min) > 0 then sum(n_acc_2_5) / sum(duracao_min) * 90 end as n_acc_2_5_90,
    case when sum(duracao_min) > 0 then sum(n_dec_3_0) / sum(duracao_min) * 90 end as n_dec_3_0_90,
    case when sum(duracao_min) > 0 then sum(active_time_min) / sum(duracao_min) * 90 end as active_time_min_90,
    max(vmax_mps) as vmax_mps_peak,
    max(peak_1m_m_min) as peak_1m_m_min_peak
from public.vw_perf_total_session
group by
    atleta_id,
    athlete_sk,
    contexto;


-- Profile by athlete for games
create or replace view public.vw_profile_game as
select *
from public.vw_perf_athlete_aggregate
where contexto = 'Jogo';


-- Profile by athlete for training
create or replace view public.vw_profile_training as
select *
from public.vw_perf_athlete_aggregate
where contexto = 'Treino';
