alter table public.performance_metrics
add column if not exists acc_per_1000_beats double precision;

alter table public.performance_metrics
add column if not exists dec_per_1000_beats double precision;

alter table public.performance_metrics
add column if not exists player_load double precision;

alter table public.performance_metrics
add column if not exists rhie_bouts integer;

alter table public.performance_metrics
add column if not exists rhie_actions integer;

alter table public.performance_metrics
add column if not exists zone1_walk_time_min double precision;

alter table public.performance_metrics
add column if not exists zone1_walk_dist_m double precision;

alter table public.performance_metrics
add column if not exists zone2_jog_time_min double precision;

alter table public.performance_metrics
add column if not exists zone2_jog_dist_m double precision;

alter table public.performance_metrics
add column if not exists zone3_run_time_min double precision;

alter table public.performance_metrics
add column if not exists zone3_run_dist_m double precision;

alter table public.performance_metrics
add column if not exists zone4_hsr_time_min double precision;

alter table public.performance_metrics
add column if not exists zone4_hsr_dist_m double precision;

alter table public.performance_metrics
add column if not exists zone5_sprint_time_min double precision;

alter table public.performance_metrics
add column if not exists zone5_sprint_dist_m double precision;

alter table public.performance_metrics
add column if not exists peak_dist_1m_m double precision;

alter table public.performance_metrics
add column if not exists peak_dist_3m_m double precision;

alter table public.performance_metrics
add column if not exists peak_dist_5m_m double precision;

alter table public.performance_metrics
add column if not exists peak_hsr_1m_m double precision;

alter table public.performance_metrics
add column if not exists peak_hsr_3m_m double precision;

alter table public.performance_metrics
add column if not exists peak_hsr_5m_m double precision;

alter table public.performance_metrics
add column if not exists peak_sprint_1m_m double precision;

alter table public.performance_metrics
add column if not exists peak_sprint_3m_m double precision;

alter table public.performance_metrics
add column if not exists peak_sprint_5m_m double precision;

alter table public.performance_metrics
add column if not exists peak_acc_actions_1m double precision;

alter table public.performance_metrics
add column if not exists peak_acc_actions_3m double precision;

alter table public.performance_metrics
add column if not exists peak_acc_actions_5m double precision;

alter table public.performance_metrics
add column if not exists peak_hi_actions_1m double precision;

alter table public.performance_metrics
add column if not exists peak_hi_actions_3m double precision;

alter table public.performance_metrics
add column if not exists peak_hi_actions_5m double precision;

alter table public.performance_metrics
add column if not exists trimp_banister double precision;

alter table public.performance_metrics
add column if not exists trimp_per_min double precision;

create table if not exists public.collective_performance_metrics (
    session_sk integer not null references public.sessions(session_sk),
    phase_id integer not null,
    fase text,
    data date,
    selecao text,
    genero text,
    contexto text,
    jogo text,
    duracao_min_total double precision,
    dist_m_total double precision,
    hsr_dist_m_total double precision,
    sprint_dist_m_total double precision,
    active_time_min_total double precision,
    m_min_avg double precision,
    hsr_pct_avg double precision,
    active_pct_avg double precision,
    n_sprints_total integer,
    n_acc_2_5_total integer,
    n_dec_3_0_total integer,
    vmax_mps_max double precision,
    peak_1m_m_min_max double precision,
    hr_avg_bpm_avg double precision,
    external_load_score_total double precision,
    total_load_score_total double precision,
    player_load_total double precision,
    rhie_bouts_total integer,
    rhie_actions_total integer,
    trimp_banister_total double precision,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    primary key (session_sk, phase_id)
);

create index if not exists idx_collective_performance_metrics_session
on public.collective_performance_metrics(session_sk);
