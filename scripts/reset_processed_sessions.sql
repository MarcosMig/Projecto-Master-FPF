-- Reset processed GPS/game-session data while preserving reference data.
--
-- Preserved tables:
--   athletes, fields, selecoes
--
-- Cleared tables:
--   session_reports, samples, quality_metrics, collective_performance_metrics,
--   performance_metrics, athlete_session, metrics, sessions, games
--
-- Run with:
--   psql "$env:DATABASE_URL" -f scripts/reset_processed_sessions.sql

BEGIN;

TRUNCATE TABLE
    session_reports,
    samples,
    quality_metrics,
    collective_performance_metrics,
    performance_metrics,
    athlete_session,
    metrics,
    sessions,
    games
RESTART IDENTITY;

ALTER TABLE samples ADD COLUMN IF NOT EXISTS x_norm DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS y_norm DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS speed_mps DOUBLE PRECISION;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS acc_mps2 DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_samples_session_phase_time
ON samples(session_sk, phase_id, time_evento_s);

COMMIT;

SELECT 'session_reports' AS table_name, COUNT(*) AS rows_remaining FROM session_reports
UNION ALL SELECT 'samples', COUNT(*) FROM samples
UNION ALL SELECT 'quality_metrics', COUNT(*) FROM quality_metrics
UNION ALL SELECT 'collective_performance_metrics', COUNT(*) FROM collective_performance_metrics
UNION ALL SELECT 'performance_metrics', COUNT(*) FROM performance_metrics
UNION ALL SELECT 'athlete_session', COUNT(*) FROM athlete_session
UNION ALL SELECT 'metrics', COUNT(*) FROM metrics
UNION ALL SELECT 'sessions', COUNT(*) FROM sessions
UNION ALL SELECT 'games', COUNT(*) FROM games
ORDER BY table_name;
