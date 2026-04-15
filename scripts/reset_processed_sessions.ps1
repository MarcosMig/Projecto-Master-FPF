param(
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$sqlFile = Join-Path $PSScriptRoot "reset_processed_sessions.sql"

if (-not (Test-Path $sqlFile)) {
    throw "SQL file not found: $sqlFile"
}

if ([string]::IsNullOrWhiteSpace($env:DATABASE_URL)) {
    Write-Host ""
    Write-Host "DATABASE_URL is not set." -ForegroundColor Red
    Write-Host "Set it with the Supabase PostgreSQL connection string, for example:" -ForegroundColor Yellow
    Write-Host '$env:DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres"' -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$psql = Get-Command psql -ErrorAction SilentlyContinue
if ($null -eq $psql) {
    Write-Host ""
    Write-Host "psql was not found in PATH." -ForegroundColor Red
    Write-Host "Install PostgreSQL client tools or run the SQL file manually in a working SQL client." -ForegroundColor Yellow
    Write-Host "SQL file: $sqlFile" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "This will permanently delete processed sessions from Supabase:" -ForegroundColor Yellow
Write-Host "  session_reports, samples, quality_metrics, collective_performance_metrics,"
Write-Host "  performance_metrics, athlete_session, metrics, sessions, games"
Write-Host ""
Write-Host "It will preserve reference data:" -ForegroundColor Green
Write-Host "  athletes, fields, selecoes"
Write-Host ""

if (-not $Yes) {
    $confirmation = Read-Host "Type RESET_PROCESSED_SESSIONS to continue"
    if ($confirmation -ne "RESET_PROCESSED_SESSIONS") {
        Write-Host "Cancelled. No database changes were made." -ForegroundColor Cyan
        exit 0
    }
}

Push-Location $repoRoot
try {
    & psql $env:DATABASE_URL -v ON_ERROR_STOP=1 -f $sqlFile
    if ($LASTEXITCODE -ne 0) {
        throw "psql failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Processed session data reset complete." -ForegroundColor Green
}
finally {
    Pop-Location
}
