param(
    [string]$RemoteHost = "qtran47@login-phoenix.pace.gatech.edu",
    [string]$RemoteRepo = "/storage/home/hcoda1/4/qtran47/AutoResearch-SMB",
    [string]$LocalRepo = (Split-Path -Parent $PSScriptRoot),
    [string]$SyncRoot = "",
    [switch]$Resume,
    [switch]$IncludeLogs,
    [switch]$StageRunsOnly,
    [switch]$IncludeResearch,
    [switch]$StrictMissing,
    [int]$ScpRetries = 3,
    [int]$RetryDelaySeconds = 2
)

$ErrorActionPreference = "Stop"

if (-not $SyncRoot) {
    $paceSyncBase = Join-Path $LocalRepo "artifacts\pace_sync"
    if ($Resume -and (Test-Path $paceSyncBase)) {
        $latest = Get-ChildItem -Path $paceSyncBase -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latest) {
            $SyncRoot = $latest.FullName
            Write-Host "Resume mode: using existing sync root $SyncRoot"
        }
    }
    if (-not $SyncRoot) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $SyncRoot = Join-Path $LocalRepo "artifacts\pace_sync\$timestamp"
    }
}

$scp = Get-Command scp -ErrorAction SilentlyContinue
if (-not $scp) {
    throw "scp not found on PATH. Install or enable OpenSSH client first."
}

$items = @("artifacts/smb_stage_runs", "artifacts/two_scientist_smb")
if (-not $StageRunsOnly) {
    $items += "artifacts/agent_runs"
}
if ($IncludeLogs) {
    $items += "logs"
}
if ($IncludeResearch) {
    $items += "research.md"
}

New-Item -ItemType Directory -Force -Path $SyncRoot | Out-Null

foreach ($item in $items) {
    $remotePath = "${RemoteRepo}/${item}"
    $localTarget = Join-Path $SyncRoot ($item -replace "/", "\")
    $localParent = Split-Path -Parent $localTarget
    New-Item -ItemType Directory -Force -Path $localParent | Out-Null

    if ($Resume -and (Test-Path $localTarget)) {
        $targetItem = Get-Item -Path $localTarget
        $hasContent = $true
        if ($targetItem.PSIsContainer) {
            $hasContent = (Get-ChildItem -Path $localTarget -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
        }
        if ($hasContent) {
            Write-Host "Skipping $item (already present): $localTarget"
            continue
        }
    }

    Write-Host "Syncing $item -> $localTarget"

    $scpExit = 1
    $scpOutput = @()
    for ($attempt = 1; $attempt -le $ScpRetries; $attempt++) {
        $scpOutput = @(& $scp.Source -r "${RemoteHost}:${remotePath}" "$localParent" 2>&1)
        $scpExit = $LASTEXITCODE
        if ($scpExit -eq 0) {
            break
        }

        $outputText = ($scpOutput -join "`n")
        $isTransportError = $outputText -match "Connection closed|Connection reset|Connection timed out|No route to host|Network is unreachable|Operation timed out|Could not resolve hostname"
        if ($isTransportError -and $attempt -lt $ScpRetries) {
            Write-Warning "Transient SSH/SCP failure for '$item' (attempt $attempt/$ScpRetries). Retrying in ${RetryDelaySeconds}s..."
            Start-Sleep -Seconds $RetryDelaySeconds
            continue
        }

        break
    }

    if ($scpExit -ne 0) {
        $outputText = ($scpOutput -join "`n")
        $missingRemote = $outputText -match "No such file or directory|not found"
        $isTransportError = $outputText -match "Connection closed|Connection reset|Connection timed out|No route to host|Network is unreachable|Operation timed out|Could not resolve hostname"

        if (($missingRemote -or $isTransportError) -and -not $StrictMissing) {
            $kind = if ($missingRemote) { "Remote path missing" } else { "Transport failure after $ScpRetries attempts" }
            Write-Warning "$kind, skipping: $remotePath"
            continue
        }

        $tail = ($scpOutput | Select-Object -Last 10) -join "`n"
        throw "scp failed while syncing '$item' from '$RemoteHost'.`n$tail"
    }
}

Write-Host ""
Write-Host "PACE sync complete."
Write-Host "Local sync root: $SyncRoot"
