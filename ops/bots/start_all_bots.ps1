<#
.SYNOPSIS
  Optional market-open launcher for bots that don't already have their own
  Task Scheduler entry.

.DESCRIPTION
  Reads ops/bots/bots.config.json and for every bot whose `launch` is non-null,
  invokes it. Bots with `launch: null` are assumed to be started by their own
  existing scheduled task (DMB, NexusAlpha, MegaMind, OmegaIntelTrader,
  AllWeather etc. all self-schedule) and are skipped here so we don't double-start.

  Intended to run from Windows Task Scheduler at 06:25 America/Los_Angeles
  (= 09:25 ET, five minutes before market open) on weekdays only.

  Idempotent: if a bot's `match` regex already matches a running python.exe,
  it is NOT relaunched. Re-running this script is safe.

.PARAMETER ConfigPath
  Path to bots.config.json. Defaults to the file alongside this script.

.PARAMETER DryRun
  List what WOULD be launched; do not actually launch.

.PARAMETER LogPath
  Append a one-line summary per run to this file. Defaults to
  $env:LOCALAPPDATA\HQC\bot_lifecycle.log.

.EXAMPLE
  pwsh -File C:\HQC\ops\bots\start_all_bots.ps1
  pwsh -File C:\HQC\ops\bots\start_all_bots.ps1 -DryRun
#>
[CmdletBinding()]
param(
  [string]$ConfigPath = (Join-Path $PSScriptRoot 'bots.config.json'),
  [switch]$DryRun,
  [string]$LogPath = (Join-Path $env:LOCALAPPDATA 'HQC\bot_lifecycle.log')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ConfigPath)) {
  throw "Config not found: $ConfigPath"
}

$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$pythonProcs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'"

$launched = New-Object System.Collections.Generic.List[object]
$alreadyRunning = New-Object System.Collections.Generic.List[object]
$noLaunch = New-Object System.Collections.Generic.List[object]
$failed = New-Object System.Collections.Generic.List[object]

foreach ($bot in $cfg.bots) {
  if (-not $bot.launch) {
    $noLaunch.Add($bot.name)
    continue
  }
  $alreadyMatches = $pythonProcs | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match $bot.match)
  }
  if ($alreadyMatches) {
    $alreadyRunning.Add($bot.name)
    continue
  }
  if ($DryRun) {
    $launched.Add([PSCustomObject]@{ Bot=$bot.name; Cmd=$bot.launch; DryRun=$true })
    continue
  }
  try {
    Invoke-Expression $bot.launch
    $launched.Add([PSCustomObject]@{ Bot=$bot.name; Cmd=$bot.launch })
  } catch {
    $failed.Add([PSCustomObject]@{ Bot=$bot.name; Error="$_" })
  }
}

$ts = (Get-Date).ToString('o')
$summary = "{0} start_all_bots: launched={1} already_running={2} no_launch_defined={3} failed={4} dry_run={5}" -f `
  $ts, $launched.Count, $alreadyRunning.Count, $noLaunch.Count, $failed.Count, $DryRun.IsPresent

try {
  $logDir = Split-Path -Parent $LogPath
  if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
  Add-Content -Path $LogPath -Value $summary
} catch {
  Write-Warning "Could not write log to ${LogPath}: $_"
}

Write-Host $summary
if ($launched.Count -gt 0)        { Write-Host "Launched:";        $launched        | Format-Table -AutoSize | Out-String | Write-Host }
if ($alreadyRunning.Count -gt 0)  { Write-Host "Already running: $($alreadyRunning -join ', ')" }
if ($noLaunch.Count -gt 0)        { Write-Host "Self-scheduled (skipped): $($noLaunch -join ', ')" }
if ($failed.Count -gt 0)          { Write-Host "Failed:";          $failed          | Format-Table -AutoSize | Out-String | Write-Host }
