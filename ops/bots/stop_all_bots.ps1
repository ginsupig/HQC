<#
.SYNOPSIS
  Universal market-close kill for the 12-bot fleet.

.DESCRIPTION
  Reads ops/bots/bots.config.json, finds every running python.exe whose
  CommandLine matches any bot's `match` regex, and force-kills it.

  Intended to run from Windows Task Scheduler at 13:05 America/Los_Angeles
  (= 16:05 ET, five minutes after market close). The five-minute buffer lets
  bots with their own EOD liquidation flow (e.g., OmegaIntelTrader-Close,
  HQC EODLiquidationManager) liquidate cleanly before the force-kill backstop.

.PARAMETER ConfigPath
  Path to bots.config.json. Defaults to the file alongside this script.

.PARAMETER DryRun
  List what WOULD be killed; do not actually kill.

.PARAMETER LogPath
  Append a one-line summary per run to this file. Defaults to
  $env:LOCALAPPDATA\HQC\bot_lifecycle.log.

.EXAMPLE
  pwsh -File C:\HQC\ops\bots\stop_all_bots.ps1
  pwsh -File C:\HQC\ops\bots\stop_all_bots.ps1 -DryRun
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

$killed = New-Object System.Collections.Generic.List[object]
$skipped = New-Object System.Collections.Generic.List[object]

foreach ($bot in $cfg.bots) {
  $matches = $pythonProcs | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match $bot.match)
  }
  foreach ($p in $matches) {
    if ($DryRun) {
      $skipped.Add([PSCustomObject]@{ Bot=$bot.name; PID=$p.ProcessId; Cmd=$p.CommandLine })
      continue
    }
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      $killed.Add([PSCustomObject]@{ Bot=$bot.name; PID=$p.ProcessId })
    } catch {
      # Process may have died between enumeration and kill — note and move on.
      $skipped.Add([PSCustomObject]@{ Bot=$bot.name; PID=$p.ProcessId; Reason="$_" })
    }
  }
}

$ts = (Get-Date).ToString('o')
$summary = "{0} stop_all_bots: killed={1} skipped={2} dry_run={3}" -f $ts, $killed.Count, $skipped.Count, $DryRun.IsPresent

# Best-effort log; never blow up the kill on log failure.
try {
  $logDir = Split-Path -Parent $LogPath
  if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
  Add-Content -Path $LogPath -Value $summary
} catch {
  Write-Warning "Could not write log to ${LogPath}: $_"
}

Write-Host $summary
if ($killed.Count -gt 0)  { $killed  | Format-Table -AutoSize | Out-String | Write-Host }
if ($skipped.Count -gt 0) { $skipped | Format-Table -AutoSize | Out-String | Write-Host }
