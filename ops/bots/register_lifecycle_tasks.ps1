<#
.SYNOPSIS
  Register / unregister the two universal bot-lifecycle scheduled tasks.

.DESCRIPTION
  Creates two Windows Task Scheduler entries:
    1. "HQC_BotFleet_Start"  — daily Mon-Fri at 06:25 local (= 09:25 ET)
    2. "HQC_BotFleet_Stop"   — daily Mon-Fri at 13:05 local (= 16:05 ET)

  Both run as the current user (no SYSTEM/elevated escalation; matches how
  each individual bot already runs). Uses pwsh.exe if available, else falls
  back to powershell.exe.

.PARAMETER Action
  Register (default) or Unregister.

.EXAMPLE
  # Install both tasks (run from an admin PowerShell once)
  pwsh -File C:\HQC\ops\bots\register_lifecycle_tasks.ps1

  # Remove them
  pwsh -File C:\HQC\ops\bots\register_lifecycle_tasks.ps1 -Action Unregister
#>
[CmdletBinding()]
param(
  [ValidateSet('Register','Unregister')]
  [string]$Action = 'Register'
)

$ErrorActionPreference = 'Stop'

$startTask = 'HQC_BotFleet_Start'
$stopTask  = 'HQC_BotFleet_Stop'

function Get-PwshExe {
  $candidates = @(
    "$env:ProgramFiles\PowerShell\7\pwsh.exe",
    "$env:ProgramFiles\PowerShell\pwsh.exe",
    "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
  )
  foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
  throw "No pwsh.exe or powershell.exe found."
}

if ($Action -eq 'Unregister') {
  foreach ($n in @($startTask, $stopTask)) {
    try {
      Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction Stop
      Write-Host "Removed task: $n"
    } catch {
      Write-Warning "Could not remove ${n}: $_"
    }
  }
  return
}

$pwsh = Get-PwshExe
$scriptsDir = $PSScriptRoot
$startScript = Join-Path $scriptsDir 'start_all_bots.ps1'
$stopScript  = Join-Path $scriptsDir 'stop_all_bots.ps1'

if (-not (Test-Path $startScript)) { throw "Missing $startScript" }
if (-not (Test-Path $stopScript))  { throw "Missing $stopScript"  }

# Daily Mon-Fri triggers in local time. America/Los_Angeles assumed; if the
# host machine is on a different timezone, edit the time strings below or use
# the time appropriate for your timezone such that the trigger fires within
# the [09:25 ET, 16:05 ET] window.
$weekdays = 'Monday','Tuesday','Wednesday','Thursday','Friday'

function Register-FleetTask {
  param([string]$Name, [string]$Script, [string]$LocalTime)
  $action  = New-ScheduledTaskAction -Execute $pwsh -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""
  $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $LocalTime
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
  Write-Host "Registered task: $Name  ->  $LocalTime  $Script"
}

Register-FleetTask -Name $startTask -Script $startScript -LocalTime '06:25'
Register-FleetTask -Name $stopTask  -Script $stopScript  -LocalTime '13:05'

Write-Host ""
Write-Host "Verify with: Get-ScheduledTask -TaskName 'HQC_BotFleet_*'"
