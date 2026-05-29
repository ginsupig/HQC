# Bot Fleet Lifecycle

Operational scripts for starting the 12-bot fleet before market open and
killing the fleet after close. Lives in `ops/bots/`.

## Files

| File | Purpose |
|---|---|
| `ops/bots/bots.config.json` | Bot registry: name, match-regex (for kill), launch-cmd (for start). Edit this to add/remove/relocate bots. |
| `ops/bots/stop_all_bots.ps1` | Universal close kill. Scans every running `python.exe` and force-stops any whose CommandLine matches a bot's regex. |
| `ops/bots/start_all_bots.ps1` | Idempotent launcher. Starts bots whose `launch` is non-null and that aren't already running. Bots with `launch: null` are assumed to self-schedule. |
| `ops/bots/register_lifecycle_tasks.ps1` | Registers/removes the two Task Scheduler entries that wire the scripts to clock-time triggers. |

## Default schedule

| Task | When (America/Los_Angeles) | ET equivalent |
|---|---|---|
| `HQC_BotFleet_Start` | 06:25 Mon–Fri | 09:25 ET — 5 min before open |
| `HQC_BotFleet_Stop`  | 13:05 Mon–Fri | 16:05 ET — 5 min after close |

The 5-minute buffer after close lets bots that have their own EOD liquidation
(HQC `EODLiquidationManager` flattens at 15:55 ET, `OmegaIntelTrader-Close`
runs at the close bell) finish liquidating *before* the force-kill backstop
catches them. Force-kill is the safety net, not the primary EOD mechanism.

If the host machine is on a timezone other than Los Angeles, edit `06:25` /
`13:05` in `register_lifecycle_tasks.ps1` so the triggers land in the
[09:25 ET, 16:05 ET] window.

## Install

Run once from an **admin PowerShell** (Task Scheduler needs admin to register
machine-scope tasks; user-scope tasks would still work but get hidden from
SYSTEM-context cleanup tools):

```powershell
pwsh -File C:\HQC\ops\bots\register_lifecycle_tasks.ps1
```

Verify:

```powershell
Get-ScheduledTask -TaskName 'HQC_BotFleet_*' | Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}}
```

## Test before the first scheduled fire

Dry-run both scripts so you can see what they would do without actually
killing or launching anything:

```powershell
pwsh -File C:\HQC\ops\bots\stop_all_bots.ps1  -DryRun
pwsh -File C:\HQC\ops\bots\start_all_bots.ps1 -DryRun
```

The output lists matched/launchable processes per bot. If a bot you expect
to be in the kill set isn't matched, fix its `match` regex in
`bots.config.json`. If a bot is double-matched (the same PID hits two regexes),
narrow one of them.

## Adding a new bot

Edit `bots.config.json` and add an entry to `bots`:

```json
{
  "name": "new_bot",
  "match": "C:\\\\new_bot\\\\.*main\\.py",
  "launch": "Start-Process -FilePath 'C:\\new_bot\\.venv\\Scripts\\python.exe' -ArgumentList 'main.py' -WorkingDirectory 'C:\\new_bot' -WindowStyle Minimized"
}
```

- `match` is a .NET regex matched against `python.exe` `CommandLine`. Double-escape
  backslashes (JSON `\\\\` → regex `\\` → literal `\`).
- `launch` is a PowerShell expression. Set to `null` if the bot has its own
  scheduled task and you don't want `start_all_bots.ps1` to double-start it.

No script changes needed — the config is the source of truth.

## Uninstall

```powershell
pwsh -File C:\HQC\ops\bots\register_lifecycle_tasks.ps1 -Action Unregister
```

## Logs

Both scripts append a one-line summary per run to
`$env:LOCALAPPDATA\HQC\bot_lifecycle.log`. Override with `-LogPath`.

## What this does NOT do

- It does not replace each bot's internal EOD liquidation flow. Bots that
  carry overnight positions because they skipped their own EOD logic will
  get force-killed mid-position; that's a bot bug, not a lifecycle-script bug.
- It does not handle half-day market closes (e.g., the day after Thanksgiving,
  Christmas Eve). On those days the 13:05 PT trigger fires 2 hours after the
  early close at 13:00 ET; if you want a tighter half-day flat, add a separate
  trigger for the half-day session list.
- It does not arbitrate Alpaca connection-limit conflicts. If two bots share
  the same Alpaca account and one is the orphan from yesterday's crash, the
  start-script will happily launch the new instance into a 406 loop. Killing
  the orphan is on you (or on `stop_all_bots.ps1` having run first).
