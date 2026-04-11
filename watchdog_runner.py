from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    restart_backoff_sec = float(os.getenv("HQC_WATCHDOG_RESTART_SEC", "15"))
    max_restarts = int(os.getenv("HQC_WATCHDOG_MAX_RESTARTS", "0"))
    always_restart = _env_flag("HQC_WATCHDOG_ALWAYS_RESTART", default=True)

    env = os.environ.copy()
    env.setdefault("HQC_RUN_FOREVER", "1")

    restart_count = 0

    while True:
        command = [sys.executable, "main.py"]
        print(f"[WATCHDOG] Launching {' '.join(command)} | restart_count={restart_count}")

        try:
            result = subprocess.run(command, cwd=os.path.dirname(__file__) or None, env=env)
        except KeyboardInterrupt:
            print("[WATCHDOG] KeyboardInterrupt received. Exiting.")
            return 130
        except Exception as exc:
            print(f"[WATCHDOG] Failed to launch process: {exc}")
            result = subprocess.CompletedProcess(command, returncode=1)

        if result.returncode == 0 and not always_restart:
            print("[WATCHDOG] Process exited cleanly. Not restarting.")
            return 0

        restart_count += 1
        if max_restarts > 0 and restart_count > max_restarts:
            print(f"[WATCHDOG] Max restarts exceeded ({max_restarts}). Exiting with {result.returncode}.")
            return result.returncode

        print(f"[WATCHDOG] Process exited with {result.returncode}. Restarting in {restart_backoff_sec:.1f}s.")
        time.sleep(restart_backoff_sec)


if __name__ == "__main__":
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())