import subprocess
import sys
import time
import os
from pathlib import Path

def print_header(title: str):
    print("\n" + "="*60)
    print(f">> RUNNING: {title}")
    print("="*60 + "\n")

def run_script(script_name: str, is_unittest: bool = False):
    """
    Automatically finds the script recursively in the project and executes it
    with PYTHONPATH set to the root directory to fix import errors.
    """
    root_dir = Path(__file__).parent.resolve()

    # 1. Find the file anywhere in the HQC folder
    found_paths = list(root_dir.rglob(script_name))
    if not found_paths:
        print(f"\n[ERROR] Could not find '{script_name}' anywhere in {root_dir}")
        print("   -> Please ensure the file has been saved.")
        return False

    script_path = found_paths[0]

    # 2. Inject the root directory into PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    env["PYTHONIOENCODING"] = "utf-8"

    # 3. Build the execution command
    cmd = [sys.executable, "-X", "utf8"]
    if is_unittest:
        cmd.extend(["-m", "unittest"])

    cmd.append(str(script_path))

    try:
        result = subprocess.run(cmd, text=True, env=env)
        if result.returncode == 0:
            print(f"\n[PASS] {script_name}")
            return True
        else:
            print(f"\n[FAIL] {script_name} returned exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n[ERROR] Failed to execute {script_name}: {e}")
        return False

def main():
    print("Initializing HQC System-Wide Test Suite...\n")
    time.sleep(1)

    tests = [
        # 1. Feature Engineering & ML Pipeline
        {"name": "feature_engineering.py", "title": "Stationary PCA Feature Engineering", "is_unittest": False},
        {"name": "hmm_classifier.py", "title": "HMM Regime Classifier", "is_unittest": False},
        {"name": "model_retraining.py", "title": "LightGBM Background Retraining Thread", "is_unittest": False},

        # 2. Market Microstructure & Risk
        {"name": "liquidity_scorer.py", "title": "Amihud Liquidity Flash-Crash Detection", "is_unittest": False},
        {"name": "equity_slope_detector.py", "title": "Virtual Equity & Drawdown Governance", "is_unittest": False},

        # 3. Strategy Engines
        {"name": "kalman_spread.py", "title": "Kalman Filter Pairs Trading (Absolute Time Gating)", "is_unittest": False},
        {"name": "hunter_state_machine.py", "title": "VWAP Bounce State Machine", "is_unittest": False},

        # 4. Core Architecture (Unit Tests)
        {"name": "test_backtest_parity.py", "title": "ORB Execution Parity (Zero Lookahead Bias)", "is_unittest": True},
        {"name": "test_historical_monte_carlo.py", "title": "Monte Carlo Historical Day Stress Test", "is_unittest": True},
    ]

    passed = 0
    failed = []

    for test in tests:
        print_header(test["title"])
        success = run_script(test["name"], test["is_unittest"])
        if success:
            passed += 1
        else:
            failed.append(test["name"])
        time.sleep(0.5)

    print("\n" + "="*60)
    print(f"TEST SUITE COMPLETE")
    print(f"Passed: {passed} / {len(tests)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"   - {f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
