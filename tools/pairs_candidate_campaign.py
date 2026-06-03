"""
Candidate-pair campaign runner with pre-registered family sizing.

This workflow keeps research candidates separate from deployment config,
batch-runs pair walk-forwards, applies family correction with the
pre-registered family size, runs the A2/A3/A4/D3 gates, and writes a
standardized audit trail plus a ranked campaign summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import a2_analyze  # noqa: E402
import a3_analyze  # noqa: E402
import analyze_walkforward  # noqa: E402
import multiple_comparisons  # noqa: E402
import promote_pairs  # noqa: E402  (stdlib + yaml only; safe to import here)

D3_WITH_TIME_STOP = "with_ts"
D3_WITHOUT_TIME_STOP = "without_ts"
UNKNOWN_BUCKET_SORT_ORDER = 9
INVALID_P_SORT_ORDER = 9


@dataclass
class GateConfig:
    enabled: bool = True
    workers: int = 1
    smoke: bool = False
    spy_csv: str = ""
    spy_symbol: str = "SPY"


@dataclass
class CandidatePair:
    y: str
    x: str
    csv_y: str
    csv_x: str
    thesis: str = ""
    entry_z: Optional[float] = None
    exit_z: Optional[float] = None
    delta: Optional[float] = None
    ve: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    nominal_stop_pct: Optional[float] = None
    target_dollar_notional: Optional[float] = None

    @property
    def label(self) -> str:
        return f"{self.y.upper()}/{self.x.upper()}"

    @property
    def slug(self) -> str:
        return self.label.replace("/", "_").lower()


@dataclass
class PromotionConfig:
    """Optional auto-promotion of APPROVED pairs into the deploy config after a
    campaign run. Off by default; dry-run by default when on."""
    enabled: bool = False
    write: bool = False
    target: str = "config/pairs.yaml"
    keep_existing: bool = False
    buying_power: Optional[float] = None
    max_leverage: float = 1.0
    allow_empty: bool = False


@dataclass
class CampaignConfig:
    output_dir: str
    pre_registered_family_size: int
    alpha: float = 0.05
    bootstrap: int = 10000
    train_days: int = 30
    test_days: int = 10
    initial_capital: float = 100000.0
    sim_max_hold_minutes: int = 240
    slippage_bps_per_side: float = 1.5
    sec_fee_rate: float = 0.000008
    short_borrow_apr: float = 0.0025
    pair_entry_z: float = 1.5
    pair_exit_z: float = 0.4
    pair_delta: float = 1e-4
    pair_ve: float = 1e-3
    pair_max_leg_staleness_sec: float = 30.0
    pair_cooldown_seconds: float = 120.0
    pair_nominal_stop_pct: float = 0.02
    pair_target_dollar_notional: float = 10000.0
    gates: Dict[str, GateConfig] = field(default_factory=dict)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    pairs: List[CandidatePair] = field(default_factory=list)


@dataclass
class GateResult:
    status: str
    detail: str
    artifact: str = ""


def load_campaign(path: Path) -> CampaignConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    campaign_raw = raw.get("campaign") or {}
    gates_raw = campaign_raw.get("gates") or {}
    gates = {
        name: GateConfig(
            enabled=bool((cfg or {}).get("enabled", True)),
            workers=int((cfg or {}).get("workers", 1)),
            smoke=bool((cfg or {}).get("smoke", False)),
            spy_csv=str((cfg or {}).get("spy_csv", "")),
            spy_symbol=str((cfg or {}).get("spy_symbol", "SPY")),
        )
        for name, cfg in gates_raw.items()
    }
    promotion_raw = campaign_raw.get("promotion") or {}
    promotion = PromotionConfig(
        enabled=bool(promotion_raw.get("enabled", False)),
        write=bool(promotion_raw.get("write", False)),
        target=str(promotion_raw.get("target", "config/pairs.yaml")),
        keep_existing=bool(promotion_raw.get("keep_existing", False)),
        buying_power=(float(promotion_raw["buying_power"]) if promotion_raw.get("buying_power") is not None else None),
        max_leverage=float(promotion_raw.get("max_leverage", 1.0)),
        allow_empty=bool(promotion_raw.get("allow_empty", False)),
    )
    pairs = [
        CandidatePair(
            y=str(p["y"]).upper(),
            x=str(p["x"]).upper(),
            csv_y=str(p["csv_y"]),
            csv_x=str(p["csv_x"]),
            thesis=str(p.get("thesis", "")),
            entry_z=float(p["entry_z"]) if p.get("entry_z") is not None else None,
            exit_z=float(p["exit_z"]) if p.get("exit_z") is not None else None,
            delta=float(p["delta"]) if p.get("delta") is not None else None,
            ve=float(p["ve"]) if p.get("ve") is not None else None,
            cooldown_seconds=float(p["cooldown_seconds"]) if p.get("cooldown_seconds") is not None else None,
            nominal_stop_pct=float(p["nominal_stop_pct"]) if p.get("nominal_stop_pct") is not None else None,
            target_dollar_notional=float(p["target_dollar_notional"]) if p.get("target_dollar_notional") is not None else None,
        )
        for p in (raw.get("pairs") or [])
    ]
    cfg = CampaignConfig(
        output_dir=str(campaign_raw["output_dir"]),
        pre_registered_family_size=int(campaign_raw["pre_registered_family_size"]),
        alpha=float(campaign_raw.get("alpha", 0.05)),
        bootstrap=int(campaign_raw.get("bootstrap", 10000)),
        train_days=int(campaign_raw.get("train_days", 30)),
        test_days=int(campaign_raw.get("test_days", 10)),
        initial_capital=float(campaign_raw.get("initial_capital", 100000.0)),
        sim_max_hold_minutes=int(campaign_raw.get("sim_max_hold_minutes", 240)),
        slippage_bps_per_side=float(campaign_raw.get("slippage_bps_per_side", 1.5)),
        sec_fee_rate=float(campaign_raw.get("sec_fee_rate", 0.000008)),
        short_borrow_apr=float(campaign_raw.get("short_borrow_apr", 0.0025)),
        pair_entry_z=float(campaign_raw.get("pair_entry_z", 1.5)),
        pair_exit_z=float(campaign_raw.get("pair_exit_z", 0.4)),
        pair_delta=float(campaign_raw.get("pair_delta", 1e-4)),
        pair_ve=float(campaign_raw.get("pair_ve", 1e-3)),
        pair_max_leg_staleness_sec=float(campaign_raw.get("pair_max_leg_staleness_sec", 30.0)),
        pair_cooldown_seconds=float(campaign_raw.get("pair_cooldown_seconds", 120.0)),
        pair_nominal_stop_pct=float(campaign_raw.get("pair_nominal_stop_pct", 0.02)),
        pair_target_dollar_notional=float(campaign_raw.get("pair_target_dollar_notional", 10000.0)),
        gates=gates,
        promotion=promotion,
        pairs=pairs,
    )
    if cfg.pre_registered_family_size < len(cfg.pairs):
        raise ValueError(
            f"pre_registered_family_size={cfg.pre_registered_family_size} is smaller than candidates={len(cfg.pairs)}"
        )
    if not cfg.pairs:
        raise ValueError("campaign config must define at least one candidate pair")
    return cfg


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _pair_value(pair: CandidatePair, attr: str, default: float) -> float:
    value = getattr(pair, attr)
    return float(default if value is None else value)


import time as _time

# Set by main() from --stream / --quiet-subprocess. When True, sub-step output
# (walk-forward, A2/A3/A4/D3) is streamed live to the console AND teed to the
# log; when False it is captured to the log only (the old silent behaviour).
STREAM_OUTPUT = True


def _run_command(cmd: List[str], log_path: Path, dry_run: bool, label: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN:\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return
    tag = label or Path(cmd[1]).stem if len(cmd) > 1 else "step"
    print(f"    -> {tag}  (log: {log_path})", flush=True)
    t0 = _time.time()

    if STREAM_OUTPUT:
        # Stream stdout+stderr live and tee to the log so the run is observable.
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd, cwd=_REPO_ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"      | {line}", end="", flush=True)
                logf.write(line)
            returncode = proc.wait()
    else:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, capture_output=True, check=False)
        log_path.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
        returncode = proc.returncode

    elapsed = _time.time() - t0
    print(f"       {tag} done in {elapsed:.0f}s (rc={returncode})", flush=True)
    if returncode != 0:
        raise RuntimeError(f"Command failed ({returncode}): {' '.join(cmd)}")


def summarize_walkforward_payload(payload: dict, bootstrap: int, alpha: float) -> dict:
    symbol_block = (payload.get("results") or [])[0]
    returns = analyze_walkforward._per_window_returns(symbol_block)
    mean, lo, hi = analyze_walkforward._bootstrap_mean_ci(returns, n_boot=bootstrap, alpha=alpha)
    raw_p = analyze_walkforward._one_sample_p_greater_than_zero(returns)
    return {
        "windows": len(returns),
        "mean_pct": round(mean * 100.0, 6),
        "ci_lo": round(lo * 100.0, 6),
        "ci_hi": round(hi * 100.0, 6),
        "raw_p": raw_p,
        "edge_plus": bool(lo > 0 and math.isfinite(raw_p) and raw_p < alpha),
        "total_test_pnl": float(symbol_block.get("total_test_pnl") or 0.0),
    }


def classify_candidate(stat_pass: bool, corrected_pass: bool, gates: Dict[str, GateResult]) -> str:
    statuses = [g.status for g in gates.values()]
    if stat_pass and corrected_pass and statuses and all(s == "PASS" for s in statuses):
        return "APPROVED"
    if stat_pass and corrected_pass and all(s in {"PASS", "PENDING", "ATTENTION"} for s in statuses):
        return "PROBATION"
    return "REJECTED"


def _summarize_a2(path: Path, pair: CandidatePair, alpha: float) -> GateResult:
    rows = a2_analyze._load(path)
    if not rows:
        return GateResult("PENDING", "no A2 rows", str(path))
    family_size = len(rows)
    bonf_threshold = alpha / family_size
    by_slice: Dict[Tuple[float, float], List[dict]] = {}
    for row in rows:
        by_slice.setdefault((row["delta"], row["ve"]), []).append(row)
    deployed = (_pair_value(pair, "entry_z", 1.5), _pair_value(pair, "exit_z", 0.4))
    deployed_slice_key = (_pair_value(pair, "delta", 1e-4), _pair_value(pair, "ve", 1e-3))
    rows_slice = by_slice.get(deployed_slice_key)
    if not rows_slice:
        return GateResult("PENDING", "deployed A2 slice missing", str(path))
    entry_vals = sorted({r["entry_z"] for r in rows_slice})
    exit_vals = sorted({r["exit_z"] for r in rows_slice})
    cells = {(r["entry_z"], r["exit_z"]): r for r in rows_slice}
    is_edge = {
        key: row["ci_lo"] > 0 and math.isfinite(row["raw_p"]) and row["raw_p"] <= bonf_threshold
        for key, row in cells.items()
    }
    region = a2_analyze._connected_region(cells, entry_vals, exit_vals, deployed, is_edge)
    if not is_edge.get(deployed, False):
        return GateResult("FAIL", "A2 deployed cell fails Bonferroni", str(path))
    if len(region) >= 4:
        return GateResult("PASS", f"A2 plateau size={len(region)}", str(path))
    return GateResult("FAIL", f"A2 spike size={len(region)}", str(path))


def _summarize_a3(path: Path, pair_label: str, alpha: float) -> GateResult:
    rows = [r for r in a3_analyze._load(path) if (r.get("pair") or "").strip() == pair_label]
    if not rows:
        return GateResult("PENDING", "no A3 rows", str(path))
    bonf = alpha / len(rows)
    cells = {(r["slippage_bps_per_side"], r["short_borrow_apr"]): r for r in rows}
    deployed = cells.get((a3_analyze.DEPLOYED_SLIPPAGE_BPS, a3_analyze.DEPLOYED_BORROW_APR))
    if deployed is None:
        return GateResult("PENDING", "deployed A3 cost cell missing", str(path))
    if not a3_analyze._is_edge(deployed, bonf):
        return GateResult("FAIL", "A3 deployed cost cell fails", str(path))
    slip_vals = sorted({r["slippage_bps_per_side"] for r in rows})
    borrow_vals = sorted({r["short_borrow_apr"] for r in rows})
    slip_breakeven = next(
        (
            s for s in slip_vals
            if s > a3_analyze.DEPLOYED_SLIPPAGE_BPS
            and (cells.get((s, a3_analyze.DEPLOYED_BORROW_APR)) is not None)
            and (not a3_analyze._is_edge(cells[(s, a3_analyze.DEPLOYED_BORROW_APR)], bonf))
        ),
        None,
    )
    borrow_breakeven = next(
        (
            b for b in borrow_vals
            if b > a3_analyze.DEPLOYED_BORROW_APR
            and (cells.get((a3_analyze.DEPLOYED_SLIPPAGE_BPS, b)) is not None)
            and (not a3_analyze._is_edge(cells[(a3_analyze.DEPLOYED_SLIPPAGE_BPS, b)], bonf))
        ),
        None,
    )
    fragile_slip = slip_breakeven is not None and (slip_breakeven - a3_analyze.DEPLOYED_SLIPPAGE_BPS) <= a3_analyze.FRAGILE_SLIPPAGE_BUMP_BPS
    fragile_borrow = borrow_breakeven is not None and (borrow_breakeven - a3_analyze.DEPLOYED_BORROW_APR) <= a3_analyze.FRAGILE_BORROW_BUMP_APR
    if fragile_slip or fragile_borrow:
        return GateResult("ATTENTION", "A3 cost-fragile", str(path))
    return GateResult("PASS", "A3 cost margin intact", str(path))


def _summarize_a4(path: Path, pair_label: str) -> GateResult:
    if not path.exists():
        return GateResult("PENDING", "A4 output missing", str(path))
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("pair") or "").strip() == pair_label:
                rows.append(row)
    if not rows:
        return GateResult("PENDING", "no A4 rows", str(path))
    edge_rows = [r for r in rows if str(r.get("edge_plus", "")).strip().lower() == "true"]
    if edge_rows:
        regimes = ",".join(sorted(r["regime"] for r in edge_rows))
        return GateResult("PASS", f"A4 edge regimes={regimes}", str(path))
    return GateResult("FAIL", "A4 found no eligible edge regime", str(path))


def _summarize_d3(path: Path, pair_label: str) -> GateResult:
    if not path.exists():
        return GateResult("PENDING", "D3 output missing", str(path))
    by_config: Dict[str, List[float]] = {D3_WITH_TIME_STOP: [], D3_WITHOUT_TIME_STOP: []}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("pair") or "").strip() != pair_label:
                continue
            cfg = str(row.get("config") or "")
            if cfg in by_config:
                by_config[cfg].append(float(row["return_pct"]))
    if not by_config[D3_WITH_TIME_STOP] or not by_config[D3_WITHOUT_TIME_STOP]:
        return GateResult("PENDING", "D3 missing one comparison arm", str(path))
    mean_with = sum(by_config[D3_WITH_TIME_STOP]) / len(by_config[D3_WITH_TIME_STOP])
    mean_without = sum(by_config[D3_WITHOUT_TIME_STOP]) / len(by_config[D3_WITHOUT_TIME_STOP])
    if abs(mean_with) <= 1e-9:
        return GateResult("PENDING", "D3 with_ts mean near zero", str(path))
    rel_gap = ((mean_with - mean_without) / mean_with) * 100.0
    if abs(rel_gap) < 10.0:
        return GateResult("PASS", f"D3 gap={rel_gap:+.1f}%", str(path))
    if abs(rel_gap) < 30.0:
        return GateResult("ATTENTION", f"D3 gap={rel_gap:+.1f}%", str(path))
    return GateResult("FAIL", f"D3 gap={rel_gap:+.1f}%", str(path))


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--promote", action="store_true",
                        help="Force auto-promotion (dry-run) after the campaign, "
                             "overriding the config's promotion.enabled.")
    parser.add_argument("--promote-write", action="store_true",
                        help="Force auto-promotion AND write the deploy config.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Disable auto-promotion even if the config enables it.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for ALL gate sweeps (A2/A3/A4/D3 — the slow "
                             "part). Defaults to the CPU count; pass --workers 1 to force "
                             "serial. Overrides per-gate workers in the config.")
    parser.add_argument("--screen", action="store_true",
                        help="Fast triage: run only the per-pair walk-forward and skip the "
                             "A2/A3/A4/D3 gates. Pairs land in PROBATION at best (no gate "
                             "evidence) — use it to see which pairs have raw edge before "
                             "paying for full validation.")
    parser.add_argument("--quiet-subprocess", action="store_true",
                        help="Capture sub-step output to logs only (old silent behaviour). "
                             "Default streams it live so the run is observable.")
    args = parser.parse_args()

    global STREAM_OUTPUT
    STREAM_OUTPUT = not args.quiet_subprocess

    config_path = args.config.resolve()
    config = load_campaign(config_path)

    # Worker count: explicit --workers, else auto-detect the CPU count so the
    # gate sweeps parallelise by default (they are the slow part). Overrides the
    # per-gate workers in the config. --screen disables the gates entirely.
    effective_workers = args.workers if args.workers is not None else max(1, os.cpu_count() or 1)
    for gate in config.gates.values():
        gate.workers = max(1, effective_workers)
    if args.screen:
        for gate in config.gates.values():
            gate.enabled = False
        print("[screen] fast mode: gates disabled, running walk-forward only.")

    base_dir = _REPO_ROOT
    output_dir = _resolve_path(config.output_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = config.pairs[: args.max_pairs] if args.max_pairs else config.pairs
    total = len(pairs)
    print(
        f"Campaign: {total} pair(s), gates="
        f"{'OFF (screen)' if args.screen else ','.join(n for n, g in config.gates.items() if g.enabled) or 'none'}"
        f", workers={effective_workers}{' (auto)' if args.workers is None else ''}, output={output_dir}"
    )
    family_rows: List[dict] = []
    pair_summaries: List[dict] = []

    campaign_t0 = _time.time()
    for idx, pair in enumerate(pairs, 1):
        print(f"\n[{idx}/{total}] {pair.label}  ({_time.strftime('%H:%M:%S')})", flush=True)
        pair_t0 = _time.time()
        pair_dir = output_dir / pair.slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        csv_y = _resolve_path(pair.csv_y, base_dir)
        csv_x = _resolve_path(pair.csv_x, base_dir)
        walkforward_json = pair_dir / "walkforward.json"
        walkforward_log = pair_dir / "walkforward.log"
        walkforward_cmd = [
            sys.executable,
            str(_REPO_ROOT / "walkforward_pairs.py"),
            "--csv-y", str(csv_y),
            "--csv-x", str(csv_x),
            "--symbol-y", pair.y,
            "--symbol-x", pair.x,
            "--train-days", str(config.train_days),
            "--test-days", str(config.test_days),
            "--initial-capital", str(config.initial_capital),
            "--pair-entry-z", str(_pair_value(pair, "entry_z", config.pair_entry_z)),
            "--pair-exit-z", str(_pair_value(pair, "exit_z", config.pair_exit_z)),
            "--pair-delta", str(_pair_value(pair, "delta", config.pair_delta)),
            "--pair-ve", str(_pair_value(pair, "ve", config.pair_ve)),
            "--pair-max-leg-staleness-sec", str(config.pair_max_leg_staleness_sec),
            "--pair-cooldown-seconds", str(_pair_value(pair, "cooldown_seconds", config.pair_cooldown_seconds)),
            "--pair-nominal-stop-pct", str(_pair_value(pair, "nominal_stop_pct", config.pair_nominal_stop_pct)),
            "--pair-target-dollar-notional", str(_pair_value(pair, "target_dollar_notional", config.pair_target_dollar_notional)),
            "--sim-max-hold-minutes", str(config.sim_max_hold_minutes),
            "--slippage-bps-per-side", str(config.slippage_bps_per_side),
            "--sec-fee-rate", str(config.sec_fee_rate),
            "--output", str(walkforward_json),
        ]
        _run_command(walkforward_cmd, walkforward_log, args.dry_run)

        gates: Dict[str, GateResult] = {}
        stats: dict
        if args.dry_run:
            stats = {"windows": 0, "mean_pct": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "raw_p": float("nan"), "edge_plus": False, "total_test_pnl": 0.0}
            gates = {name: GateResult("PENDING", "dry run") for name in ("a2", "a3", "a4", "d3")}
        else:
            payload = json.loads(walkforward_json.read_text(encoding="utf-8"))
            stats = summarize_walkforward_payload(payload, bootstrap=config.bootstrap, alpha=config.alpha)

            gate_cfg = config.gates.get("a2", GateConfig())
            if gate_cfg.enabled:
                a2_csv = pair_dir / "a2_surface.csv"
                _run_command(
                    [
                        sys.executable, str(_REPO_ROOT / "tools" / "a2_parameter_sensitivity.py"),
                        "--csv-y", str(csv_y), "--csv-x", str(csv_x),
                        "--symbol-y", pair.y, "--symbol-x", pair.x,
                        "--train-days", str(config.train_days), "--test-days", str(config.test_days),
                        "--initial-capital", str(config.initial_capital),
                        "--sim-max-hold-minutes", str(config.sim_max_hold_minutes),
                        "--slippage-bps-per-side", str(config.slippage_bps_per_side),
                        "--sec-fee-rate", str(config.sec_fee_rate),
                        "--short-borrow-apr", str(config.short_borrow_apr),
                        "--pair-max-leg-staleness-sec", str(config.pair_max_leg_staleness_sec),
                        "--pair-cooldown-seconds", str(_pair_value(pair, "cooldown_seconds", config.pair_cooldown_seconds)),
                        "--pair-nominal-stop-pct", str(_pair_value(pair, "nominal_stop_pct", config.pair_nominal_stop_pct)),
                        "--pair-target-dollar-notional", str(_pair_value(pair, "target_dollar_notional", config.pair_target_dollar_notional)),
                        "--workers", str(gate_cfg.workers),
                        "--output", str(a2_csv),
                    ] + (["--smoke"] if gate_cfg.smoke else []),
                    pair_dir / "a2.log",
                    False,
                )
                gates["a2"] = _summarize_a2(a2_csv, pair, config.alpha)

            gate_cfg = config.gates.get("a3", GateConfig())
            if gate_cfg.enabled:
                a3_csv = pair_dir / "a3_surface.csv"
                _run_command(
                    [
                        sys.executable, str(_REPO_ROOT / "tools" / "a3_cost_stress.py"),
                        "--csv-y", str(csv_y), "--csv-x", str(csv_x),
                        "--symbol-y", pair.y, "--symbol-x", pair.x,
                        "--train-days", str(config.train_days), "--test-days", str(config.test_days),
                        "--initial-capital", str(config.initial_capital),
                        "--sim-max-hold-minutes", str(config.sim_max_hold_minutes),
                        "--sec-fee-rate", str(config.sec_fee_rate),
                        "--pair-entry-z", str(_pair_value(pair, "entry_z", config.pair_entry_z)),
                        "--pair-exit-z", str(_pair_value(pair, "exit_z", config.pair_exit_z)),
                        "--pair-delta", str(_pair_value(pair, "delta", config.pair_delta)),
                        "--pair-ve", str(_pair_value(pair, "ve", config.pair_ve)),
                        "--pair-max-leg-staleness-sec", str(config.pair_max_leg_staleness_sec),
                        "--pair-cooldown-seconds", str(_pair_value(pair, "cooldown_seconds", config.pair_cooldown_seconds)),
                        "--pair-nominal-stop-pct", str(_pair_value(pair, "nominal_stop_pct", config.pair_nominal_stop_pct)),
                        "--pair-target-dollar-notional", str(_pair_value(pair, "target_dollar_notional", config.pair_target_dollar_notional)),
                        "--workers", str(gate_cfg.workers),
                        "--output", str(a3_csv),
                    ],
                    pair_dir / "a3.log",
                    False,
                )
                gates["a3"] = _summarize_a3(a3_csv, pair.label, config.alpha)

            gate_cfg = config.gates.get("a4", GateConfig())
            if gate_cfg.enabled:
                a4_csv = pair_dir / "a4_regimes.csv"
                spy_csv = _resolve_path(gate_cfg.spy_csv, base_dir) if gate_cfg.spy_csv else None
                if spy_csv and spy_csv.exists():
                    _run_command(
                        [
                            sys.executable, str(_REPO_ROOT / "tools" / "a4_regime_split.py"),
                            "--walkforward", str(walkforward_json),
                            "--spy-csv", str(spy_csv),
                            "--spy-symbol", gate_cfg.spy_symbol,
                            "--alpha", str(config.alpha),
                            "--output", str(a4_csv),
                        ],
                        pair_dir / "a4.log",
                        False,
                    )
                    gates["a4"] = _summarize_a4(a4_csv, pair.label)
                else:
                    gates["a4"] = GateResult("PENDING", "spy_csv not configured", str(a4_csv))

            gate_cfg = config.gates.get("d3", GateConfig())
            if gate_cfg.enabled:
                d3_csv = pair_dir / "d3_fidelity.csv"
                _run_command(
                    [
                        sys.executable, str(_REPO_ROOT / "tools" / "d3_bracket_fidelity.py"),
                        "--csv-y", str(csv_y), "--csv-x", str(csv_x),
                        "--symbol-y", pair.y, "--symbol-x", pair.x,
                        "--train-days", str(config.train_days), "--test-days", str(config.test_days),
                        "--initial-capital", str(config.initial_capital),
                        "--slippage-bps-per-side", str(config.slippage_bps_per_side),
                        "--sec-fee-rate", str(config.sec_fee_rate),
                        "--short-borrow-apr", str(config.short_borrow_apr),
                        "--pair-entry-z", str(_pair_value(pair, "entry_z", config.pair_entry_z)),
                        "--pair-exit-z", str(_pair_value(pair, "exit_z", config.pair_exit_z)),
                        "--pair-delta", str(_pair_value(pair, "delta", config.pair_delta)),
                        "--pair-ve", str(_pair_value(pair, "ve", config.pair_ve)),
                        "--pair-max-leg-staleness-sec", str(config.pair_max_leg_staleness_sec),
                        "--pair-cooldown-seconds", str(_pair_value(pair, "cooldown_seconds", config.pair_cooldown_seconds)),
                        "--pair-nominal-stop-pct", str(_pair_value(pair, "nominal_stop_pct", config.pair_nominal_stop_pct)),
                        "--pair-target-dollar-notional", str(_pair_value(pair, "target_dollar_notional", config.pair_target_dollar_notional)),
                        "--alpha", str(config.alpha),
                        "--workers", str(gate_cfg.workers),
                        "--output", str(d3_csv),
                    ],
                    pair_dir / "d3.log",
                    False,
                )
                gates["d3"] = _summarize_d3(d3_csv, pair.label)

        family_rows.append({"pair": pair.label, "raw_p": "" if not math.isfinite(stats["raw_p"]) else round(stats["raw_p"], 6)})
        pair_summary = {
            "pair": pair.label,
            "thesis": pair.thesis,
            "raw_stats": stats,
            "gates": {name: asdict(result) for name, result in gates.items()},
            "artifacts_dir": str(pair_dir),
        }
        (pair_dir / "summary.json").write_text(json.dumps(pair_summary, indent=2), encoding="utf-8")
        pair_summaries.append(pair_summary)
        gate_tags = " ".join(f"{n}={g.status}" for n, g in gates.items()) or "(no gates)"
        print(
            f"[{idx}/{total}] {pair.label} done in {_time.time() - pair_t0:.0f}s  "
            f"raw_p={'' if not math.isfinite(stats['raw_p']) else round(stats['raw_p'], 4)}  "
            f"mean={stats['mean_pct']:+.3f}%  {gate_tags}",
            flush=True,
        )

    print(f"\nAll {total} pair(s) processed in {_time.time() - campaign_t0:.0f}s.", flush=True)
    family_csv = output_dir / "family.csv"
    _write_csv(family_csv, family_rows, ["pair", "raw_p"])

    valid_family_tests = [
        multiple_comparisons.TestResult(pair=row["pair"], raw_p=float(row["raw_p"]))
        for row in family_rows
        if row.get("raw_p") not in ("", None)
    ]
    corrected = multiple_comparisons.apply_corrections(
        valid_family_tests,
        alpha=config.alpha,
        family_size=config.pre_registered_family_size,
    ) if valid_family_tests else []
    corrected_by_pair = {row.pair: row for row in corrected}

    ranking_rows: List[dict] = []
    for summary in pair_summaries:
        pair_label = summary["pair"]
        stats = summary["raw_stats"]
        corrected_row = corrected_by_pair.get(pair_label)
        stat_pass = bool(stats["ci_lo"] > 0 and math.isfinite(stats["raw_p"]))
        corrected_pass = bool(corrected_row and corrected_row.bonferroni_pass)
        gate_map = {
            name: GateResult(**gate)
            for name, gate in (summary.get("gates") or {}).items()
        }
        bucket = classify_candidate(stat_pass, corrected_pass, gate_map)
        ranking_rows.append({
            "pair": pair_label,
            "bucket": bucket,
            "mean_pct": stats["mean_pct"],
            "ci_lo": stats["ci_lo"],
            "ci_hi": stats["ci_hi"],
            "raw_p": "" if not math.isfinite(stats["raw_p"]) else round(stats["raw_p"], 6),
            "bonferroni_pass": bool(corrected_row and corrected_row.bonferroni_pass),
            "bonferroni_threshold": corrected_row.bonferroni_threshold if corrected_row else "",
            "a2": gate_map.get("a2", GateResult("PENDING", "not run")).status,
            "a3": gate_map.get("a3", GateResult("PENDING", "not run")).status,
            "a4": gate_map.get("a4", GateResult("PENDING", "not run")).status,
            "d3": gate_map.get("d3", GateResult("PENDING", "not run")).status,
            "artifacts_dir": summary["artifacts_dir"],
        })

    bucket_order = {"APPROVED": 0, "PROBATION": 1, "REJECTED": 2}
    ranking_rows.sort(
        key=lambda row: (
            bucket_order.get(row["bucket"], UNKNOWN_BUCKET_SORT_ORDER),
            row["raw_p"] if row["raw_p"] != "" else INVALID_P_SORT_ORDER,
            -float(row["mean_pct"]),
        )
    )
    ranking_csv = output_dir / "ranking.csv"
    _write_csv(
        ranking_csv,
        ranking_rows,
        ["pair", "bucket", "mean_pct", "ci_lo", "ci_hi", "raw_p", "bonferroni_pass", "bonferroni_threshold", "a2", "a3", "a4", "d3", "artifacts_dir"],
    )

    campaign_summary = {
        "config": {
            "alpha": config.alpha,
            "pre_registered_family_size": config.pre_registered_family_size,
            "candidate_pairs": len(pairs),
            "output_dir": str(output_dir),
        },
        "family_csv": str(family_csv),
        "ranking_csv": str(ranking_csv),
        "pair_summaries": pair_summaries,
        "ranking": ranking_rows,
    }
    (output_dir / "campaign_summary.json").write_text(json.dumps(campaign_summary, indent=2), encoding="utf-8")

    header = f"{'pair':<12} {'bucket':<10} {'raw_p':>8} {'mean%':>9} {'A2':>10} {'A3':>10} {'A4':>10} {'D3':>10}"
    print(header)
    print("-" * len(header))
    for row in ranking_rows:
        raw_p = f"{float(row['raw_p']):.4f}" if row["raw_p"] != "" else "n/a"
        print(
            f"{row['pair']:<12} {row['bucket']:<10} {raw_p:>8} {float(row['mean_pct']):>+8.3f}% "
            f"{row['a2']:>10} {row['a3']:>10} {row['a4']:>10} {row['d3']:>10}"
        )
    print(f"\nArtifacts: {output_dir}")

    _maybe_auto_promote(args, config, config_path, ranking_csv)


def _maybe_auto_promote(
    args: argparse.Namespace,
    config: CampaignConfig,
    config_path: Path,
    ranking_csv: Path,
) -> None:
    """Run the selection->deploy promoter after the campaign, if enabled.

    Resolution order: --no-promote wins (off); else --promote-write / --promote
    force it on; else the config's promotion block decides. Skipped during
    --dry-run (the ranking is all-REJECTED placeholder data)."""
    promo = config.promotion
    enabled = promo.enabled
    write = promo.write
    if args.no_promote:
        enabled = False
    if args.promote:
        enabled, write = True, write
    if args.promote_write:
        enabled, write = True, True
    if not enabled:
        return
    if args.dry_run:
        print("\n[promote] skipped: campaign --dry-run produced placeholder ranking.")
        return

    target = promo.target
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = _REPO_ROOT / target_path

    print("\n" + "=" * 70)
    print(f"[promote] auto-promotion ({'WRITE' if write else 'dry-run'})")
    print("=" * 70)
    try:
        result = promote_pairs.promote(
            campaign_config=config_path,
            target=target_path,
            ranking=ranking_csv,
            write=write,
            keep_existing=promo.keep_existing,
            buying_power=promo.buying_power,
            max_leverage=promo.max_leverage,
            allow_empty=promo.allow_empty,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[promote] ERROR: {exc}")
        return
    # Print the summary but not the full YAML body (campaign output is already long).
    promote_pairs.print_promotion_result(result, write=write, show_yaml=False)


if __name__ == "__main__":
    main()
