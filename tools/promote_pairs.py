"""
promote_pairs — the mechanical bridge from the research campaign to the
deployed basket.

Roadmap reference: this is the executor of A1's deploy decision. The campaign
runner (``tools/pairs_candidate_campaign.py``) produces a ``ranking.csv`` that
classifies every candidate pair as APPROVED / PROBATION / REJECTED. This tool
takes that ranking, keeps ONLY the strictly-approved pairs, resolves the exact
parameter set each one was validated at (from the campaign config), and writes
a multi-pair ``config/pairs.yaml`` ready for ``main_pairs.py`` to deploy.

It exists because promoting validated pairs into the live basket was a manual,
error-prone hand-edit. Hand-editing the deploy config is exactly how an
un-validated or PROBATION pair sneaks into the basket. This tool makes the
promotion deterministic, auditable, and conservative-by-default:

  - **Dry-run by default.** It prints the diff (added / kept / dropped) and the
    YAML it WOULD write, but writes nothing unless ``--write`` is passed.
  - **Strict gate, re-verified.** It does not trust the ``bucket`` label
    blindly. A row is promotable only if it is APPROVED *and* independently
    re-passes every strict condition (Bonferroni pass, bootstrap CI lower
    bound > 0, and A2/A3/A4/D3 gates all PASS). A row labelled APPROVED that
    fails any of these is refused with a loud warning — that mismatch means
    the ranking was edited or is stale.
  - **Faithful record.** The emitted YAML carries per-pair provenance comments
    (raw p, Bonferroni threshold, gate statuses, thesis) so the deploy config
    stays the same kind of honest audit trail the repo already maintains.
  - **Never flips paper→live.** The target's ``paper`` / ``feed`` / ``risk``
    blocks are preserved as-is; this tool only ever rewrites the ``pairs``
    list. Going live still requires the two existing gates in main_pairs.py.

Dependency-light on purpose: only the stdlib + PyYAML. It does NOT import the
campaign module (which pulls statsmodels), so it runs anywhere.

Usage:
    # Dry-run: show what WOULD be promoted (default; writes nothing)
    python tools/promote_pairs.py \\
        --campaign-config config/pairs_research.yaml \\
        --target config/pairs.yaml

    # Actually write config/pairs.yaml with the approved basket
    python tools/promote_pairs.py \\
        --campaign-config config/pairs_research.yaml \\
        --target config/pairs.yaml \\
        --write

    # Point at an explicit ranking.csv (otherwise derived from the
    # campaign config's output_dir)
    python tools/promote_pairs.py \\
        --campaign-config config/pairs_research.yaml \\
        --ranking state/pairs_campaign/ranking.csv \\
        --target config/pairs.yaml --write
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Strategy defaults — must match strategies/mean_reversion/kalman_spread.py and
# main_pairs.PairConfig. Used both as fallbacks when the campaign config omits a
# knob and to decide whether delta/ve are worth emitting (we keep the YAML clean
# by only writing delta/ve when they differ from the strategy default).
_DEFAULT_PARAMS = {
    "entry_z": 1.5,
    "exit_z": 0.4,
    "delta": 1e-4,
    "ve": 1e-3,
    "cooldown_seconds": 120.0,
    "nominal_stop_pct": 0.02,
    "target_dollar_notional": 10000.0,
}

# The gate columns that must read exactly "PASS" for a strict promotion.
_REQUIRED_GATES = ("a2", "a3", "a4", "d3")


@dataclass
class CandidateRow:
    """One row parsed from the campaign ranking.csv."""

    pair: str
    bucket: str
    mean_pct: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    raw_p: Optional[float]
    bonferroni_pass: bool
    bonferroni_threshold: str
    gates: Dict[str, str] = field(default_factory=dict)

    def strict_failures(self) -> List[str]:
        """Reasons this row is NOT strictly promotable. Empty list == promotable.

        Defense-in-depth: we re-verify every condition that the campaign's
        APPROVED bucket is supposed to encode, rather than trusting the label.
        """
        reasons: List[str] = []
        if self.bucket != "APPROVED":
            reasons.append(f"bucket={self.bucket or 'EMPTY'}")
        if not self.bonferroni_pass:
            reasons.append("bonferroni_pass=False")
        if self.ci_lo is None or self.ci_lo <= 0:
            reasons.append(f"ci_lo={self.ci_lo}<=0")
        for gate in _REQUIRED_GATES:
            status = self.gates.get(gate, "")
            if status != "PASS":
                reasons.append(f"{gate}={status or 'MISSING'}")
        return reasons


@dataclass
class DeployPair:
    """An approved pair resolved to its validated parameter set + provenance."""

    y: str
    x: str
    params: Dict[str, float]
    thesis: str
    provenance: Dict[str, str]

    @property
    def label(self) -> str:
        return f"{self.y}/{self.x}"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _to_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def load_ranking(path: Path) -> List[CandidateRow]:
    """Parse the campaign ranking.csv into CandidateRow objects."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: List[CandidateRow] = []
    for r in rows:
        pair = (r.get("pair") or "").strip().upper()
        if not pair:
            continue
        out.append(
            CandidateRow(
                pair=pair,
                bucket=(r.get("bucket") or "").strip().upper(),
                mean_pct=_to_float(r.get("mean_pct")),
                ci_lo=_to_float(r.get("ci_lo")),
                ci_hi=_to_float(r.get("ci_hi")),
                raw_p=_to_float(r.get("raw_p")),
                bonferroni_pass=_to_bool(r.get("bonferroni_pass")),
                bonferroni_threshold=(r.get("bonferroni_threshold") or "").strip(),
                gates={g: (r.get(g) or "").strip().upper() for g in _REQUIRED_GATES},
            )
        )
    return out


def load_campaign_params(
    path: Path,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]], Optional[Path]]:
    """Parse the campaign YAML for: global default params, per-pair overrides +
    thesis (keyed by "Y/X" label), and the campaign output_dir (to locate
    ranking.csv when --ranking is not given).

    Parsed with plain yaml.safe_load so this tool never imports the
    statsmodels-heavy campaign module.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    campaign_raw = raw.get("campaign") or {}

    defaults: Dict[str, float] = {}
    for key, fallback in _DEFAULT_PARAMS.items():
        defaults[key] = float(campaign_raw.get(f"pair_{key}", fallback))

    output_dir: Optional[Path] = None
    if campaign_raw.get("output_dir"):
        od = Path(str(campaign_raw["output_dir"]))
        output_dir = od if od.is_absolute() else (_REPO_ROOT / od)

    per_pair: Dict[str, Dict[str, object]] = {}
    for p in raw.get("pairs") or []:
        y = str(p["y"]).upper()
        x = str(p["x"]).upper()
        label = f"{y}/{x}"
        entry: Dict[str, object] = {"thesis": str(p.get("thesis", ""))}
        for key in _DEFAULT_PARAMS:
            if p.get(key) is not None:
                entry[key] = float(p[key])
        per_pair[label] = entry
    return defaults, per_pair, output_dir


def resolve_params(
    label: str,
    defaults: Dict[str, float],
    per_pair: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    """Per-pair override wins over campaign default wins over strategy default."""
    overrides = per_pair.get(label, {})
    return {
        key: float(overrides.get(key, defaults.get(key, _DEFAULT_PARAMS[key])))
        for key in _DEFAULT_PARAMS
    }


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_approved(
    ranking: List[CandidateRow],
    campaign_defaults: Dict[str, float],
    campaign_pairs: Dict[str, Dict[str, object]],
) -> Tuple[List[DeployPair], List[str]]:
    """Return (promotable pairs, warnings).

    A warning is emitted for any row labelled APPROVED that fails the
    independent strict re-check (stale/edited ranking), and for any approved
    pair missing from the campaign config (can't resolve its params).
    """
    approved: List[DeployPair] = []
    warnings: List[str] = []
    for row in ranking:
        failures = row.strict_failures()
        if row.bucket == "APPROVED" and failures:
            warnings.append(
                f"REFUSED {row.pair}: labelled APPROVED but fails strict re-check "
                f"({'; '.join(failures)}). Ranking may be stale or edited."
            )
            continue
        if failures:
            continue  # not approved; silently skip PROBATION/REJECTED
        if "/" not in row.pair:
            warnings.append(f"SKIP {row.pair}: malformed pair label.")
            continue
        if row.pair not in campaign_pairs:
            warnings.append(
                f"SKIP {row.pair}: APPROVED but absent from campaign config; "
                f"cannot resolve validated parameters."
            )
            continue
        y, x = row.pair.split("/", 1)
        approved.append(
            DeployPair(
                y=y,
                x=x,
                params=resolve_params(row.pair, campaign_defaults, campaign_pairs),
                thesis=str(campaign_pairs.get(row.pair, {}).get("thesis", "")),
                provenance={
                    "raw_p": "" if row.raw_p is None else f"{row.raw_p:.6f}",
                    "ci_lo": "" if row.ci_lo is None else f"{row.ci_lo:.4f}",
                    "ci_hi": "" if row.ci_hi is None else f"{row.ci_hi:.4f}",
                    "mean_pct": "" if row.mean_pct is None else f"{row.mean_pct:.4f}",
                    "bonferroni_threshold": row.bonferroni_threshold,
                    "gates": " ".join(f"{g}={row.gates.get(g, '')}" for g in _REQUIRED_GATES),
                },
            )
        )
    return approved, warnings


# --------------------------------------------------------------------------- #
# Existing-target inspection + YAML rendering
# --------------------------------------------------------------------------- #
def load_existing_target(path: Path) -> dict:
    """Read the current deploy config (if any) so we preserve paper/feed/risk
    and can diff the pairs list. Returns {} if the file does not exist."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _existing_labels(existing: dict) -> List[str]:
    labels: List[str] = []
    for p in existing.get("pairs") or []:
        try:
            labels.append(f"{str(p['y']).upper()}/{str(p['x']).upper()}")
        except (KeyError, TypeError):
            continue
    return labels


def _fmt_num(value: float) -> str:
    """Render a number without trailing-zero noise as a YAML-parseable decimal.

    Avoids scientific notation (e.g. ``5e-05``), which PyYAML's float resolver
    does not parse back to a float — small knobs like delta=1e-4 are written as
    ``0.0001`` so the emitted config round-trips cleanly."""
    if value == int(value):
        return str(int(value))
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _render_pair_block(pair: DeployPair) -> List[str]:
    prov = pair.provenance
    lines = [
        f"  # {pair.label} — promoted from campaign ranking (APPROVED).",
    ]
    if pair.thesis:
        lines.append(f"  #   thesis: {pair.thesis}")
    lines.append(
        f"  #   raw_p={prov.get('raw_p') or 'n/a'}  "
        f"ci=[{prov.get('ci_lo') or 'n/a'}, {prov.get('ci_hi') or 'n/a'}]%  "
        f"bonferroni_threshold={prov.get('bonferroni_threshold') or 'n/a'}"
    )
    lines.append(f"  #   gates: {prov.get('gates') or 'n/a'}")
    lines.append(f"  - y: {pair.y}")
    lines.append(f"    x: {pair.x}")
    lines.append(f"    entry_z: {_fmt_num(pair.params['entry_z'])}")
    lines.append(f"    exit_z: {_fmt_num(pair.params['exit_z'])}")
    # delta/ve only when they differ from the strategy default (keeps YAML clean,
    # matching the existing hand-curated config which omits them at default).
    if pair.params["delta"] != _DEFAULT_PARAMS["delta"]:
        lines.append(f"    delta: {_fmt_num(pair.params['delta'])}")
    if pair.params["ve"] != _DEFAULT_PARAMS["ve"]:
        lines.append(f"    ve: {_fmt_num(pair.params['ve'])}")
    lines.append(f"    cooldown_seconds: {_fmt_num(pair.params['cooldown_seconds'])}")
    lines.append(f"    nominal_stop_pct: {_fmt_num(pair.params['nominal_stop_pct'])}")
    lines.append(f"    target_dollar_notional: {_fmt_num(pair.params['target_dollar_notional'])}")
    return lines


def _render_risk_block(existing: dict) -> List[str]:
    risk = existing.get("risk") or {}
    return [
        "risk:",
        f"  initial_capital: {_fmt_num(float(risk.get('initial_capital', 100000.0)))}",
        f"  daily_loss_pct_kill: {_fmt_num(float(risk.get('daily_loss_pct_kill', 0.02)))}",
        f"  beta_drift_pct_kill: {_fmt_num(float(risk.get('beta_drift_pct_kill', 0.30)))}",
        f"  beta_drift_window_min: {int(risk.get('beta_drift_window_min', 60))}",
    ]


def render_yaml(
    pairs: List[DeployPair],
    existing: dict,
    source_campaign: str,
    ranking_path: str,
) -> str:
    paper = existing.get("paper", True)
    feed = existing.get("feed", "iex")
    today = dt.date.today().isoformat()

    lines: List[str] = [
        "# Kalman pairs deployment config.",
        "#",
        "# GENERATED by tools/promote_pairs.py — the selection->deploy bridge.",
        "# Only campaign-APPROVED pairs (strict gate: Bonferroni pass, bootstrap",
        "# CI lower bound > 0, and A2/A3/A4/D3 all PASS) are promoted here.",
        "# Re-running the promoter after a new campaign overwrites this pairs list;",
        "# edit candidates in the campaign config, not here.",
        "#",
        f"#   promoted_on : {today}",
        f"#   source      : {source_campaign}",
        f"#   ranking     : {ranking_path}",
        f"#   basket_size : {len(pairs)} pair(s)",
        "#",
        "# To go LIVE you must set BOTH paper: false (here) AND",
        "# HQC_ENABLE_LIVE_TRADING=1 (env). The promoter never flips paper->live.",
        "",
        f"paper: {str(bool(paper)).lower()}",
        f"feed: {feed}",
        "",
        "pairs:",
    ]
    if not pairs:
        lines.append("  []  # no APPROVED pairs in the ranking")
    else:
        for i, pair in enumerate(pairs):
            if i:
                lines.append("")
            lines.extend(_render_pair_block(pair))
    lines.append("")
    lines.extend(_render_risk_block(existing))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gross-exposure sanity check (a promotion guard, not the full risk engine)
# --------------------------------------------------------------------------- #
def gross_exposure_report(
    pairs: List[DeployPair], buying_power: float, max_leverage: float
) -> Tuple[float, float, Optional[str]]:
    """Estimate total gross notional of the promoted basket (2 legs/pair) and
    compare to the allowed exposure (buying_power * max_leverage). Returns
    (gross, limit, warning_or_None).

    This is a cheap guard so promotion never silently produces a basket whose
    gross exposure exceeds the account's margin. It does NOT auto-scale
    notionals — that is the (separate) portfolio risk-engine deliverable.
    Leverage is expressed explicitly so an account comfortable at, e.g., 3x
    can pass ``--max-leverage 3`` rather than fighting a 1x default."""
    limit = max(0.0, float(buying_power) * float(max_leverage))
    gross = sum(2.0 * float(p.params["target_dollar_notional"]) for p in pairs)
    warning = None
    if limit > 0 and gross > limit:
        warning = (
            f"gross notional ${gross:,.0f} across {len(pairs)} pairs exceeds the "
            f"allowed ${limit:,.0f} (buying_power ${float(buying_power):,.0f} x "
            f"{float(max_leverage):g}x). Lower target_dollar_notional in the "
            f"campaign config, reduce the basket, or raise --max-leverage."
        )
    return gross, limit, warning


# --------------------------------------------------------------------------- #
# Programmatic core (used by the CLI and by the campaign auto-promotion hook)
# --------------------------------------------------------------------------- #
@dataclass
class PromotionResult:
    approved: List[DeployPair]
    final_pairs: List[DeployPair]
    warnings: List[str]
    existing_labels: List[str]
    new_labels: List[str]
    gross: float
    gross_limit: float
    gross_warning: Optional[str]
    rendered: str
    written: bool
    refused_empty: bool
    ranking_rows: int
    campaign_path: Path
    ranking_path: Path
    target: Path
    buying_power: float
    max_leverage: float


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO_ROOT)) if path.is_relative_to(_REPO_ROOT) else str(path)


def promote(
    *,
    campaign_config: Path,
    target: Path,
    ranking: Optional[Path] = None,
    write: bool = False,
    keep_existing: bool = False,
    buying_power: Optional[float] = None,
    max_leverage: float = 1.0,
    allow_empty: bool = False,
) -> PromotionResult:
    """Resolve, gate, render (and optionally write) the deploy basket.

    Pure of stdout — returns a PromotionResult the caller renders. Raises
    FileNotFoundError for a missing campaign config / ranking, ValueError if
    the ranking location can't be derived. ``refused_empty`` is True (and
    nothing is written) when the basket is empty and ``allow_empty`` is False.
    """
    campaign_path = Path(campaign_config).resolve()
    if not campaign_path.exists():
        raise FileNotFoundError(f"campaign config not found: {campaign_path}")

    defaults, campaign_pairs, output_dir = load_campaign_params(campaign_path)

    ranking_path = ranking
    if ranking_path is None:
        if output_dir is None:
            raise ValueError("ranking not given and campaign config has no output_dir.")
        ranking_path = output_dir / "ranking.csv"
    ranking_path = Path(ranking_path).resolve()
    if not ranking_path.exists():
        raise FileNotFoundError(f"ranking not found: {ranking_path}")

    ranking_rows = load_ranking(ranking_path)
    approved, warnings = select_approved(ranking_rows, defaults, campaign_pairs)

    target = Path(target)
    existing = load_existing_target(target)
    existing_labels = _existing_labels(existing)

    # Approved order preserved; kept-existing non-approved pairs appended as-is.
    final_pairs = list(approved)
    if keep_existing:
        approved_labels = {dp.label for dp in approved}
        for raw_pair in existing.get("pairs") or []:
            try:
                y = str(raw_pair["y"]).upper()
                x = str(raw_pair["x"]).upper()
            except (KeyError, TypeError):
                continue
            label = f"{y}/{x}"
            if label in approved_labels:
                continue
            params = {
                key: float(raw_pair.get(key, defaults.get(key, _DEFAULT_PARAMS[key])))
                for key in _DEFAULT_PARAMS
            }
            final_pairs.append(
                DeployPair(
                    y=y, x=x, params=params, thesis="",
                    provenance={"raw_p": "", "ci_lo": "", "ci_hi": "", "mean_pct": "",
                                "bonferroni_threshold": "", "gates": "retained (--keep-existing)"},
                )
            )

    resolved_bp = (
        buying_power if buying_power is not None
        else float((existing.get("risk") or {}).get("initial_capital", 100000.0))
    )
    gross, gross_limit, gross_warning = gross_exposure_report(final_pairs, resolved_bp, max_leverage)

    rendered = render_yaml(
        final_pairs, existing,
        source_campaign=_rel(campaign_path),
        ranking_path=_rel(ranking_path),
    )

    refused_empty = not final_pairs and not allow_empty
    written = False
    if write and not refused_empty:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        written = True

    return PromotionResult(
        approved=approved,
        final_pairs=final_pairs,
        warnings=warnings,
        existing_labels=existing_labels,
        new_labels=[dp.label for dp in final_pairs],
        gross=gross,
        gross_limit=gross_limit,
        gross_warning=gross_warning,
        rendered=rendered,
        written=written,
        refused_empty=refused_empty,
        ranking_rows=len(ranking_rows),
        campaign_path=campaign_path,
        ranking_path=ranking_path,
        target=target,
        buying_power=resolved_bp,
        max_leverage=max_leverage,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_diff(new_labels: List[str], old_labels: List[str]) -> None:
    new_set, old_set = set(new_labels), set(old_labels)
    added = [l for l in new_labels if l not in old_set]
    kept = [l for l in new_labels if l in old_set]
    dropped = [l for l in old_labels if l not in new_set]
    print("Basket diff (promoted vs. current target):")
    print(f"  + added  ({len(added)}): {', '.join(added) or '-'}")
    print(f"  = kept   ({len(kept)}): {', '.join(kept) or '-'}")
    print(f"  - dropped({len(dropped)}): {', '.join(dropped) or '-'}")
    if dropped:
        print(
            "  NOTE: dropped pairs are currently deployed but are NOT APPROVED in "
            "this ranking. Use --keep-existing to retain them."
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-config", required=True, type=Path,
                   help="Campaign YAML (e.g. config/pairs_research.yaml) — source of "
                        "validated per-pair params + the ranking output_dir.")
    p.add_argument("--ranking", type=Path, default=None,
                   help="ranking.csv path. Defaults to <campaign output_dir>/ranking.csv.")
    p.add_argument("--target", type=Path, default=_REPO_ROOT / "config" / "pairs.yaml",
                   help="Deploy config to write (default: config/pairs.yaml).")
    p.add_argument("--write", action="store_true",
                   help="Actually write --target. Without this, dry-run (print only).")
    p.add_argument("--keep-existing", action="store_true",
                   help="Union the approved basket with pairs already in --target "
                        "(don't drop a currently-deployed pair that isn't in this ranking).")
    p.add_argument("--buying-power", type=float, default=None,
                   help="Account buying power for the gross-exposure guard. "
                        "Defaults to risk.initial_capital from the target config.")
    p.add_argument("--max-leverage", type=float, default=1.0,
                   help="Allowed gross leverage on buying power (default 1.0). "
                        "An account comfortable at 2-3x should pass e.g. --max-leverage 3.")
    p.add_argument("--allow-empty", action="store_true",
                   help="Permit writing a config with zero pairs (default: refuse).")
    args = p.parse_args(argv)

    try:
        result = promote(
            campaign_config=args.campaign_config,
            target=args.target,
            ranking=args.ranking,
            write=args.write,
            keep_existing=args.keep_existing,
            buying_power=args.buying_power,
            max_leverage=args.max_leverage,
            allow_empty=args.allow_empty,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, FileNotFoundError) and "ranking" in str(exc):
            print(
                "       Run tools/pairs_candidate_campaign.py first "
                "(it needs market data in data/alpaca/).",
                file=sys.stderr,
            )
        return 2

    print_promotion_result(result, write=args.write)
    if result.refused_empty:
        print(
            "REFUSING to write a zero-pair basket. No pairs passed the strict gate.\n"
            "  Pass --allow-empty to override (this disables all pair trading).",
            file=sys.stderr,
        )
        return 1
    return 0


def print_promotion_result(result: PromotionResult, *, write: bool, show_yaml: bool = True) -> None:
    """Render a PromotionResult to stdout (shared by the CLI and campaign hook)."""
    print(f"Campaign : {result.campaign_path}")
    print(f"Ranking  : {result.ranking_path}  ({result.ranking_rows} rows)")
    print(f"Target   : {result.target}")
    print(f"Approved : {len(result.approved)}  |  final basket: {len(result.final_pairs)}")
    print()
    for w in result.warnings:
        print(f"  WARN: {w}")
    if result.warnings:
        print()

    _print_diff(result.new_labels, result.existing_labels)
    print()

    print(
        f"Gross notional (2 legs/pair): ${result.gross:,.0f}  vs allowed "
        f"${result.gross_limit:,.0f} (buying_power ${result.buying_power:,.0f} "
        f"x {result.max_leverage:g}x)"
    )
    if result.gross_warning:
        print(f"  WARN: {result.gross_warning}")
    print()

    if show_yaml:
        print("=" * 70)
        print(f"{'WROTE' if result.written else ('WOULD WRITE' if not write else 'NOT WRITTEN')} {result.target}:")
        print("=" * 70)
        print(result.rendered)

    if result.written:
        print(f"Wrote {result.target} ({len(result.final_pairs)} pair(s)).")
    elif not result.refused_empty:
        print("Dry-run: nothing written. Re-run with --write to apply.")


if __name__ == "__main__":
    raise SystemExit(main())
