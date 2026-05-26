# A4 — Regime conditioning

**Roadmap reference:** Workstream A, task A4. Final A-track gate. Splits the validated walk-forward by market regime to answer: does JPM/BAC's marginal A1 edge concentrate in a particular slice (e.g. trending-down low-vol months)? Bank pairs are rate-regime sensitive, so concentration is the expected pattern.

**Standing-rule recap:** Significance bar is bootstrap 95% CI lower bound > 0 AND family-corrected p < 0.05. The family in A4 is the number of regimes with enough samples (default ≥4 windows). Bonferroni threshold = `α / family_size`.

---

## What A4 measures

For each walk-forward window in an existing `walkforward_pairs.py` or `walkforward_basket.py` result:

1. Take the window's `test_start` → `test_end` date range.
2. Look up the dominant SPY regime over those days from `RegimeTagger`.
3. Bucket the window's OOS `total_return_pct` by regime.

Per bucket: bootstrap 95% CI on the mean, Newey-West HAC p-value, EDGE+ verdict under Bonferroni at `α / #eligible_regimes`.

The deployment rule emitter then collapses the table into a one-line trading directive: "trade only when SPY regime is X" if exactly one regime concentrates the edge; "trade when one of {X, Y}" if 2+ concentrate; "no concentration rule" if all regimes pass; "no rule emitted" if none pass.

---

## Regime axes

Deliberately coarse, defensible, computable without any extra data fetch:

| Axis | How it's computed |
|---|---|
| **Trend** | Daily SPY close vs 20-day EMA. `up` if close > EMA, `down` otherwise. |
| **Vol bucket** | 20-day rolling std of SPY daily log returns, bucketed into terciles across the dataset. Proxy for VIX. Labels: `low`, `mid`, `high`. |

6 combined regimes: `up_low`, `up_mid`, `up_high`, `down_low`, `down_mid`, `down_high`.

If the user has a VIX CSV they want to use instead of realized vol, that's an A4-v2 enhancement; the current implementation uses SPY-realized-vol only because it requires no additional data.

---

## How to run

Requires SPY OHLCV at the same date range as the walk-forward (typically `data/alpaca/spy_730d_1m.csv`). If you don't have it yet, fetch:

```bash
python fetch_alpaca.py --symbols SPY --days 730 --out data/alpaca
```

Then for each pair you've walk-forwarded:

```bash
python tools/a4_regime_split.py \
    --walkforward result_wf_pair_jpm_bac_2y.json \
    --spy-csv data/alpaca/spy_730d_1m.csv \
    --min-n 4 \
    --output a4_jpm_bac_regime.csv
```

The `--walkforward` argument accepts the JSON output of either `walkforward_pairs.py` (single pair) or `walkforward_basket.py` (multiple symbols); per-pair splits are emitted in either case.

---

## Interpreting the output

Sample (synthetic):

```
JPM/BAC  (windows=24)
    regime       n     mean%                   95%CI     raw_p    Bonf<=  edge+
    ---------------------------------------------------------------------------
    down_high    2   +0.341%      [+0.320%, +0.363%]       n/a  0.0125  (insufficient n)
    down_low     8   +0.288%      [+0.094%, +0.467%]    0.0021  0.0125  EDGE+
    down_mid     3   +0.148%      [-0.247%, +0.517%]    0.1154  0.0125
    up_high      5   +0.010%      [-0.233%, +0.318%]    0.4687  0.0125
    up_mid       6   -0.088%      [-0.279%, +0.089%]    0.7977  0.0125

DEPLOYMENT RULES
  JPM/BAC: edge concentrates in regime 'down_low' (n=8, mean=+0.288%, p=0.002138).
          DEPLOYMENT RULE: trade JPM/BAC only when current SPY regime is 'down_low'.
          Live: PairsRiskMonitor should suspend the pair when the live regime differs.
```

The deployment rule means: don't trade the pair on every tick — only when the live SPY regime matches. This is a runtime-side change (not in scope for A4 itself); the rule is the spec.

---

## Verdict matrix

| What A4 produces | What it means |
|---|---|
| No regime is EDGE+ | A1's marginal pass doesn't hold up under regime split. **DEMOTE** the pair. |
| Exactly one regime EDGE+ | Edge is conditional. **Deploy with regime gate** in PairsRiskMonitor. Sizing should reflect the smaller subset of trading time. |
| 2-3 regimes EDGE+ | Edge holds in part of the regime space. Deploy with regime gate restricted to those. |
| All regimes EDGE+ | Edge is unconditional. **No gate needed**; continue as-is. (Rare for a marginal-A1 pair.) |

---

## What A4 does NOT do

- Does not change `main_pairs.py` to enforce the deployment rule. That's a follow-up implementation task. A4 produces the SPEC; the runtime change ships separately, after A4's verdict is in.
- Does not retroactively re-run walk-forward. It splits existing results — cheap re-analysis only.
- Does not test alternative regime definitions (HMM, factor rotation, day-of-week). The roadmap asked for SPY EMA20 + vol bucket; that's what's implemented.

---

## Reusable library

`tools/regime_tagger.RegimeTagger` is importable. The same tagger backs the analyzer; it can also be wired into the live runtime later to enforce the deployment rule:

```python
from datetime import date
from pathlib import Path
from tools.regime_tagger import RegimeTagger

tagger = RegimeTagger.from_csv(Path("data/alpaca/spy_730d_1m.csv"), ema_span=20, vol_window=20)
todays_regime = tagger.label_for_date(date.today())
if todays_regime != "down_low":
    # suspend JPM/BAC trading for the session
    ...
```
