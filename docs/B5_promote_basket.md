# B5 — Promote the validated basket (selection → deploy)

**Roadmap reference:** the mechanical executor of A1's deploy decision. The
research pipeline ends with a `ranking.csv` that labels every candidate pair
APPROVED / PROBATION / REJECTED. `tools/promote_pairs.py` turns the APPROVED
rows — and only those — into the deployed `config/pairs.yaml` that
`main_pairs.py` runs. It replaces the error-prone hand-edit that was the only
previous way to grow the basket.

---

## Pipeline position

```
B1 (correlation)        -> b1_*.csv
B2 (cointegration)      -> b2_*.csv  (screen_pass=1)
B3 (batch walk-forward) -> b3_*.csv  + returns.json
B4 (pre-register bar)   -> docs/B4_*.md  (family-size K locked)
A1 (corrected verdict)  -> campaign ranking.csv (APPROVED/PROBATION/REJECTED)
B5 (THIS TOOL)          -> config/pairs.yaml   (APPROVED pairs only)
main_pairs.py           -> trades the basket
```

The campaign runner (`tools/pairs_candidate_campaign.py`) produces the
`ranking.csv`; B5 consumes it.

## Why it exists

Hand-editing `config/pairs.yaml` is exactly how an un-validated or PROBATION
pair slips into the live basket. B5 makes promotion deterministic, auditable,
and conservative by default.

## The strict gate (re-verified, not trusted)

B5 does **not** trust the `bucket` label alone. A row is promotable only if it
is APPROVED **and** independently re-passes every condition that APPROVED is
supposed to encode:

- `bucket == APPROVED`
- `bonferroni_pass == True`
- bootstrap CI lower bound `ci_lo > 0`
- `a2 == a3 == a4 == d3 == PASS`

A row labelled APPROVED that fails any check is **refused with a loud warning**
— that mismatch means the ranking is stale or was edited by hand. An APPROVED
pair that is missing from the campaign config (so its validated parameters
can't be resolved) is skipped with a warning.

## Conservative behaviour

- **Dry-run by default.** Prints the basket diff (added / kept / dropped) and
  the YAML it *would* write, but writes nothing unless `--write` is passed.
- **Never flips paper → live.** The target's `paper` / `feed` / `risk` blocks
  are preserved verbatim; B5 only ever rewrites the `pairs:` list. Going live
  still needs both `paper: false` and `HQC_ENABLE_LIVE_TRADING=1`.
- **Refuses an empty basket** unless `--allow-empty` (so a ranking with zero
  survivors can't silently disable all pair trading).
- **Faithful record.** Each emitted pair carries provenance comments — raw p,
  bootstrap CI, Bonferroni threshold, gate statuses, thesis — so the deploy
  config stays the same honest audit trail the rest of the repo maintains.

## Parameter resolution

The validated parameter set for each pair is resolved as:

> per-pair override (campaign config) → campaign default (`pair_*`) → strategy default

so a pair validated at, e.g., `entry_z: 1.8` is deployed at `1.8`, not the
1.5 default. `delta`/`ve` are written only when they differ from the strategy
default, keeping the YAML clean.

## Gross-exposure guard

B5 sums the basket's gross notional (2 legs × `target_dollar_notional` per
pair) and compares it to `buying_power × max_leverage`:

- `--buying-power` defaults to `risk.initial_capital` from the target config.
- `--max-leverage` defaults to `1.0`.

It **warns** if the basket exceeds the limit; it does **not** auto-scale
notionals (that is the separate portfolio risk-engine deliverable). An account
running at 3x passes `--max-leverage 3` rather than fighting a 1x default.

## How to run

```bash
# Dry-run: show what WOULD be promoted (writes nothing)
python tools/promote_pairs.py \
    --campaign-config config/pairs_research.yaml \
    --target config/pairs.yaml

# Apply: write config/pairs.yaml with the approved basket
python tools/promote_pairs.py \
    --campaign-config config/pairs_research.yaml \
    --target config/pairs.yaml \
    --write \
    --buying-power 450000 --max-leverage 3

# Keep currently-deployed pairs that aren't in this ranking
python tools/promote_pairs.py \
    --campaign-config config/pairs_research.yaml \
    --target config/pairs.yaml --write --keep-existing
```

`--ranking` overrides the default `ranking.csv` location (otherwise derived
from the campaign config's `output_dir`).

## Flags

| flag | meaning |
|---|---|
| `--campaign-config` | Campaign YAML; source of validated params + ranking `output_dir`. (required) |
| `--ranking` | Explicit `ranking.csv` path. Default: `<output_dir>/ranking.csv`. |
| `--target` | Deploy config to write. Default: `config/pairs.yaml`. |
| `--write` | Actually write. Without it, dry-run. |
| `--keep-existing` | Union approved pairs with pairs already in `--target`. |
| `--buying-power` | Account buying power for the gross guard. Default: target `initial_capital`. |
| `--max-leverage` | Allowed gross leverage (default 1.0). |
| `--allow-empty` | Permit writing a zero-pair config. |

## Tests

`tests/test_promote_pairs.py` covers the strict gate, mislabeled-APPROVED
refusal, parameter-override resolution, YAML round-trip + meta preservation,
the gross-exposure guard (including the leverage path), and the dry-run /
write / refuse-empty / keep-existing behaviour of `main()`. It runs on
synthetic artifacts — no market data, no statsmodels.
