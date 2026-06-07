# win-gate-model-trainer

Trains the win-gate logistic model that predicts `P(fire wins)` at fire time for the Polymarket 5-minute crypto up/down markets (BTC/XRP).

In a binary market `P(win)` implies the side, so one calibrated probability drives three actions: fire as-is when high, fire the opposite side when low, skip in between. The live bot loads the resulting `win_gate_model.pkl` to gate fires.

## Contents

| File | What it does |
|------|--------------|
| `win_gate_train.py` | Entry point. Loads gate-clean ("FREE") fires, builds the feature matrix, runs a 70/30 chronological walk-forward logistic regression, writes `win_gate_model.pkl`. |
| `flip_features.py` | Runtime feature extractor: per-window aggregates plus cd-snapshot features. Mirrors `vps_feature_extraction.py` so training and live values match. Edit upstream, not here. |

## Thresholds

- `P(win) >= 0.70`: fire as-is (current side wins)
- `P(win) <= 0.30`: fire the opposite side
- otherwise: skip

## Validation

C-grid 3-fold cross-validation on the train split, walk-forward AUC on the held-out test split, Bonferroni-corrected t-tests across the 7 fire and 5 flip thresholds, and a 5,000-iteration permutation test. Deploy verdict requires `AUC >= 0.65`, the best-threshold `t` above the Bonferroni critical value, and `WR >= 55%`.

## Usage

```bash
python3 win_gate_train.py
```

Prints dataset stats, walk-forward AUC, per-threshold fire/flip tables, Bonferroni and permutation results, and a DEPLOY / DO NOT DEPLOY verdict. On success writes `win_gate_model.pkl` (model, scaler, feature list, AUC, chosen fire/flip thresholds).

## Requirements

- Python 3.8+, `numpy`, `scikit-learn`
- No keys or network. Trains offline, handles no funds.

## Data

Code only. The `.jsonl`/`.pkl` inputs are gitignored and come from the private `polymarket-data` repo. `win_gate_train.py` reads `market_history.jsonl` (per-market ticks) and `market_recap_history.jsonl` (fires) from `$DATA_DIR`, and writes `win_gate_model.pkl` back there. The script currently hard-codes the engine layout via `sys.path.insert` and `os.chdir`; point both at your local `polymarket-data` checkout before running.
