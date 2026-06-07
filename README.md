# win-gate-model-trainer

Trains the per-fire **win-gate** logistic model (cross-validated) for the Polymarket strategy stack — a classifier that predicts `P(fire wins)` at fire time for **5-minute crypto up/down markets** (BTC/XRP).

## Why it exists
The strategy stack emits many candidate "fires," but not all of them are worth taking. In a binary market `P(win)` directly implies the correct side, so a single calibrated probability gives three actions: fire as-is when confident, fire the **opposite** side when confident the current side loses, or skip when uncertain. This repo produces the model (`win_gate_model.pkl`) that the live bot loads to gate fires.

## How it works
| File | Role |
|------|------|
| `win_gate_train.py` | **Entry point.** Loads gate-clean ("FREE") fires, builds the feature matrix, runs a walk-forward (70/30 chronological) logistic regression, and writes the model + thresholds to `win_gate_model.pkl`. |
| `flip_features.py` | Runtime feature extractor (per-window aggregates + cd-snapshot features). Vendored from the engine — it mirrors `vps_feature_extraction.py` exactly so training and live values match. **Edit upstream, not here.** |

The trainer's decision logic:
- `P(win) >= ~0.70` → fire as-is (current side wins)
- `P(win) <= ~0.30` → fire the opposite side
- in between → skip

**Statistical rigor** (so a model only ships if real): C-grid 3-fold cross-validation on the train split, walk-forward AUC on held-out test, Bonferroni-corrected t-tests across the 7 fire / 5 flip thresholds, and a 5,000-iteration permutation test. Deployment verdict requires `AUC >= 0.65`, best-threshold `t` over the Bonferroni critical value, and `WR >= 55%`.

## Requirements
- Python 3.8+
- `numpy`, `scikit-learn` (LogisticRegression, StandardScaler, cross-val, roc_auc)
- Read access to the private **`polymarket-data`** repo (training inputs — see Data)
- No wallet, key, or network access — this trains offline and handles no funds.

## Usage
```bash
# run from the data directory the trainer chdir's into
python3 win_gate_train.py
```
The script prints dataset stats, the walk-forward AUC, per-threshold fire/flip tables, the Bonferroni + permutation results, and a final DEPLOY / DO NOT DEPLOY verdict. On success it writes `win_gate_model.pkl` (model, scaler, feature list, AUC, and the chosen fire/flip thresholds).

## Data
This repo ships **code only** — all `.jsonl`/`.pkl` inputs are gitignored and come from the private **`polymarket-data`** repo. `win_gate_train.py` currently hard-codes the engine layout:
- `sys.path.insert(0, '/home/polybot/polymarket-bot')`
- `os.chdir('/home/polybot/polymarket-bot/data')`

and reads `market_history.jsonl` (per-market ticks) + `market_recap_history.jsonl` (fires) from that data dir, writing `win_gate_model.pkl` back into it. Point these paths at your local checkout of `polymarket-data` before running.

> Private research software. No warranty; trades/handles real funds at your own risk.
