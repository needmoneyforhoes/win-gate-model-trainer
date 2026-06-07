#!/usr/bin/env python3
"""
WIN-GATE CLASSIFIER + SIDE-FLIP DETECTION

Trains a classifier that predicts P(fire wins) at fire time.
Insight: in binary markets, P(win) directly implies the right side.
  - P(win) >= 0.70 → fire as-is (high confidence current side wins)
  - P(win) <= 0.30 → fire OPPOSITE side (high confidence current side loses)
  - 0.30 < P(win) < 0.70 → skip (uncertain)

Trains on ALL gate-clean (FREE) fires from market_recap_history.jsonl
matched to ticks in market_history.jsonl.

RIGOR:
  - Walk-forward chronological split (70/30)
  - Bonferroni correction for multi-threshold testing
  - Permutation test (5000 iters)
  - Only deploy if AUC >= 0.65 AND best threshold passes Bonferroni

USAGE on VPS:
  python3 win_gate_train.py
"""
import json, math, sys, os, pickle, random
from collections import defaultdict

sys.path.insert(0, '/home/polybot/polymarket-bot')
os.chdir('/home/polybot/polymarket-bot/data')
random.seed(42)

print("Loading market_history.jsonl (ticks)...")
ticks_by_slug = {}
with open('market_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get('slug') and r.get('ticks'):
            ticks_by_slug[r['slug']] = r['ticks']
print(f"  Markets: {len(ticks_by_slug)}")

print("Loading market_recap_history.jsonl (fires)...")
free_fires = []
with open('market_recap_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        slug = r.get('slug')
        ts = r.get('ts')
        if slug not in ticks_by_slug: continue
        for fire in r.get('fires', []):
            cd = fire.get('cd', 0)
            if cd is None or cd < 15: continue
            # FREE = gate-clean = passed all gates
            if (fire.get('pre_gate_held') or fire.get('opp_gate_held')
                or fire.get('dedup_excluded') or fire.get('pricecap_excluded')
                or fire.get('model_vetoed') or fire.get('flip_gate_blocked')
                or fire.get('bn_vetoed')):
                continue
            free_fires.append({
                'slug': slug, 'ts': ts,
                'strategy': fire.get('strategy'),
                'side': fire.get('side'),
                'cd': cd,
                'entry': fire.get('entry_price'),
                'hypo': fire.get('hypo_pnl', 0),
                'won': fire.get('hypo_pnl', 0) > 0,
            })

print(f"  Gate-clean fires: {len(free_fires)}")
print(f"  Win rate:         {sum(1 for f in free_fires if f['won'])/len(free_fires)*100:.1f}%")

# Compute features
print("\nComputing features...")
import flip_features
import datetime

data = []
errors = 0
for f in free_fires:
    ticks = ticks_by_slug[f['slug']]
    visible = [t for t in ticks if t[0] >= f['cd']]
    if len(visible) < 10:
        errors += 1; continue
    try:
        feats = flip_features.build_features(
            visible,
            current_time=datetime.datetime.fromtimestamp(f['ts']) if f['ts'] else None
        )
        feats['fire_cd']      = float(f['cd'])
        feats['fire_entry']   = float(f['entry']) if f['entry'] else 0.0
        feats['fire_side_dn'] = 1.0 if f['side'] == 'DN' else 0.0
    except Exception:
        errors += 1; continue
    
    data.append({
        'features': feats,
        'won': f['won'],
        'hypo': f['hypo'],
        'ts': f['ts'],
        'strategy': f['strategy'],
        'side': f['side'],
    })

print(f"  Computed: {len(data)}  errors: {errors}")

# Build matrices
all_keys = set(data[0]['features'].keys())
for d in data[:50]:
    all_keys = all_keys & set(d['features'].keys())
feat_names = sorted(all_keys)

import numpy as np
X = []; y = []; hypos = []; tss = []; sides = []; strats = []
for d in data:
    row = []
    for fname in feat_names:
        v = d['features'].get(fname)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        row.append(float(v))
    X.append(row); y.append(1 if d['won'] else 0)
    hypos.append(d['hypo']); tss.append(d['ts'])
    sides.append(d['side']); strats.append(d['strategy'])

X = np.array(X); y = np.array(y); hypos = np.array(hypos)
tss = np.array(tss); sides = np.array(sides); strats = np.array(strats)

print(f"\nDataset: {X.shape}  WR={y.mean()*100:.1f}%  features={len(feat_names)}")

# Walk-forward chronological split
order = np.argsort(tss)
X = X[order]; y = y[order]; hypos = hypos[order]; tss = tss[order]
sides = sides[order]; strats = strats[order]

split = int(len(X) * 0.70)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
h_te = hypos[split:]
sides_te = sides[split:]

print(f"Train: {len(X_tr)} fires  WR={y_tr.mean()*100:.1f}%")
print(f"Test:  {len(X_te)} fires  WR={y_te.mean()*100:.1f}%")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Try multiple regularization strengths, pick best on TRAIN cross-val
from sklearn.model_selection import cross_val_score
best_C = 0.5; best_cv = -1
for C in [0.1, 0.5, 1.0, 2.0]:
    m = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
    cv = cross_val_score(m, X_tr_s, y_tr, cv=3, scoring='roc_auc').mean()
    if cv > best_cv: best_cv = cv; best_C = C
print(f"  Best C: {best_C} (cv-AUC: {best_cv:.3f})")

model = LogisticRegression(C=best_C, max_iter=2000, class_weight='balanced')
model.fit(X_tr_s, y_tr)
proba = model.predict_proba(X_te_s)[:, 1]
auc = roc_auc_score(y_te, proba)

print(f"\n{'='*55}")
print(f"WALK-FORWARD AUC (test): {auc:.4f}")
print(f"{'='*55}")

# Test thresholds for FIRE-AS-IS strategy (P >= thr)
print(f"\n[FIRE AS-IS] (current side wins)  thr → fire when P(win) >= thr:")
print(f"{'thr':>5s} {'n':>5s} {'WR':>5s} {'sum':>9s} {'mean':>8s} {'t':>6s}")
print('-'*55)
fire_results = []
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = proba >= thr
    if mask.sum() < 5: continue
    pnls = h_te[mask]
    n = len(pnls); s = pnls.sum(); m = s/n
    sd = pnls.std(ddof=1) if n > 1 else 0
    t = m/(sd/math.sqrt(n)) if sd else 0
    wr = (pnls > 0).sum() / n
    fire_results.append({'thr':thr,'n':n,'mean':m,'t':t,'wr':wr,'sum':s})
    print(f"  {thr:.2f}  {n:>4d}  {wr*100:>3.0f}%  ${s:>+6.2f}  ${m:>+5.2f}  {t:>+5.2f}")

# Test FLIP strategy: when P(win) very low, fire OPPOSITE side
print(f"\n[FLIP SIDE] (low P → fire opposite, expect win)  P <= thr → flip:")
print(f"{'thr':>5s} {'n':>5s} {'flip_WR':>8s} {'flip_sum':>10s} {'flip_mean':>10s}")
print('-'*55)
# When original side LOSES, opposite WINS in binary market.
# So P(opposite wins) = 1 - P(original wins). If we flip, hypo flips sign.
flip_results = []
for thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
    mask = proba <= thr
    if mask.sum() < 5: continue
    # Flipped fire: opposite hypo. If original was -1.7 (lost), flipped wins ~+2.5
    # Flipped hypo magnitude varies but in binary settle, win ≈ entry_inverse * shares
    # Simplification: flipped_pnl ≈ -original_pnl (fees aside)
    flipped_pnls = -h_te[mask]
    n = len(flipped_pnls); s = flipped_pnls.sum(); m = s/n
    wr = (flipped_pnls > 0).sum() / n
    flip_results.append({'thr':thr,'n':n,'mean':m,'wr':wr,'sum':s})
    print(f"  {thr:.2f}  {n:>4d}  {wr*100:>5.0f}%   ${s:>+7.2f}   ${m:>+6.2f}")

# Bonferroni — testing 7 fire thresholds + 5 flip thresholds = 12 tests
n_tests = len(fire_results) + len(flip_results)
bonf_t = 2.85 if n_tests <= 10 else 3.10
print(f"\nBonferroni critical t (for {n_tests} tests): {bonf_t:.2f}")

best_fire = max(fire_results, key=lambda r: r['t']) if fire_results else None
if best_fire:
    print(f"\nBest FIRE threshold: {best_fire['thr']:.2f}")
    print(f"  n={best_fire['n']}  WR={best_fire['wr']*100:.0f}%  mean=${best_fire['mean']:+.2f}  t={best_fire['t']:+.2f}")
    pass_fire = best_fire['t'] > bonf_t
    print(f"  Bonferroni: {'✅ PASSES' if pass_fire else '❌ FAILS'}")

# Compute fire-side t-test for flip results too
if flip_results:
    # Use flip_results best by mean
    best_flip = max(flip_results, key=lambda r: r['mean']) 
    print(f"\nBest FLIP threshold: P(win)<={best_flip['thr']:.2f}")
    print(f"  n={best_flip['n']}  flip_WR={best_flip['wr']*100:.0f}%  flip_mean=${best_flip['mean']:+.2f}")

# Permutation
if best_fire:
    print(f"\nPermutation test on best FIRE threshold (5000 shuffles)...")
    actual = best_fire['mean']
    extreme = 0
    proba_list = list(proba)
    for _ in range(5000):
        random.shuffle(proba_list)
        ps = np.array(proba_list)
        mask = ps >= best_fire['thr']
        if mask.sum() < 5: continue
        if h_te[mask].mean() >= actual:
            extreme += 1
    p_perm = (extreme + 1) / 5001
    print(f"  Permutation p: {p_perm:.4f}  ({'✅ PASSES' if p_perm<0.01 else '❌ FAILS'} at α=0.01)")

# Combined fire-and-flip strategy expected EV
if best_fire and flip_results:
    best_flip = max(flip_results, key=lambda r: r['mean'])
    n_fire = best_fire['n']
    n_flip = best_flip['n']
    n_skip = len(X_te) - n_fire - n_flip
    total_pnl = best_fire['sum'] + best_flip['sum']
    print(f"\n[COMBINED STRATEGY] fire@{best_fire['thr']:.2f} + flip@{best_flip['thr']:.2f}")
    print(f"  Fires:  {n_fire}  (WR={best_fire['wr']*100:.0f}%, sum=${best_fire['sum']:+.2f})")
    print(f"  Flips:  {n_flip}  (WR={best_flip['wr']*100:.0f}%, sum=${best_flip['sum']:+.2f})")
    print(f"  Skips:  {n_skip}")
    print(f"  Total PnL: ${total_pnl:+.2f} across {len(X_te)} test fires")
    print(f"  Per fire: ${total_pnl/(n_fire+n_flip):+.3f}")

# Save
out = {
    'model': model, 'scaler': scaler, 'features': feat_names,
    'auc': auc,
    'best_C': best_C,
    'best_fire_threshold': best_fire['thr'] if best_fire else 0.65,
    'best_fire_wr':        best_fire['wr']  if best_fire else 0,
    'best_fire_mean':      best_fire['mean'] if best_fire else 0,
    'best_flip_threshold': best_flip['thr'] if 'best_flip' in dir() else 0.30,
    'best_flip_wr':        best_flip['wr']  if 'best_flip' in dir() else 0,
}
pickle.dump(out, open('win_gate_model.pkl', 'wb'))
print(f"\n✅ Model saved to win_gate_model.pkl")

# VERDICT
print(f"\n{'='*55}")
print(f"DEPLOYMENT VERDICT")
print(f"{'='*55}")
deploy_fire = (auc >= 0.65) and best_fire and (best_fire['t'] > bonf_t) and best_fire['wr'] >= 0.55
if deploy_fire:
    print(f"✅ DEPLOY win_gate (FIRE mode)")
    print(f"   AUC: {auc:.3f}")
    print(f"   Threshold: {best_fire['thr']:.2f}  WR: {best_fire['wr']*100:.0f}%  mean: ${best_fire['mean']:+.2f}")
else:
    print(f"❌ DO NOT DEPLOY")
    print(f"   AUC: {auc:.3f} (need >= 0.65)")
    if best_fire:
        print(f"   Best WR: {best_fire['wr']*100:.0f}% (need >= 55%)")
        print(f"   Best t: {best_fire['t']:.2f} (need > {bonf_t:.2f})")
