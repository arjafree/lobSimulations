"""Validate the price-based aggressive/passive classifier against ground truth.

The baseline runs record the TWAP's chosen action code on every step, so each
fill can be attributed to either the agent's own market order (aggressive) or a
resting limit order being taken (passive). The RL-present runs have no action
codes saved, so the headline 38.8%-vs-28% number relies on inferring fill type
from price vs the touch -- which is what this checks.
"""
import numpy as np, glob

COLS = dict(t=0, bid=1, ask=2, cash=7, inv=8, act=9)
ACTIONS = ["lo_deep_Ask","co_deep_Ask","lo_top_Ask","co_top_Ask","mo_Ask","lo_inspread_Ask",
           "lo_inspread_Bid","mo_Bid","co_top_Bid","lo_top_Bid","co_deep_Bid","lo_deep_Bid","noop"]
MO = {4, 7}

for side in ("buy", "sell"):
    hist = {}
    tp = fp = tn = fn = 0
    agg_true = pas_true = 0.0
    for fp_ in sorted(glob.glob(f'HawkesRLTrading/twap_baseline_obs_out/vary/{side}/ep*/stepobs_{side}.npy')):
        for a in np.load(fp_, allow_pickle=True):
            for r in a[a[:, 9] != -2]:
                hist[int(r[9])] = hist.get(int(r[9]), 0) + 1
            for i in range(1, len(a)):
                if a[i, 9] != -2:  continue
                dI = a[i, COLS['inv']] - a[i-1, COLS['inv']]
                if dI == 0:        continue
                dC = a[i, COLS['cash']] - a[i-1, COLS['cash']]
                px = dC / -dI
                q  = abs(dI)
                truth_agg = int(a[i-1, COLS['act']]) in MO          # ground truth
                pred_agg  = (px >= a[i, COLS['ask']] - 1e-9) if side == 'buy' \
                            else (px <= a[i, COLS['bid']] + 1e-9)   # price-based guess
                if truth_agg: agg_true += q
                else:         pas_true += q
                tp += (truth_agg and pred_agg);   fn += (truth_agg and not pred_agg)
                fp += (not truth_agg and pred_agg); tn += (not truth_agg and not pred_agg)
    n = tp + fp + tn + fn
    tot = agg_true + pas_true
    print(f"\n=== {side.upper()} TWAP alone (independent-seed arm, 17 eps)")
    print("  actions chosen: " + ", ".join(f"{ACTIONS[k]}={v}" for k, v in sorted(hist.items()) if k >= 0))
    print(f"  TRUE aggressive share (from action codes): {100*agg_true/tot:.1f}%   passive {100*pas_true/tot:.1f}%")
    print(f"  classifier agreement: {100*(tp+tn)/n:.1f}%  (tp {tp} fp {fp} tn {tn} fn {fn}, n={n} fills)")
