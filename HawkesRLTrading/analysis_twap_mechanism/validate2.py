"""Same validation, but comparing the fill price against the PRE-step touch.

A market order executes against the book as it stood *before* the step; the
post-step touch already reflects the level the order just consumed, so
comparing against it misses exactly the fills that moved the price -- which is
the one-directional false-negative pattern seen in validate.py.
"""
import numpy as np, glob
C = dict(t=0, bid=1, ask=2, cash=7, inv=8, act=9)
MO = {4, 7}

def classify(px, side, pre_bid, pre_ask, post_bid, post_ask, mode):
    if mode == 'post':  return (px >= post_ask - 1e-9) if side=='buy' else (px <= post_bid + 1e-9)
    if mode == 'pre':   return (px >= pre_ask  - 1e-9) if side=='buy' else (px <= pre_bid  + 1e-9)
    # 'mid': fill at or beyond the pre-step far touch, tolerant of book motion
    return (px >= pre_ask - 1e-9) if side=='buy' else (px <= pre_bid + 1e-9)

for mode in ('post','pre'):
    print(f"\n########## classifier mode = {mode}-step touch")
    for side in ("buy","sell"):
        tp=fp=tn=fn=0; agg_t=pas_t=agg_p=pas_p=0.0
        for f in sorted(glob.glob(f'HawkesRLTrading/twap_baseline_obs_out/vary/{side}/ep*/stepobs_{side}.npy')):
            for a in np.load(f, allow_pickle=True):
                post = a[a[:,9]==-2]; pre = a[a[:,9]!=-2]
                m = min(len(post), len(pre))
                for i in range(1, m):
                    dI = post[i,C['inv']] - post[i-1,C['inv']]
                    if dI == 0: continue
                    px = (post[i,C['cash']] - post[i-1,C['cash']]) / -dI
                    q = abs(dI)
                    truth = int(pre[i,C['act']]) in MO
                    pred = classify(px, side, pre[i,C['bid']], pre[i,C['ask']],
                                    post[i,C['bid']], post[i,C['ask']], mode)
                    if truth: agg_t += q
                    else:     pas_t += q
                    if pred:  agg_p += q
                    else:     pas_p += q
                    tp += (truth and pred); fn += (truth and not pred)
                    fp += (not truth and pred); tn += (not truth and not pred)
        n = tp+fp+tn+fn
        print(f"  {side.upper():4s} true-agg {100*agg_t/(agg_t+pas_t):5.1f}%  pred-agg {100*agg_p/(agg_p+pas_p):5.1f}%  "
              f"agreement {100*(tp+tn)/n:5.1f}%  (fp {fp} fn {fn}, n={n})")
