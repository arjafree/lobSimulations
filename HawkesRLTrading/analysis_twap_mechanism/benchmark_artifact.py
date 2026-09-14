"""Is the buy/sell slippage asymmetry a property of the agent, or of the benchmark?

Every episode -- baseline and RL-present -- starts from mid = 99.9550 at
t=100 (same Hawkes seed, RL not yet unlocked, verified sd=0 across all runs).
The RL agent then trades for 150s before the TWAP's arrival benchmark is
struck at t=250, so its pre-window price impact is baked into the benchmark
rather than measured against it.

Re-pricing both arms against the common t=100 mid removes that contamination.
Arrival-benchmarked slippage has OPPOSITE sign-sensitivity to the benchmark on
the two sides: raising the arrival price lowers a buyer's measured cost and
raises a seller's. So one and the same physical action -- the agent pushing
price up before t=250 -- flatters the buy number and inflates the sell number.
"""
import numpy as np, glob, os
RL = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
P100, CASH0, INV0 = 99.9550, 1_000_000.0, 500.0

def load(d, pat):
    return np.array(np.load(glob.glob(f'{RL}/{d}/{pat}')[0], allow_pickle=True), float).ravel()

def slips(d, side, name, bl_pat=False):
    fc = load(d, f'*final_cash*{side}.npy' if bl_pat else '*final_cash*.npy')
    te = load(d, f'*total_executed*{side}.npy' if bl_pat else '*total_executed*.npy')
    sm = load(d, f'*start_midprices*{side}.npy' if bl_pat else '*start_midprices*.npy')
    n = min(len(fc), len(te), len(sm)); fc, te, sm = fc[:n], te[:n], sm[:n]
    got = (fc - CASH0) if side == 'sell' else (CASH0 - fc)      # earned / spent
    def bps(ref):
        b = ref * te
        return ((b - got) if side == 'sell' else (got - b)) * 1e4 / b
    a, c = bps(sm), bps(np.full(n, P100))
    ka = np.abs(a - np.median(a)) < 20; kc = np.abs(c - np.median(c)) < 20
    print(f"  {name:26s} arrival-benchmarked {a[ka].mean():+6.2f}   |   t=100-benchmarked {c[kc].mean():+6.2f}"
          f"    (benchmark moved {(sm.mean()-P100)/P100*1e4:+5.2f} bps)")
    return a[ka], c[kc]

from scipy import stats as st
for side, rows in [('sell', [('baseline_sell','TWAP alone [seed-1]',True), ('sell_ep16','vs RL ckpt16',False)]),
                   ('buy',  [('baseline_buy','TWAP alone [seed-1]',True), ('buy_ep16','vs RL ckpt16',False),
                             ('buy_ep64','vs RL ckpt64',False), ('buy_ep40_lim25','vs RL ckpt40 lim25',False),
                             ('buy_ep40_lim50','vs RL ckpt40 lim50',False)])]:
    print(f"\n{'='*104}\n{side.upper()} meta-order   (bps, positive = worse for the meta-order)\n{'='*104}")
    base_a = base_c = None
    for d, nm, bp in rows:
        a, c = slips(d, side, nm, bp)
        if base_a is None: base_a, base_c = a, c; continue
        pa = st.ttest_ind(a, base_a, equal_var=False).pvalue
        pc = st.ttest_ind(c, base_c, equal_var=False).pvalue
        print(f"  {'':26s}   delta {a.mean()-base_a.mean():+6.2f} (p={pa:.3f})"
              f"   |   delta {c.mean()-base_c.mean():+6.2f} (p={pc:.3f})")
