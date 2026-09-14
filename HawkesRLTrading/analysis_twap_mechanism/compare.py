"""RL-present vs TWAP-alone baseline: how the meta-order actually gets executed.

Both arms are reduced to the same three statistics, computed the same way:
  * fill mix   - each inventory change is priced at dCash/dInv and classified
                 aggressive (executed at/through the far touch) or passive.
  * price path - mid relative to the arrival price (mid at the TWAP's first
                 wake), in bps, on a common 10s grid.
  * slippage   - AR_RL_runner.py:586-595, robust mean excludes episodes more
                 than 20bps from the median (the paper's convention).
"""
import numpy as np, glob, os, sys

RL_BASE = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
BL_BASE = '/Users/alirazajafree/lobSimulations-1/HawkesRLTrading/twap_baseline_obs_out'
GRID = np.arange(250, 551, 10.)

def stats(episodes, side, name):
    """episodes: list of (t, bid, ask, cash, inv) arrays."""
    agg_q = pas_q = 0.0
    paths, slips = [], []
    for t, bid, ask, cash, inv in episodes:
        dI, dC = np.diff(inv), np.diff(cash)
        f = dI != 0
        if f.any():
            px = dC[f] / -dI[f]
            q = np.abs(dI[f]); b, a = bid[1:][f], ask[1:][f]
            aggressive = (px >= a - 1e-9) if side == 'buy' else (px <= b + 1e-9)
            agg_q += q[aggressive].sum(); pas_q += q[~aggressive].sum()
        mid = 0.5 * (bid + ask)
        paths.append(np.interp(GRID, t, mid))
        arrival = mid[0]
        executed = abs(inv[0] - inv[-1])
        if executed:
            bench = arrival * executed
            earned_or_spent = (cash[-1] - cash[0]) if side == 'sell' else (cash[0] - cash[-1])
            slips.append(((bench - earned_or_spent) if side == 'sell'
                          else (earned_or_spent - bench)) * 1e4 / bench)
    P = np.array(paths); bps = (P - P[:, [0]]) / P[:, [0]] * 1e4
    s = np.array(slips); keep = np.abs(s - np.median(s)) < 20
    tot = agg_q + pas_q
    print(f"{name:26s} n={len(episodes):2d}  exec {tot/max(len(episodes),1):5.1f}/ep  "
          f"AGGRESSIVE {100*agg_q/tot:4.1f}%  passive {100*pas_q/tot:4.1f}%")
    print(f"{'':26s} mid vs arrival:  t300 {bps[:,5].mean():+6.2f}  t350 {bps[:,10].mean():+6.2f}  "
          f"t400 {bps[:,15].mean():+6.2f}  t550 {bps[:,-1].mean():+6.2f} bps")
    print(f"{'':26s} slippage robust {s[keep].mean():+6.2f} +/- {s[keep].std(ddof=1):.2f} "
          f"(n={keep.sum()}), raw mean {s.mean():+.2f}")
    return dict(agg=100*agg_q/tot, bps=bps, slip=s[keep], name=name)

def load_rl(d):
    eps = np.load(glob.glob(f'{RL_BASE}/{d}/*twap_observations.npy')[0], allow_pickle=True)
    out = []
    for ep in eps:
        out.append((np.array([o['current_time'] for o in ep]),
                    np.array([o['LOB0']['Bid_L1'][0] for o in ep], float),
                    np.array([o['LOB0']['Ask_L1'][0] for o in ep], float),
                    np.array([o['Cash'] for o in ep], float),
                    np.array([o['Inventory'] for o in ep], float)))
    return out

def load_baseline(arm, side):
    """arm: 'vary' (per-episode dirs) or 'matched' (single dir)."""
    files = sorted(glob.glob(f'{BL_BASE}/{arm}/{side}/ep*/stepobs_{side}.npy')) \
            or glob.glob(f'{BL_BASE}/{arm}/{side}/stepobs_{side}.npy')
    out = []
    for fp in files:
        for a in np.load(fp, allow_pickle=True):
            a = a[a[:, 9] == -2]                    # post-step rows only
            if len(a) < 3: continue
            out.append((a[:,0], a[:,1], a[:,2], a[:,7], a[:,8]))
    return out

if __name__ == '__main__':
    arm = sys.argv[1] if len(sys.argv) > 1 else 'vary'
    for side, rl_dirs in [('sell', [('sell_ep16','SELL RL ckpt16')]),
                          ('buy',  [('buy_ep16','BUY RL ckpt16'), ('buy_ep64','BUY RL ckpt64'),
                                    ('buy_ep40_lim25','BUY RL ckpt40 lim25'),
                                    ('buy_ep40_lim50','BUY RL ckpt40 lim50')])]:
        bl = load_baseline(arm, side)
        print(f"\n{'='*92}\n{side.upper()} meta-order\n{'='*92}")
        if bl: stats(bl, side, f'TWAP alone ({arm})')
        else:  print(f"  [baseline '{arm}' not ready yet]")
        for d, nm in rl_dirs: stats(load_rl(d), side, nm)
