import numpy as np, glob, os
base = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
TWAP_ON, TWAP_OFF = 250., 400.

def mid(o):
    pa = o['LOB0']['Ask_L1'][0]; pb = o['LOB0']['Bid_L1'][0]
    return 0.5*(pa+pb)

def analyse(path, name):
    eps = np.load(path, allow_pickle=True)
    rows = []
    for ep in eps:
        t   = np.array([o['current_time'] for o in ep])
        inv = np.array([o['Inventory'] for o in ep], dtype=float)
        cash= np.array([o['Cash'] for o in ep], dtype=float)
        m   = np.array([mid(o) for o in ep])
        d   = np.diff(inv)
        td  = t[1:]
        pre = td < TWAP_ON; dur = (td>=TWAP_ON)&(td<=TWAP_OFF); post = td > TWAP_OFF
        def gb(mask): return d[mask][d[mask]>0].sum()
        def gs(mask): return -d[mask][d[mask]<0].sum()
        pnl = cash + inv*m*(1-1e-4*np.sign(inv))
        rows.append(dict(
            inv_at250 = inv[np.searchsorted(t,TWAP_ON)-1] if t[-1]>TWAP_ON else np.nan,
            inv_at400 = inv[np.searchsorted(t,TWAP_OFF)-1] if t[-1]>TWAP_OFF else np.nan,
            inv_term  = inv[-1],
            inv_max   = inv.max(), inv_min = inv.min(),
            mean_pre  = inv[t<TWAP_ON].mean(),
            mean_dur  = inv[(t>=TWAP_ON)&(t<=TWAP_OFF)].mean(),
            mean_post = inv[t>TWAP_OFF].mean() if (t>TWAP_OFF).any() else np.nan,
            buys_pre=gb(pre), sells_pre=gs(pre),
            buys_dur=gb(dur), sells_dur=gs(dur),
            buys_post=gb(post), sells_post=gs(post),
            pnl_term = pnl[-1]-pnl[0], t_end=t[-1], n=len(ep)))
    keys = rows[0].keys()
    agg = {k: np.nanmean([r[k] for r in rows]) for k in keys}
    print(f"\n=== {name}   ({len(rows)} episodes)")
    print(f"  inventory: pre-mean {agg['mean_pre']:+.2f} | during-mean {agg['mean_dur']:+.2f} | post-mean {agg['mean_post']:+.2f}")
    print(f"             @t=250 {agg['inv_at250']:+.2f} | @t=400 {agg['inv_at400']:+.2f} | TERMINAL {agg['inv_term']:+.2f}   (range {agg['inv_min']:+.1f}..{agg['inv_max']:+.1f})")
    print(f"  gross vol: pre  B {agg['buys_pre']:6.1f} / S {agg['sells_pre']:6.1f}")
    print(f"             dur  B {agg['buys_dur']:6.1f} / S {agg['sells_dur']:6.1f}")
    print(f"             post B {agg['buys_post']:6.1f} / S {agg['sells_post']:6.1f}")
    print(f"  terminal MTM pnl {agg['pnl_term']:+.3f}   steps/ep {agg['n']:.0f}, t_end {agg['t_end']:.0f}")
    print("  per-episode terminal inv: " + " ".join(f"{r['inv_term']:+.0f}" for r in rows))
    return rows

for d, nm in [('sell_ep16','SELL TWAP  ckpt16'), ('buy_ep16','BUY TWAP  ckpt16'),
              ('buy_ep64','BUY TWAP  ckpt64'), ('buy_ep40_lim25','BUY TWAP ckpt40 lim25'),
              ('buy_ep40_lim50','BUY TWAP ckpt40 lim50')]:
    g = glob.glob(os.path.join(base,d,'*RL_observations.npy'))
    if g: analyse(g[0], nm)
