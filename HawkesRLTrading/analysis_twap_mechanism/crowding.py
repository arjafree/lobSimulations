"""Where the RL agent's liquidity actually sits during the meta-order window.

'Positions' in each recorded observation is the agent's live resting orders by
level, so this is standing depth, not the order-count proxy used in the paper's
Table 9 (limit orders minus cancellations).
"""
import numpy as np, glob, os
base = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')

def qty(orders):
    tot = 0.0
    for o in orders:
        for attr in ('quantity', 'size', 'total_quantity'):
            v = getattr(o, attr, None)
            if v is not None: tot += float(v); break
        else:
            tot += 1.0
    return tot

def run(d, name):
    eps = np.load(glob.glob(f'{base}/{d}/*RL_observations.npy')[0], allow_pickle=True)
    acc = {k: [] for k in ('ask_pre','bid_pre','ask_dur','bid_dur')}
    for ep in eps:
        for o in ep:
            t = o['current_time']; p = o.get('Positions', {}) or {}
            a = qty(p.get('Ask_L1', [])) + qty(p.get('Ask_L2', []))
            b = qty(p.get('Bid_L1', [])) + qty(p.get('Bid_L2', []))
            if t < 250:              acc['ask_pre'].append(a); acc['bid_pre'].append(b)
            elif t <= 400:           acc['ask_dur'].append(a); acc['bid_dur'].append(b)
    m = {k: np.mean(v) if v else np.nan for k, v in acc.items()}
    print(f"{name:22s} pre-window  asks {m['ask_pre']:6.2f}  bids {m['bid_pre']:6.2f}   "
          f"|| in-window  asks {m['ask_dur']:6.2f}  bids {m['bid_dur']:6.2f}   "
          f"(ask/bid ratio in window {m['ask_dur']/max(m['bid_dur'],1e-9):.2f})")

for d, nm in [('sell_ep16','SELL ckpt16'),('buy_ep16','BUY ckpt16'),('buy_ep64','BUY ckpt64'),
              ('buy_ep40_lim25','BUY ckpt40 lim25'),('buy_ep40_lim50','BUY ckpt40 lim50')]:
    run(d, nm)
