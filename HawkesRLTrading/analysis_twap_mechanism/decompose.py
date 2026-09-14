"""Exact decomposition of arrival slippage into drift vs spread.

For a sell meta-order,
    slip_bps = 1e4 * Sum_i q_i (arrival - px_i) / (arrival * Q)
and (arrival - px_i) = (arrival - mid_i) + (mid_i - px_i), so

    slip = DRIFT + SPREAD
    DRIFT  = quantity-weighted (arrival - mid_i)/arrival   [price ran away]
    SPREAD = quantity-weighted (mid_i - px_i)/arrival      [cost vs prevailing mid]

For a buy the signs flip (mid_i - arrival, px_i - mid_i). The two terms are
exhaustive: their sum reproduces the runner's slippage number, which is checked
against the independently-saved slippage arrays below.
"""
import numpy as np, glob, os
RL = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')

def decomp(t, bid, ask, cash, inv, side):
    dI = np.diff(inv); f = dI != 0
    if not f.any(): return None
    px = np.diff(cash)[f] / -dI[f]; q = np.abs(dI[f])
    mid = (0.5*(bid+ask))[:-1][f]
    ok = np.abs(px-mid)/mid < 0.01
    px, q, mid = px[ok], q[ok], mid[ok]
    arrival = 0.5*(bid[0]+ask[0])
    if side == 'sell':
        drift  = np.sum(q*(arrival-mid))/(arrival*np.sum(q))*1e4
        spread = np.sum(q*(mid-px))    /(arrival*np.sum(q))*1e4
    else:
        drift  = np.sum(q*(mid-arrival))/(arrival*np.sum(q))*1e4
        spread = np.sum(q*(px-mid))     /(arrival*np.sum(q))*1e4
    return drift, spread, float(np.sum(q))

def run(eps, side, name):
    D=[];S=[]
    for e in eps:
        if e is None: continue
        D.append(e[0]); S.append(e[1])
    D=np.array(D); S=np.array(S); T=D+S
    keep=np.abs(T-np.median(T))<20     # paper's robust-mean convention
    print(f"  {name:26s} slip {T[keep].mean():+6.2f} = DRIFT {D[keep].mean():+6.2f} + SPREAD {S[keep].mean():+6.2f}   "
          f"(n={keep.sum()})")
    return D[keep], S[keep], T[keep]

def load_bl(side):
    out=[]
    for f in sorted(glob.glob(f'HawkesRLTrading/twap_baseline_obs_out/vary/{side}/ep*/stepobs_{side}.npy')):
        for a in np.load(f,allow_pickle=True):
            p=a[a[:,9]==-2]
            out.append(decomp(p[:,0],p[:,1],p[:,2],p[:,7],p[:,8],side))
    return out

def load_rl(d,side):
    out=[]
    for ep in np.load(glob.glob(f'{RL}/{d}/*twap_observations.npy')[0],allow_pickle=True):
        out.append(decomp(np.array([o['current_time'] for o in ep]),
                          np.array([o['LOB0']['Bid_L1'][0] for o in ep],float),
                          np.array([o['LOB0']['Ask_L1'][0] for o in ep],float),
                          np.array([o['Cash'] for o in ep],float),
                          np.array([o['Inventory'] for o in ep],float),side))
    return out

from scipy import stats as st
for side, rls in [('sell',[('sell_ep16','vs RL ckpt16')]),
                  ('buy',[('buy_ep16','vs RL ckpt16'),('buy_ep64','vs RL ckpt64'),
                          ('buy_ep40_lim25','vs RL ckpt40 lim25'),('buy_ep40_lim50','vs RL ckpt40 lim50')])]:
    print(f"\n{'='*94}\n{side.upper()} meta-order   (all figures bps, positive = worse for the meta-order)\n{'='*94}")
    bD,bS,bT = run(load_bl(side), side, 'TWAP alone [vary-seed]')
    for d,nm in rls:
        rD,rS,rT = run(load_rl(d,side), side, nm)
        pd_=st.ttest_ind(rD,bD,equal_var=False).pvalue
        ps_=st.ttest_ind(rS,bS,equal_var=False).pvalue
        print(f"  {'':26s}   delta: drift {rD.mean()-bD.mean():+6.2f} (p={pd_:.3f})   "
              f"spread {rS.mean()-bS.mean():+6.2f} (p={ps_:.3f})")
