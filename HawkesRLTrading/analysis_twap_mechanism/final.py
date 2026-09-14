"""Fill-level analysis with pathological rows filtered.

~2% of reconstructed fills price out near zero (dCash and dInv recorded out of
step with each other). They are rejected by requiring the implied price to sit
within 1% of the prevailing mid. Both the classifier and the effective-cost
metric are re-derived on the surviving fills, and the classifier is re-scored
against the recorded action codes on the baseline arm.
"""
import numpy as np, glob, os
RL = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
C = dict(t=0,bid=1,ask=2,cash=7,inv=8,act=9); MO={4,7}

def fills(t,bid,ask,cash,inv,side,acts=None):
    dI=np.diff(inv); f=dI!=0
    if not f.any(): return None
    px=np.diff(cash)[f]/-dI[f]; q=np.abs(dI[f])
    pb,pa=bid[:-1][f],ask[:-1][f]; mid=0.5*(pb+pa)
    ok=np.abs(px-mid)/mid < 0.01                     # reject reconstruction failures
    out=dict(px=px[ok],q=q[ok],pb=pb[ok],pa=pa[ok],mid=mid[ok],
             nbad=int((~ok).sum()),ntot=int(f.sum()))
    if acts is not None: out['truth']=np.isin(acts[:-1][f][ok].astype(int),list(MO))
    return out

def summarize(d,side,name):
    agg=pas=0.0; costs=[]; bad=tot=0
    for ep in d:
        if ep is None: continue
        pred=(ep['px']>=ep['pa']-1e-9) if side=='buy' else (ep['px']<=ep['pb']+1e-9)
        agg+=ep['q'][pred].sum(); pas+=ep['q'][~pred].sum()
        sgn=1.0 if side=='buy' else -1.0
        c=sgn*(ep['px']-ep['mid'])/ep['mid']*1e4
        costs.append(np.sum(ep['q']*c)/np.sum(ep['q']))
        bad+=ep['nbad']; tot+=ep['ntot']
    c=np.array(costs)
    print(f"  {name:26s} agg {100*agg/(agg+pas):5.1f}%   eff.cost {c.mean():+6.2f} +/- {c.std(ddof=1)/np.sqrt(len(c)):4.2f} bps/unit   (dropped {100*bad/tot:.1f}% of fills)")
    return 100*agg/(agg+pas), c

def load_bl(side):
    eps=[]
    for f in sorted(glob.glob(f'HawkesRLTrading/twap_baseline_obs_out/vary/{side}/ep*/stepobs_{side}.npy')):
        for a in np.load(f,allow_pickle=True):
            post=a[a[:,9]==-2]; pre=a[a[:,9]!=-2]; m=min(len(post),len(pre))
            eps.append(fills(post[:m,0],post[:m,1],post[:m,2],post[:m,7],post[:m,8],side,pre[:m,9]))
    return eps

def load_rl(d,side):
    out=[]
    for ep in np.load(glob.glob(f'{RL}/{d}/*twap_observations.npy')[0],allow_pickle=True):
        out.append(fills(np.array([o['current_time'] for o in ep]),
                         np.array([o['LOB0']['Bid_L1'][0] for o in ep],float),
                         np.array([o['LOB0']['Ask_L1'][0] for o in ep],float),
                         np.array([o['Cash'] for o in ep],float),
                         np.array([o['Inventory'] for o in ep],float),side))
    return out

for side in ('sell','buy'):
    bl=load_bl(side)
    tp=fp=tn=fn=0; at=pt=0.0
    for ep in bl:
        pred=(ep['px']>=ep['pa']-1e-9) if side=='buy' else (ep['px']<=ep['pb']+1e-9)
        tr=ep['truth']
        at+=ep['q'][tr].sum(); pt+=ep['q'][~tr].sum()
        tp+=int((tr&pred).sum()); fn+=int((tr&~pred).sum())
        fp+=int((~tr&pred).sum()); tn+=int((~tr&~pred).sum())
    n=tp+fp+tn+fn
    print(f"\n{'='*88}\n{side.upper()} meta-order")
    print(f"  [classifier check on baseline] TRUE agg {100*at/(at+pt):.1f}%  "
          f"agreement {100*(tp+tn)/n:.1f}%  fp {fp} fn {fn}  n={n}")
    summarize(bl,side,'TWAP alone [vary-seed]')
    rls=[('sell_ep16','vs RL ckpt16')] if side=='sell' else \
        [('buy_ep16','vs RL ckpt16'),('buy_ep64','vs RL ckpt64'),
         ('buy_ep40_lim25','vs RL ckpt40 lim25'),('buy_ep40_lim50','vs RL ckpt40 lim50')]
    for d,nm in rls: summarize(load_rl(d,side),side,nm)

# --- per-episode aggressive share, with tests -------------------------------
from scipy import stats as st
def per_ep_agg(eps, side):
    v=[]
    for ep in eps:
        if ep is None: continue
        pred=(ep['px']>=ep['pa']-1e-9) if side=='buy' else (ep['px']<=ep['pb']+1e-9)
        v.append(100*ep['q'][pred].sum()/ep['q'].sum())
    return np.array(v)

print(f"\n\n{'#'*88}\nPer-episode aggressive share: RL-present vs TWAP-alone\n{'#'*88}")
for side, rls in [('sell',[('sell_ep16','RL ckpt16')]),
                  ('buy',[('buy_ep16','RL ckpt16'),('buy_ep64','RL ckpt64'),
                          ('buy_ep40_lim25','RL ckpt40 lim25'),('buy_ep40_lim50','RL ckpt40 lim50')])]:
    b=per_ep_agg(load_bl(side),side)
    print(f"\n{side.upper():4s} baseline {b.mean():5.1f}% +/- {b.std(ddof=1):4.1f} (n={len(b)})")
    for d,nm in rls:
        r=per_ep_agg(load_rl(d,side),side)
        w=st.ttest_ind(r,b,equal_var=False); u=st.mannwhitneyu(r,b)
        print(f"     {nm:18s} {r.mean():5.1f}% +/- {r.std(ddof=1):4.1f}   delta {r.mean()-b.mean():+5.1f}pp   "
              f"Welch p={w.pvalue:.4f}  MWU p={u.pvalue:.4f}")
