"""Is dCash/dInv actually the fill price? Test on fills ground truth calls MOs."""
import numpy as np, glob
C = dict(t=0,bid=1,ask=2,cash=7,inv=8,act=9); MO={4,7}
for side in ("buy","sell"):
    dev=[]; gaps=[]; nfill=[]
    for f in sorted(glob.glob(f'HawkesRLTrading/twap_baseline_obs_out/vary/{side}/ep*/stepobs_{side}.npy')):
        for a in np.load(f,allow_pickle=True):
            post=a[a[:,9]==-2]; pre=a[a[:,9]!=-2]; m=min(len(post),len(pre))
            gaps.append(np.diff(post[:m,0]))
            for i in range(1,m):
                dI=post[i,C['inv']]-post[i-1,C['inv']]
                if dI==0: continue
                nfill.append(abs(dI))
                if int(pre[i,C['act']]) not in MO: continue
                px=(post[i,C['cash']]-post[i-1,C['cash']])/-dI
                ref = pre[i,C['ask']] if side=='buy' else pre[i,C['bid']]
                dev.append(px-ref)
    dev=np.array(dev); g=np.concatenate(gaps); nf=np.array(nfill)
    print(f"\n{side.upper()}: MO fill price minus pre-step far touch")
    print(f"  exact match (|dev|<1e-9): {100*np.mean(np.abs(dev)<1e-9):5.1f}%   "
          f"within 1 tick: {100*np.mean(np.abs(dev)<=0.0101):5.1f}%")
    print(f"  dev mean {dev.mean():+.4f}  sd {dev.std():.4f}  min {dev.min():+.3f} max {dev.max():+.3f}")
    print(f"  time gap between TWAP wakes: median {np.median(g):.3f}s  p95 {np.percentile(g,95):.3f}s")
    print(f"  fill size: median {np.median(nf):.0f}  p95 {np.percentile(nf,95):.0f}  max {nf.max():.0f}")
