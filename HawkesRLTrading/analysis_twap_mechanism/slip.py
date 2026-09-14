import numpy as np, glob, os
base=os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
def load(d,pat):
    g=glob.glob(os.path.join(base,d,pat))
    return np.load(g[0],allow_pickle=True) if g else None
def slip(d,side,name):
    fc=load(d,'*final_cash*.npy'); te=load(d,'*total_executed*.npy'); sm=load(d,'*start_midprices*.npy')
    if fc is None: return print(f"{name}: missing"); 
    fc=np.array(fc,float).ravel(); te=np.array(te,float).ravel(); sm=np.array(sm,float).ravel()
    n=min(len(fc),len(te),len(sm)); fc,te,sm=fc[:n],te[:n],sm[:n]
    bench=sm*te
    s=((1e6-fc)-bench)*1e4/bench if side=='buy' else (bench-(fc-1e6))*1e4/bench
    med=np.median(s); rob=s[np.abs(s-med)<=20]
    print(f"{name:24s} n={n:2d} exec~{te.mean():5.1f} | raw {s.mean():+6.2f}±{s.std(ddof=1)/np.sqrt(n):.2f} | robust {rob.mean():+6.2f}±{rob.std(ddof=1):.2f} (n={len(rob)})")
for d,side,nm in [('baseline_sell','sell','BASELINE sell'),('sell_ep16','sell','SELL ckpt16'),
                  ('baseline_buy','buy','BASELINE buy'),('buy_ep16','buy','BUY ckpt16'),
                  ('buy_ep64','buy','BUY ckpt64'),('buy_ep40_lim25','buy','BUY ckpt40 lim25'),
                  ('buy_ep40_lim50','buy','BUY ckpt40 lim50')]:
    slip(d,side,nm)
