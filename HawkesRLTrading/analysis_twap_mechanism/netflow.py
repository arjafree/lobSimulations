import numpy as np, glob, os
base=os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
def f(path,name):
    eps=np.load(path,allow_pickle=True); out={}
    for lab,(a,b) in {'pre[100,250)':(100,250),'dur[250,400]':(250,400),'post(400,550]':(400,552)}.items():
        v=[]
        for ep in eps:
            t=np.array([o['current_time'] for o in ep]); inv=np.array([o['Inventory'] for o in ep],float)
            v.append(np.interp(b,t,inv)-np.interp(a,t,inv))
        v=np.array(v); out[lab]=(v.mean(), v.std(ddof=1)/np.sqrt(len(v)))
    s=" | ".join(f"{k} {m:+6.2f}±{e:.2f}" for k,(m,e) in out.items())
    print(f"{name:20s} net inventory change: {s}")
for d,nm in [('sell_ep16','SELL ckpt16'),('buy_ep16','BUY ckpt16'),('buy_ep64','BUY ckpt64'),
             ('buy_ep40_lim25','BUY ckpt40 lim25'),('buy_ep40_lim50','BUY ckpt40 lim50')]:
    g=glob.glob(os.path.join(base,d,'*RL_observations.npy'))
    if g: f(g[0],nm)
