import numpy as np, glob, os
base = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
def prof(path,name):
    eps=np.load(path,allow_pickle=True)
    grid=np.arange(100,552,10.)
    M=[]
    for ep in eps:
        t=np.array([o['current_time'] for o in ep]); inv=np.array([o['Inventory'] for o in ep],float)
        M.append(np.interp(grid,t,inv))
    M=np.array(M); m=M.mean(0)
    def w(a,b): return m[(grid>=a)&(grid<b)].mean()
    print(f"{name:26s} t100-150 {w(100,150):+6.2f} | 200-250 {w(200,250):+6.2f} | 250-400 {w(250,400):+6.2f} | 400-550 {w(400,551):+6.2f} "
          f"|| paper-tilt {w(250,400)-w(100,250):+5.2f}  clean-tilt(vs 200-250) {w(250,400)-w(200,250):+5.2f}")
for d,nm in [('sell_ep16','SELL ckpt16'),('buy_ep16','BUY ckpt16'),('buy_ep64','BUY ckpt64'),
             ('buy_ep40_lim25','BUY ckpt40 lim25'),('buy_ep40_lim50','BUY ckpt40 lim50')]:
    g=glob.glob(os.path.join(base,d,'*RL_observations.npy'))
    if g: prof(g[0],nm)
