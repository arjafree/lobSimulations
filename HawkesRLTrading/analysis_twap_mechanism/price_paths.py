"""Mid-price paths around the meta-order, RL-present vs TWAP-alone.

All runs share an identical mid of 99.9550 at t=100 (same Hawkes seed, RL not
yet unlocked, verified sd = 0 across every run), so prices are shown in bps
relative to that common origin. The arrival benchmark used by the slippage
measure is the mid at t=250, marked on each panel: the gap between the two
curves at that instant is price displacement the benchmark absorbs rather than
measures.

The TWAP-alone arm only records the book from the meta-order's first wake at
t=250 -- nothing wakes the agent before then -- so its curve starts there. Its
t=100 value is not interpolated; it is the same deterministic 99.9550.
"""
import numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RL   = os.path.expanduser('~/Downloads/new_value_function/ckpt_eval')
BL   = 'HawkesRLTrading/twap_baseline_obs_out/seed1'
OUT  = os.path.expanduser('~/Tackling-Execution-paper')
P100 = 99.9550
BLUE, ORANGE = '#3b6fd4', '#c8531f'     # validated categorical pair (dataviz)
INK, MUTED   = '#1a1a1a', '#6b6b6b'

def bps(p): return (np.asarray(p) - P100) / P100 * 1e4

def rl_paths(d, grid):
    eps = np.load(glob.glob(f'{RL}/{d}/*RL_observations.npy')[0], allow_pickle=True)
    M = []
    for ep in eps:
        t = np.array([o['current_time'] for o in ep])
        m = np.array([0.5*(o['LOB0']['Ask_L1'][0]+o['LOB0']['Bid_L1'][0]) for o in ep])
        M.append(np.interp(grid, t, m))
    return bps(np.array(M))

def bl_paths(side, grid):
    """Baseline stepobs: per-episode dirs (array job) or one stacked file."""
    files = sorted(glob.glob(f'{BL}/{side}/ep*/stepobs_{side}.npy')) or \
            glob.glob(f'{BL}/{side}/stepobs_{side}.npy')
    if not files: return None
    M = []
    for f in files:
        for a in np.load(f, allow_pickle=True):
            p = a[a[:,9] == -2]
            if len(p) < 5: continue
            t, m = p[:,0], 0.5*(p[:,1]+p[:,2])
            # The first record lands at t~251 (the meta-order's first wake); the
            # window opens at 250. np.interp holds m[0] flat across that <1s gap
            # rather than inventing a value. Before 250 there is genuinely no
            # recorded book, so those points stay NaN.
            row = np.full(len(grid), np.nan)
            sel = grid >= 250
            row[sel] = np.interp(grid[sel], t, m)
            M.append(row)
    return bps(np.array(M))

def band(ax, grid, M, color, label, ls='-'):
    if M is None: return
    mu = np.nanmean(M, axis=0)
    lo, hi = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(grid, mu, color=color, lw=2.0, ls=ls, label=label, solid_capstyle='round')
    return mu

grid = np.arange(100, 551, 2.0)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)

panels = [('buy',  'buy_ep16',  'Buy meta-order',  axes[0]),
          ('sell', 'sell_ep16', 'Sell meta-order', axes[1])]

for side, d, title, ax in panels:
    ax.axvspan(250, 400, color='#000000', alpha=0.045, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.7, ls=':', zorder=1)
    mb = band(ax, grid, bl_paths(side, grid), BLUE,   'TWAP alone')
    mr = band(ax, grid, rl_paths(d, grid),    ORANGE, 'TWAP + RL agent')
    ax.plot([100], [0], marker='o', ms=6, color=INK, zorder=5, clip_on=False)
    ax.annotate('common start\n99.9550', xy=(100, 0), xytext=(112, 1.9),
                fontsize=7.5, color=INK, ha='left')
    # arrival benchmark: the level the slippage measure is struck at
    i250 = int(np.where(grid == 250)[0][0])
    for M, c in ((bl_paths(side, grid), BLUE), (rl_paths(d, grid), ORANGE)):
        if M is None: continue
        ax.plot([250], [np.nanmean(M, axis=0)[i250]], marker='D', ms=5.5,
                color=c, zorder=6, markeredgecolor='white', markeredgewidth=1.0)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel('Simulation time (s)', fontsize=9.5, color=INK)
    ax.set_xlim(100, 550); ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color('#cccccc')
    ax.grid(axis='y', color='#e8e8e8', lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')

axes[0].set_ylabel('Mid price vs $t{=}100$ (bps)', fontsize=9.5, color=INK)
axes[1].annotate('TWAP window', xy=(325, axes[1].get_ylim()[1]*0.93),
                 fontsize=8, color=MUTED, ha='center')
axes[0].annotate('arrival benchmark\nstruck here ($t{=}250$)', xy=(250, -3.6),
                 xytext=(285, -5.6), fontsize=7.5, color=MUTED,
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'advopt_price_paths.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
print('saved ->', os.path.join(OUT, 'advopt_price_paths.png'))

# numbers quoted in the caption / text
for side, d, title, _ in panels:
    B, R = bl_paths(side, grid), rl_paths(d, grid)
    i250 = int(np.where(grid == 250)[0][0]); i400 = int(np.where(grid == 400)[0][0])
    b250 = np.nanmean(B, axis=0)[i250] if B is not None else float('nan')
    print(f"{title:16s} t=250: alone {b250:+5.2f} | +RL {np.nanmean(R,axis=0)[i250]:+5.2f} bps   "
          f"t=400: alone {np.nanmean(B,axis=0)[i400] if B is not None else float('nan'):+5.2f} | "
          f"+RL {np.nanmean(R,axis=0)[i400]:+5.2f} bps")
