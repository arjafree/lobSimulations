"""
Plot the two TWAP market-impact figures from the local twap-alone run output,
mirroring the original npyreader logic:

  (A) price_quantity_graph_predecay_mortised  -> SQL during execution,
      sampled every 50s, sqrt fit.  Figure: TWAP_predecay_50s_intervals_local.png
  (B) aggregatePricePaths_decay_fitted_propogatormodel -> post-execution decay,
      propagator model fit z^(1-b)-(z-1)^(1-b) on z=(t-100)/1200 in [1,2].
      Figure: TWAP_Price_Impact_Decay_local.png

Reads OUT_DIR/times.npy and OUT_DIR/pricepath_ep*.npy (raw midprices), and
converts to relative impact (P_t - P_0)/P_0 * 100  (P_0 = midprice at execution
start). beta is scale-invariant so it matches the paper regardless of this scaling.
"""
import os
import glob
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/alirazajafree/lobSimulations-1"
OUT_DIR = os.path.join(REPO, "HawkesRLTrading", "twap_alone_out")

EXEC_START = 100      # start_trading_lag
T_EXEC = 1200
EXEC_END = EXEC_START + T_EXEC  # 1300

times = np.load(os.path.join(OUT_DIR, "times.npy"))
files = sorted(glob.glob(os.path.join(OUT_DIR, "pricepath_ep*.npy")),
               key=lambda f: int(f.split("_ep")[-1].split(".")[0]))
raw = [np.load(f) for f in files]
print(f"loaded {len(raw)} episodes, times[{times[0]:.0f}..{times[-1]:.0f}] n={len(times)}")

# relative impact (%) vs P_0 (price at execution start)
data_list = []
for d in raw:
    p0 = d[0]
    data_list.append((d - p0) / p0 * 100.0)


def predecay_sql():
    """SQL during execution, sampled every 50s, sqrt fit (vs executed quantity)."""
    end_idx = np.where(times <= EXEC_END)[0]
    if len(end_idx) == 0:
        print("no data <= EXEC_END"); return
    end_idx = end_idx[-1]

    # pad/average across episodes up to end_idx
    seg = [d[:end_idx] for d in data_list]
    max_len = max(len(s) for s in seg)
    padded = np.full((len(seg), max_len), np.nan)
    for i, s in enumerate(seg):
        padded[i, :len(s)] = s
    mean_path = np.nanmean(padded, axis=0)
    seg_times = times[:max_len]

    # sample every 50s with tolerance
    target_times = np.arange(EXEC_START, EXEC_END, 50)
    tol = 0.5
    idxs = []
    for tt in target_times:
        diffs = np.abs(seg_times - tt)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol:
            idxs.append(j)
    idxs = np.array(sorted(set(idxs)))
    samp_t = seg_times[idxs]
    samp_y = mean_path[idxs]
    # executed quantity proxy: linear with time since exec start (q-rate = Q/T = 1/s)
    qty = samp_t - EXEC_START

    valid = ~np.isnan(samp_y) & (qty >= 0)
    qty, samp_y = qty[valid], samp_y[valid]

    plt.figure(figsize=(12, 8))
    plt.scatter(qty, samp_y, color="tab:blue", s=35, alpha=0.8, label="Sampled (50s)")

    def sqrt_func(q, a, b):
        return a * np.sqrt(q + 1e-6) + b
    try:
        popt, _ = curve_fit(sqrt_func, qty, samp_y, p0=[0.001, 0])
        qg = np.linspace(qty.min(), qty.max(), 200)
        plt.plot(qg, sqrt_func(qg, *popt), "r--", lw=2,
                 label=f"Sqrt fit: {popt[0]:.4g}*sqrt(q) + {popt[1]:.4g}")
        yhat = sqrt_func(qty, *popt)
        ss_res = np.sum((samp_y - yhat) ** 2)
        ss_tot = np.sum((samp_y - samp_y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        plt.text(0.02, 0.98, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
                 va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        print(f"[predecay SQL] a={popt[0]:.5g} b={popt[1]:.5g} R2={r2:.4f} n={len(qty)}")
    except Exception as e:
        print("sqrt fit failed:", e)

    plt.xlabel("Quantity executed")
    plt.ylabel("Price impact  (P_t - P_0)/P_0  [%]")
    plt.title("TWAP Quantity vs Price Impact During Execution (regular POV)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = os.path.join(OUT_DIR, "TWAP_predecay_50s_intervals_local.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("saved", out)


def decay_propagator():
    """Post-execution decay, propagator model on z=(t-100)/1200 in [1,2]."""
    start = np.where(times >= EXEC_END)[0]
    if len(start) == 0:
        print("no data >= EXEC_END"); return
    start = start[0]

    seg = [d[start:] for d in data_list if len(d) > start]
    if not seg:
        print("no decay segment"); return
    max_len = max(len(s) for s in seg)
    padded = np.full((len(seg), max_len), np.nan)
    for i, s in enumerate(seg):
        padded[i, :len(s)] = s
    mean_path = np.nanmean(padded, axis=0)
    seg_times = times[start:start + len(mean_path)]

    z = (seg_times - EXEC_START) / T_EXEC
    zmask = (z >= 1.0) & (z <= 2.0) & ~np.isnan(mean_path)
    zf = z[zmask]; yf = mean_path[zmask]
    print(f"[decay] z range {zf.min():.3f}..{zf.max():.3f}  n={len(zf)}")
    if len(zf) < 3:
        print("not enough decay points in z in [1,2] (need longer STOP_TIME)"); return

    def propagator(zarr, a, beta):
        out = []
        for zv in zarr:
            if zv <= 0:
                term = 0
            elif zv >= 1:
                term = zv ** (1 - beta) - (zv - 1) ** (1 - beta)
            else:
                term = zv ** (1 - beta) - (1 - zv) ** (1 - beta) * (-1)
            out.append(a * term)
        return np.array(out)

    plt.figure(figsize=(12, 8))
    plt.plot(zf, yf, color="orange", lw=3, alpha=0.8, label="Average")
    try:
        popt, pcov = curve_fit(propagator, zf, yf, p0=[yf.max() if yf.max() > 0 else 0.01, 0.22],
                               bounds=([0, 0.1], [np.inf, 1.9]), maxfev=5000)
        a_fit, beta_fit = popt
        zs = np.linspace(zf.min(), zf.max(), 200)
        plt.plot(zs, propagator(zs, *popt), "r--", lw=2,
                 label=f"Propagator: {a_fit:.4g}*(z^{1-beta_fit:.3f} - (z-1)^{1-beta_fit:.3f})")
        yhat = propagator(zf, *popt)
        ss_res = np.sum((yf - yhat) ** 2); ss_tot = np.sum((yf - yf.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        perr = np.sqrt(np.diag(pcov))
        plt.text(0.02, 0.98, f"R² = {r2:.4f}\nβ = {beta_fit:.3f} ± {perr[1]:.3f}\na = {a_fit:.4g}",
                 transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
        print(f"[decay propagator] a={a_fit:.5g} beta={beta_fit:.3f}+-{perr[1]:.3f} R2={r2:.4f}")
    except Exception as e:
        print("propagator fit failed:", e)

    plt.xlabel("z = (t-100)/1200"); plt.ylabel("Price impact  (P_t - P_0)/P_0  [%]")
    plt.title("TWAP Price Impact Decay Post Execution (regular POV)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.xlim(1.0, 2.0); plt.tight_layout()
    out = os.path.join(OUT_DIR, "TWAP_Price_Impact_Decay_local.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("saved", out)


if __name__ == "__main__":
    predecay_sql()
    decay_propagator()
