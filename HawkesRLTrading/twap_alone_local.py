"""
Local TWAP-alone market-impact runner.

Reproduces the price-impact figures (SQL during execution + post-execution
decay) from the paper, but with the *regular*-POV market configuration
(Pis dirac=40 / Pi_Q0=200  =>  high market volume v~22  =>  POV ~4.5%).

Saves, into OUT_DIR:
    times.npy                  shared per-step time grid (seconds)
    pricepath_ep{e}.npy        midprice at each recorded step for episode e

Execution window: [start_lag, start_lag + T] = [100, 1300], so the plotting
code's hardcoded t=1300 execution-end threshold lines up. The sim runs to
STOP_TIME (> 1300) so the decay/relaxation tail is captured.

Env-tunable (for smoke testing):
    TWAP_EPISODES   (default 20)
    TWAP_STOP_TIME  (default 1600)
    TWAP_SIDE       (default "buy")
    TWAP_T          (default 1200)   total execution time
    TWAP_Q          (default 1200)   total order size  (=> q=1 at action_freq=1)
"""
import sys
import os
import pickle
import copy
import numpy as np

# --- local paths -----------------------------------------------------------
REPO = "/Users/alirazajafree/lobSimulations-1"
sys.path.append(os.path.abspath(REPO))
PARAM_FILE = "/Users/alirazajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10"
OUT_DIR = os.path.join(REPO, "HawkesRLTrading", "twap_alone_out")
os.makedirs(OUT_DIR, exist_ok=True)

from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *  # noqa: F401,F403

# --- config knobs ----------------------------------------------------------
EPISODES  = int(os.environ.get("TWAP_EPISODES", 20))
STOP_TIME = float(os.environ.get("TWAP_STOP_TIME", 2500))  # decay fit needs z=(t-100)/1200 up to 2 => t=2500
TWAP_SIDE = os.environ.get("TWAP_SIDE", "buy")
T_EXEC    = int(os.environ.get("TWAP_T", 1200))
Q_TOTAL   = int(os.environ.get("TWAP_Q", 1200))
START_LAG = 100

# --- Hawkes kernel params --------------------------------------------------
with open(PARAM_FILE, "rb") as f:
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)

cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
faketod = {k: {k1: 1.0 for k1 in np.arange(13)} for k in cols}
tod = np.zeros(shape=(len(cols), 13))
for i in range(len(cols)):
    tod[i] = [faketod[cols[i]][k] for k in range(13)]

# Regular-POV (high market volume): background order sizes ~40 / queue ~200.
Pis = {'Bid_L2': [0., [(40, 1.)]],
       'Bid_inspread': [0., [(40, 1.)]],
       'Bid_L1': [0., [(40, 1.)]],
       'Bid_MO': [0., [(40, 1.)]]}
Pis["Ask_MO"] = Pis["Bid_MO"]
Pis["Ask_L1"] = Pis["Bid_L1"]
Pis["Ask_inspread"] = Pis["Bid_inspread"]
Pis["Ask_L2"] = Pis["Bid_L2"]
Pi_Q0 = {'Ask_L1': [0., [(200, 1.)]],
         'Ask_L2': [0., [(200, 1.)]],
         'Bid_L1': [0., [(200, 1.)]],
         'Bid_L2': [0., [(200, 1.)]]}

kwargs = {
    "TradingAgent": [],
    "GymTradingAgent": [{
        "cash": 1_000_000,
        "cashlimit": 1_000_000_000,
        "strategy": "TWAP",
        "on_trade": False,
        "total_order_size": Q_TOTAL,
        "order_target": "INTC",
        "total_time": T_EXEC,
        "window_size": 50,            # seconds
        "action_freq": 1,             # => q = Q/(T*f) = 1
        "Inventory": {"INTC": max(2000, Q_TOTAL + 500)},
        "start_trading_lag": START_LAG,
        "side": TWAP_SIDE,
        "off_time": START_LAG + T_EXEC,   # TWAP stops at execution end (t=1300)
        "wake_on_MO": False,
        "wake_on_Spread": False,
    }],
    "Exchange": {"symbol": "INTC", "ticksize": 0.01, "LOBlevels": 2,
                 "numOrdersPerLevel": 10, "PriceMid0": 100, "spread0": 0.03},
    "Arrival_model": {"name": "Hawkes",
                      "parameters": {"kernelparams": kernelparams, "tod": tod, "Pis": Pis,
                                     "beta": 0.941, "avgSpread": 0.0101, "Pi_Q0": Pi_Q0,
                                     "expApprox": True}},
}


def midprice_of(obs):
    return float((obs.get('LOB0').get('Ask_L1')[0] + obs.get('LOB0').get('Bid_L1')[0]) / 2)


def run():
    print(f"[twap-alone] side={TWAP_SIDE} Q={Q_TOTAL} T={T_EXEC} stop={STOP_TIME} "
          f"episodes={EPISODES} -> {OUT_DIR}", flush=True)
    saved_times = None
    for episode in range(EPISODES):
        kwargs["GymTradingAgent"][0]["side"] = TWAP_SIDE
        kwargs["GymTradingAgent"][0]["Inventory"] = {"INTC": max(2000, Q_TOTAL + 500)}
        kwargs["GymTradingAgent"][0]["cash"] = 1_000_000

        env = tradingEnv(stop_time=STOP_TIME, wall_time_limit=23400, **kwargs)
        Simstate, observations, termination, truncation = env.step(action=None)

        ep_times = []
        ep_mid = []
        while Simstate["Done"] is False and termination is not True:
            AgentsIDs = [k for k, v in Simstate["Infos"].items() if v is True]
            agents = [env.getAgent(ID=aid) for aid in AgentsIDs]
            for agent in agents:
                mid = midprice_of(observations)
                ep_times.append(float(Simstate["TimeCode"]))
                ep_mid.append(mid)
                agentAction = agent.get_action(data=env.getobservations(agentID=agent.id))
                Simstate, observations, termination, truncation = env.step(action=(agent.id, agentAction))

        ep_times = np.array(ep_times)
        ep_mid = np.array(ep_mid)
        np.save(os.path.join(OUT_DIR, f"pricepath_ep{episode}.npy"), ep_mid)
        if saved_times is None or len(ep_times) > len(saved_times):
            saved_times = ep_times
            np.save(os.path.join(OUT_DIR, "times.npy"), saved_times)
        print(f"[ep {episode}] steps={len(ep_mid)} t=[{ep_times.min():.0f},{ep_times.max():.0f}] "
              f"mid0={ep_mid[0]:.3f} midEnd={ep_mid[-1]:.3f}", flush=True)
    print("[twap-alone] done.", flush=True)


if __name__ == "__main__":
    run()
