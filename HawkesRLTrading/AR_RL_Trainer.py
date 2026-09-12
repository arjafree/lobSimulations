import sys
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Device-Side Assertions (even more detail)

sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations1'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent

import torch
import time

# Run identity is env-overridable so a submission script fully determines its
# own config. Without this the config is whatever ~/lobSimulations happens to be
# checked out to when the job STARTS (not when it is queued), which means only
# one run can safely be queued at a time -- a queued job can dequeue after a
# later pull and silently run the wrong config. Defaults reproduce today's run.
_DEFAULT_RUN = '/home/ajafree/LSTM_fRL/wout_expo/new_value_function/gae_lambda/training/sell'
log_dir = os.environ.get('RUN_LOG_DIR', _DEFAULT_RUN + '/logs/')
model_dir = os.environ.get('RUN_MODEL_DIR', _DEFAULT_RUN + '/model')

start_trading_lag = 100
twap_off_time = 400

twap_side = "sell"

# TWAP_SIDE_MODE: "sell" | "buy" | "alternate" | "random".
#   "alternate" flips buy/sell across TWAP-PRESENT episodes only -- counting all
#   episodes would let the absent ones consume side-slots and skew the balance.
# TWAP_ON / TWAP_OFF: the presence cycle. TWAP_ON episodes with the meta-order
#   active, then TWAP_OFF episodes with it neutered (off_time=0, TWAPPresent
#   pinned to 0 throughout), repeating. Absent episodes force the agent to be a
#   viable market maker standalone rather than only a front-runner.
#     1/0 -> present every episode (today's behaviour, the default)
#     1/1 -> alternating
#     2/1 -> two on, one off (keeps time-against-TWAP at 2/3 rather than 1/2)
TWAP_SIDE_MODE = os.environ.get("TWAP_SIDE_MODE", "sell")
TWAP_ON = int(os.environ.get("TWAP_ON", 1))
TWAP_OFF = int(os.environ.get("TWAP_OFF", 0))
assert TWAP_ON >= 1 and TWAP_OFF >= 0, "TWAP_ON must be >=1 and TWAP_OFF >=0"

# SEED_MODE: "fixed" reproduces today's behaviour exactly (tradingEnv's default
#   seed=1 on every episode). "vary" gives each episode its own seed, so the
#   agent cannot overfit one realisation of the background order flow.
# The seed list is logged and saved so any run can be replayed exactly.
SEED_MODE = os.environ.get("SEED_MODE", "fixed")
SEED_BASE = int(os.environ.get("SEED_BASE", 1000))

# Number of training episodes; env-overridable so a short smoke run can
# validate a new config before committing a multi-day job to it.
N_EPISODES = int(os.environ.get("N_EPISODES", 80))

# --- Arm parameters. Previously hardcoded, which is how all six of the
# 2026-09-07 runs silently launched on the gae_lambda arm (exploration_bonus=0)
# when the plan called for the combined explo_gae arm. Making them env-overridable
# means the arm is explicit per job and travels with the submission script.
#   EXPLORATION_BONUS: explo_gae arm = 0.1, gae_lambda-only arm = 0.
#   GAE_LAMBDA: 0.95 in every arm run to date.
#   EXP_APPROX: exponential-approximation simulator regime (TAU=10 vs 500).
#     ~13x faster but a DIFFERENT simulator regime -- fast iteration only,
#     never mix with expApprox=False results.
EXPLORATION_BONUS = float(os.environ.get("EXPLORATION_BONUS", 0.1))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", 0.95))
EXP_APPROX = os.environ.get("EXP_APPROX", "false").strip().lower() in ("1", "true", "yes")

# --- Reward-shaping arm. Measured on the 2026-09-07 runs: per-step PnL change
# has median EXACTLY 0 (85-88% of steps move no money), mean |dW| 0.006-0.018,
# and an episode terminal PnL of +-0.2 dollars over ~2,480 steps. Against that,
# ACTION_BONUS=0.5 per acted step is worth hundreds per episode and is
# deterministic and always positive, while PnL is near-zero-mean noise. The
# count-based exploration bonus (0.2 first visit, 0.1/sqrt(N) after) beats the
# mean PnL signal until a state-action pair has been seen ~132 times, which
# most never are. So the objective being maximised is "act often, visit novel
# states"; PnL is a rounding error, which is why no arm has ever produced
# positive standalone PnL.
#   ACTION_BONUS: 0.5 reproduces every run to date. ~1e-3 puts the episode
#     total near 0.25, an order of magnitude under the ~2.5 dollars a
#     successful front-run of a 500-share meta-order is worth at 10 bps.
#   RUNNING_INVPENALTY: per-step lambda*inv**2. 0 reproduces today's behaviour,
#     where the ONLY inventory control is a terminal penalty ~2,480 steps away.
ACTION_BONUS = float(os.environ.get("ACTION_BONUS", 0.5))
RUNNING_INVPENALTY = float(os.environ.get("RUNNING_INVPENALTY", 0.0))
ENTROPY_COEF = float(os.environ.get("ENTROPY_COEF", 0.0))

#the time that the TWAP agent will kick in:
twap_start_time = 150 + start_trading_lag

twap_end_time = 300 + start_trading_lag

label = os.environ.get('RUN_LABEL', 'train_new_vf_gae_lambda_sell')
layer_widths=100
n_layers=3
eta = 5

checkpoint_params = None

def graphInventories(beforetwap, withtwap_buy, withtwap_sell, episode_num):
    plt.figure(figsize=(12, 8))
    
    # Flatten the lists of lists to get all inventory values
    all_before = []
    all_buy = []
    all_sell = []
    
    # Flatten beforetwap (list of lists across episodes)
    for episode_inventories in beforetwap:
        all_before.extend(episode_inventories)
    
    # Flatten withtwap_buy (list of lists across episodes) 
    for episode_inventories in withtwap_buy:
        all_buy.extend(episode_inventories)
        
    # Flatten withtwap_sell (list of lists across episodes)
    for episode_inventories in withtwap_sell:
        all_sell.extend(episode_inventories)
    
    # Create normalized histograms (density=True gives probability density)
    # weights parameter normalizes to show ratios/proportions that sum to 1
    if all_before:
        weights_before = np.ones(len(all_before)) / len(all_before)
        plt.hist(all_before, bins=30, alpha=0.7, label=f'Before TWAP (n={len(all_before)})', 
                 color='blue', edgecolor='black', weights=weights_before)
    
    if all_buy:
        weights_buy = np.ones(len(all_buy)) / len(all_buy)
        plt.hist(all_buy, bins=30, alpha=0.7, label=f'With TWAP Buy (n={len(all_buy)})', 
                 color='green', edgecolor='black', weights=weights_buy)
    
    if all_sell:
        weights_sell = np.ones(len(all_sell)) / len(all_sell)
        plt.hist(all_sell, bins=30, alpha=0.7, label=f'With TWAP Sell (n={len(all_sell)})', 
                 color='red', edgecolor='black', weights=weights_sell)
    
    # Add median lines
    if all_before:
        plt.axvline(np.median(all_before), color='blue', linestyle='--', linewidth=2, 
                   label=f'Before Median: {np.median(all_before):.1f}')
    if all_buy:
        plt.axvline(np.median(all_buy), color='green', linestyle='--', linewidth=2,
                   label=f'Buy Median: {np.median(all_buy):.1f}')
    if all_sell:
        plt.axvline(np.median(all_sell), color='red', linestyle='--', linewidth=2,
                   label=f'Sell Median: {np.median(all_sell):.1f}')
    
    plt.xlabel('RL Agent Inventory')
    plt.ylabel('Proportion') 
    plt.title('RL Agent Inventory Distribution: Before vs With TWAP (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir + f'_all_inventory_distributions_episode_{episode_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_avg_inventory_trajectories(buy_trajectories, sell_trajectories, episode_num, save_dir, label_prefix, twap_start, twap_end):
    """Plot avg inventory trajectories colored by training batch (every 4 episodes), separated by buy/sell."""
    import matplotlib.colors
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    cmap = plt.cm.viridis
    max_batch = max(episode_num // 4, 1)

    for ax, trajectories, side_label in zip(axes, [buy_trajectories, sell_trajectories], ['Buy TWAP', 'Sell TWAP']):
        if not trajectories:
            ax.set_title(f'{side_label} - No episodes')
            ax.grid(True, alpha=0.3)
            continue

        ax.axvspan(twap_start, twap_end, alpha=0.1, color='lightcoral', label='TWAP active')

        all_interp = []
        max_time = max(times[-1] for _, times, _ in trajectories if len(times) > 0)
        common_times = np.linspace(0, max_time, 500)

        for ep_num, times, invs in trajectories:
            batch_idx = ep_num // 4
            color = cmap(batch_idx / max_batch)
            ax.plot(times, invs, alpha=0.5, color=color, linewidth=1.5)
            all_interp.append(np.interp(common_times, times, invs))

        if all_interp:
            avg = np.mean(all_interp, axis=0)
            ax.plot(common_times, avg, color='black', linewidth=2.5, label=f'Avg (n={len(trajectories)})')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Inventory')
        ax.set_title(f'{side_label} Episodes (n={len(trajectories)})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_batch)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Training batch (every 4 episodes)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label_prefix}_avg_inv_trajectories_ep{episode_num}.png'), dpi=150)
    plt.close()


# with open("/Users/alirazajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
with open("/home/ajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)

cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
       "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]

faketod = {}
for k in cols:
    faketod[k] = {}
    for k1 in np.arange(13):
        faketod[k][k1] = 1.0
tod=np.zeros(shape=(len(cols), 13))
for i in range(len(cols)):
    tod[i]=[faketod[cols[i]][k] for k in range(13)]
Pis={'Bid_L2': [0.,
                [(40, 1.)]],
     'Bid_inspread': [0.,
                      [(40, 1.)]],
     'Bid_L1': [0.,
                [(40, 1.)]],
     'Bid_MO': [0.,
                [(40, 1.)]]}
Pis["Ask_MO"] = Pis["Bid_MO"]
Pis["Ask_L1"] = Pis["Bid_L1"]
Pis["Ask_inspread"] = Pis["Bid_inspread"]
Pis["Ask_L2"] = Pis["Bid_L2"]
Pi_Q0= {'Ask_L1': [0.,
                   [(400, 1.)]],
        'Ask_L2': [0.,
                   [(400, 1.)]],
        'Bid_L1': [0.,
                   [(400, 1.)]],
        'Bid_L2': [0.,
                   [(400, 1.)]]}

kwargs={
    "TradingAgent": [],
    "GymTradingAgent": [
                        {"cash": 2500,
                         "strategy": "ICRL",
                         "action_freq": 0.213,
                         "rewardpenalty": eta,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": 25,
                         'start_trading_lag': start_trading_lag,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}
                         ,
                         {"cash":1000000,
                          "cashlimit": 100000000000,
                          "strategy": "TWAP",
                          "on_trade":False,
                          "total_order_size":150,
                          "order_target":"INTC",
                          "total_time":150,
                          "window_size":25, #window size, measured in seconds
                          "action_freq":1,
                          "Inventory": {"INTC":500},
                          'start_trading_lag': start_trading_lag,
                          "wake_on_MO": False,
                          "wake_on_Spread": False,
                          "off_time": twap_off_time}
                          ],
    "Exchange": {"symbol": "INTC",
                 "ticksize":0.01,
                 "LOBlevels": 2,
                 "numOrdersPerLevel": 10,
                 "PriceMid0": 100,
                 "spread0": 0.03},
    "Arrival_model": {"name": "Hawkes",
                      "parameters": {"kernelparams": kernelparams,
                                     "tod": tod,
                                     "Pis": Pis,
                                     "beta": 0.941,
                                     "avgSpread": 0.0101,
                                     "Pi_Q0": Pi_Q0,
                                     'expApprox' : EXP_APPROX}}
}

agents = kwargs['GymTradingAgent']
j = agents[0]
tc = 0.0001
RLagentInstance = AdversarialPPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = j["rewardpenalty"], epochs = 100, transaction_cost=1e-4, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=GAE_LAMBDA, gamma=0.999, truncation_enabled=False, action_space_config = 1, alt_state=True, enhance_state=True, include_time=True, optim_type='ADAM',entropy_coef=ENTROPY_COEF, exploration_bonus = EXPLORATION_BONUS, hidden_activation='sigmoid',
                          typeNN = "LSTM", lr = 3e-4, chunk_length=64, TWAPPresent=0, cem_full_episode=True, terminal_invpenalty=5*eta, two_sided_reward=False,
                          action_bonus=ACTION_BONUS, running_invpenalty=RUNNING_INVPENALTY)

# Config banner. The six 2026-09-07 runs could not be told apart from their .o
# files because nothing recorded which arm they were on; this makes every job
# self-documenting.
print("=" * 72, flush=True)
print("RUN CONFIG", flush=True)
print(f"  label              = {label}", flush=True)
print(f"  log_dir            = {log_dir}", flush=True)
print(f"  model_dir          = {model_dir}", flush=True)
print(f"  N_EPISODES         = {N_EPISODES}", flush=True)
print(f"  TWAP_SIDE_MODE     = {TWAP_SIDE_MODE}", flush=True)
print(f"  TWAP_ON/TWAP_OFF   = {TWAP_ON}/{TWAP_OFF}", flush=True)
print(f"  SEED_MODE/BASE     = {SEED_MODE}/{SEED_BASE}", flush=True)
print(f"  exploration_bonus  = {EXPLORATION_BONUS}", flush=True)
print(f"  gae_lambda         = {GAE_LAMBDA}", flush=True)
print(f"  expApprox          = {EXP_APPROX}", flush=True)
print(f"  action_bonus       = {ACTION_BONUS}", flush=True)
print(f"  running_invpenalty = {RUNNING_INVPENALTY}", flush=True)
print(f"  entropy_coef       = {ENTROPY_COEF}", flush=True)
print(f"  eta/rewardpenalty  = {eta}/{j['rewardpenalty']}", flush=True)
print(f"  inventorylimit     = {j['inventorylimit']}", flush=True)
print("=" * 72, flush=True)

inventories_with_twap_buy = []
inventories_with_twap_sell = []
inventories_without_twap = []

j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents
i_eps=0
t, t_with_twap_buy, t_with_twap_sell, t_without_twap = [], [], [], []
avgEpisodicRewards, stdEpisodicRewards, finalcash, finalcash2, profit_with_twap_sell, profit_with_twap_buy, profit_without_twap = [], [], [], [], [], [], []
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}
RLagentID = 1

start_midprices = []
total_executeds = []

TWAP_obsv = []
RL_obsv = []

sides = []
twap_presents = []
episode_seeds = []
n_twap_present = 0

eps_with_buy = []
eps_with_sell = []

total_RL_obsv = []

final_cashs = []
total_executeds = []
episode_inv_trajectories_buy = []
episode_inv_trajectories_sell = []
# Per-episode scalar metrics, rewritten every episode. The arrays that carry the
# primary metric (inventory level inside the TWAP window) are only written when
# the run FINISHES, which is why every in-flight run to date could only be
# judged by eyeballing the avg_inv_trajectories PNGs. This file makes the three
# criteria readable numerically while a job is still running, and keeps
# TWAP-absent episodes separated from the out-of-window parts of present
# episodes -- `profit_without_twap` mixes the two and is not a clean readout.
episode_metrics = []


def _save_episode_metrics():
    import json
    with open(log_dir + "episode_metrics_" + label + ".json", "w") as fh:
        json.dump({"label": label,
                   "exploration_bonus": EXPLORATION_BONUS,
                   "gae_lambda": GAE_LAMBDA,
                   "expApprox": EXP_APPROX,
                   "action_bonus": ACTION_BONUS,
                   "running_invpenalty": RUNNING_INVPENALTY,
                   "entropy_coef": ENTROPY_COEF,
                   "twap_on": TWAP_ON, "twap_off": TWAP_OFF,
                   "twap_side_mode": TWAP_SIDE_MODE,
                   "seed_mode": SEED_MODE, "seed_base": SEED_BASE,
                   "n_episodes_planned": N_EPISODES,
                   "inventorylimit": j["inventorylimit"],
                   "rewardpenalty": j["rewardpenalty"],
                   "episodes": episode_metrics}, fh, indent=1)


def _twap_slippage_bps(side, twap_final_cash, executed, start_mid):
    """TWAP execution cost in bps vs the arrival midprice. Positive = the
    meta-order paid more (buy) or received less (sell) than the arrival mid,
    i.e. higher transaction cost. Criterion 3 wants this to go UP versus the
    TWAP-alone baseline. Same convention as AR_RL_runner.py:605-613."""
    # np.nan is the sentinel total_executed carries on TWAP-absent episodes, and
    # `not nan` is False -- guard on non-finiteness explicitly or nan propagates
    # into every downstream mean.
    if executed is None or start_mid is None:
        return None
    if not (np.isfinite(executed) and np.isfinite(start_mid)):
        return None
    benchmark = start_mid * executed
    if benchmark <= 0:
        return None
    if side == "sell":
        return (benchmark - (twap_final_cash - 1000000)) * 10000 / benchmark
    return ((1000000 - twap_final_cash) - benchmark) * 10000 / benchmark


for episode in range(N_EPISODES):
    inventory_with_twap_buy = []
    inventory_with_twap_sell = []
    inventory_without_twap = []
    episode_times_rl = []
    episode_invs_rl = []

    RL_agent_obsv = []
    TWAP_agent_obsv = []

    twap_diff = 0
    starting_midprice = 0
    new_midprice = True
    kwargs["GymTradingAgent"][1]["Inventory"] = {"INTC": 500}
    kwargs["GymTradingAgent"][1]["cash"] = 1000000

    # Is the TWAP meta-order present at all this episode?
    twap_present = (episode % (TWAP_ON + TWAP_OFF)) < TWAP_ON
    if TWAP_SIDE_MODE == "alternate":
        # Indexed by how many TWAP-present episodes have already run, so the
        # buy/sell split stays exact regardless of the presence cycle.
        twap_side = "buy" if (n_twap_present % 2 == 0) else "sell"
    elif TWAP_SIDE_MODE == "random":
        twap_side = str(np.random.choice(["buy", "sell"]))
    else:
        twap_side = TWAP_SIDE_MODE
    if twap_present:
        n_twap_present += 1
    twap_presents.append(twap_present)
    sides.append(twap_side if twap_present else "none")
    if twap_present:
        eps_with_buy.append(episode) if twap_side == "buy" else eps_with_sell.append(episode)
    # off_time=0 neuters the TWAP: it still wakes on its 1s clock (so the
    # sampling cadence and starting_midprice capture are unchanged) but returns
    # (12,0) every time and never trades.
    kwargs["GymTradingAgent"][1]["off_time"] = twap_off_time if twap_present else 0
    kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_start_time
    #randomise buy or sell
    kwargs["GymTradingAgent"][1]["side"] = twap_side
    i = 0
    action_num = 0
    episode_seed = (SEED_BASE + episode) if SEED_MODE == "vary" else 1
    episode_seeds.append(episode_seed)
    env=tradingEnv(stop_time=550, wall_time_limit=23400, seed=episode_seed, **kwargs)
    print(f"Start of episode {episode}. TWAP present: {twap_present}, side: "
          f"{twap_side if twap_present else 'none'}, seed: {episode_seed}")
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    if episode == 0:
        for agent in agents:
            if isinstance(agent, PPOAgent):
                agent.setupNNs(observations)
        if checkpoint_params is not None:
            loaded_models = model_manager.load_models(timestamp=checkpoint_params[0], epoch = checkpoint_params[1], d = agent.Actor_Critic_d, u = agent.Actor_Critic_u)
            agent.Actor_Critic_d = loaded_models['d']
            agent.Actor_Critic_u = loaded_models['u']
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    episode_start_time = time.time()
    while Simstate["Done"]==False and termination!=True:
        counter_profit +=1
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        # action:list[Tuple] = []
        if twap_present and (twap_end_time >= Simstate['TimeCode'] >= twap_start_time):
            if not RLagentInstance.TWAPPresent:
                RLagentInstance.TWAPPresent = -1 if twap_side == 'sell' else 1
        else:
            RLagentInstance.TWAPPresent = 0

        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
            #check if agent is an RL agent or not
            
            if not isinstance(agent, PPOAgent):
                if(new_midprice):
                    starting_midprice = float((observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)
                    new_midprice = False
                action_num+=1
                agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
                action = (agent.id, agentAction)
                print(f"Action: {action}")
                
                print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
                print(f"Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")
                
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                observationsDict.update({agent.id:observations})
                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})

                # On a TWAP-absent episode the meta-order never trades, so
                # total_executed is 0 and any slippage derived from it would
                # divide by zero. Record NaN so downstream analysis skips the
                # episode instead of silently producing inf/garbage.
                total_executed = abs(500 - agent.Inventory["INTC"]) if twap_present else np.nan
                final_cash = agent.cash if twap_present else np.nan
                   
            else:
                print(f"Twap present: {RLagentInstance.TWAPPresent}")
                action_num+=1
                RLagentID = agent.id
                agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id), epsilon = 0.5 if i_eps < 100 else 0.1)
                # Snapshot the state the action was actually chosen from (set inside get_action).
                # Using this instead of prev_readData ensures the stored (s, a) pair is aligned —
                # critical when other agents (TWAP) act between RL steps and shift env state.
                state_at_action = agent.last_state.clone() if agent.last_state is not None else None
                action = (agent.id, (agentAction[0],1))
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                episode_times_rl.append(Simstate['TimeCode'])
                episode_invs_rl.append(observations["Inventory"])
                # Key on ACTUAL TWAP presence, not just the clock window: on a
                # TWAP-absent episode the [twap_start, twap_end] window is
                # ordinary market-making time and must not be filed as
                # "with TWAP", which would contaminate the very comparison this
                # design exists to make.
                if twap_present and (twap_end_time > Simstate['TimeCode'] > twap_start_time):
                    if twap_side == "sell":
                        inventory_with_twap_sell.append(observations["Inventory"])
                    else:
                        inventory_with_twap_buy.append(observations["Inventory"])
                else:
                    inventory_without_twap.append(observations["Inventory"])
                observationsDict.update({agent.id:observations})
                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                if len(t) > 0 and Simstate['TimeCode'] < t[-1]:
                    # Episode has reset - mark boundary BEFORE appending the new episode's first entry
                    episode_boundaries.append(len(cashs.get(agent.id, [])))
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
                t += [Simstate['TimeCode']]
                current_readData = agent.readData(observations)
                if state_at_action is not None:
                    agent.store_transition(episode, state_at_action, agentAction[1], agent.calculaterewards(termination), current_readData, (termination or truncation))
                print(f'Current reward: {agent.calculaterewards(termination):0.4f}')
                # print(f'Prev avg reward: {np.mean([r[2] for r in agent.experience_replay[-100:]]):0.4f}')
                i_eps+=1
                logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                # Calculate current PnL (cash + inventory value)
                current_pnl = cashs[agent.id][-1] + inventories[agent.id][-1] * agent.mid * (1 - tc*np.sign(inventories[agent.id][-1]))
                finalcash2.append(current_pnl)

                # Same presence-vs-clock fix as the inventory buckets above.
                if twap_present and (twap_end_time >= Simstate['TimeCode'] >= twap_start_time):
                    if twap_side == "buy":
                        profit_with_twap_buy.append(current_pnl)
                        t_with_twap_buy += [Simstate['TimeCode']]
                    else:
                        profit_with_twap_sell.append(current_pnl)
                        t_with_twap_sell += [Simstate['TimeCode']]
                else:
                    profit_without_twap.append(current_pnl)
                    t_without_twap += [Simstate['TimeCode']]
                
                #Sharpe ratio
                all_log_returns = []
                pft = np.array(finalcash2)
                # Calculate log returns for each episode separately
                for i in range(len(episode_boundaries)):
                    start_idx = episode_boundaries[i]
                    end_idx = episode_boundaries[i + 1] if i + 1 < len(episode_boundaries) else len(pft)

                    if end_idx - start_idx > 1:  # Need at least 2 points for log returns
                        episode_pnl = pft[start_idx:end_idx]
                        episode_log_returns = np.diff(np.log(episode_pnl))
                        all_log_returns.extend(episode_log_returns)

                # Calculate Sharpe on concatenated log returns from all episodes
                if len(all_log_returns) > 0:
                    all_log_returns = np.array(all_log_returns)
                    sr = np.mean(all_log_returns) / np.std(all_log_returns) if np.std(all_log_returns) > 0 else 0
                else:
                    sr = 0

                # Plotting logic
                if (counter_profit % 100 == 0):
                    plt.figure(figsize=(12, 8))

                    # Create episode-aware plotting
                    pft = np.array(finalcash2)
                    t_array = np.array(t)


                    # Plot each episode as a separate line
                    for i in range(len(episode_boundaries)):
                        start_idx = episode_boundaries[i]
                        end_idx = episode_boundaries[i + 1] if i + 1 < len(episode_boundaries) else len(pft)

                        if end_idx > start_idx:  # Valid episode
                            episode_t = t_array[start_idx:end_idx]
                            episode_pnl = pft[start_idx:end_idx]
                            episode_profit = episode_pnl - j["cash"] # Profit relative to starting capital

                            # Plot this episode
                            if i == len(episode_boundaries) - 1:
                                plt.plot(episode_t, episode_profit, alpha=0.7, label=f'Sharpe:{sr:0.4f}', marker = 'X')
                            else:
                                plt.plot(episode_t, episode_profit, alpha=0.7)

                plt.legend()
                plt.ticklabel_format(useOffset=False, style='plain')
                plt.xlabel('Time in seconds')
                plt.ylabel('Profit in Dollars')
                plt.title('Final Profit - All Episodes Overlaid')
                plt.savefig(log_dir + label + '_profit.png')

                np.save(log_dir + "sharpe_" + label + '_profit', np.array([t, finalcash2]))
                if len(t_with_twap_buy) > 0:
                    np.save(log_dir + "sharpe_" + label + "_profit_w_twap_buy", np.array([t_with_twap_buy, profit_with_twap_buy]))
                if len(t_with_twap_sell) > 0:
                    np.save(log_dir + "sharpe_" + label + "_profit_w_twap_sell", np.array([t_with_twap_sell, profit_with_twap_sell]))
                np.save(log_dir + "sharpe_" + label + "_profit_wout_twap", np.array([t_without_twap, profit_without_twap]))

                RL_agent_obsv.append(observationsDict[RLagentID])
            
            print(agent.current_time)
            print(f"ACTION DONE{action_num}")
    
    total_RL_obsv.append(RL_agent_obsv)

    if len(inventory_with_twap_buy) > 0:
        inventories_with_twap_buy.append(inventory_with_twap_buy)
        if len(episode_times_rl) > 0:
            episode_inv_trajectories_buy.append((episode, episode_times_rl, episode_invs_rl))
    if len(inventory_with_twap_sell) > 0:
        inventories_with_twap_sell.append(inventory_with_twap_sell)
        if len(episode_times_rl) > 0:
            episode_inv_trajectories_sell.append((episode, episode_times_rl, episode_invs_rl))
    inventories_without_twap.append(inventory_without_twap)

    final_cashs.append(final_cash)
    total_executeds.append(total_executed)
    start_midprices.append(starting_midprice)

    # Terminal PnL straight off this episode's last sample, so no episode
    # boundary has to be reconstructed from non-monotonic time downstream.
    _term_pnl = (finalcash2[-1] - j["cash"]) if len(finalcash2) else None
    _inw = inventory_with_twap_sell if twap_side == "sell" else inventory_with_twap_buy
    episode_metrics.append({
        "episode": episode,
        "twap_present": bool(twap_present),
        "side": twap_side if twap_present else "none",
        "seed": episode_seed,
        # PRIMARY metric for criterion 1: mean inventory LEVEL inside the TWAP
        # window. Want > 0 for a buying TWAP, < 0 for a selling TWAP. Level, not
        # the paired shift -- they disagree and level is the one that asks
        # whether the agent actually HOLDS the profitable position.
        "inv_level_in_window": float(np.mean(_inw)) if (twap_present and len(_inw)) else None,
        "n_in_window": int(len(_inw)) if twap_present else 0,
        "inv_level_out_window": float(np.mean(inventory_without_twap)) if len(inventory_without_twap) else None,
        "n_out_window": int(len(inventory_without_twap)),
        "inv_terminal": float(inventory_without_twap[-1]) if len(inventory_without_twap) else None,
        # Criterion 2: terminal PnL. On a TWAP-absent episode this is the clean
        # standalone-market-maker readout.
        "terminal_pnl": float(_term_pnl) if _term_pnl is not None else None,
        # Criterion 3: TWAP execution cost. None on absent episodes.
        "twap_slippage_bps": _twap_slippage_bps(twap_side, final_cash, total_executed, starting_midprice)
            if twap_present else None,
        "twap_total_executed": float(total_executed) if twap_present else None,
        "start_midprice": float(starting_midprice) if starting_midprice else None,
        "episode_seconds": None,
    })

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass
    episode_total_time = time.time() - episode_start_time
    print(f"Episode took {episode_total_time} seconds to run")
    if episode_metrics:
        episode_metrics[-1]["episode_seconds"] = float(episode_total_time)
    _save_episode_metrics()

    if ((episode) % 4 == 0):
        if ('test' not in label) and ((checkpoint_params is None) or (episode >= 0)):
            for epoch in range(1):
                start_time = time.time()
                d_policy_loss, d_value_loss, d_entropy_loss, u_policy_loss, u_value_loss, u_entropy_loss = agent.train(train_logger, use_CEM = bool((episode) % 8) and (episode >= 10))
                train_time = time.time() - start_time
                # store timing on the train_logger (create list if necessary)
                print(f"Agent.train took {train_time:.4f}s for episode {episode}")
                train_logger.save_logs()
            train_logger.plot_losses(show=False, save=True)

        model_manager.save_models(epoch = episode, u = RLagentInstance.Actor_Critic_u, d= RLagentInstance.Actor_Critic_d)
        plot_avg_inventory_trajectories(episode_inv_trajectories_buy, episode_inv_trajectories_sell,
                                        episode, log_dir, label, twap_start_time, twap_end_time)
    for agent in agents:
        if isinstance(agent, PPOAgent):
            agent.current_time = 0
            agent.istruncated = False
            agent.cash = j['cash']
            agent.Inventory = {"INTC": 0}
            agent.positions = {'INTC':{}}
            agent.last_state = None  # Triggers LSTM reset on next get_action()
            agent.breach = False  # Clear stale breach flag; otherwise a prior episode that ended over the inventory limit forces an early-return MO on the new episode's first action (with last_state=None)
            agent.profit = 0
            agent.statelog = [(0, agent.cash, agent.profit, agent.Inventory.copy(), agent.positions.copy(), agent.mid)]
            j['agent_instance'] = agent
            kwargs['GymTradingAgent'][0] = j



    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(np.arange(len(cashs[(RLagentID)])), cashs[(RLagentID)])
    plt.title('Cash')
    plt.subplot(222)
    plt.plot(np.arange(len(cashs[(RLagentID)])), inventories[(RLagentID)])
    plt.title('Inventory')
    plt.subplot(223)
    plt.scatter(np.arange(len(cashs[(RLagentID)])), actionss[(RLagentID)])
    plt.yticks(np.arange(0,13), agent.actions)
    plt.title('Actions')
    plt.savefig(log_dir + label+'_policy.png')
    episodic_rewards = []
    r=0
    tmp = agent.trajectory_buffer[0][0]
    for ij in agent.trajectory_buffer:
        if ij[0] == tmp:
            r+=ij[1][3]
        else:
            episodic_rewards.append(r)
            r = ij[1][3]
            tmp=ij[0]
    avgEpisodicRewards.append(np.mean(episodic_rewards[-4:]))
    stdEpisodicRewards.append(np.std(episodic_rewards[-4:]))
    finalcash.append(cashs[(RLagentID)][-1] + inventories[(RLagentID)][-1]*agent.mid )
    pft = np.array(finalcash) - j["cash"]
    ma = np.convolve(pft, np.ones(5)/5, mode='valid')

    np.save(log_dir+label+"_trajectory_buffer", np.array(agent.trajectory_buffer, dtype=object), allow_pickle=True)

    plt.figure(figsize=(12,8))
    plt.subplot(311)
    plt.plot(np.arange(len(avgEpisodicRewards)),avgEpisodicRewards)
    plt.fill_between(np.arange(len(avgEpisodicRewards)),np.array(avgEpisodicRewards) - np.array(stdEpisodicRewards),np.array(avgEpisodicRewards) + np.array(stdEpisodicRewards), alpha=0.3  )
    plt.title('Moving Avg Episodic Rewards')
    plt.subplot(312)
    plt.plot(np.arange(len(ma)), ma)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Final Profit MA')
    plt.subplot(313)
    plt.plot(np.arange(len(pft)), pft)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Final Profit Raw')

    plt.savefig(log_dir + label+'_avgepisodicreward.png')
    torch.cuda.empty_cache()
    # torch.mps.empty_cache()

np.save(log_dir + "inventorydists_" + label+ "_inventory_without_twap.npy", np.array(inventories_without_twap, dtype=object), allow_pickle=True)

if len(inventories_with_twap_sell) > 0:
    np.save(log_dir + "inventorydists_" + label+ "_inventory_with_twap_sell.npy", np.array(inventories_with_twap_sell, dtype=object), allow_pickle=True)
if len(inventories_with_twap_buy) > 0:
    np.save(log_dir + "inventorydists_" + label+ "_inventory_with_twap_buy.npy", np.array(inventories_with_twap_buy, dtype=object), allow_pickle=True)

np.save(log_dir + "slippages_"+label+"episode_seeds.npy", np.array(episode_seeds))
np.save(log_dir + "slippages_"+label+"twap_present.npy", np.array(twap_presents))
np.save(log_dir + "slippages_"+label+"sides.npy", np.array(sides))
np.save(log_dir + "slippages_"+label+"total_executed.npy", np.array(total_executeds, dtype=float))
np.save(log_dir + "slippages_"+label+"final_cash.npy", np.array(final_cashs))
np.save(log_dir + "slippages_"+label+"start_midprice.npy", np.array(start_midprices, dtype=float))
np.save(log_dir+label+"RL_observations.npy", np.array(total_RL_obsv, dtype=object), allow_pickle=True)
