import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *

import torch
import random

# log_dir = '/home/ajafree/untrained_rl_testing/outputs'
# log_dir = '/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/outputs/'
# model_dir = '/home/ajafree/testing_adversarial/models'
# model_dir = '/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/models/'
log_dir = '/home/ajafree/LSTM_fRL/with_expo/testing/logs/'
model_dir = '/home/ajafree/LSTM_fRL/with_expo/testing/model'

start_trading_lag = 100
twap_off_time = 400

# Network architecture parameters - MUST match training
layer_widths = 50
n_layers = 3

label = f'test_LSTM_fRL_52withExpo'

checkpoint_params = ("20251115_164214_train_LSTMRLAgent_vs_TWAP_withExpo", 52)

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
                         "rewardpenalty": 1,
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
                                     'expApprox' : True}}
}

agents = kwargs['GymTradingAgent']
j = agents[0]
tc = 0.0001
RLagentInstance = AdversarialPPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = j["rewardpenalty"], epochs = 100, transaction_cost=1e-4, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=0.5, truncation_enabled=False, action_space_config = 1, alt_state=True, enhance_state=True, include_time=True, optim_type='ADAM',entropy_coef=0, exploration_bonus = 0, TWAPPresent=0 , hidden_activation='sigmoid', typeNN = "LSTM")


# RLagentInstance = PPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
#                               wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
#                               layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = 1e-4, epochs = 1000, transaction_cost=tc, start_trading_lag = j['start_trading_lag'],
#                               gae_lambda=0.5, truncation_enabled=False, action_space_config = 1, alt_state=True, include_time=False, optim_type='ADAM',entropy_coef=0,lr=1e-5)

j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents
i_eps=0
# cash, inventory, t, actions = [], [], [], []
t, t_with_twap_buy, t_with_twap_sell, t_without_twap = [], [], [], []
avgEpisodicRewards, stdEpisodicRewards, finalcash, finalcash2, profit_with_twap_sell, profit_with_twap_buy, profit_without_twap = [], [], [], [], [], [], []
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}
start_midprices = []
twap_agent_executions_by_episode:Dict[int, List] = {}
RLagentID = 1

# start_times = np.load("/home/ajafree/twap_testing_final/start_times.npy")



inventory_without_twap = []
inventory_with_twap_sell = []
inventory_with_twap_buy = []


percentage_of_volume = []
total_TWAP_obsv = []
total_RL_obsv = []


final_cashs = []
total_executeds = []

# Data structures for inventory time series plotting
all_episode_inventories = []
all_episode_times = []

# Data structures for slippage tracking
sell_slippage_by_episode = []  # Store (episode_num, slippage) for sell episodes
buy_slippage_by_episode = []   # Store (episode_num, slippage) for buy episodes

# Data structures for phase-based Sharpe ratio tracking
# Each list stores tuples of (episode_num, sharpe_ratio, side)
sharpe_pre_twap = []    # Sharpe before TWAP starts
sharpe_during_twap = [] # Sharpe during TWAP execution
sharpe_post_twap = []   # Sharpe after TWAP ends

# Data structure for inventory statistics by phase
# Each entry: (episode, side, phase, mean, std, min, max, median)
inventory_stats_by_phase = []

# Data structures for TWAP execution price tracking
all_episode_twap_prices = []  # List of price arrays for each episode
all_episode_twap_times = []   # List of time arrays for each episode
all_episode_twap_start_prices = []  # Starting price for each episode

def plot_inventory_timeseries(episode_num, all_inventories, all_times, twap_start, twap_stop, stop_time, side, save_dir, label_prefix):
    """
    Plot inventory time series for all episodes with individual traces (alpha=0.1) and average overlay.
    
    Args:
        episode_num: Current episode number
        all_inventories: List of inventory arrays for each episode
        all_times: List of time arrays for each episode
        twap_start: Start time of TWAP metaorder
        twap_stop: Stop time of TWAP metaorder (absolute time, not duration)
        stop_time: End time of simulation
        side: 'buy' or 'sell' - determines sign adjustment
        save_dir: Directory to save the plot
        label_prefix: Prefix for the saved filename
    """
    if len(all_inventories) == 0:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Determine sign multiplier
    sign_multiplier = -1 if side == 'sell' else 1
    
    # Add background rectangles for phases
    # Pre-metaorder phase (0 to twap_start)
    plt.axvspan(0, twap_start, alpha=0.15, color='lightblue', label='Pre-metaorder')
    # During metaorder phase (twap_start to twap_stop)
    plt.axvspan(twap_start, twap_stop, alpha=0.15, color='lightcoral', label='During metaorder')
    # Post-metaorder phase (twap_stop to stop_time)
    plt.axvspan(twap_stop, stop_time, alpha=0.15, color='lightgreen', label='Post-metaorder')
    
    # Plot individual episode trajectories with low alpha
    for i, (times, inventories) in enumerate(zip(all_times, all_inventories)):
        adjusted_inventory = np.array(inventories) * sign_multiplier
        plt.plot(times, adjusted_inventory, alpha=0.1, color='gray', linewidth=0.8)
    
    # Compute and plot average trajectory
    # Find common time grid (use the longest episode's time array)
    max_len = max(len(t) for t in all_times)
    max_time = max(t[-1] for t in all_times)
    
    # Create interpolated inventories on a common time grid
    common_times = np.linspace(0, max_time, max_len)
    interpolated_inventories = []
    
    for times, inventories in zip(all_times, all_inventories):
        # Interpolate each episode onto common time grid
        adjusted_inventory = np.array(inventories) * sign_multiplier
        interp_inv = np.interp(common_times, times, adjusted_inventory)
        interpolated_inventories.append(interp_inv)
    
    # Compute average across episodes
    avg_inventory = np.mean(interpolated_inventories, axis=0)
    
    # Plot average with full opacity and thicker line
    plt.plot(common_times, avg_inventory, alpha=1.0, color='darkblue', linewidth=2.5, label=f'Average (n={len(all_inventories)})')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Inventory (sign-adjusted)', fontsize=12)
    plt.title(f'RL Agent Inventory Time Series - Episodes 1-{episode_num+1} ({side.capitalize()} Side)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{label_prefix}_inventory_timeseries_ep{episode_num+1}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved inventory time series plot to {save_path}")

def plot_twap_execution_prices(episode_num, all_prices, all_times, all_start_prices, side, save_dir, label_prefix):
    """
    Plot TWAP execution prices relative to start price with individual traces (alpha=0.1) and average overlay.
    
    Args:
        episode_num: Current episode number
        all_prices: List of price arrays for each episode (relative to start)
        all_times: List of time arrays for each episode
        all_start_prices: List of starting prices for each episode
        side: 'buy' or 'sell' - determines sign adjustment
        save_dir: Directory to save the plot
        label_prefix: Prefix for the saved filename
    """
    if len(all_prices) == 0:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Determine sign multiplier (up = bad, down = good)
    # For buy: price increase is bad (positive), so multiply by +1
    # For sell: price decrease is bad (negative), so multiply by -1 to flip it to positive
    sign_multiplier = 1 if side == 'buy' else -1
    
    # Plot individual episode trajectories with low alpha
    for i, (times, prices) in enumerate(zip(all_times, all_prices)):
        adjusted_prices = np.array(prices) * sign_multiplier
        plt.plot(times, adjusted_prices, alpha=0.1, color='gray', linewidth=0.8)
    
    # Compute and plot average trajectory
    if len(all_times) > 0:
        max_len = max(len(t) for t in all_times)
        max_time = max(t[-1] for t in all_times if len(t) > 0)
        
        # Create interpolated prices on a common time grid
        common_times = np.linspace(0, max_time, max_len)
        interpolated_prices = []
        
        for times, prices in zip(all_times, all_prices):
            if len(times) > 0 and len(prices) > 0:
                adjusted_prices = np.array(prices) * sign_multiplier
                interp_prices = np.interp(common_times, times, adjusted_prices)
                interpolated_prices.append(interp_prices)
        
        if len(interpolated_prices) > 0:
            # Compute average across episodes
            avg_prices = np.mean(interpolated_prices, axis=0)
            
            # Plot average with full opacity and thicker line
            plt.plot(common_times, avg_prices, alpha=1.0, color='darkred', linewidth=2.5, label=f'Average (n={len(all_prices)})')
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Start Price')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Execution Price - Start Price (sign-adjusted)', fontsize=12)
    plt.title(f'TWAP Execution Prices Relative to Start - Episodes 1-{episode_num+1} (Buy & Sell, Sign-Adjusted)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{label_prefix}_twap_exec_prices_ep{episode_num+1}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved TWAP execution price plot to {save_path}")

for episode in range(50):
    RL_agent_obsv = []
    TWAP_agent_obsv = []
    total_executed = 0
    final_cash = 0
    twap_side = np.random.choice(["buy", "sell"])
    kwargs["GymTradingAgent"][1]["Inventory"] = {"INTC": 500}
    kwargs["GymTradingAgent"][1]["cash"] = 1000000
    kwargs["GymTradingAgent"][1]["side"] = twap_side
    starting_midprice = 0
    new_midprice = True
    if(isinstance(RLagentInstance, AdversarialPPOAgent)):
        RLagentInstance.TWAPPresent = False

    twap_time = 250
    kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_time

    twap_agent_executions_by_episode[episode] = []
    episode_twap_exec_prices = []  # Track execution prices for this episode
    episode_twap_exec_times = []   # Track execution times for this episode
    i = 0
    action_num = 0
    env=tradingEnv(stop_time=550, wall_time_limit=23400, **kwargs)
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
    TWAPagentid = 0
    prev_inventory = 500
    while Simstate["Done"]==False and termination!=True:
        counter_profit +=1
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        
        # Update TWAPPresent based on time window (matches trainer logic)
        if isinstance(RLagentInstance, AdversarialPPOAgent):
            if twap_off_time >= Simstate['TimeCode'] >= twap_time:
                if not RLagentInstance.TWAPPresent:
                    RLagentInstance.TWAPPresent = -1 if twap_side == 'sell' else 1
            else:
                RLagentInstance.TWAPPresent = 0
        
        # action:list[Tuple] = []
        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
            #check if agent is an RL agent or not
            
            if not isinstance(agent, PPOAgent):
                if(new_midprice):
                    starting_midprice = float((observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)
                    start_midprices.append(starting_midprice)
                    new_midprice = False
                action_num+=1
                agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
                action = (agent.id, agentAction)
                print(f"TWAP Action: {action}")
                print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
                print(f"TWAP Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")

                TWAPagentid = agent.id
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                observationsDict.update({agent.id:observations})

                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})

                diff = abs(inventories[agent.id][-1] - prev_inventory)

                if(diff != 0):
                    #inventory has changed, order has gone through
                    percentage_of_volume.append(diff/observations["market_volume"])
                    if twap_side == 'sell':
                        exec_price = observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1')[0]
                        twap_agent_executions_by_episode[episode].append((exec_price, diff, twap_side))
                    else:
                        exec_price = observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1')[0]
                        twap_agent_executions_by_episode[episode].append((exec_price, diff, twap_side))
                    
                    # Track execution price relative to start price and time
                    episode_twap_exec_prices.append(exec_price - starting_midprice)
                    episode_twap_exec_times.append(Simstate['TimeCode'])


                prev_inventory = observations['Inventory']
                total_executed = abs(500 - agent.Inventory["INTC"])
                final_cash = agent.cash #if we make the twap stop trading early, this will have to be changed
                
            else:
                action_num+=1
                RLagentID = agent.id
                agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id), epsilon = 0.5 if i_eps < 100 else 0.1)
                action = (agent.id, (agentAction[0],1))
                print(f"RL Action: {action}")
                print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
                print(f"RL Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")
                observations_prev = observationsDict[agent.id].copy() if i != 0 else observations.copy()
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                if(Simstate["TimeCode"] >= twap_time):
                    if twap_side == "sell":
                        inventory_with_twap_sell.append(observations["Inventory"])
                    else:
                        inventory_with_twap_buy.append(observations["Inventory"])
                else:
                    inventory_without_twap.append(observations["Inventory"])
                observationsDict.update({agent.id:observations})
                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
                t += [Simstate['TimeCode']]
                if 'test' in label:
                    observations['current_time'] = 100+((observations['current_time'] - 100)%300)
                # agent.appendER((agent.readData(observations_prev), agentAction, agent.calculaterewards(termination), agent.readData(observations_prev), (termination or truncation)))
                agent.store_transition(episode, agent.readData(observations_prev), agentAction[1], agent.calculaterewards(termination), agent.readData(observations), (termination or truncation))
                print(f'Current reward: {agent.calculaterewards(termination):0.4f}')
                # print(f'Prev avg reward: {np.mean([r[2] for r in agent.experience_replay[-100:]]):0.4f}')
                i_eps+=1
                logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                if len(t) > 0 and Simstate['TimeCode'] < t[-1]:
                    # Episode has reset - mark boundary
                    episode_boundaries.append(len(cashs[agent.id]))
                # Calculate current PnL (cash + inventory value)
                current_pnl = cashs[agent.id][-1] + inventories[agent.id][-1] * agent.mid * (1 - tc*np.sign(inventories[agent.id][-1]))
                finalcash2.append(current_pnl)

                if(Simstate['TimeCode'] >= twap_time):
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
                np.save(log_dir + label + '_profit', np.array([t, finalcash2]))
                # Save buy/sell profits with their respective time arrays
                np.save(log_dir+label+"_profit_w_twap_buy", np.array([t_with_twap_buy, profit_with_twap_buy]))
                np.save(log_dir+label+"_profit_w_twap_sell", np.array([t_with_twap_sell, profit_with_twap_sell]))
                np.save(log_dir+label+"_profit_wout_twap", np.array([t_without_twap, profit_without_twap]))
                if TWAPagentid in observationsDict:
                    TWAP_agent_obsv.append(observationsDict[TWAPagentid])
                RL_agent_obsv.append(observationsDict[RLagentID])
                
            
            print(agent.current_time)
            print(f"ACTION DONE{action_num}")
    
    total_RL_obsv.append(RL_agent_obsv)
    total_TWAP_obsv.append(TWAP_agent_obsv)

    # Get TWAP agent's final cash (not RL agent's cash)
    twap_agent = env.getAgent(ID=TWAPagentid)
    
    if(twap_side == "sell"):
        cash_earned = twap_agent.cash - 1000000
        benchmark_earned = starting_midprice * total_executed
        slip = (benchmark_earned - cash_earned)*10000/benchmark_earned
        sell_slippage_by_episode.append((episode, slip))
    else:
        cash_spent = 1000000 - twap_agent.cash 
        benchmark_spent = starting_midprice * total_executed
        slip = (cash_spent - benchmark_spent)*10000/benchmark_spent
        buy_slippage_by_episode.append((episode, slip))

    inventory_and_cash = ()
    final_cashs.append(final_cash)
    total_executeds.append(total_executed)

    # Calculate phase-based Sharpe ratios for this episode
    if RLagentID in cashs and len(cashs[RLagentID]) > 0:
        # Get the starting index for this episode
        if len(all_episode_inventories) > 0 and len(episode_boundaries) > len(all_episode_inventories):
            start_idx = episode_boundaries[len(all_episode_inventories)]
        else:
            start_idx = 0
        
        end_idx = len(cashs[RLagentID])
        
        # Extract episode data
        episode_cash = np.array(cashs[RLagentID][start_idx:end_idx])
        episode_inv = np.array(inventories[RLagentID][start_idx:end_idx])
        episode_times = np.array(t[start_idx:end_idx])
        
        if len(episode_times) > 0:
            # Get agent reference for mid price
            agent_ref = [a for a in agents if isinstance(a, PPOAgent)][0] if any(isinstance(a, PPOAgent) for a in agents) else None
            if agent_ref is not None:
                # Calculate PnL time series for this episode
                episode_pnl_series = episode_cash + episode_inv * agent_ref.mid * (1 - tc * np.sign(episode_inv))
                
                # Split into three phases based on time
                pre_mask = episode_times < twap_time
                during_mask = (episode_times >= twap_time) & (episode_times < twap_off_time)
                post_mask = episode_times >= twap_off_time
                
                # Calculate Sharpe for each phase
                def calc_sharpe(pnl_series):
                    if len(pnl_series) > 1:
                        log_returns = np.diff(np.log(pnl_series))
                        if len(log_returns) > 0 and np.std(log_returns) > 0:
                            return np.mean(log_returns) / np.std(log_returns)
                    return 0.0
                
                sharpe_pre = calc_sharpe(episode_pnl_series[pre_mask]) if np.sum(pre_mask) > 1 else 0.0
                sharpe_during = calc_sharpe(episode_pnl_series[during_mask]) if np.sum(during_mask) > 1 else 0.0
                sharpe_post = calc_sharpe(episode_pnl_series[post_mask]) if np.sum(post_mask) > 1 else 0.0
                
                # Store with episode number and side
                sharpe_pre_twap.append((episode, sharpe_pre, twap_side))
                sharpe_during_twap.append((episode, sharpe_during, twap_side))
                sharpe_post_twap.append((episode, sharpe_post, twap_side))
                
                # Calculate inventory statistics for each phase
                def calc_inv_stats(inv_series, phase_name):
                    if len(inv_series) > 0:
                        return {
                            'episode': episode,
                            'side': twap_side,
                            'phase': phase_name,
                            'mean': np.mean(inv_series),
                            'std': np.std(inv_series),
                            'min': np.min(inv_series),
                            'max': np.max(inv_series),
                            'median': np.median(inv_series),
                            'count': len(inv_series)
                        }
                    return None
                
                # Calculate stats for each phase
                pre_stats = calc_inv_stats(episode_inv[pre_mask], 'pre')
                during_stats = calc_inv_stats(episode_inv[during_mask], 'during')
                post_stats = calc_inv_stats(episode_inv[post_mask], 'post')
                
                # Store statistics
                for stats in [pre_stats, during_stats, post_stats]:
                    if stats is not None:
                        inventory_stats_by_phase.append(stats)
                
                # Save statistics to npy on-the-fly
                npy_path = os.path.join(log_dir, f'{label}_inventory_stats_by_phase_ep{episode+1}.npy')
                np.save(npy_path, np.array(inventory_stats_by_phase, dtype=object), allow_pickle=True)
                print(f"Saved inventory statistics to {npy_path}")

    # Collect inventory time series for this episode
    # Extract the current episode's inventory and time data
    # Get the starting index for this episode
    if len(all_episode_inventories) > 0 and len(episode_boundaries) > len(all_episode_inventories):
        start_idx = episode_boundaries[len(all_episode_inventories)]
    else:
        start_idx = 0
        
    # Get the ending index (current length)
    end_idx = len(inventories[RLagentID])
    
    # Extract this episode's inventory trajectory
    episode_inventory = inventories[RLagentID][start_idx:end_idx]
    
    # Extract corresponding times
    episode_times = t[start_idx:end_idx]
    
    # Store for cumulative plotting
    if len(episode_inventory) > 0:
        all_episode_inventories.append(episode_inventory)
        all_episode_times.append(episode_times)
        
        # Plot inventory time series with all episodes so far
        plot_inventory_timeseries(
            episode_num=episode,
            all_inventories=all_episode_inventories,
            all_times=all_episode_times,
            twap_start=twap_time,
            twap_stop=twap_off_time,
            stop_time=550,  # env stop_time
            side=twap_side,
            save_dir=log_dir,
            label_prefix=label
        )
    
    # Store and plot TWAP execution prices
    if len(episode_twap_exec_prices) > 0:
        all_episode_twap_prices.append(episode_twap_exec_prices)
        all_episode_twap_times.append(episode_twap_exec_times)
        all_episode_twap_start_prices.append(starting_midprice)
        
        # Plot TWAP execution prices
        plot_twap_execution_prices(
            episode_num=episode,
            all_prices=all_episode_twap_prices,
            all_times=all_episode_twap_times,
            all_start_prices=all_episode_twap_start_prices,
            side=twap_side,
            save_dir=log_dir,
            label_prefix=label
        )
    
    if len(episode_inventory) > 0:
        
        # Plot slippage by episode (on-the-fly)
        plt.figure(figsize=(12, 6))
        
        # Plot sell slippage
        if len(sell_slippage_by_episode) > 0:
            sell_episodes, sell_slips = zip(*sell_slippage_by_episode)
            plt.scatter(sell_episodes, sell_slips, color='red', alpha=0.6, s=50, label='Sell Slippage', marker='o')
        
        # Plot buy slippage
        if len(buy_slippage_by_episode) > 0:
            buy_episodes, buy_slips = zip(*buy_slippage_by_episode)
            plt.scatter(buy_episodes, buy_slips, color='blue', alpha=0.6, s=50, label='Buy Slippage', marker='s')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Slippage (bps)', fontsize=12)
        plt.title(f'TWAP Slippage by Episode - Episodes 1-{episode+1} (Buy vs Sell)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(log_dir + label + f'_slippage_by_episode_ep{episode+1}.png', dpi=150)
        plt.close()
        print(f"Saved slippage plot for episode {episode+1}")
        
        # Plot phase-based Sharpe ratios (3 subplots)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        phases = [
            (sharpe_pre_twap, 'Pre-TWAP Sharpe', axes[0]),
            (sharpe_during_twap, 'During-TWAP Sharpe', axes[1]),
            (sharpe_post_twap, 'Post-TWAP Sharpe', axes[2])
        ]
        
        for phase_data, phase_title, ax in phases:
            if len(phase_data) > 0:
                # Separate buy and sell episodes
                buy_eps = [(ep, sr) for ep, sr, side in phase_data if side == 'buy']
                sell_eps = [(ep, sr) for ep, sr, side in phase_data if side == 'sell']
                
                if len(buy_eps) > 0:
                    buy_episodes, buy_sharpes = zip(*buy_eps)
                    ax.scatter(buy_episodes, buy_sharpes, color='blue', alpha=0.6, s=50, label='Buy', marker='s')
                
                if len(sell_eps) > 0:
                    sell_episodes, sell_sharpes = zip(*sell_eps)
                    ax.scatter(sell_episodes, sell_sharpes, color='red', alpha=0.6, s=50, label='Sell', marker='o')
            
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Sharpe Ratio', fontsize=11)
            ax.set_title(phase_title, fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        plt.suptitle(f'RL Agent Sharpe Ratios by Phase - Episodes 1-{episode+1}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(log_dir + label + f'_sharpe_phases_ep{episode+1}.png', dpi=150)
        plt.close()
        print(f"Saved phase Sharpe plot for episode {episode+1}")

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass

    if ((episode) % 4 == 0):
        if ('test' not in label) and ((checkpoint_params is None) or (episode >= 0)):
            for epoch in range(1):
                d_policy_loss, d_value_loss, d_entropy_loss, u_policy_loss, u_value_loss, u_entropy_loss = agent.train(train_logger) #, use_CEM = bool((episode+1) % 4))
                train_logger.save_logs()
            train_logger.plot_losses(show=False, save=True)

        model_manager.save_models(epoch = episode, u = agent.Actor_Critic_u, d= agent.Actor_Critic_d)
    for agent in agents:
        if isinstance(agent, PPOAgent):
            agent.current_time = 0
            agent.istruncated = False
            agent.cash = j['cash']
            agent.Inventory = {"INTC": 0}
            agent.positions = {'INTC':{}}
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

start_midprices_array = np.array(start_midprices)

# executions_data = {}
# for episode, executions in twap_agent_executions_by_episode.items():
#     if executions:
#         executions_data[f"episode_{episode}"] = np.array(executions, dtype=[('price', 'f8'), ('quantity', 'f8'), ('side', 'U4')])

np.save(log_dir + label+ "_inventory_without_twap.npy", np.array(inventory_without_twap))

if(twap_side == "sell"):
    np.save(log_dir + label+ "_inventory_with_twap_sell.npy", np.array(inventory_with_twap_sell))
else:
    np.save(log_dir + label+ "_inventory_with_twap_buy.npy", np.array(inventory_with_twap_buy))

np.save(log_dir+label+"total_executed.npy", np.array(total_executeds))
np.save(log_dir+label+"final_cash.npy", np.array(final_cashs))
np.save(log_dir+label+"twap_observations.npy", np.array(total_TWAP_obsv, dtype=object), allow_pickle=True)
np.save(log_dir+label+"RL_observations.npy", np.array(total_RL_obsv, dtype=object), allow_pickle=True)

np.save(log_dir + label + '_start_midprices.npy', start_midprices_array)
# np.savez(log_dir + label + '_twap_executions.npz', **executions_data)

